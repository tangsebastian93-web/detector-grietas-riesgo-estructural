import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import mysql.connector as mysql  # type: ignore[import-not-found]
except ImportError:
    mysql = None


st.set_page_config(page_title="Detector Multimodal de Grietas", page_icon="🔍", layout="wide")

DB_CONFIG = {
    "host": "localhost", "port": 3306, "user": "root", "password": "",
    "database": "detector_grietas",
}

# ------------------------------------------------------------------
# Parametros de deteccion y clasificacion
# ------------------------------------------------------------------
SIGMAS = (1.0, 1.5, 2.0, 3.0)        # escalas del filtro de vasos (px de trabajo)
FRANGI_BETA = 0.5                    # sensibilidad a formas no lineales (manchas)
HIGH_PCT = 96.0                      # histeresis: semilla fuerte (percentil de la respuesta)
LOW_PCT = 85.0                       # histeresis: extension debil (percentil de la respuesta)
PRUNE_REL = 0.03                     # ramas del esqueleto mas cortas que esto / lado mayor
MIN_CONTRAST = 0.04                  # contraste minimo en log-ratio para aceptar una lesion
MIN_LENGTH_REL = 0.03                # longitud minima de una lesion / lado mayor
MIN_EXTENT_REL = 0.08                # extension minima (bbox) / lado mayor: descarta manchas compactas
MAX_WIDTH_REL = 0.05                 # ancho maximo plausible / lado mayor (descarta sombras)
BORDER_BAND_REL = 0.02               # banda lateral donde se ignoran respuestas (marco)
FISURA_MAX_MM = 1.0                  # calibrado: ancho < 1 mm = fisura; >= 1 mm = grieta
FISURA_MAX_REL = 0.004               # sin calibrar: ancho / lado mayor < 0.4 % = fisura
TORT_MAX = 1.6                       # tortuosidad local maxima (longitud/cuerda por ventana)
FOCAL_FACTOR = 0.79                  # distancia focal estimada en px = 0.79 * ancho (camara movil ~65 deg)
COLOR_FISURA = (0, 255, 255)         # BGR amarillo
COLOR_GRIETA = (0, 0, 255)           # BGR rojo
COLOR_EJE = (255, 0, 255)            # BGR magenta


def guardar_inspeccion(distancia, ancho, freq_pico, centroide, ubicacion, diagnostico, riesgo, probabilidad):
    """Guarda una inspeccion; los campos historicos de audio reciben 0.0."""
    if mysql is None:
        st.error("Falta instalar mysql-connector-python.")
        return False
    conn = None
    cursor = None
    try:
        conn = mysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO inspecciones
            (distancia_cm, ancho_mm, frecuencia_hz, centroide, ubicacion,
             diagnostico, riesgo, probabilidad_grieta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (distancia, ancho, freq_pico, centroide, ubicacion, diagnostico, riesgo, probabilidad),
        )
        conn.commit()
        return True
    except Exception as exc:
        st.error(f"Error al guardar en MySQL. Detalles: {exc}")
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


# ------------------------------------------------------------------
# Analisis de imagen
# ------------------------------------------------------------------
def _as_bgr(img_array: np.ndarray) -> np.ndarray:
    image = np.asarray(img_array)
    if image.size == 0:
        raise ValueError("La imagen esta vacia o no se pudo decodificar.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe ser gris, BGR o BGRA.")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("La imagen contiene valores no numericos o no finitos.")
        if float(image.min()) >= 0 and float(image.max()) <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _dark_ridge_response(log_img: np.ndarray) -> np.ndarray:
    """Filtro de vasos (Frangi) para valles oscuros, multiescala.

    Un valle oscuro tiene curvatura fuerte a lo ancho (autovalor grande y
    positivo) y casi nula a lo largo (autovalor pequeno). Las manchas y el
    ruido de textura tienen curvatura en ambas direcciones y se atenuan.
    """
    best = np.zeros(log_img.shape, np.float64)
    for sigma in SIGMAS:
        smooth = cv2.GaussianBlur(log_img, (0, 0), sigma)
        s2 = sigma * sigma
        hxx = cv2.Sobel(smooth, cv2.CV_64F, 2, 0, ksize=3) * s2
        hyy = cv2.Sobel(smooth, cv2.CV_64F, 0, 2, ksize=3) * s2
        hxy = cv2.Sobel(smooth, cv2.CV_64F, 1, 1, ksize=3) * s2
        root = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        l_small = 0.5 * (hxx + hyy - root)
        l_big = 0.5 * (hxx + hyy + root)
        rb = np.abs(l_small) / (np.abs(l_big) + 1e-9)
        s_norm = np.sqrt(l_small ** 2 + l_big ** 2)
        c = max(1e-9, 0.5 * float(np.percentile(s_norm, 99.5)))
        v = np.exp(-(rb ** 2) / (2 * FRANGI_BETA ** 2)) * (1 - np.exp(-(s_norm ** 2) / (2 * c * c)))
        v[l_big <= 0] = 0
        best = np.maximum(best, v)
    return best


def _zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Adelgazamiento Zhang-Suen vectorizado: linea central de 1 px."""
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for paso in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            p8, p9 = p[1:-1, :-2], p[:-2, :-2]
            vecinos = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            anillo = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transiciones = sum(((anillo[i] == 0) & (anillo[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if paso == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0
            borrar = (img == 1) & (vecinos >= 2) & (vecinos <= 6) & (transiciones == 1) & c1 & c2
            if borrar.any():
                img[borrar] = 0
                changed = True
    return img * 255


def _count_neighbors(sk: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    return cv2.filter2D(sk.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)


def _prune_spurs(sk: np.ndarray, longitud: int) -> np.ndarray:
    """Elimina ramas cortas (bigotes) y restaura los extremos reales de las lineas."""
    k3 = np.ones((3, 3), np.uint8)
    x = sk.copy()
    for _ in range(longitud):
        nb = _count_neighbors(x)
        extremos = x & (nb <= 1)
        if not extremos.any():
            break
        x = x & ~extremos
    nb = _count_neighbors(x)
    crecer = x & (nb <= 1)
    for _ in range(longitud):
        crecer = (cv2.dilate(crecer.astype(np.uint8), k3) > 0) & sk
    return x | crecer


def _local_normals(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normal local de la lesion en cada pixel, por PCA en una ventana.

    Usa coordenadas escaladas por el lado mayor para evitar perdida de
    precision en float. El eje principal de la ventana es la direccion de la
    lesion; la normal es perpendicular a el.
    """
    h, w = skeleton.shape
    scale = float(max(h, w))
    sk = skeleton.astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx /= scale
    yy /= scale
    k = (2 * 6 + 1, 2 * 6 + 1)

    def box(a):
        return cv2.boxFilter(a, -1, k, normalize=False)

    n = box(sk)
    nn = np.maximum(n, 1.0)
    mx = box(sk * xx) / nn
    my = box(sk * yy) / nn
    cxx = box(sk * xx * xx) / nn - mx * mx
    cyy = box(sk * yy * yy) / nn - my * my
    cxy = box(sk * xx * yy) / nn - mx * my
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    tx, ty = np.cos(theta), np.sin(theta)
    nx = -ty
    ny = tx
    ok = n >= 4
    nx = np.where(ok, nx, 1.0).astype(np.float32)
    ny = np.where(ok, ny, 0.0).astype(np.float32)
    return nx, ny


def _profile_widths(log_img: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                    nx: np.ndarray, ny: np.ndarray, half_len: int):
    """Ancho a mitad de altura (FWHM) sobre la normal local de cada punto."""
    smooth = cv2.GaussianBlur(log_img, (0, 0), 0.7)
    dark = -smooth
    t = np.arange(-half_len, half_len + 1, dtype=np.float32)
    ux = np.ascontiguousarray(nx[ys, xs], dtype=np.float32)[:, None]
    uy = np.ascontiguousarray(ny[ys, xs], dtype=np.float32)[:, None]
    map_x = np.ascontiguousarray(xs.astype(np.float32)[:, None] + t[None, :] * ux)
    map_y = np.ascontiguousarray(ys.astype(np.float32)[:, None] + t[None, :] * uy)
    profile = cv2.remap(dark, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    k = profile.shape[1]
    c = half_len
    pk = (c - 1) + np.argmax(profile[:, c - 1:c + 2], axis=1)
    peak = profile[np.arange(profile.shape[0]), pk]
    below = profile < (0.5 * peak)[:, None]
    cols = np.arange(k)[None, :]
    left = np.where(below & (cols < pk[:, None]), cols, -1).max(axis=1)
    right = np.where(below & (cols > pk[:, None]), cols, k).min(axis=1)
    valid = (left >= 0) & (right < k) & (peak > 0)
    # Muestras discretas por encima de la mitad del pico: un trazo continuo de
    # W px produce ~W+1 muestras, por eso se resta 1 (correccion de discretizacion).
    widths = np.maximum(right - left - 2, 1).astype(np.float32)
    return widths, peak, valid


def _geodesic_path(ys: np.ndarray, xs: np.ndarray):
    """Camino mas largo dentro de la lesion, de punta a punta."""
    pixels = list(zip(ys.tolist(), xs.tolist()))
    index = set(pixels)

    def farthest(start):
        parent = {start: None}
        queue = deque([start])
        last = start
        while queue:
            cur = queue.popleft()
            last = cur
            cy, cx = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cy + dy, cx + dx)
                    if (dy or dx) and nb in index and nb not in parent:
                        parent[nb] = cur
                        queue.append(nb)
        return last, parent

    a, _ = farthest(pixels[0])
    b, parent = farthest(a)
    path = []
    node: Optional[Tuple[int, int]] = b
    while node is not None:
        path.append(node)
        node = parent[node]
    return path, a, b


def _tortuosidad(path, start=None, end=None, ventana: int = 20) -> float:
    """Tortuosidad local de la traza de una lesion (mediana por ventanas).

    Por cada ventana a lo largo del camino: longitud del tramo / cuerda del
    tramo. Una grieta que cambia de direccion a gran escala sigue siendo casi
    recta localmente (valor cercano a 1); un trazo que serpentea a pequena
    escala tiene valores altos.
    """
    P = np.asarray(path, dtype=np.float32)
    if len(P) < 3:
        return 1.0
    W = min(ventana, len(P) - 1)
    paso = max(1, W // 2)
    cocientes = []
    for i in range(0, len(P) - W, paso):
        seg = P[i:i + W + 1]
        largo = float(np.sum(np.hypot(*np.diff(seg, axis=0).T)))
        cuerda = float(np.hypot(*(seg[-1] - seg[0])))
        cocientes.append(largo / max(cuerda, 1.0))
    return float(np.median(cocientes)) if cocientes else 1.0


def _angulo_vertical(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Angulo respecto a la vertical en [-90, 90]; 0 = vertical, + = hacia la derecha."""
    dy, dx = b[0] - a[0], b[1] - a[1]
    if dy < 0:
        dy, dx = -dy, -dx
    return float(np.degrees(np.arctan2(dx, dy)))


def _detect_cracks(image: np.ndarray, umbral_work: float,
                   trazas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Detecta cada fisura o grieta como un componente conectado del esqueleto.

    La clasificacion se hace por lesion completa (ancho mediano de la lesion),
    no por pixel. Si `trazas` es una lista, se anotan los motivos de rechazo.
    """
    h, w = image.shape[:2]
    long_side = float(max(h, w))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    background = cv2.GaussianBlur(gray, (0, 0), max(12.0, min(h, w) / 10.0))
    log_img = (np.log(gray) - np.log(background)).astype(np.float32)

    response = _dark_ridge_response(log_img)
    band = max(2, int(BORDER_BAND_REL * w))
    response[:, :band] = 0
    response[:, w - band:] = 0

    positive = response[response > 0]
    if positive.size == 0:
        return []
    high = float(np.percentile(positive, HIGH_PCT))
    low = float(np.percentile(positive, LOW_PCT))
    lo = response >= low
    fuerte = lo & (response >= high)
    if not fuerte.any():
        return []

    # Histeresis: se conservan solo regiones que contienen una semilla fuerte.
    _, region_labels = cv2.connectedComponents(lo.astype(np.uint8), connectivity=8)
    semillas = np.unique(region_labels[fuerte])
    region = np.isin(region_labels, semillas[semillas > 0])
    skeleton = _zhang_suen(region.astype(np.uint8)) > 0
    skeleton = _prune_spurs(skeleton, max(4, int(PRUNE_REL * long_side)))
    if not skeleton.any():
        return []

    nx, ny = _local_normals(skeleton)
    half_len = int(min(60, max(8, 0.035 * long_side)))
    sk_y, sk_x = np.nonzero(skeleton)
    _, _, valid_all = _profile_widths(log_img, sk_y, sk_x, nx, ny, half_len)
    valid_map = np.zeros((h, w), bool)
    valid_map[sk_y[valid_all], sk_x[valid_all]] = True
    skeleton_valid = (skeleton & valid_map).astype(np.uint8)

    min_len = max(10.0, MIN_LENGTH_REL * long_side)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton_valid, connectivity=8)
    cracks: List[Dict[str, Any]] = []
    for lab in range(1, count):
        x0, y0 = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP])
        bw, bh = int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])
        sub = labels[y0:y0 + bh, x0:x0 + bw] == lab
        ly, lx = np.nonzero(sub)
        ys, xs = ly + y0, lx + x0
        motivo = None
        extent = float(max(bw, bh))
        frame_frac = np.mean((xs < 0.03 * w) | (xs > 0.97 * w) | (ys < 0.03 * h) | (ys > 0.97 * h))
        if ys.size < min_len:
            motivo = f"corta ({ys.size} px)"
        elif extent < MIN_EXTENT_REL * long_side:
            motivo = f"no lineal (extension {extent:.0f} px)"
        elif frame_frac >= 0.8:
            motivo = "marco o borde fotografico"

        widths, peaks, valid = (None, None, None)
        if motivo is None:
            widths, peaks, valid = _profile_widths(log_img, ys, xs, nx, ny, half_len)
            if valid.mean() < 0.3:
                motivo = "perfil no valido"
            else:
                contraste = float(np.median(peaks[valid]))
                if contraste < MIN_CONTRAST:
                    motivo = f"contraste bajo ({contraste:.3f})"

        if motivo is None:
            w_valid = widths[valid]
            med = float(np.median(w_valid))
            mad = float(np.median(np.abs(w_valid - med)))
            tolerancia = max(1.0, 3.5 * 1.4826 * mad)
            kept = w_valid[np.abs(w_valid - med) <= tolerancia]
            if kept.size == 0:
                kept = w_valid
            if float(np.median(kept)) > MAX_WIDTH_REL * long_side:
                motivo = f"ancho excesivo ({float(np.median(kept)):.0f} px)"

        if motivo is not None:
            if trazas is not None:
                trazas.append(f"lesion {lab} ({ys.size} px, x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}): {motivo}")
            continue

        path, start, end = _geodesic_path(ys, xs)
        tort = _tortuosidad(path, start, end)
        if tort > TORT_MAX:
            if trazas is not None:
                trazas.append(f"lesion {lab}: tortuosa (tortuosidad local {tort:.2f})")
            continue
        longitud_traza = float(np.sum(np.hypot(*np.diff(np.asarray(path, dtype=np.float32), axis=0).T))) if len(path) > 1 else 1.0
        cracks.append({
            "tortuosidad": tort,
            "longitud": longitud_traza,
            "ancho_medio": float(kept.mean()),
            "ancho_max": float(np.percentile(kept, 95)),
            "contraste": contraste,
            "ruta": path,
            "inicio": start,
            "fin": end,
            "pixeles": (ys, xs),
            "clase_trabajo": "Grieta" if float(np.median(kept)) >= umbral_work else "Fisura",
        })
        if trazas is not None:
            trazas.append(f"ACEPTADA lesion {lab}: {ys.size} px, ancho {kept.mean():.2f} px")

    cracks.sort(key=lambda c: c["longitud"], reverse=True)
    return cracks


def _clasificar(ancho_px: float, ancho_mm: Optional[float], lado_mayor_px: float) -> str:
    if ancho_mm is not None:
        return "Grieta" if ancho_mm >= FISURA_MAX_MM else "Fisura"
    return "Grieta" if ancho_px / max(1.0, lado_mayor_px) >= FISURA_MAX_REL else "Fisura"


def _orientacion_texto(angulo: float) -> str:
    absoluto = abs(angulo)
    categoria = "Vertical" if absoluto <= 15 else ("Horizontal" if absoluto >= 75 else "Diagonal")
    sentido = "derecha" if angulo > 0.5 else ("izquierda" if angulo < -0.5 else "centrada")
    return f"{categoria} ({angulo:+.1f}° desde vertical, {sentido})"


def analizar_imagen_avanzada(img_array, distancia_cm, ubicacion_estructural,
                             referencia_mm=None, referencia_px=None,
                             ) -> Dict[str, Any]:
    """Detecta todas las fisuras y grietas y mide cada una por separado."""
    original = _as_bgr(img_array)
    h0, w0 = original.shape[:2]
    long_orig = float(max(h0, w0))
    scale = min(1.0, 1600.0 / long_orig)
    image = cv2.resize(original, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # Escala mm/px: prioridad a la calibracion fisica; si no, modelo pinhole con la
    # distancia medida a la superficie (aproximado, depende de la focal estimada).
    factor = None
    metodo = "relativa (px)"
    if referencia_mm and referencia_px and referencia_mm > 0 and referencia_px > 0:
        factor = float(referencia_mm) / float(referencia_px)
        metodo = "referencia física"
    elif distancia_cm and distancia_cm > 0:
        focal_px = FOCAL_FACTOR * w0
        factor = float(distancia_cm) * 10.0 / focal_px
        metodo = "estimación por distancia"

    long_work = float(max(image.shape[:2]))
    if factor:
        umbral_work = FISURA_MAX_MM / factor * scale
    else:
        umbral_work = FISURA_MAX_REL * long_work

    cracks = _detect_cracks(image, umbral_work)

    lesiones: List[Dict[str, Any]] = []
    for numero, c in enumerate(cracks, start=1):
        ancho_px = c["ancho_medio"] / scale
        ancho_max_px = c["ancho_max"] / scale
        ancho_mm = ancho_px * factor if factor else None
        ancho_max_mm = ancho_max_px * factor if factor else None
        clase = _clasificar(ancho_px, ancho_mm, long_orig)
        angulo = _angulo_vertical(c["inicio"], c["fin"])
        lesiones.append({
            "n": numero,
            "clase": clase,
            "tortuosidad": c["tortuosidad"],
            "longitud_px": c["longitud"] / scale,
            "ancho_px": ancho_px,
            "ancho_maximo_px": ancho_max_px,
            "ancho_mm": ancho_mm,
            "ancho_maximo_mm": ancho_max_mm,
            "angulo": angulo,
            "orientacion": _orientacion_texto(angulo),
            "inicio": (int(round(c["inicio"][1] / scale)), int(round(c["inicio"][0] / scale))),
            "fin": (int(round(c["fin"][1] / scale)), int(round(c["fin"][0] / scale))),
            "_ruta": c["ruta"],
            "_pixeles": c["pixeles"],
            "_inicio_trabajo": c["inicio"],
        })

    visual = image.copy()
    if not lesiones:
        return {
            "imagen": visual, "lesiones": [], "numero_lesiones": 0,
            "ancho_px": None, "ancho_maximo_px": None, "ancho_mm": None, "ancho_maximo_mm": None,
            "angulo": 0.0, "orientacion": "No determinada",
            "tipo_patologia": "Sin lesion identificable",
            "requiere_referencia_metrica": factor is None, "riesgo": "No concluyente",
            "probabilidad_morfologica": 0.0, "roi_aplicada": False, "metodo": metodo,
        }

    principal = lesiones[0]
    for lesion in lesiones:
        ys, xs = lesion["_pixeles"]
        visual[ys, xs] = COLOR_GRIETA if lesion["clase"] == "Grieta" else COLOR_FISURA

    ruta = np.array([[x, y] for (y, x) in principal["_ruta"]], np.int32)
    cv2.polylines(visual, [ruta], False, COLOR_EJE, 2, cv2.LINE_AA)

    hh, ww = visual.shape[:2]
    for lesion in lesiones:
        py, px = lesion["_inicio_trabajo"]
        ox = int(np.clip(px + 8, 2, max(2, ww - 22)))
        oy = int(np.clip(py - 8, 14, max(14, hh - 4)))
        cv2.rectangle(visual, (ox - 2, oy - 13), (ox + 16, oy + 3), (0, 0, 0), -1)
        cv2.putText(visual, str(lesion["n"]), (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    prob = float(np.clip(40 + 60 * min(1.0, (principal["longitud_px"] / long_orig) / 0.25), 1, 99))
    if principal["clase"] == "Grieta":
        riesgo = "Abertura igual o mayor al umbral de grieta; evaluacion estructural recomendada."
    else:
        riesgo = "Abertura menor al umbral de grieta; registrar y vigilar."

    for lesion in lesiones:
        lesion.pop("_ruta", None)
        lesion.pop("_pixeles", None)
        lesion.pop("_inicio_trabajo", None)

    return {
        "imagen": visual,
        "lesiones": lesiones,
        "numero_lesiones": len(lesiones),
        "ancho_px": principal["ancho_px"],
        "ancho_maximo_px": principal["ancho_maximo_px"],
        "ancho_mm": principal["ancho_mm"],
        "ancho_maximo_mm": principal["ancho_maximo_mm"],
        "angulo": principal["angulo"],
        "orientacion": principal["orientacion"],
        "tipo_patologia": principal["clase"],
        "requiere_referencia_metrica": factor is None,
        "metodo": metodo,
        "riesgo": riesgo,
        "probabilidad_morfologica": prob,
        "roi_aplicada": False,
    }


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
modo = st.sidebar.radio("Apariencia", ["Día", "Noche"], horizontal=True)
oscuro = modo == "Noche"

fondo = "#111918" if oscuro else "#f8f9fa"
panel = "#192321" if oscuro else "#ffffff"
texto_color = "#e7efeb" if oscuro else "#111827"
secundario = "#a8b8b0" if oscuro else "#4b5563"
borde = "#30423b" if oscuro else "#d1d5db"
fondo_inputs = "#26342f" if oscuro else "#ffffff"

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background: {fondo}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    p, span, label, div[data-testid="stMarkdownContainer"] p, .stRadio label {{ color: {texto_color} !important; }}
    section[data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {borde}; }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.2rem !important; white-space: normal !important;
        word-wrap: break-word !important; overflow: visible !important; line-height: 1.3 !important;
    }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.9rem !important; white-space: normal !important; overflow: visible !important; }}
    svg {{ fill: {secundario} !important; }}
    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {panel} !important; border-color: {borde} !important; border-radius: 8px; padding: 10px !important;
    }}
    .stButton > button, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: {fondo_inputs} !important; border-color: {borde} !important; color: {texto_color} !important;
    }}
    .inspection-hero {{ border-bottom: 1px solid {borde}; padding: 0.4rem 0 1.2rem; margin-bottom: 1.3rem; }}
    .inspection-hero p {{ color: {secundario} !important; margin: 0.3rem 0 0; }}
    .inspection-kicker {{ color: #18866b; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; }}
    .inspection-hero h1 {{ color: {texto_color} !important; font-size: 2rem; line-height: 1.15; margin: 0.35rem 0; }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="inspection-hero"><div class="inspection-kicker">INSPECCIÓN DE SUPERFICIES</div>'
    '<h1>Evaluación de fisuras y grietas</h1>'
    '<p>Análisis visual orientativo con medición calibrada cuando hay referencia física.</p></div>',
    unsafe_allow_html=True,
)
vista = st.sidebar.radio("Sección", ["Inspección", "Historial"], horizontal=True)

with st.sidebar.expander("Hardware ESP8266", expanded=False):
    esp_ip = st.text_input("Dirección IP del NodeMCU", value="http://192.168.1.50")
    angulo_servo = st.slider("Ángulo del sensor", 0, 180, 90)
    if st.button("Mover servomotor"):
        try:
            respuesta = requests.get(f"{esp_ip}/servo?angle={angulo_servo}", timeout=3)
            if respuesta.status_code == 200:
                st.success(f"Servo en posición {angulo_servo}°")
            else:
                st.error(f"El ESP8266 respondió con estado {respuesta.status_code}.")
        except requests.RequestException:
            st.error("No se pudo conectar con el ESP8266.")

if vista == "Inspección":
    col1, col2 = st.columns([1.1, 1.9])

    with col1:
        st.subheader("Parámetros de inspección")
        if "distancia_manual" not in st.session_state:
            st.session_state.distancia_manual = 15.0

        distancia_lectura = st.number_input(
            "Distancia a la superficie (cm)", min_value=0.0, max_value=2000.0, step=1.0, key="distancia_manual",
            help="Se usa para estimar la escala (mm por píxel) cuando no hay calibración física.",
        )
        ubicacion = st.selectbox("Ubicación estructural", ["Pared/Muro", "Viga", "Columna", "Losa/Techo"])
        imagen = None
        archivo_imagen = st.file_uploader("Selecciona una foto", type=["jpg", "jpeg", "png", "webp", "bmp"])

        if archivo_imagen is not None:
            bytes_imagen = np.frombuffer(archivo_imagen.getvalue(), np.uint8)
            imagen = cv2.imdecode(bytes_imagen, cv2.IMREAD_COLOR)

        with st.expander("Calibración métrica", expanded=False):
            referencia_presente = st.checkbox("Incluye referencia física")
            referencia_mm = referencia_px = None
            if referencia_presente:
                st.caption("Introduce la longitud real y cuántos píxeles ocupa.")
                referencia_mm = st.number_input("Longitud real de referencia (mm)", min_value=0.1, value=50.0)
                referencia_px = st.number_input("Longitud en la imagen (px)", min_value=0.1, value=100.0)

    with col2:
        st.subheader("Resultado")
        if imagen is None:
            st.info("Captura una imagen o carga un archivo para iniciar la inspección.")
        else:
            try:
                resultado = analizar_imagen_avanzada(
                    imagen, distancia_lectura, ubicacion, referencia_mm, referencia_px
                )
            except ValueError as exc:
                st.error(str(exc))
                resultado = None

            if resultado is not None:
                st.image(
                    cv2.cvtColor(resultado["imagen"], cv2.COLOR_BGR2RGB),
                    caption="Rojo: grieta · Amarillo: fisura · Magenta: eje de la lesión principal · Números: identificador",
                    width="stretch",
                )

                if resultado["numero_lesiones"] == 0:
                    st.info("No se identificó ninguna fisura o grieta en la imagen. Prueba con mejor iluminación o un encuadre más cercano.")
                else:
                    if resultado["metodo"] == "referencia física":
                        st.success("Medición calibrada con referencia física.")
                    elif resultado["metodo"] == "estimación por distancia":
                        st.info("Escala estimada a partir de la distancia a la superficie (aproximada).")
                    else:
                        st.warning("Medición relativa en píxeles: indica la distancia para estimar mm.")
                    st.caption(f"Lesiones detectadas: {resultado['numero_lesiones']}")

                    m1, m2 = st.columns(2)
                    m1.metric("Clasificación principal", resultado["tipo_patologia"])
                    m2.metric("Orientación", resultado["orientacion"])

                    ancho_texto = f"{resultado['ancho_px']:.1f} px"
                    ancho_max_texto = f"{resultado['ancho_maximo_px']:.1f} px"
                    if resultado["ancho_mm"] is not None:
                        ancho_texto += f" ({resultado['ancho_mm']:.2f} mm)"
                        ancho_max_texto += f" ({resultado['ancho_maximo_mm']:.2f} mm)"
                    st.metric("Ancho medio (principal)", ancho_texto)
                    st.metric("Ancho máximo (P95, principal)", ancho_max_texto)

                    filas = []
                    for lesion in resultado["lesiones"]:
                        filas.append({
                            "N°": lesion["n"],
                            "Clasificación": lesion["clase"],
                            "Longitud (px)": round(lesion["longitud_px"], 1),
                            "Ancho medio (px)": round(lesion["ancho_px"], 2),
                            "Ancho máximo (px)": round(lesion["ancho_maximo_px"], 2),
                            "Ancho medio (mm)": None if lesion["ancho_mm"] is None else round(lesion["ancho_mm"], 2),
                            "Ángulo (°)": round(lesion["angulo"], 1),
                            "Tortuosidad": round(lesion["tortuosidad"], 2),
                        })
                    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)

                    st.caption(f"**Riesgo evaluado:** {resultado['riesgo']}")

                    recomendacion = (
                        "Evaluación estructural recomendada." if resultado["tipo_patologia"] == "Grieta"
                        else "Registrar y vigilar cambios mediante inspección en sitio."
                    )

                    informe = {
                        "numero_lesiones": resultado["numero_lesiones"],
                        "lesion_principal": {
                            "clasificacion": resultado["tipo_patologia"],
                            "orientacion": resultado["orientacion"],
                            "ancho_px_promedio": resultado["ancho_px"],
                            "ancho_px_maximo_p95": resultado["ancho_maximo_px"],
                            "ancho_mm": resultado["ancho_mm"],
                            "ancho_mm_maximo_p95": resultado["ancho_maximo_mm"],
                        },
                        "lesiones": resultado["lesiones"],
                        "metodo": "referencia física" if resultado["ancho_mm"] is not None else "Relativa (px)",
                        "recomendacion_tecnica": recomendacion,
                        "ubicacion": ubicacion,
                        "distancia_cm": distancia_lectura,
                    }

                    with st.expander("Informe estructurado (JSON)"):
                        st.json(informe)

                    st.download_button(
                        "Descargar informe JSON",
                        data=json.dumps(informe, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="informe_inspeccion.json",
                        mime="application/json",
                        icon=":material/download:",
                    )

                    if resultado["ancho_mm"] is None:
                        st.button(
                            "Guardar en historial MySQL",
                            disabled=True,
                            help="Requiere una distancia mayor que 0 o una calibración física.",
                        )
                    elif st.button("Guardar inspección en MySQL", icon=":material/save:"):
                        # Campos de audio (frecuencia, centroide) en 0.0 para preservar la estructura de la BD.
                        if guardar_inspeccion(
                            distancia_lectura,
                            resultado["ancho_mm"],
                            0.0,
                            0.0,
                            ubicacion,
                            resultado["tipo_patologia"],
                            resultado["riesgo"],
                            resultado["probabilidad_morfologica"],
                        ):
                            st.success("Guardado en base de datos correctamente.")

else:
    st.subheader("Historial de inspecciones")
    if mysql is None:
        st.error("Falta instalar mysql-connector-python.")
    else:
        conn = None
        try:
            conn = mysql.connect(**DB_CONFIG)
            df_historial = pd.read_sql("SELECT * FROM inspecciones ORDER BY id DESC LIMIT 50", conn)
            if df_historial.empty:
                st.info("Todavía no hay inspecciones guardadas.")
            else:
                st.dataframe(df_historial, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"No se pudo consultar MySQL: {exc}")
        finally:
            if conn is not None:
                conn.close()
