import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import mysql.connector as mysql  # type: ignore[import-not-found]
except ImportError:
    mysql = None


st.set_page_config(page_title="Evaluación de Grietas Estructurales", page_icon="🔍", layout="wide")

# ------------------------------------------------------------------
# Configuracion general
# ------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
MODELO_PATH = RAIZ / "modelo_grietas.h5"   # generado por notebooks/01_analisis_y_linea_base.ipynb
IMG_SIZE_IA = 224                          # mismo tamano de entrada usado en el entrenamiento
UMBRAL_IA = 0.5                            # probabilidad minima para diagnosticar "Con Grieta"

DB_CONFIG = {
    "host": "localhost", "port": 3306, "user": "root", "password": "",
    "database": "detector_grietas",
}

# Medicion de fisuras (umbrales relativos al tamano de la imagen)
SIGMAS = (1.0, 1.5, 2.0, 3.0)    # escalas del filtro de valles oscuros
FRANGI_BETA = 0.5                # sensibilidad a formas no lineales
HIGH_PCT = 96.0                  # histeresis: semilla fuerte (percentil de la respuesta)
LOW_PCT = 85.0                   # histeresis: extension debil (percentil de la respuesta)
MIN_CONTRASTE = 0.04             # diferencia minima de log-intensidad respecto al entorno
MIN_EXTENSION_REL = 0.08         # extension minima de una lesion / lado mayor (descarta textura del yeso)
MAX_ANCHO_REL = 0.05             # ancho maximo plausible / lado mayor (descarta sombras)
ELONGACION_MIN = 4.0             # extension / ancho: una grieta es larga y delgada
BORDE_REL = 0.02                 # banda lateral ignorada (marco fotografico)
FISURA_MAX_MM = 1.0              # criterio del proyecto: ancho >= 1 mm = grieta

# Desplome: bordes verticales del muro o columna
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_VOTOS = 80
HOUGH_LONG_MIN_REL = 0.30        # longitud minima de un segmento / alto de la imagen
HOUGH_HUECO_MAX = 10             # px de hueco permitido para unir segmentos
VERTICAL_TOLERANCIA_DEG = 20.0   # solo segmentos a menos de 20 grados de la vertical
LINEAS_PRINCIPALES = 6           # numero de lineas verticales mas largas que se promedian

COLOR_FISURA = (0, 255, 255)     # BGR amarillo
COLOR_GRIETA = (0, 0, 255)       # BGR rojo
COLOR_BORDE = (255, 0, 255)      # BGR magenta
COLOR_REFERENCIA = (0, 255, 0)   # BGR verde: vertical de referencia (0 grados)


# ------------------------------------------------------------------
# Persistencia MySQL
# ------------------------------------------------------------------
def guardar_inspeccion(ancho, freq_pico, centroide, ubicacion, diagnostico, riesgo, probabilidad):
    """Guarda una inspeccion. Los campos heredados de audio (frecuencia, centroide) reciben 0.0.

    La distancia ya no se usa en la interfaz; se conserva el campo por compatibilidad
    de la tabla y se guarda como 0.0.
    """
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
            (0.0, ancho, freq_pico, centroide, ubicacion, diagnostico, riesgo, probabilidad),
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
# Modelo de IA (clasificacion Con Grieta / Sin Grieta)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo_ia():
    """Carga modelo_grietas.h5 una sola vez por proceso del servidor.

    TensorFlow se importa aqui y no al inicio: si no esta instalado, solo
    se desactiva la IA y la medicion clasica sigue funcionando.
    """
    import tensorflow as tf  # importacion diferida

    if not MODELO_PATH.is_file():
        raise FileNotFoundError(
            f"No existe '{MODELO_PATH.name}'. Ejecuta el cuaderno de entrenamiento para generarlo."
        )
    return tf.keras.models.load_model(MODELO_PATH, compile=False)


def diagnosticar_con_ia(modelo, imagen: np.ndarray) -> Tuple[str, float]:
    """Devuelve (diagnostico, probabilidad_de_grieta) para una imagen BGR."""
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    # Mismo redimensionado que el entrenamiento (interpolacion bilineal, imagen completa)
    entrada = cv2.resize(rgb, (IMG_SIZE_IA, IMG_SIZE_IA), interpolation=cv2.INTER_LINEAR)
    lote = entrada.astype(np.float32)[None, ...]  # forma (1, 224, 224, 3), rango 0-255
    # El modelo normaliza internamente. La salida es P(Positive), es decir, P(con grieta)
    probabilidad = float(modelo.predict(lote, verbose=0)[0][0])
    diagnostico = "Con Grieta" if probabilidad >= UMBRAL_IA else "Sin Grieta"
    return diagnostico, probabilidad


# ------------------------------------------------------------------
# Medicion de fisuras con OpenCV (ancho con distanceTransform)
# ------------------------------------------------------------------
def decodificar_imagen(datos: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)


def _respuesta_valle_oscuro(log_img: np.ndarray) -> np.ndarray:
    """Filtro de vasos (Frangi) para valles oscuros, multiescala.

    Un valle oscuro tiene curvatura fuerte a lo ancho (autovalor grande y positivo)
    y casi nula a lo largo (autovalor pequeno). Las manchas no lineales se atenuan.
    """
    mejor = np.zeros(log_img.shape, np.float64)
    for sigma in SIGMAS:
        suave = cv2.GaussianBlur(log_img, (0, 0), sigma)
        s2 = sigma * sigma
        hxx = cv2.Sobel(suave, cv2.CV_64F, 2, 0, ksize=3) * s2
        hyy = cv2.Sobel(suave, cv2.CV_64F, 0, 2, ksize=3) * s2
        hxy = cv2.Sobel(suave, cv2.CV_64F, 1, 1, ksize=3) * s2
        raiz = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        l_pequeno = 0.5 * (hxx + hyy - raiz)
        l_grande = 0.5 * (hxx + hyy + raiz)
        rb = np.abs(l_pequeno) / (np.abs(l_grande) + 1e-9)
        s_norm = np.sqrt(l_pequeno ** 2 + l_grande ** 2)
        c = max(1e-9, 0.5 * float(np.percentile(s_norm, 99.5)))
        v = np.exp(-(rb ** 2) / (2 * FRANGI_BETA ** 2)) * (1 - np.exp(-(s_norm ** 2) / (2 * c * c)))
        v[l_grande <= 0] = 0
        mejor = np.maximum(mejor, v)
    return mejor


def _lesiones_desde_mascara(mascara: np.ndarray, log_img: np.ndarray, lado: float,
                            escala: float, mm_por_px: Optional[float]) -> List[Dict[str, Any]]:
    """Mide cada componente conectada de la mascara de fisuras.

    cv2.distanceTransform calcula, para cada pixel de la fisura, la distancia al
    fondo mas cercano. Los maximos locales de esa distancia forman el eje central
    de la fisura, y en ese eje el ancho local vale 2 * distancia.
    """
    h, w = mascara.shape
    distancia = cv2.distanceTransform(mascara, cv2.DIST_L2, 3)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    kernel = np.ones((3, 3), np.uint8)
    lesiones: List[Dict[str, Any]] = []

    for lab in range(1, n):
        x0, y0 = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP])
        bw, bh = int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])
        extension = float(max(bw, bh))
        if extension < MIN_EXTENSION_REL * lado:
            continue

        sub = etiquetas[y0:y0 + bh, x0:x0 + bw] == lab
        d_sub = distancia[y0:y0 + bh, x0:x0 + bw] * sub
        eje = sub & (d_sub > 0) & (d_sub >= cv2.dilate(d_sub, kernel))
        if not eje.any():
            continue

        anchos = 2.0 * d_sub[eje]                    # ancho local en cada punto del eje
        ancho_medio = float(np.median(anchos))
        ancho_max = float(np.percentile(anchos, 95))
        if ancho_medio > MAX_ANCHO_REL * lado or extension / ancho_medio < ELONGACION_MIN:
            continue  # demasiado ancha (sombra) o no alargada (mancha)

        ys, xs = np.nonzero(sub)
        ys, xs = ys + y0, xs + x0
        en_marco = np.mean((xs < 0.03 * w) | (xs > 0.97 * w) | (ys < 0.03 * h) | (ys > 0.97 * h))
        if en_marco >= 0.8:
            continue  # borde de la fotografia, no una lesion del elemento

        contraste = float(np.median(-log_img[ys, xs]))
        if contraste < MIN_CONTRASTE:
            continue

        ancho_px = ancho_medio / escala              # convertido a pixeles de la imagen original
        ancho_max_px = ancho_max / escala
        ancho_mm = float(ancho_px * mm_por_px) if mm_por_px else None
        clase = None
        if ancho_mm is not None:
            clase = "Grieta" if ancho_mm >= FISURA_MAX_MM else "Fisura"

        lesiones.append({
            "extension_px": extension / escala,
            "ancho_medio_px": float(ancho_px),
            "ancho_maximo_px": float(ancho_max_px),
            "ancho_medio_mm": ancho_mm,
            "ancho_maximo_mm": float(ancho_max_px * mm_por_px) if mm_por_px else None,
            "clase": clase,
            "contraste": contraste,
            "_pixeles": (ys, xs),
            "_centro": (int(xs.mean()), int(ys.mean())),
        })
    return lesiones


def analizar_grietas(imagen: np.ndarray, mm_por_px: Optional[float]) -> Dict[str, Any]:
    """Detecta fisuras y grietas y mide el ancho de cada una.

    mm_por_px solo viene de una calibracion fisica (referencia conocida en la imagen).
    Sin ella, los anchos se reportan en pixeles y no se clasifican.
    """
    h0, w0 = imagen.shape[:2]
    escala = min(1.0, 1600.0 / max(h0, w0))
    trabajo = cv2.resize(imagen, (max(1, round(w0 * escala)), max(1, round(h0 * escala))),
                         interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR)
    h, w = trabajo.shape[:2]
    lado = float(max(h, w))

    gris = cv2.cvtColor(trabajo, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    fondo = cv2.GaussianBlur(gris, (0, 0), max(12.0, min(h, w) / 10.0))
    log_img = (np.log(gris) - np.log(fondo)).astype(np.float32)  # negativo = mas oscuro que el entorno

    respuesta = _respuesta_valle_oscuro(log_img)
    banda = max(2, int(BORDE_REL * w))
    respuesta[:, :banda] = 0
    respuesta[:, w - banda:] = 0

    lesiones: List[Dict[str, Any]] = []
    positivos = respuesta[respuesta > 0]
    if positivos.size:
        alto = float(np.percentile(positivos, HIGH_PCT))
        bajo = float(np.percentile(positivos, LOW_PCT))
        candidata = (respuesta >= bajo).astype(np.uint8)
        semilla = respuesta >= alto
        if semilla.any():
            # Histeresis: solo se conservan las regiones que contienen una semilla fuerte
            _, etiq_region = cv2.connectedComponents(candidata, connectivity=8)
            ids = np.unique(etiq_region[semilla])
            ids = ids[ids > 0]
            mascara = np.isin(etiq_region, ids).astype(np.uint8)
            lesiones = _lesiones_desde_mascara(mascara, log_img, lado, escala, mm_por_px)

    lesiones.sort(key=lambda les: les["extension_px"], reverse=True)
    for numero, les in enumerate(lesiones, start=1):
        les["n"] = numero

    return {
        "imagen": _dibujar_fisuras(trabajo, lesiones),
        "lesiones": lesiones,
        "principal": lesiones[0] if lesiones else None,
    }


def _dibujar_fisuras(imagen: np.ndarray, lesiones: List[Dict[str, Any]]) -> np.ndarray:
    visual = imagen.copy()
    for les in lesiones:
        ys, xs = les["_pixeles"]
        visual[ys, xs] = COLOR_GRIETA if les["clase"] == "Grieta" else COLOR_FISURA
    for les in lesiones:
        cx, cy = les["_centro"]
        cv2.putText(visual, str(les["n"]), (cx + 6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return visual


# ------------------------------------------------------------------
# Desplome de muro o columna (independiente de la fisura)
# ------------------------------------------------------------------
def _mediana_ponderada(valores: np.ndarray, pesos: np.ndarray) -> float:
    orden = np.argsort(valores)
    v = valores[orden]
    acumulado = np.cumsum(pesos[orden])
    return float(v[np.searchsorted(acumulado, acumulado[-1] / 2.0)])


def describir_desplome(angulo: float) -> str:
    if abs(angulo) < 0.05:
        return "0.0° (vertical)"
    lado = "izquierda" if angulo > 0 else "derecha"
    return f"{abs(angulo):.1f}° (parte superior hacia la {lado})"


def medir_desplome(imagen: np.ndarray) -> Dict[str, Any]:
    """Angulo de desplome del muro o columna respecto a la vertical (0 grados).

    Procedimiento:
      1. Canny detecta los bordes de la imagen.
      2. HoughLinesP encuentra segmentos rectos largos.
      3. Se conservan solo los segmentos casi verticales (bordes del elemento).
      4. El desplome es la mediana ponderada por longitud de sus angulos.

    La fisura NO se usa: su angulo describe la patologia local, no el desplome
    de la estructura. Positivo = la parte superior queda a la izquierda de la base.
    """
    h, _ = imagen.shape[:2]
    gris = cv2.GaussianBlur(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    bordes = cv2.Canny(gris, CANNY_LOW, CANNY_HIGH)
    segmentos = cv2.HoughLinesP(
        bordes, rho=1, theta=np.pi / 180, threshold=HOUGH_VOTOS,
        minLineLength=HOUGH_LONG_MIN_REL * h, maxLineGap=HOUGH_HUECO_MAX,
    )
    resultado: Dict[str, Any] = {
        "valido": False, "motivo": "", "angulo_deg": None, "dispersion_deg": None,
        "lineas_usadas": 0, "lineas": [], "punto_referencia": None,
    }
    if segmentos is None:
        resultado["motivo"] = "No se detectaron segmentos rectos en la imagen."
        return resultado

    candidatas = []
    for x1, y1, x2, y2 in segmentos.reshape(-1, 4):  # acepta formas (N,1,4) y (N,4) segun la version de OpenCV
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1  # orientar el segmento de arriba hacia abajo
        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            continue
        angulo = float(np.degrees(np.arctan2(dx, dy)))  # 0 = vertical
        if abs(angulo) > VERTICAL_TOLERANCIA_DEG:
            continue
        candidatas.append((angulo, float(np.hypot(dx, dy)), (x1, y1, x2, y2)))

    if not candidatas:
        resultado["motivo"] = "No hay lineas verticales suficientemente largas. Revisa el encuadre."
        return resultado

    candidatas.sort(key=lambda c: c[1], reverse=True)
    principales = candidatas[:LINEAS_PRINCIPALES]
    angulos = np.array([c[0] for c in principales])
    longitudes = np.array([c[1] for c in principales])
    x1, y1, _, _ = principales[0][2]

    resultado.update({
        "valido": True,
        "angulo_deg": _mediana_ponderada(angulos, longitudes),
        "dispersion_deg": float(np.std(angulos)),
        "lineas_usadas": len(principales),
        "lineas": [c[2] for c in principales],
        "punto_referencia": (x1, y1),
    })
    return resultado


def _dibujar_muro(imagen: np.ndarray, desplome: Dict[str, Any]) -> np.ndarray:
    visual = imagen.copy()
    if not desplome["valido"]:
        return visual
    grosor = max(2, visual.shape[1] // 400)
    for x1, y1, x2, y2 in desplome["lineas"]:
        cv2.line(visual, (x1, y1), (x2, y2), COLOR_BORDE, grosor, cv2.LINE_AA)
    x_ref, y_ref = desplome["punto_referencia"]
    cv2.line(visual, (x_ref, y_ref), (x_ref, visual.shape[0] - 1), COLOR_REFERENCIA, grosor, cv2.LINE_AA)
    return visual


# ------------------------------------------------------------------
# Evaluacion de riesgo estructural (NSR-10 / ACI 224R-01)
# ------------------------------------------------------------------
def evaluar_riesgo(ubicacion: str, material: str, ancho_critico_mm: Optional[float],
                   desplome: Dict[str, Any]) -> Dict[str, Any]:
    """Dictamen de riesgo segun criterios simplificados NSR-10 / ACI 224R-01.

    Los umbrales son valores de tamizaje para una inspeccion visual; no reemplazan
    la evaluacion de un ingeniero estructural.

    Umbrales:
      - Pañete / Estuco: BAJO (no es elemento portante).
      - Columna / Muro de Carga: > 0.3 mm o desplome > 1.0° -> ALTO/CRITICO
                                   (ACI 224R-01, tolerancias estrictas para elementos
                                    de carga en exposicion severa).
      - Viga / Losa: > 0.4 mm -> ALTO (ACI 224R-01, limite a flexion).
      - Pared Divisoria: > 1.0 mm -> MODERADO (asentamiento / mortero de pega).
                         <= 1.0 mm -> BAJO.
      - Desplome > 1.0°: tamizaje visual conservador; NSR-10 limita derivas
        del orden de 0.5 % - 1.0 % (aprox. 0.3° - 0.6°).
    """
    # Helper para acortar el texto que va a la base de datos (columna VARCHAR 100)
    def resumen_bd(nivel: str, accion: str) -> str:
        return f"RIESGO {nivel} - {accion}"[:100]

    # 1) Acabados: no son portantes
    if material == "Pañete / Estuco":
        return {
            "nivel": "BAJO",
            "color": "verde",
            "dictamen": "Fisuración superficial en capa de acabado (retracción plástica). No compromete la estructura.",
            "justificacion": "El pañete/estuco no es elemento portante; no aplican los límites de ancho de ACI 224R-01 a elementos estructurales.",
            "recomendacion": "Sellado cosmético y repaño del acabado.",
            "riesgo_bd": resumen_bd("BAJO", "Sellado cosmético y repaño del acabado"),
        }

    # 2) Sin calibracion: no se puede comparar con umbrales normativos
    if ancho_critico_mm is None:
        return {
            "nivel": "NO CONCLUYENTE",
            "color": "gris",
            "dictamen": "Sin calibración métrica no se puede comparar el ancho con los límites normativos.",
            "justificacion": "ACI 224R-01 y NSR-10 expresan los límites de fisuración en milímetros.",
            "recomendacion": "Repetir la inspección incluyendo una referencia física (regla, moneda) en la foto.",
            "riesgo_bd": resumen_bd("NO CONCLUYENTE", "Repetir con calibración métrica"),
        }

    # Extraer desplome valido, si existe
    desplome_deg = abs(desplome["angulo_deg"]) if desplome["valido"] else None

    if ubicacion == "Columna / Muro de Carga":
        excede_ancho = ancho_critico_mm > 0.3
        excede_desplome = desplome_deg is not None and desplome_deg > 1.0
        if excede_ancho or excede_desplome:
            return {
                "nivel": "ALTO / CRÍTICO",
                "color": "rojo",
                "dictamen": "Excede el límite de 0.3 mm (ACI 224R-01) o presenta desaplome. Peligro estructural.",
                "justificacion": "ACI 224R-01 recomienda anchos inferiores a 0.3 mm para elementos de carga en exposición severa; NSR-10 C.10.6 exige control de fisuración en concreto reforzado.",
                "recomendacion": "Apuntalamiento inmediato y evaluación por ingeniero estructural.",
                "riesgo_bd": resumen_bd("ALTO / CRITICO", "Apuntalamiento inmediato y evaluación estructural"),
            }
        return {
            "nivel": "MODERADO",
            "color": "amarillo",
            "dictamen": "Ancho dentro de límites pero requiere vigilancia en elemento de carga.",
            "justificacion": "No se excede el umbral de 0.3 mm de ACI 224R-01 ni se detecta desplome significativo, pero la fisura debe seguirse.",
            "recomendacion": "Instalar fisurómetros de testigo para monitorear apertura durante 30 días.",
            "riesgo_bd": resumen_bd("MODERADO", "Fisurómetros de testigo por 30 días"),
        }

    if ubicacion == "Viga / Losa":
        if ancho_critico_mm > 0.4:
            return {
                "nivel": "ALTO",
                "color": "rojo",
                "dictamen": "Excede el límite a flexión de 0.4 mm. Riesgo de corrosión en acero.",
                "justificacion": "ACI 224R-01 cita 0.4 mm como ancho tolerable en elementos a flexión con exposición normal; su superación aumenta el riesgo de ingreso de humedad y corrosión.",
                "recomendacion": "Requiere inyección epóxica y revisión de refuerzo.",
                "riesgo_bd": resumen_bd("ALTO", "Inyección epóxica y revisión de refuerzo"),
            }
        return {
            "nivel": "MODERADO",
            "color": "amarillo",
            "dictamen": "Ancho a flexión dentro de límites, pero requiere monitoreo.",
            "justificacion": "No supera los 0.4 mm recomendados por ACI 224R-01 para elementos a flexión.",
            "recomendacion": "Instalar fisurómetros de testigo para monitorear apertura durante 30 días.",
            "riesgo_bd": resumen_bd("MODERADO", "Fisurómetros de testigo por 30 días"),
        }

    if ubicacion == "Pared Divisoria":
        if ancho_critico_mm > 1.0:
            return {
                "nivel": "MODERADO",
                "color": "amarillo",
                "dictamen": "Asentamiento o falla en mortero de pega. Sellado de juntas.",
                "justificacion": "Elemento no portante; aberturas superiores a 1.0 mm indican movimientos de asentamiento o desprendimiento de mortero.",
                "recomendacion": "Verificar nivelaciones y realizar sellado de juntas.",
                "riesgo_bd": resumen_bd("MODERADO", "Verificar nivelación y sellar juntas"),
            }
        return {
            "nivel": "BAJO",
            "color": "verde",
            "dictamen": "Fisura leve en elemento no portante; no compromete la estructura.",
            "justificacion": "Pared divisoria no estructural con abertura menor a 1.0 mm.",
            "recomendacion": "Sellado cosmético de juntas.",
            "riesgo_bd": resumen_bd("BAJO", "Sellado cosmético de juntas"),
        }

    # Fallback (no deberia ocurrir)
    return {
        "nivel": "NO CONCLUYENTE",
        "color": "gris",
        "dictamen": "No se pudo clasificar el riesgo con los parámetros ingresados.",
        "justificacion": "La ubicación estructural no está dentro de las categorías programadas.",
        "recomendacion": "Revisar los datos de inspección.",
        "riesgo_bd": resumen_bd("NO CONCLUYENTE", "Revisar datos de inspección"),
    }


def _publico(registro: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in registro.items() if not k.startswith("_")}


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
if "oscuro" not in st.session_state:
    st.session_state.oscuro = False

oscuro = st.session_state.oscuro
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
    .risk-box {{ padding: 1rem; border-radius: 8px; border-left: 6px solid; }}
    .risk-rojo {{ background: rgba(255, 75, 75, 0.12); border-color: #ff4b4b; }}
    .risk-amarillo {{ background: rgba(255, 200, 50, 0.12); border-color: #ffc800; }}
    .risk-verde {{ background: rgba(40, 167, 69, 0.12); border-color: #28a745; }}
    .risk-gris {{ background: rgba(150, 150, 150, 0.12); border-color: #888; }}
</style>
""", unsafe_allow_html=True)

# Encabezado con boton de tema a la derecha
col_titulo, col_tema = st.columns([6, 1])
with col_titulo:
    st.markdown(
        '<div class="inspection-hero"><div class="inspection-kicker">INSPECCIÓN ESTRUCTURAL</div>'
        '<h1>Evaluación de fisuras y riesgo</h1>'
        '<p>Clasificación con IA, medición geométrica y dictamen NSR-10 / ACI 224R-01.</p></div>',
        unsafe_allow_html=True,
    )
with col_tema:
    icono = "🌙" if not oscuro else "☀️"
    texto_boton = f"{icono} {'Oscuro' if not oscuro else 'Claro'}"
    if st.button(texto_boton, key="toggle_tema", use_container_width=True):
        st.session_state.oscuro = not oscuro
        st.rerun()

# ------------------------------------------------------------------
# Panel lateral: parametros de inspeccion
# ------------------------------------------------------------------
st.sidebar.header("Parámetros de inspección")

ubicacion = st.sidebar.selectbox(
    "Ubicación estructural",
    ["Pared Divisoria", "Viga / Losa", "Columna / Muro de Carga"],
    help="Elemento donde se encuentra la fisura.",
)
material = st.sidebar.selectbox(
    "Material / acabado",
    ["Concreto Reforzado", "Mampostería (Ladrillo/Bloque)", "Pañete / Estuco"],
    help="Material del elemento afectado. Los acabados no son portantes.",
)

archivo_imagen = st.sidebar.file_uploader(
    "Cargar fotografía de la obra",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Toma la foto externamente y súbela aquí.",
)

mm_por_px = None
with st.sidebar.expander("Calibración métrica", expanded=False):
    referencia_presente = st.checkbox("Incluye referencia física")
    if referencia_presente:
        st.caption("Introduce la longitud real de un objeto de medida visible y cuántos píxeles ocupa en la imagen.")
        referencia_mm = st.number_input("Longitud real de referencia (mm)", min_value=0.1, value=50.0)
        referencia_px = st.number_input("Longitud en la imagen (px)", min_value=0.1, value=100.0)
        mm_por_px = referencia_mm / referencia_px

# ------------------------------------------------------------------
# Area principal: resultados
# ------------------------------------------------------------------
imagen = None
if archivo_imagen is not None:
    imagen = decodificar_imagen(archivo_imagen.getvalue())
    if imagen is None:
        st.error("No se pudo leer el archivo como imagen.")

if imagen is None:
    st.info("Carga una fotografía desde la barra lateral para iniciar la inspección.")
else:
    # Modelo IA
    modelo = None
    error_modelo = None
    try:
        modelo = cargar_modelo_ia()
    except Exception as exc:
        error_modelo = str(exc)
    if error_modelo:
        st.warning(f"Diagnóstico con IA no disponible: {error_modelo}")

    diagnostico_ia, prob_grieta = None, None
    if modelo is not None:
        diagnostico_ia, prob_grieta = diagnosticar_con_ia(modelo, imagen)

    # Analisis de grietas y desplome
    analisis = analizar_grietas(imagen, mm_por_px)
    desplome = medir_desplome(imagen)
    principal = analisis["principal"]

    # Ancho critico: maximo ancho medio entre TODAS las ramas calibradas
    ancho_critico_mm = None
    if analisis["lesiones"]:
        anchos_calibrados = [
            les["ancho_medio_mm"]
            for les in analisis["lesiones"]
            if les["ancho_medio_mm"] is not None
        ]
        if anchos_calibrados:
            ancho_critico_mm = max(anchos_calibrados)

    # Dictamen de riesgo estructural
    riesgo = evaluar_riesgo(ubicacion, material, ancho_critico_mm, desplome)

    # Imagenes procesadas
    c_fisura, c_muro = st.columns(2)
    c_fisura.image(
        cv2.cvtColor(analisis["imagen"], cv2.COLOR_BGR2RGB),
        caption="Segmentación de fisuras: rojo = grieta, amarillo = fisura (sin calibrar o < 1 mm).",
        width="stretch",
    )
    c_muro.image(
        cv2.cvtColor(_dibujar_muro(imagen, desplome), cv2.COLOR_BGR2RGB),
        caption="Desplome: magenta = bordes verticales usados, verde = vertical de referencia (0°).",
        width="stretch",
    )

    # Estado de calibracion
    if mm_por_px is None:
        st.info("Sin calibración física: los anchos se muestran en píxeles y no se calcula el riesgo estructural.")
    else:
        st.success(f"Medición calibrada ({mm_por_px:.4f} mm/px). El dictamen normativo usa el ancho crítico máximo entre ramas.")

    # Métricas principales
    m1, m2, m3 = st.columns(3)
    m1.metric("Diagnóstico IA", diagnostico_ia or "No disponible")
    m2.metric("Prob. de grieta (IA)",
              f"{prob_grieta * 100:.1f} %" if prob_grieta is not None else "No disponible")
    m3.metric("Desplome del muro",
              describir_desplome(desplome["angulo_deg"]) if desplome["valido"] else "No determinado")

    m4, m5, m6 = st.columns(3)
    m4.metric("Ancho medio",
              f"{principal['ancho_medio_mm']:.2f} mm" if principal and principal["ancho_medio_mm"] is not None
              else (f"{principal['ancho_medio_px']:.1f} px" if principal else "No identificado"))
    m5.metric("Ancho crítico (máx.)",
              f"{ancho_critico_mm:.2f} mm" if ancho_critico_mm is not None else "Sin calibrar",
              help="Máximo ancho medio entre todas las ramas detectadas. Se usa para el dictamen normativo.")
    m6.metric("Clasificación geométrica", principal["clase"] if principal and principal["clase"] else "Sin calibrar")

    # Tabla de lesiones detectadas
    if principal is None:
        st.info("No se midió ninguna fisura con la visión clásica. Prueba con mejor iluminación o un encuadre más cercano.")
    else:
        filas = []
        for les in analisis["lesiones"]:
            filas.append({
                "N°": les["n"],
                "Clasificación": les["clase"] or "Sin calibrar",
                "Extensión (px)": round(les["extension_px"], 1),
                "Ancho medio (px)": round(les["ancho_medio_px"], 2),
                "Ancho máximo P95 (px)": round(les["ancho_maximo_px"], 2),
                "Ancho medio (mm)": None if les["ancho_medio_mm"] is None else round(les["ancho_medio_mm"], 2),
                "Contraste": round(les["contraste"], 3),
            })
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)

    # Cuadro destacado de riesgo estructural
    color = riesgo["color"]
    emoji = {"rojo": "🔴", "amarillo": "🟡", "verde": "🟢", "gris": "⚪"}[color]
    box_class = f"risk-box risk-{color}"
    texto_ancho_critico = (
        f"Ancho crítico evaluado: {ancho_critico_mm:.2f} mm (máximo entre ramas)."
        if ancho_critico_mm is not None
        else "Sin calibración métrica: el dictamen no usa ancho en milímetros."
    )
    st.markdown(f"""
    <div class="{box_class}">
        <h4 style="margin:0 0 0.5rem 0; color:{texto_color};">{emoji} NIVEL DE RIESGO: {riesgo['nivel']}</h4>
        <p style="margin:0.25rem 0; color:{secundario};"><strong>{texto_ancho_critico}</strong></p>
        <p style="margin:0.25rem 0; color:{texto_color};"><strong>Dictamen:</strong> {riesgo['dictamen']}</p>
        <p style="margin:0.25rem 0; color:{secundario};"><strong>Justificación normativa:</strong> {riesgo['justificacion']}</p>
        <p style="margin:0.25rem 0 0 0; color:{texto_color};"><strong>Intervención recomendada:</strong> {riesgo['recomendacion']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Informe estructurado JSON
    informe = {
        "diagnostico_ia": diagnostico_ia,
        "probabilidad_grieta": prob_grieta,
        "modelo": MODELO_PATH.name if modelo is not None else None,
        "parametros_inspeccion": {
            "ubicacion": ubicacion,
            "material": material,
            "calibracion_mm_por_px": mm_por_px,
        },
        "fisuras": {
            "numero": len(analisis["lesiones"]),
            "ancho_critico_mm": ancho_critico_mm,
            "principal": _publico(principal) if principal is not None else None,
            "lesiones": [_publico(les) for les in analisis["lesiones"]],
        },
        "desplome": {
            "valido": desplome["valido"],
            "angulo_deg": desplome["angulo_deg"],
            "dispersion_deg": desplome["dispersion_deg"],
            "lineas_usadas": desplome["lineas_usadas"],
            "motivo": desplome["motivo"],
        },
        "riesgo_estructural": {
            "nivel": riesgo["nivel"],
            "dictamen": riesgo["dictamen"],
            "justificacion": riesgo["justificacion"],
            "recomendacion": riesgo["recomendacion"],
        },
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

    # Guardar en MySQL
    if principal is None or ancho_critico_mm is None:
        st.button(
            "Guardar inspección en MySQL",
            disabled=True,
            help="Requiere una fisura medida con calibración física.",
        )
    elif prob_grieta is None:
        st.button(
            "Guardar inspección en MySQL",
            disabled=True,
            help="Requiere el modelo de IA (modelo_grietas.h5).",
        )
    elif st.button("Guardar inspección en MySQL", icon=":material/save:"):
        # Campos de audio (frecuencia, centroide) en 0.0 para preservar la estructura de la BD
        if guardar_inspeccion(
            ancho_critico_mm,
            0.0,
            0.0,
            ubicacion,
            diagnostico_ia,
            riesgo["riesgo_bd"],
            prob_grieta,
        ):
            st.success("Guardado en base de datos correctamente.")
