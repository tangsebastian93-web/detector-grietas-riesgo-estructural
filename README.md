#  Sistema Multimodal de Detección de Grietas, Inclinación y Riesgo Estructural

**Asignatura:** Algoritmos y Programación (2026-2)  
**Institución:** Universidad Industrial de Santander (UIS)  
**Proyecto:** Reto de Detección de Grietas, Inclinación y Riesgo en Edificaciones (Fase 1)

---

##  Descripción del Proyecto
Este proyecto consiste en un prototipo funcional de extremo a extremo (*end-to-end*) diseñado para asistir en la inspección visual de patologías en concreto (muros, vigas, columnas y losas). Combina técnicas avanzadas de **Visión por Computadora** en OpenCV, análisis morfológico multiescala y una interfaz interactiva desplegada en **Streamlit**.

### Capacidades Principales:
1. **Diagnóstico con IA:** Clasificador binario `Con Grieta` / `Sin Grieta` con Transfer Learning (MobileNetV2), entrenado en `notebooks/01_analisis_y_linea_base.ipynb` y guardado como `modelo_grietas.h5`. Devuelve la probabilidad de grieta.
2. **Medición de Fisuras:** Detección por filtro de valles oscuros e histéresis. El ancho se mide con `cv2.distanceTransform` sobre la máscara: ancho local = 2 × distancia en el eje central. Se expresa en píxeles y en mm si hay calibración física.
3. **Desplome del Muro o Columna:** Canny + `cv2.HoughLinesP` sobre los bordes verticales del elemento. El ángulo respecto a la vertical ($0^\circ$) es la mediana ponderada de esas líneas. La fisura no interviene en este cálculo.
4. **Persistencia e Integración:** Exportación de informes estructurados en JSON, registro histórico en base de datos MySQL e integración de hardware mediante ESP8266 / HC-SR04.

> **Dataset:** colocar las imágenes en `data/dataset_grietas/Positive/` (con grieta) y `data/dataset_grietas/Negative/` (sin grieta) antes de ejecutar el cuaderno.

---

##  Estructura del Repositorio

```text
Proyecto ia/
├── data/
│   └── imagenes_propias/          # Fotografías de prueba en entorno real
├── notebooks/
│   └── 01_analisis_y_linea_base.ipynb # EDA, preprocesamiento y entrenamiento CNN
├── sql/
│   └── detctor_de_grietas.SQL     # Script de creación de la base de datos MySQL
├── src/
│   ├── paginaweb.py               # Aplicación principal en Streamlit
│   └── index.html                 # Vista HTML secundaria
├── .gitignore                     # Archivos excluidos del control de versiones
├── README.md                      # Documentación del proyecto
└── requirements.txt               # Librerías y dependencias necesarias
