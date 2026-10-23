# Sistema Multimodal de Detección de Grietas, Inclinación y Riesgo Estructural

**Institución:** Universidad Industrial de Santander (UIS)  
**Asignatura:** Algoritmos y Programación (2026-2)  
**Proyecto:** Sistema de Inspección Patológica Estructural  
**Normativa de Referencia:** NSR-10 (Colombia) / ACI 224R-01  

---

## Descripción del Proyecto

Este proyecto consiste en una plataforma de software *end-to-end* diseñada para la inspección y diagnóstico preliminar de patologías en estructuras de concreto (muros, columnas, vigas y losas). El sistema utiliza una arquitectura **híbrida** que integra:

1. **Inteligencia Artificial (Visión por Computadora):** Clasificación contextual mediante redes neuronales convolucionales con aprendizaje por transferencia.
2. **Procesamiento Digital de Imágenes (PDI):** Caracterización geométrica de alta precisión (extracción de ramas, medición de espesores locales y estimación de ángulos de desaplome).
3. **Matriz de Diagnóstico Normativo:** Evaluación automatizada del nivel de riesgo estructural bajo los parámetros de tolerancia de la norma colombiana **NSR-10** y el comité internacional **ACI 224R-01**.

---

## Capacidades Principales

* **Diagnóstico Contextual por IA:** Clasificación binaria (*Con Grieta* / *Sin Grieta*) utilizando el modelo **MobileNetV2** (`modelo_grietas.h5`), entregando la probabilidad de certeza de la lesión.
* **Cálculo de Ancho Crítico ($w_{\text{máx}}$):** Segmentación multirrama con filtros morfológicos (Frangi / Valles Oscuros) y medición por **Transformada de Distancia ($L_2$)**. El dictamen evalúa el ancho medio máximo entre todas las ramas detectadas para garantizar seguridad técnica.
* **Estimación de Inclinación y Desaplome:** Detección de bordes verticales mediante **Canny** y la **Transformada de Hough Lineal Adaptativa** (`cv2.HoughLinesP`), calculando el ángulo de desviación respecto a la vertical ($0^\circ$).
* **Matriz de Riesgo Normativo:** Clasificación en cuatro niveles de severidad (*Bajo*, *Moderado*, *Alto*, *Crítico*) al comparar la apertura medida ($mm$) y el desaplome ($^\circ$) según el tipo de elemento estructural:
  * **Columnas / Muros de Carga:** Umbral crítico de $0.3\text{ mm}$.
  * **Vigas / Losas:** Umbral crítico de $0.4\text{ mm}$.
  * **Paredes Divisorias / Tabiques:** Umbral de $1.0\text{ mm}$.
* **Persistencia e Informes:** Integración con base de datos **MySQL** para registro de auditoría e historial de inspecciones, más exportación de diagnósticos detallados en formato **JSON**.

---

## Estructura del Repositorio

```text
Proyecto ia/
├── data/                       # Dataset de imágenes de entrenamiento y prueba
├── notebooks/                  # Cuadernos Jupyter (EDA, entrenamiento de MobileNetV2)
│   └── 01_analisis_y_linea_base.ipynb
├── sql/                        # Script DDL para la base de datos MySQL
│   └── detctor_de_grietas.SQL
├── src/                        # Código fuente de la aplicación
│   ├── paginaweb.py            # Interfaz gráfica interactiva en Streamlit
│   └── index.html              # Vista landing secundaria
├── modelo_grietas.h5           # Pesos del modelo de IA entrenado (MobileNetV2)
├── requirements.txt            # Dependencias y librerías de Python
├── .gitignore                  # Reglas de exclusión de Git
└── README.md                   # Documentación principal del repositorio

Requisitos e Instalación
1. Prerrequisitos
Asegúrate de tener instalado Python 3.10 o superior en tu sistema.

2. Clonar el repositorio
Bash
git clone [https://github.com/tangsebastian93-web/detector-grietas-riesgo-estructural.git](https://github.com/tangsebastian93-web/detector-grietas-riesgo-estructural.git)
cd detector-grietas-riesgo-estructural
3. Crear y activar un entorno virtual (Opcional pero recomendado)
Windows:

Bash
python -m venv .venv
.venv\Scripts\activate
Linux / macOS:

Bash
python3 -m venv .venv
source .venv/bin/activate
4. Instalar dependencias
Bash
pip install -r requirements.txt
5. Ejecutar la aplicación
Bash
streamlit run src/paginaweb.py
Flujo de Procesamiento
Fragmento de código
graph TD
    A[Carga de Imagen] --> B[Clasificación IA: MobileNetV2]
    B --> C[Segmentación y Filtrado OpenCV]
    C --> D[Transformada de Distancia y Ancho Crítico]
    D --> E[Transformada de Hough y Desaplome]
    E --> F[Matriz de Riesgo NSR-10 / ACI 224R-01]
    F --> G[Visualización Streamlit / Reporte JSON / MySQL]