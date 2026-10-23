# Sistema Multimodal de Detección de Grietas, Inclinación y Riesgo Estructural

**Institución:** Universidad Industrial de Santander (UIS)  
**Asignatura:** Algoritmos y Programación  
**Normativa de Referencia:** NSR-10 (Colombia) / ACI 224R-01  

---

## Descripción del Proyecto

Sistema experto de inspección patológica *end-to-end* diseñado para evaluar fisuras e inclinaciones en elementos de concreto (columnas, vigas, losas y muros)[cite: 13]. El software integra un enfoque **híbrido** que combina Inteligencia Artificial (redes neuronales convolucionales) para el reconocimiento de contexto[cite: 13] y Visión por Computadora clásica para la caracterización geométrica de precisión y evaluación de riesgo bajo norma técnica[cite: 13].

---

## Capacidades Principales

1. **Diagnóstico con IA:** Clasificación binaria (*Con Grieta* / *Sin Grieta*) empleando Transfer Learning (**MobileNetV2**)[cite: 13] entrenado sobre imágenes estructurales reales (`modelo_grietas.h5`).
2. **Evaluación de Ancho Crítico ($w_{\text{máx}}$):** Aislamiento multirrama mediante filtros morfológicos (Frangi / Valles oscuros) y medición por **Transformada de Distancia ($L_2$)**. El dictamen evalúa el espesor medio máximo entre todas las ramas detectadas para garantizar seguridad en la inspección[cite: 13].
3. **Cálculo de Desaplome / Inclinación:** Detección de bordes verticales mediante **Canny** y la **Transformada de Hough Lineal Adaptativa** (`cv2.HoughLinesP`) para medir desviaciones angulares respecto a la vertical ($0^\circ$) en elementos portantes[cite: 13].
4. **Matriz de Riesgo Normativo (NSR-10 / ACI 224R-01):** Clasificación del nivel de peligro (*Bajo*, *Moderado*, *Alto*, *Crítico*) cruzando la apertura en milímetros y el desplome frente al elemento inspeccionado ($0.3\text{ mm}$ en columnas, $0.4\text{ mm}$ en vigas y $1.0\text{ mm}$ en muros divisorios).
5. **Persistencia e Informes:** Integración con base de datos **MySQL** para registro de auditoría e historial de inspecciones, más exportación de diagnósticos en formato **JSON**[cite: 13].

---

## Estructura del Repositorio

```text
Proyecto ia/
├── data/                       # Dataset e imágenes de prueba en entorno real
├── notebooks/                  # EDA, procesamiento e hiperparámetros de MobileNetV2
├── sql/                        # Script DDL de la base de datos MySQL
├── src/                        # Código fuente de la aplicación
│   ├── paginaweb.py            # Interfaz interactiva principal (Streamlit)
│   └── index.html              # Vista secundaria / landing
├── modelo_grietas.h5           # Pesos del modelo de IA entrenado
├── requirements.txt            # Dependencias del proyecto Python
├── .gitignore                  # Archivos excluidos de control de versiones
└── README.md                   # Documentación principal del sistema
