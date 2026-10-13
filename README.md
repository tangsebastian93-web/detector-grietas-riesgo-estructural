#  Sistema Multimodal de Detección de Grietas, Inclinación y Riesgo Estructural

**Asignatura:** Algoritmos y Programación (2026-2)  
**Institución:** Universidad Industrial de Santander (UIS)  
**Proyecto:** Reto de Detección de Grietas, Inclinación y Riesgo en Edificaciones (Fase 1)

---

##  Descripción del Proyecto
Este proyecto consiste en un prototipo funcional de extremo a extremo (*end-to-end*) diseñado para asistir en la inspección visual de patologías en concreto (muros, vigas, columnas y losas). Combina técnicas avanzadas de **Visión por Computadora** en OpenCV, análisis morfológico multiescala y una interfaz interactiva desplegada en **Streamlit**.

### Capacidades Principales:
1. **Detección y Medición de Lesiones:** Preprocesamiento adaptativo con espacio de color $L*a*b*$, ecualización CLAHE y filtrado Black-Hat para aislar cavidades oscuras sin sufrir sobreestimación por concreto erosionado.
2. **Medición Calibrada:** Cálculo de **Ancho Promedio** ($\text{Ancho} = \frac{2 \times \text{Área}}{\text{Perímetro}}$) y **Ancho Máximo** (Distance Transform), con opción de calibración en milímetros reales ($mm$).
3. **Análisis de Inclinación y Desaplome:** Cálculo del eje longitudinal mediante geometría envolvente (`minAreaRect`) y ángulo respecto a la vertical ($0^\circ$), indicando la orientación exacta (Vertical, Diagonal o Horizontal) y su dirección.
4. **Persistencia e Integración:** Exportación de informes estructurados en JSON, registro histórico en base de datos MySQL e integración de hardware mediante ESP8266 / HC-SR04.

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
