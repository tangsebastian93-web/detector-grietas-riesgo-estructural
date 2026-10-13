CREATE DATABASE IF NOT EXISTS detector_grietas;
USE detector_grietas;

-- Eliminamos la tabla anterior si existe para evitar conflictos con las columnas viejas
DROP TABLE IF EXISTS inspecciones;

CREATE TABLE inspecciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    distancia_cm FLOAT NOT NULL,
    ancho_mm FLOAT NOT NULL,
    frecuencia_hz FLOAT NOT NULL,
    centroide FLOAT NOT NULL,
    ubicacion VARCHAR(50) NOT NULL,
    diagnostico VARCHAR(50) NOT NULL,
    riesgo VARCHAR(100) NOT NULL,
    probabilidad_grieta FLOAT NOT NULL
);