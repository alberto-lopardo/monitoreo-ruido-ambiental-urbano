🔊 Monitoreo de Ruido Ambiental Urbano
Análisis Estadístico, Espectral y Detección de Eventos Atípicos
Python
NumPy
Pandas
Matplotlib
Norma

![image alt](https://github.com/alberto-lopardo/monitoreo-ruido-ambiental-urbano/blob/main/fig1_serie_7dias.png?raw=true)

⚠️ Aviso: Proyecto de portafolio con datos 100% sintéticos generados
matemáticamente. No guarda relación con proyectos reales, clientes,
metodologías propietarias ni datos confidenciales de empleadores actuales
o anteriores. Las normativas citadas son de conocimiento público y su
aplicación es meramente ilustrativa.

📋 Descripción
Pipeline completo de análisis de ruido ambiental urbano sobre una campaña
simulada de 7 días continuos (168 horas, resolución horaria). El proyecto
reproduce el flujo de trabajo real de una consultora ambiental: desde la
ingesta de datos crudos del sonómetro hasta el reporte automatizado de
anomalías con criterios normativos.

El escenario modela una zona residencial/mixta con perfil de tráfico
diurno realista (doble pico mañana/tarde), ruido de fondo nocturno diferenciado,
y tres eventos atípicos inyectados deliberadamente para validar la detección.

🎯 Objetivos Técnicos
Simular la generación y limpieza de datos crudos de sonómetro
Calcular estadísticos horarios: Leq, L10, L90, Lmax, Lmin
Comparar niveles medidos contra límites de ISO 1996 / DS 38/11 (Chile) / Res. SRT 85/2012 (Argentina)
Detectar automáticamente anomalías por tres criterios independientes
Generar visualizaciones publicables y reporte de consola estructurado
📊 Visualizaciones Generadas
Figura	Contenido
fig1_serie_7dias.png	Serie temporal 7 días — Leq, Lmax, L90 + anomalías etiquetadas
fig2_perfil_horario.png	Perfil horario promedio 24 h con banda L10–L90
fig3_espectro_tob.png	Espectro en bandas de tercio de octava (diurno / nocturno / 24 h)
fig4_heatmap.png	Mapa de calor hora × frecuencia (espectro-temporal)
fig5_dashboard.png	Dashboard 4 paneles: excedencia, estadísticos, curva Ln, tabla de anomalías
🔍 Lógica de Detección de Anomalías
Se implementan tres criterios independientes basados en ISO 1996-2:2017:

Criterio A — Superación de Leq normativo
  Nocturno (22–07 h): Leq > 50 dBA
  Diurno   (07–22 h): Leq > 55 dBA

Criterio B — Evento impulsivo nocturno
  (Lmax − L90) > 20 dB en período nocturno

Criterio C — Superación de Lmax absoluto nocturno
  Lmax > 65 dBA en período nocturno
Eventos inyectados para validación:

03/10 03:00 — Recolección nocturna de residuos (+18 dB Lmax)
05/10 02:00 — Evento social / música amplificada (+22 dB Lmax)
06/10 06:00 — Inicio anticipado de obra (+15 dB Lmax)
⚙️ Tecnologías
Librería	Uso
NumPy	Generación de señales sintéticas, operaciones vectoriales
Pandas	Estructura de datos, agrupación horaria, exportación CSV
Matplotlib	Todas las visualizaciones (serie temporal, heatmap, dashboard)
SciPy	Estadísticos, percentiles, interpolación
🚀 Instalación y Uso
# Clonar el repositorio
git clone https://github.com/alberto-lopardo/monitoreo-acustico-urbano.git
cd monitoreo-acustico-urbano

# Instalar dependencias
pip install numpy pandas matplotlib scipy

# Ejecutar
python monitoreo_acustico_urbano.py
Las figuras se guardan automáticamente en el directorio de trabajo.
Para exportar el dataset sintético, descomentar la última línea del script.

📐 Normativa de Referencia
ISO 1996-1:2016 — Descripción, medición y evaluación del ruido ambiental
ISO 1996-2:2017 — Determinación de los niveles de ruido ambiental
DS 38/11 Chile — Norma de emisión de ruidos molestos generados por fuentes fijas
Res. SRT 85/2012 Argentina — Límites de exposición al ruido en el trabajo
👤 Autor
Alberto Lopardo — Especialista en Acústica y Análisis de Señales
LinkedIn: https://www.linkedin.com/in/alberto-lopardo-acustica/

Proyecto de portafolio — datos 100% sintéticos — sin relación con mediciones reales.
