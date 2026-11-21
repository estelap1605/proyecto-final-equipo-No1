# proyecto-final-equipo-No1
Proyecto para diplomado de IA, Ciencia de datos y Business Intelligence de la Universidad Anáhuac Querétaro

Proyecto: Pronóstico de demanda semanal en centros comerciales de Estambul

Autores: Ing. Maria Estela Pérez Hernández / Ing. Andres Roque Rangel

Resumen:

El presente estudio aborda la optimización de la gestión de inventarios y la planeación de compras en el
sector de retail físico, analizando transacciones de diez centros comerciales en Estambul, Turquía. El
objetivo principal es predecir la demanda diaria y semanal para mitigar faltantes de stock (*stockouts*) y
reducir el capital inmovilizado. Se utilizó el Customer Shopping Dataset, compuesto por 99,457 transacciones
limpias registradas a lo largo de 797 días continuos entre enero de 2021 y marzo de 2023. Para la
estimación, se desarrolló y comparó una arquitectura de Red Neuronal Recurrente (SimpleRNN / LSTM) frente
a modelos estadísticos tradicionales de series temporales (SARIMAX) y regresiones lineales de referencia.
Adicionalmente, se conectó la solución a un modelo semántico estrella en Microsoft Fabric y un tablero en
Power BI. Los resultados experimentales demostraron la superioridad de la red recurrente, alcanzando un
Error Porcentual Absoluto Ponderado (WAPE) del 7.93%, un Error Absoluto Medio (MAE) de 29.83
unidades/día y una Raíz del Error Cuadrático Medio (RMSE) de 36.47 unidades/día en el conjunto de prueba,
superando ampliamente a SARIMAX (WAPE del 16.63%). El pronóstico recursivo proyectó una demanda
estable de 11,275 unidades a 30 días. Se recomienda un piloto operativo de seis semanas en Compras y
Logística


Resumen Técnico Ordenado del Desarrollo del Sistema (14 Etapas)
De acuerdo con el plan del proyecto registrado en la arquitectura de software del repositorio, el desarrollo del
sistema de pronóstico de demanda siguió un flujo estructurado en 14 etapas ordenadas:

<img width="672" height="341" alt="image" src="https://github.com/user-attachments/assets/7d923843-8340-43a8-8636-3f760562b44a" />
<img width="672" height="75" alt="image" src="https://github.com/user-attachments/assets/c7dbc6c2-84d4-4595-ad1b-5825f2481528" />

Estructura del proyecto:

<img width="362" height="860" alt="image" src="https://github.com/user-attachments/assets/b494cd2c-4411-4dbe-b347-0c111f74474d" />



Instalación y configuración

Se recomienda utilizar Python 3.12 y un entorno virtual independiente.

Activar el entorno virtual en Windows:

.\.venv\Scripts\Activate.ps1

Instalar las dependencias:

python -m pip install numpy pandas scikit-learn scipy pyyaml holidays statsmodels tensorflow matplotlib

Verificar las dependencias:

python -c "import numpy, pandas, sklearn, scipy, yaml, holidays, statsmodels, tensorflow, matplotlib; print('Dependencias OK')"

Orden de ejecución

Todos los comandos deben ejecutarse desde la raíz del proyecto.

Paso 1. Construcción del dataset de ventas

python -m src.ingestion.build_sales_dataset

Paso 2. Descarga de información meteorológica

python -m src.ingestion.download_weather

Paso 3. Generación de días festivos

python -m src.ingestion.download_holidays

Paso 4. Integración de datasets

python -m src.ingestion.merge_datasets

Paso 5. Preparación de datasets para modelado

python -m src.modeling.prepare_dataset

Paso 6. Entrenamiento y evaluación de modelos

python -m src.modeling.train_models

Es importante respetar este orden, ya que cada etapa genera los archivos requeridos por la siguiente.

Resultados y entregables

Los resultados del proyecto se almacenan en:

data/processed/model_results/

Se contemplan los siguientes entregables:

Modelos entrenados.

Predicciones de demanda.

Métricas de evaluación.

Comparación entre algoritmos.

Resultados de validación temporal.

Resultados de prueba final.

Gráficas de pérdida de entrenamiento.

Tecnologías utilizadas

Python 3.12.

NumPy.

Pandas.

Scikit-learn.

SciPy.

Statsmodels.

TensorFlow / Keras.

Matplotlib.

PyYAML.

Holidays.

Open-Meteo API.

Consideraciones finales

El proyecto implementa un flujo reproducible de procesamiento, preparación, entrenamiento y evaluación de modelos de pronóstico de demanda.

La integración de información histórica de ventas con variables meteorológicas y de calendario permite analizar distintos factores asociados al comportamiento de la demanda.

La comparación de Regresión Lineal, SARIMAX y LSTM permite evaluar diferentes enfoques estadísticos y de aprendizaje automático para identificar el modelo con mejor desempeño predictivo.

El objetivo final es proporcionar una herramienta de análisis y pronóstico que facilite la planeación de inventarios, la identificación de patrones de consumo y la toma de decisiones comerciales.
