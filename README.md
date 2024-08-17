# Modelado predictivo de demanda energética en España orientado a la comercialización en el mercado mayorista

Este repositorio contiene el desarrollo de un proyecto de modelado predictivo de la demanda nacional de energía de España con un enfoque específico en las necesidades de los agentes que participan en el mercado mayorista de energía. Los modelos fueron diseñados para optimizar la creación de ofertas en el mercado diario e intradiario.

## Estructura del Repositorio

El proyecto sigue un pipeline de modelado predictivo, que se ha dividido en 4 fases. El objetivo de este enfoque es proporcionar una estructura flexible y modular que permita la reutilización del código para un entorno productivo de predicciones a nivel de negocio.

### Fases del Pipeline

| Fase | Inputs | Archivos de Código | Outputs |
|------|--------|--------------------|---------|
| **1: Análisis y limpieza de datos** | Datos originales de energía y características meteorológicas | `EDA_AND_DATA_PREPARATION.ipynb`, `eda_auxiliary_functions.py` | `clean_datasets/energy_clean_dataset.pkl`, `clean_datasets/weather_clean_dataset.pkl` |
| **2: Ingeniería de variables** | Datos limpios (output de la Fase 1) | `FEATURE_ENGINEERING.ipynb`, `fe_auxiliary_functions.py` | `modeling_datasets/basic_dataset.pkl`, `modeling_datasets/daily_market_dataset.pkl`, `modeling_datasets/third_session_dataset.pkl`, `modeling_datasets/sixth_session_dataset.pkl` |
| **3: Modelado, entrenamiento y predicción** | Conjuntos de datos de ingeniería de variables (output de la Fase 2) | `DATA_MODELING.ipynb`, `modeling_auxiliary_functions.py` | `results/results.csv` |
| **4: Evaluación de resultados** | Resultados de la fase de modelado (output de la Fase 3) | `results_analysis_dashboard.pbix` | (No aplica) |

## Uso del Proyecto

Cada fase del pipeline está documentada y ejecutada en archivos de Jupyter Notebook. Para reproducir los resultados, sigue el flujo de las fases indicadas.

### Requerimientos

Para ejecutar el proyecto, es necesario contar con las siguientes librerías de Python:
- Pandas
- Numpy
- Scikit-learn
- Scipy
- Seaborn
- Statsmodels
- Tqdm
- Xgboost


### Ejecución

1. Realizar el análisis y limpieza de datos con `EDA_AND_DATA_PREPARATION.ipynb`.
2. Ejecutar la ingeniería de variables en `FEATURE_ENGINEERING.ipynb`.
3. Entrenar los modelos y hacer predicciones en `DATA_MODELING.ipynb`.
4. Evaluar los resultados con el tablero de análisis en `results_analysis_dashboard.pbix`.

## Contribuciones

Este proyecto está abierto a mejoras y contribuciones. Si tienes ideas o deseas reportar algún problema, por favor abre un issue o envía un pull request.
