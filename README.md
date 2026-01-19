# TFG — Imbalanced Datasets

Repositorio que agrupa código, datos y configuraciones para evaluar métodos de balanceo y clasificación sobre datasets desbalanceados. Incluye utilidades para extraer metadatos, generar JSONs por artículo, *wrappers* para balanceadores y clasificadores, y *pipelines* reproducibles. El punto de entrada principal es `main.py`. 

## Estructura principal

```
/TFG-imbalanced-datasets
├── README.md
├── requirements.txt
├── config/parameters.py
├── data/
├── img/
├── json/
├── src/
└── main.py
```

Los módulos clave están en `src/` (ingestión, balanceo, clasificadores, core, visualización, utilidades). El fichero de parámetros globales es `config/parameters.py`.  

## Requisitos

1. Dependencias listadas en `requirements.txt`.

Instalación rápida:

```bash
python -m venv .venv
source .venv/bin/activate    # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

## Configuración

Parámetros globales (rutas, métricas por defecto, métodos, opciones de análisis, etc.) están centralizados en `config/parameters.py`. Modifique ahí índices, rutas y *flags* (por ejemplo `EXTENDED_ANALYSIS`, `JSON_PATH`, `PATH_ARTICLES_XLSX`, `RESULTS_OUTPUT_CSV`) para reconfigurar la ejecución sin tocar la lógica fuente. 

## Uso

Ejecutar el menú interactivo:

```bash
python main.py
```

El programa muestra un menú con las opciones principales (1–9). Las acciones implementadas en `main.py` son, de forma concisa:

1. Registro manual de artículos (GUI).
2. Ingestión automática de artículos desde `PATH_ARTICLES_XLSX`.
3. Cargar JSONs y listar técnicas por artículo.
4. Resumen de métricas extraídas.
5. Análisis/validación de JSONs.
6. Construcción del meta-dataset y exportación a CSV (`TRAIN_CSV_OUTPUT`, `TEST_CSV_OUTPUT`).
7. Métodos para selección del número de clústeres (Elbow, Silhouette).
8. Evaluación de modelos de clustering (configurable mediante `MODELS`).
9. Ejecución de experimentos de balanceo y clasificación.  

## Rutas y ficheros relevantes

* JSONs generados: `JSON_PATH` (por defecto `json/`). 
* XLSX artículos: `PATH_ARTICLES_XLSX`. 
* CSVs meta-dataset: `TRAIN_CSV_OUTPUT`, `TEST_CSV_OUTPUT`.
* Resultados de experimentos: `RESULTS_OUTPUT_CSV`.

## Reproducibilidad

* Todas las constantes y rutas se gestionan desde `config/parameters.py` para facilitar la re-ejecución controlada. 
* Las *pipelines* del experimento se coordinan desde `src/core.py` y `main.py`.
* Los JSONs normalizados actúan como fuente para construir el meta-dataset. 

## Notas rápidas de desarrollo

* Para añadir un balanceador o clasificador, implemente el *wrapper* en `src/balance_methods.py` o `src/classification_methods.py` respectivamente, respetando la interfaz usada por el *pipeline*. 
* `src/visualization.py` genera las figuras de clústeres.