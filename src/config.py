
import os
from dotenv import load_dotenv
from src.paths import ROOT_DIR

# Cargamos las variables de entorno
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Bloque de depuración temporal
print("DEBUG ENV HOPSWORKS_PROJECT_NAME:", os.environ.get("HOPSWORKS_PROJECT_NAME"))
print("DEBUG ENV PATH:", os.path.join(ROOT_DIR, ".env"))

try:
    HOPSWORKS_PROJECT_NAME = os.environ["HOPSWORKS_PROJECT_NAME"]
    HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

except KeyError as e:
    raise Exception(f'Create a .env file on the {ROOT_DIR} directory with the following variables: {e}')

# Eliminar FEATURE_GROUP_NAME and FEATURE_GROUP:_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = 'times_series_bolleria_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'times_series_bolleria_feature_view'
FEATURE_VIEW_VERSION = 1

# Configuración de lags y columnas objetivo

COLUMNA_TARGET = "base_imponible"
COLS_EXOGENAS = ["is_easter", "is_summer_peak"]
# Configuración de periodos
PERIODOS_ADELANTE = 1  # Número de semanas a predecir

ELIMINAR_NULOS = True

LAGS = [1, 2, 3, 52]

MODEL_NAME = "fleca_bolleria_predictor_next_week"
MODEL_VERSION = 1
MODEL_FILE = "xgboost_hopsworks.pkl"

# Configuración del feature group y feature view de predicciones
PRED_FEATURE_GROUP_NAME = "bolleria_predicciones_feature_group"
PRED_FEATURE_GROUP_VERSION = 1
PRED_FEATURE_VIEW_NAME = "bolleria_predicciones_feature_view" 
PRED_FEATURE_VIEW_VERSION = 1

# Metadatos para feature groups y feature views
PRED_FEATURE_GROUP_METADATA = {
    "name": PRED_FEATURE_GROUP_NAME,
    "version": PRED_FEATURE_GROUP_VERSION,
    "description": "Predicciones semanales de base_imponible para bolleria",
    "primary_key": ["week_start"],
    "online_enabled": False,
    "statistics_config": {"enabled": True},
    "event_time": "week_start"
}

PRED_FEATURE_VIEW_METADATA = {
    "name": PRED_FEATURE_VIEW_NAME,
    "version": PRED_FEATURE_VIEW_VERSION,
    "description": "Vista de predicciones semanales de bolleria"
}

# También definimos metadatos para los datos históricos (ejemplo de metadatos reutilizables)
HISTORICAL_FEATURE_GROUP_METADATA = {
    "name": FEATURE_GROUP_NAME,
    "version": FEATURE_GROUP_VERSION,
    "description": "Datos históricos de ventas de bolleria",
    "primary_key": ["week_start"],
    "online_enabled": False,
    "statistics_config": {"enabled": True},
    "event_time": "week_start"
}

HISTORICAL_FEATURE_VIEW_METADATA = {
    "name": FEATURE_VIEW_NAME,
    "version": FEATURE_VIEW_VERSION,
    "description": "Vista de datos históricos de ventas de bolleria"
}


# ML FLOW
# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Cambia esto si usas un servidor remoto