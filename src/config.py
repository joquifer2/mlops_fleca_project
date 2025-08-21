import os
from dotenv import load_dotenv
from src.paths import ROOT_DIR

# Cargamos las variables de entorno
load_dotenv(os.path.join(ROOT_DIR, ".env"))

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

MODEL_NAME = "fleca_bolleria_predictor_next_week"
MODEL_VERSION = 1
MODEL_FILE = "xgboost_hopsworks.pkl"



