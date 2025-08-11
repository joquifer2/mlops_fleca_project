import os
from dotenv import load_dotenv
from paths import ROOT_DIR

# Cargamos las variables de entorno
load_dotenv(os.path.join(ROOT_DIR, ".env"))

try:
    HOPSWORKS_PROJECT_NAME = os.environ["HOPSWORKS_PROJECT_NAME"]
    HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]

except KeyError as e:
    raise Exception(f'Create a .env file on the {ROOT_DIR} directory with the following variables: {e}')

# Eliminar FEATURE_GROUP_NAME and FEATURE_GROUP:_VERSION, and use FEATURE_GROUP_METADATA instead
FEATURE_GROUP_NAME = 'time_series_bolleria_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_VERSION = 1

