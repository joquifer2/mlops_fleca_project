
import sys
from pathlib import Path
import os

# Añadir la carpeta raíz del proyecto al sys.path si no está presente
ROOT_DIR = Path(__file__).resolve().parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.append(root_str)

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'

# Directorios de modelos
MODELS_DIR = ROOT_DIR / 'models'


# Otros paths útiles
MODELS_DIR = ROOT_DIR / 'models'
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'
REPORTS_DIR = ROOT_DIR / 'reports'

# Puedes añadir aquí más rutas según se necesiten

if not Path(MODELS_DIR).exists():
    os.makedirs(MODELS_DIR)
if not Path(PROCESSED_DIR).exists():
    os.makedirs(PROCESSED_DIR)
if not Path(INTERIM_DIR).exists():
    os.makedirs(INTERIM_DIR)
if not Path(RAW_DIR).exists():
    os.makedirs(RAW_DIR)
