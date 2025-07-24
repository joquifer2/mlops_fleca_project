from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_DIR = ROOT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'

# Archivos raw
RAW_BQ_PARQUET = RAW_DIR / 'raw_data_bq_forecasting_20250630.parquet'


# Archivos interim
VALIDATED_RANGE_FECHA_FAMILIA = INTERIM_DIR / 'validated_range_fecha_familia_20250630.parquet'
VALIDATED_RANGE_SEMANAL_FAMILIA = INTERIM_DIR / 'validated_range_semanal_familia_20250630.parquet'
VALIDATED_RANGE_MONTHLY_FAMILIA = INTERIM_DIR / 'validated_range_monthly_familia_20250630.parquet'

# Archivos processed
TS_DF_BOLLERIA_SEMANAL = PROCESSED_DIR / 'ts_df_bolleria_semanal.parquet'
TS_DF_BOLLERIA_BASELINE = PROCESSED_DIR / 'ts_df_bolleria_baseline.parquet'

# Otros paths útiles
MODELS_DIR = ROOT_DIR / 'models'
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'
REPORTS_DIR = ROOT_DIR / 'reports'

# Puedes añadir aquí más rutas según se necesiten
