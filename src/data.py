import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

FAMILIA = 'BOLLERIA'  # Cambia aquí la familia que desees procesar

# Importación robusta de paths centralizados
try:
    from src.paths import RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from src.paths import RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA
    except ImportError:
        print("AVISO: Usando rutas alternativas para paths centralizados")
        data_dir = Path(__file__).resolve().parent.parent / 'data'
        RAW_BQ_PARQUET = data_dir / 'raw' / 'raw_data_bq_forecasting_20250630.parquet'
        VALIDATED_RANGE_SEMANAL_FAMILIA = data_dir / 'interim' / 'validated_range_semanal_familia_20250630.parquet'

def cargar_datos_raw(parquet_file):
    print(f"Cargando datos desde: {parquet_file}")
    table = pq.read_table(parquet_file)
    schema = table.schema
    dbdate_cols = [field.name for field in schema if str(field.type) == 'dbdate']
    for col in dbdate_cols:
        table = table.set_column(table.schema.get_field_index(col), col, table.column(col).cast('string'))
    df = table.to_pandas(ignore_metadata=True)
    return df

# --- Agregación semanal para notebook 01 ---
def get_weekly_dataset(
    parquet_path=RAW_BQ_PARQUET,
    familia=FAMILIA,
    fecha_inicio='2023-01-01',
    fecha_fin='2025-06-30',
    output_path=None
):
    """
    Carga, limpia y transforma los datos diarios, y devuelve el DataFrame semanal.
    Guarda el resultado en output_path si se indica.
    """
   
    #-----------------------
    # Cargar datos raw
    # -----------------------

    df_raw = cargar_datos_raw(parquet_path)
    df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])
    # Filtrar fechas
    df_raw = df_raw[(df_raw['fecha'] >= fecha_inicio) & (df_raw['fecha'] <= fecha_fin)]
    # Homogeneizar familia
    if 'familia' in df_raw.columns:
        df_raw.loc[df_raw['familia'] == 'BEBIDA', 'familia'] = 'BEBIDAS'
    # Imputar nulos básicos
    for col in ['base_imponible', 'total']:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(0)
    # Variables exógenas mínimas
    if 'is_summer_peak' not in df_raw.columns:
        df_raw['is_summer_peak'] = df_raw['fecha'].dt.month.isin([7,8]).astype(int)
    if 'is_easter' not in df_raw.columns:
        df_raw['is_easter'] = 0
    # Agregación semanal
    df_weekly = (
        df_raw
        .set_index('fecha')
        .groupby('familia')[['base_imponible','total','is_easter','is_summer_peak']]
        .resample('W')
        .sum()
        .reset_index()
    )
    # Filtrar familia si se indica
    if familia:
        df_weekly = df_weekly[df_weekly['familia'] == familia].copy()
    # Guardar si se indica
    if output_path is not None:
        df_weekly.to_parquet(str(output_path), index=False)
    return df_weekly



if __name__ == "__main__":
    df_weekly = get_weekly_dataset(familia=FAMILIA)
    print(f"Mostrando las primeras filas para la familia: {FAMILIA}")
    print(df_weekly.head())