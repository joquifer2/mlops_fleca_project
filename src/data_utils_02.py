# --- Imports ---
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from google.cloud import bigquery

# --- Variables globales y paths ---
FAMILIA = 'BOLLERIA'  # Cambia aquí la familia que desees procesar

# --- Función para importación robusta de paths centralizados ---
def get_paths():
    try:
        from src.paths import RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA
        return RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        try:
            from src.paths import RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA
            return RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA
        except ImportError:
            print("AVISO: Usando rutas alternativas para paths centralizados")
            data_dir = Path(__file__).resolve().parent.parent / 'data'
            RAW_BQ_PARQUET = data_dir / 'raw' / 'raw_data_bq_forecasting_20250630.parquet'
            VALIDATED_RANGE_SEMANAL_FAMILIA = data_dir / 'interim' / 'validated_range_semanal_familia_20250630.parquet'
            return RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA

RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA = get_paths()

def transformar_a_series_temporales(
    df_raw,
    fecha_inicio='2023-01-02',
    fecha_fin='2025-06-29',
    familia=None,
    output_path=None,
    min_dias_semana=7,
    guardar_interim=False
):
    """
    Limpia, homogeneiza y agrega los datos diarios a series semanales completas para la familia indicada o todas si familia=None.
    """
    df = df_raw.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df[(df['fecha'] >= fecha_inicio) & (df['fecha'] <= fecha_fin)]
    if 'familia' in df.columns:
        df.loc[df['familia'] == 'BEBIDA', 'familia'] = 'BEBIDAS'
    for col in ['base_imponible', 'total']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if 'is_summer_peak' not in df.columns:
        df['is_summer_peak'] = df['fecha'].dt.month.isin([7, 8]).astype(int)
    if 'is_easter' not in df.columns:
        easter_ranges = {
            2023: pd.date_range('2023-04-03', '2023-04-09'),
            2024: pd.date_range('2024-03-25', '2024-03-31'),
            2025: pd.date_range('2025-04-14', '2025-04-20'),
        }
        df['is_easter'] = 0
        for year, dates in easter_ranges.items():
            df.loc[df['fecha'].dt.date.isin(dates.date), 'is_easter'] = 1
    easter_weeks = []
    for year, dates in {
            2023: pd.date_range('2023-04-03', '2023-04-09'),
            2024: pd.date_range('2024-03-25', '2024-03-31'),
            2025: pd.date_range('2025-04-14', '2025-04-20'),
        }.items():
        for date in dates:
            easter_weeks.append((date.isocalendar().year, date.isocalendar().week))
    easter_weeks = list(set(easter_weeks))
    iso = df['fecha'].dt.isocalendar()
    df['year_iso'] = iso['year']
    df['week_iso'] = iso['week']
    if familia is not None and isinstance(familia, str) and familia != '' and familia.lower() != 'none':
        conteo_dias = df.groupby(['year_iso','week_iso','familia'])['fecha'].nunique().reset_index(name='dias_semana')
        df_semanal = (
            df.groupby(['year_iso','week_iso','familia'], as_index=False)
              .agg({
                 'base_imponible': 'sum',
                 'is_summer_peak': 'max',
                 'is_easter':      'max'
              })
            .merge(conteo_dias, on=['year_iso','week_iso','familia'])
        )
        df_semanal = df_semanal[df_semanal['dias_semana'] >= min_dias_semana]
        df_familia_semanal = df_semanal.query(f"familia=='{familia}'").copy()
    else:
        conteo_dias = df.groupby(['year_iso','week_iso'])['fecha'].nunique().reset_index(name='dias_semana')
        df_semanal = (
            df.groupby(['year_iso','week_iso'], as_index=False)
              .agg({
                 'base_imponible': 'sum',
                 'is_summer_peak': 'max',
                 'is_easter':      'max'
              })
            .merge(conteo_dias, on=['year_iso','week_iso'])
        )
        df_semanal = df_semanal[df_semanal['dias_semana'] >= min_dias_semana]
        df_familia_semanal = df_semanal.copy()
    for year_iso, week_iso in easter_weeks:
        mask = (df_familia_semanal['year_iso'] == year_iso) & (df_familia_semanal['week_iso'] == week_iso)
        if len(df_familia_semanal[mask]) > 0:
            df_familia_semanal.loc[mask, 'is_easter'] = 1
    df_familia_semanal.rename(columns={'year_iso':'year','week_iso':'week'}, inplace=True)
    df_familia_semanal['week_start'] = pd.to_datetime(
        df_familia_semanal['year'].astype(str) + '-W' + df_familia_semanal['week'].astype(str) + '-1',
        format='%G-W%V-%u'
    )
    df_familia_semanal = df_familia_semanal.sort_values('week_start').reset_index(drop=True)
    if output_path:
        df_familia_semanal.to_parquet(str(output_path), index=False)
        print(f"Series temporales guardadas en: {output_path}")
    if guardar_interim:
        guardar_time_series_interim(df_familia_semanal, familia if familia else 'TODAS')
    return df_familia_semanal

def guardar_time_series_interim(df, familia, interim_dir=None):
    """
    Guarda el DataFrame de series temporales en la carpeta interim con nombre:
    time_series_{familia}_weekly_{timestamp}.parquet
    Utiliza una ruta absoluta a la carpeta data/interim del proyecto.
    """
    import os
    import datetime
    from pathlib import Path
    if interim_dir is None:
        try:
            module_path = Path(__file__).resolve().parent
            project_root = module_path.parent
            interim_dir = project_root / 'data' / 'interim'
        except:
            print("AVISO: No se pudo determinar la ruta del proyecto, usando 'data/interim'")
            interim_dir = 'data/interim'
    os.makedirs(interim_dir, exist_ok=True)
    date_only = datetime.datetime.now().strftime('%Y%m%d')
    filename = f"time_series_{familia}_weekly_{date_only}.parquet"
    filepath = os.path.join(str(interim_dir), filename)
    df.to_parquet(filepath, index=False)
    print(f"Archivo guardado en: {filepath}")
    return filepath

def cargar_datos_raw(parquet_file):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_file)
    schema = table.schema
    dbdate_cols = [field.name for field in schema if str(field.type) == 'dbdate']
    for col in dbdate_cols:
        table = table.set_column(table.schema.get_field_index(col), col, table.column(col).cast('string'))
    df_result = table.to_pandas(ignore_metadata=True)
    return df_result


def load_raw_data(
    parquet_path=None,
    fecha_inicio='2023-01-01',
    fecha_fin='2025-06-30',
    descargar_bq=False
):
    raw_bq_parquet, _ = get_paths()
    if parquet_path is None:
        parquet_path = raw_bq_parquet
    df_raw = cargar_datos_raw(parquet_path)
    df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])
    df_raw = df_raw[(df_raw['fecha'] >= fecha_inicio) & (df_raw['fecha'] <= fecha_fin)]
    if 'familia' in df_raw.columns:
        df_raw.loc[df_raw['familia'] == 'BEBIDA', 'familia'] = 'BEBIDAS'
    for col in ['base_imponible', 'total']:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(0)
    if 'is_summer_peak' not in df_raw.columns:
        df_raw['is_summer_peak'] = df_raw['fecha'].dt.month.isin([7,8]).astype(int)
    if 'is_easter' not in df_raw.columns:
        df_raw['is_easter'] = 0
    return df_raw


def generar_lags(df, lags_list, columna='base_imponible'):
    df_lags = df.copy()
    if 'week_start' in df_lags.columns:
        df_lags = df_lags.sort_values('week_start')
    elif 'year' in df_lags.columns and 'week' in df_lags.columns:
        df_lags = df_lags.sort_values(['year', 'week'])
    for lag in lags_list:
        df_lags[f'{columna}_lag{lag}'] = df_lags[columna].shift(lag)
    return df_lags


def generar_target(df, columna='base_imponible', periodos_adelante=1):
    df_target = df.copy()
    if 'week_start' in df_target.columns:
        df_target = df_target.sort_values('week_start')
    elif 'year' in df_target.columns and 'week' in df_target.columns:
        df_target = df_target.sort_values(['year', 'week'])
    target_name = f'{columna}_next{periodos_adelante}'
    df_target[target_name] = df_target[columna].shift(-periodos_adelante)
    return df_target, target_name


def transformar_features_target(
    df, 
    lags_list=[1, 2, 3, 4], 
    columna_target='base_imponible',
    cols_exogenas=None,
    periodos_adelante=1,
    eliminar_nulos=True
):
    if cols_exogenas is None:
        cols_exogenas = []
    df_features = generar_lags(df, lags_list, columna_target)
    df_completo, target_name = generar_target(
        df_features, 
        columna_target, 
        periodos_adelante
    )
    cols_lags = [f'{columna_target}_lag{lag}' for lag in lags_list]
    extra_cols = []
    if 'week_start' in df_completo.columns:
        extra_cols.append('week_start')
    drop_cols = ['familia', 'year', 'week', 'dias_semana']
    df_completo = df_completo.drop(columns=[col for col in drop_cols if col in df_completo.columns])
    X = df_completo[cols_lags + cols_exogenas + extra_cols]
    y = df_completo[target_name]
    if eliminar_nulos:
        mask_completos = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask_completos]
        y = y[mask_completos]
        df_completo = df_completo[mask_completos]
    return X, y, df_completo


def guardar_datos_procesados(X, y, df_completo, familia='BOLLERIA', processed_dir=None):
    from pathlib import Path
    import os
    import datetime
    if processed_dir is None:
        module_path = Path(__file__).resolve().parent
        project_root = module_path.parent
        processed_dir = project_root / 'data' / 'processed'
    os.makedirs(processed_dir, exist_ok=True)
    date_only = datetime.datetime.now().strftime('%Y%m%d')
    files = {}
    x_filename = f"ts_X_{familia.lower()}_{date_only}.parquet"
    x_filepath = os.path.join(str(processed_dir), x_filename)
    X.to_parquet(x_filepath, index=False)
    files['X'] = x_filepath
    y_filename = f"ts_y_{familia.lower()}_{date_only}.parquet"
    y_filepath = os.path.join(str(processed_dir), y_filename)
    y.to_frame().to_parquet(y_filepath, index=False)
    files['y'] = y_filepath
    df_filename = f"ts_df_{familia.lower()}_{date_only}.parquet"
    df_filepath = os.path.join(str(processed_dir), df_filename)
    df_completo.to_parquet(df_filepath, index=False)
    files['df_completo'] = df_filepath
    print(f"Datos procesados guardados en la carpeta: {processed_dir}")
    print(f"- Features (X): {x_filename}")
    print(f"- Target (y): {y_filename}")
    print(f"- Dataset completo: {df_filename}")
    return files
