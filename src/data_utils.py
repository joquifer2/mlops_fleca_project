

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
    """
    Devuelve los paths centralizados RAW_BQ_PARQUET y VALIDATED_RANGE_SEMANAL_FAMILIA de forma robusta.
    """
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

# Obtener los paths al inicio para uso global (mantiene compatibilidad)
RAW_BQ_PARQUET, VALIDATED_RANGE_SEMANAL_FAMILIA = get_paths()

# --- Descargar datos desde BigQuery ---
def descargar_datos_bigquery():
    """
    Descarga los datos desde BigQuery y guarda el DataFrame como archivo parquet en el path centralizado.
    """
    print("Iniciando conexión con BigQuery...")
    client = bigquery.Client()
    print("Conexión establecida.")
    print("Ejecutando consulta SQL...")
    query = """
    SELECT *
    FROM `fleca-del-port.varios.raw_data_bq_forecasting_20250630`
    WHERE fecha < '2025-07-01'
    """
    df = client.query(query).to_dataframe()
    print(f"Consulta finalizada. Filas descargadas: {len(df)}")
    output_path = RAW_BQ_PARQUET
    print(f"Guardando archivo en {output_path} ...")
    df.to_parquet(str(output_path), index=False)
    print("Archivo guardado correctamente.")
    return df


# ----------------------
# Cargar datos raw
# ----------------------

def cargar_datos_raw(parquet_file):
    print(f"Cargando datos desde: {parquet_file}")
    table = pq.read_table(parquet_file)
    schema = table.schema
    dbdate_cols = [field.name for field in schema if str(field.type) == 'dbdate']
    for col in dbdate_cols:
        table = table.set_column(table.schema.get_field_index(col), col, table.column(col).cast('string'))
    df = table.to_pandas(ignore_metadata=True)
    return df

#-----------------------
# Validar datos 
# -----------------------
def validar_fechas_completas(df, fecha_col='fecha', fecha_inicio='2023-01-01', fecha_fin='2025-06-30'):
    """
    Valida que todas las fechas diarias entre fecha_inicio y fecha_fin estén presentes en el DataFrame.
    Imprime un resumen y devuelve la lista de fechas faltantes.
    """
    
    fechas_completas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    fechas_presentes = pd.to_datetime(df[fecha_col].unique())
    missing_dates = np.setdiff1d(fechas_completas, fechas_presentes)
    print(f"Total de fechas faltantes: {len(missing_dates)}")
    if len(missing_dates) > 0:
        print("Fechas faltantes:", missing_dates)
    else:
        print("No faltan fechas en el rango especificado.")
    return missing_dates


# -----------------------
# Cargar datos raw y validar
# -----------------------   

def load_raw_data(
    parquet_path=None,
    fecha_inicio='2023-01-01',
    fecha_fin='2025-06-30',
    descargar_bq=False
):
    """
    Descarga, carga y valida los datos raw antes de la agregación semanal.
    Devuelve el DataFrame limpio y listo para agregación.
    """
    # Importación robusta de paths centralizados
    raw_bq_parquet, _ = get_paths()
    if parquet_path is None:
        parquet_path = raw_bq_parquet
    # 1. Descargar datos desde BigQuery y guardar parquet si se indica
    if descargar_bq:
        descargar_datos_bigquery()
    # 2. Cargar datos raw desde parquet
    df_raw = cargar_datos_raw(parquet_path)
    # 3. Validar continuidad temporal de fechas
    df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])
    df_raw = df_raw[(df_raw['fecha'] >= fecha_inicio) & (df_raw['fecha'] <= fecha_fin)]
    validar_fechas_completas(df_raw, fecha_col='fecha', fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    # 4. Homogeneizar familia
    if 'familia' in df_raw.columns:
        df_raw.loc[df_raw['familia'] == 'BEBIDA', 'familia'] = 'BEBIDAS'
    # 5. Imputar nulos básicos
    for col in ['base_imponible', 'total']:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(0)
    # 6. Variables exógenas mínimas
    if 'is_summer_peak' not in df_raw.columns:
        df_raw['is_summer_peak'] = df_raw['fecha'].dt.month.isin([7,8]).astype(int)
    if 'is_easter' not in df_raw.columns:
        df_raw['is_easter'] = 0
    return df_raw


# -----------------------
# Imputación de fechas faltantes
# -----------------------

def impute_missing_dates(df, missing_dates):
    """
    Imputa fechas faltantes en el DataFrame df, añadiendo filas con valores por defecto.
    Marca las filas imputadas con is_closed=1 y las originales con is_closed=0.
    """
    import numpy as np
    import pandas as pd

    # Crear DataFrame con fechas faltantes y valores por defecto
    df_missing_dates = pd.DataFrame({'fecha': missing_dates})
    df_missing_dates['n_factura'] = 'cerrado'
    df_missing_dates['zona_de_venta'] = 'cerrado'
    df_missing_dates['producto'] = 'cerrado'
    df_missing_dates['familia'] = 'cerrado'
    df_missing_dates['cantidad'] = 0.0
    df_missing_dates['base_imponible'] = 0.0
    df_missing_dates['tipo_IVA'] = 0.0
    df_missing_dates['total'] = 0.0
    df_missing_dates['is_closed'] = 1

    # Añadir columna is_closed al DataFrame original si no existe
    if 'is_closed' not in df.columns:
        df['is_closed'] = 0

    # Concatenar ambos DataFrames
    df_out = pd.concat([df, df_missing_dates], ignore_index=True)

    # Eliminar duplicados por fecha, dejando la primera ocurrencia
    df_out = df_out.drop_duplicates(subset=['fecha'], keep='first')

    # Ordenar por fecha
    df_out = df_out.sort_values('fecha').reset_index(drop=True)

    return df_out

# -----------------------
# Tratamiento e imputación de valores nulos
# -----------------------

def impute_null_values (df, cols_nulos=None, verbose=True):
    """
    Imputa valores nulos en columnas numéricas por familia y mes, usando medias de meses cercanos y anual.
    Marca las imputaciones en la columna 'is_imputed'.
    Args:
        df (pd.DataFrame): DataFrame con columnas 'fecha', 'familia' y columnas numéricas a imputar.
        cols_nulos (list): Lista de columnas a imputar. Si None, usa ['base_imponible', 'tipo_IVA', 'total'].
        verbose (bool): Si True, muestra progreso y resumen.
    Returns:
        pd.DataFrame: DataFrame con valores imputados y columna 'is_imputed'.
    """
    if cols_nulos is None:
        cols_nulos = ['base_imponible', 'tipo_IVA', 'total']
    df = df.copy()
    df['mes'] = df['fecha'].dt.to_period('M')
    if 'is_imputed' not in df.columns:
        df['is_imputed'] = 0
    meses_cercanos = [pd.Period('2023-10', 'M'), pd.Period('2023-12', 'M')]
    familias = df['familia'].unique()
    if verbose:
        print(f"Procesando {len(familias)} familias...")
    # Imputar por meses cercanos
    for columna in cols_nulos:
        for familia in familias:
            # Calcular la media de la categoría en los meses cercanos
            medias_cercanas = []
            for mes in meses_cercanos:
                media_categoria = df[(df['familia'] == familia) & (df['mes'] == mes)][columna].mean()
                if not pd.isna(media_categoria):
                    medias_cercanas.append(media_categoria)
            # Usar la media de los meses cercanos si existe
            if medias_cercanas:
                media_final = np.mean(medias_cercanas)
                mask_nov = (df['familia'] == familia) & (df['mes'] == pd.Period('2023-11', 'M')) & (df[columna].isnull())
                df.loc[mask_nov, columna] = media_final
                df.loc[mask_nov, 'is_imputed'] = 1
    # Imputar por media anual si sigue habiendo nulos
    for columna in cols_nulos:
        for familia in familias:
            media_anual = df[(df['familia'] == familia) & (df['mes'].dt.year == 2023)][columna].mean()
            if not pd.isna(media_anual):
                mask_nov = (df['familia'] == familia) & (df['mes'] == pd.Period('2023-11', 'M')) & (df[columna].isnull())
                df.loc[mask_nov, columna] = media_anual
                df.loc[mask_nov, 'is_imputed'] = 1
    if verbose:
        print("Imputación completada.")
        print("Valores nulos después de la imputación:")
        print(df[cols_nulos].isnull().sum())
    # Eliminar filas con nulos restantes en las columnas relevantes
    df = df.dropna(subset=cols_nulos)
    return df


# -----------------------
# Homogeneización de columnas clave
# -----------------------
def homogenization(df):
    """
    Homogeneiza las columnas clave del DataFrame df_fleca:
    - Convierte 'fecha' a datetime
    - Cambia 'BEBIDA' a 'BEBIDAS' en 'familia'
    - Normaliza 'producto' a mayúsculas y sin espacios
    - Filtra solo las familias relevantes
    """
    df = df.copy()
    # Homogeneizar fecha
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Homogeneizar familia 'BEBIDA' -> 'BEBIDAS'
    df.loc[df['familia'] == 'BEBIDA', 'familia'] = 'BEBIDAS'
    # Normalizar producto
    if 'producto' in df.columns:
        df['producto'] = df['producto'].str.upper().str.strip()
    # Filtrar familias relevantes
    familias_relevantes = [
        'cerrado', 'BEBIDAS', 'BOLLERIA', 'CAFES', 'VARIOS', 'PASTELERIA',
        'BOCADILLOS', 'PAN', 'AÑADIDOS', 'TES & INFUSIONES', 'LICORES',
        'CERVEZAS', 'TOSTADAS'
    ]
    df = df[df['familia'].isin(familias_relevantes)]
    return df

