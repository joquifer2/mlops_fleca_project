

# --- Imports ---
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from google.cloud import bigquery
from datetime import datetime, timedelta

from src.paths import RAW_DIR, INTERIM_DIR

# --- Utilidad para obtener el último domingo ---
def get_last_sunday():
    """
    Devuelve el último domingo anterior o igual a la fecha actual (UTC).
    """
    today = pd.to_datetime(datetime.utcnow()).floor('D')
    # 6 = Sunday, 0 = Monday
    days_since_sunday = (today.weekday() + 1) % 7
    last_sunday = today - timedelta(days=days_since_sunday)
    return last_sunday

# --- Cliente BigQuery centralizado ---
def get_bigquery_client(cred_path=None, project="fleca-del-port"):
    """
    Inicializa y retorna un cliente de BigQuery usando la ruta de credenciales especificada.
    Si no se pasa cred_path, busca en la ruta por defecto del proyecto.
    """
    import os
    if cred_path is None:
        # Busca desde la raíz del proyecto
        cred_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "credentials", "fleca-del-port-978701b834a4.json"))
    if os.path.exists(cred_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    else:
        raise FileNotFoundError(f"No se encontró el archivo de credenciales en {cred_path}")
    return bigquery.Client(project=project)



# --- Función para importación robusta de paths centralizados ---

from src.paths import RAW_DIR, INTERIM_DIR


def get_paths():
    """
    Devuelve todos los paths centralizados definidos en src.paths como un diccionario.
    Lanza un error si no puede importar el módulo.
    """
    import sys
    from pathlib import Path
    try:
        from src import paths as _paths
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        try:
            from src import paths as _paths
        except ImportError as e:
            raise ImportError("No se pudo importar src.paths. Revisa que el archivo y los nombres de variables existen y están actualizados.") from e

    # Recoger todos los atributos en mayúsculas definidos en paths.py
    path_dict = {k: getattr(_paths, k) for k in dir(_paths) if k.isupper()}
    return path_dict

# Ejemplo de uso: obtener los paths como diccionario
PATHS = get_paths()
RAW_BQ_PARQUET = PATHS.get('RAW_BQ_PARQUET', None)
VALIDATED_RANGE_SEMANAL_FAMILIA = PATHS.get('VALIDATED_RANGE_SEMANAL_FAMILIA', None)
RAW_DIR = PATHS.get('RAW_DIR', None)
INTERIM_DIR = PATHS.get('INTERIM_DIR', None)
PROCESSED_DIR = PATHS.get('PROCESSED_DIR', None)

# --- Descargar datos desde BigQuery (todo el histórico) ---
def descargar_datos_bigquery_histórico():

    from datetime import datetime
    """
    Descarga los datos desde dos tablas de BigQuery, concatena los resultados y guarda el DataFrame como archivo parquet en data/raw con la fecha actual en el nombre.
    """
    print("Iniciando conexión con BigQuery...")
    client = bigquery.Client()
    print("Conexión establecida.")

    # 1. Descargar todos los datos de la tabla varios.raw_data_bq_forecasting_20250630
    print("Descargando datos de fleca-del-port.varios.raw_data_bq_forecasting_20250630 ...")
    query1 = """
    SELECT *
    FROM `fleca-del-port.varios.raw_data_bq_forecasting_20250630`
    """
    df1 = client.query(query1).to_dataframe()
    print(f"Filas descargadas de la primera tabla: {len(df1)}")

    # 2. Descargar solo los campos seleccionados de la tabla t_facturas_dia_extendida_2023
    print("Descargando datos de fleca-del-port.fleca_ventas_dia.t_facturas_dia_extendida_2023 ...")
    query2 = """
    SELECT 
        fecha,
        n_factura,
        zona_de_venta,
        producto,
        familia,
        cantidad,
        base_imponible,
        tipo_IVA,
        total
    FROM `fleca-del-port.fleca_ventas_dia.t_facturas_dia_extendida_2023`
    WHERE fecha >= '2025-07-01'
    """
    df2 = client.query(query2).to_dataframe()
    print(f"Filas descargadas de la segunda tabla: {len(df2)}")

    # 3. Concatenar ambos DataFrames (unión vertical)
    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    print(f"Total de filas tras concatenar: {len(df)}")

    # 4. Guardar el DataFrame combinado en parquet con fecha en el nombre
    fecha_actual = datetime.now().strftime("%Y%m%d")
    output_path = RAW_DIR / f"raw_data_bq_forecasting_{fecha_actual}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Asegura que la carpeta existe
    print(f"Guardando archivo en {output_path} ...")
    df.to_parquet(str(output_path), index=False)
    print("Archivo guardado correctamente.")
    return df


def descargar_datos_bigquery():
    from datetime import datetime
    print("Iniciando conexión con BigQuery...")
    client = bigquery.Client()
    print("Conexión establecida.")

    # Solo descargar la segunda tabla
    print("Descargando datos de fleca-del-port.fleca_ventas_dia.t_facturas_dia_extendida_2023 ...")
    query = """
    SELECT 
        fecha,
        n_factura,
        zona_de_venta,
        producto,
        familia,
        cantidad,
        base_imponible,
        tipo_IVA,
        total
    FROM `fleca-del-port.fleca_ventas_dia.t_facturas_dia_extendida_2023`
    WHERE fecha >= '2025-08-04' 
    """
    df = client.query(query).to_dataframe()
    print(f"Filas descargadas de la segunda tabla: {len(df)}")

    # Guardar el DataFrame en la carpeta RAW con fecha en el nombre
    from pathlib import Path
    fecha_actual = datetime.now().strftime("%Y%m%d")
    # Determinar la ruta absoluta a la carpeta data/raw
    module_path = Path(__file__).resolve().parent  # src/
    project_root = module_path.parent  # raíz del proyecto
    raw_dir = project_root / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"raw_data_bq_forecasting_{fecha_actual}.parquet"
    print(f"Guardando archivo en {output_path} ...")
    df.to_parquet(str(output_path), index=False)
    print("Archivo guardado correctamente.")
    return df

# ----------------------
# Cargar datos raw
# ----------------------

def cargar_datos_raw(parquet_file):
    """
    Carga datos desde un archivo parquet y maneja la conversión de tipos dbdate.
    
    Parámetros:
    - parquet_file: Ruta al archivo parquet
    
    Retorna:
    - DataFrame con los datos cargados
    """
    print(f"Cargando datos desde: {parquet_file}")
    # Usar PyArrow para leer el archivo
    table = pq.read_table(parquet_file)
    # Identificar columnas de tipo dbdate
    schema = table.schema
    dbdate_cols = [field.name for field in schema if str(field.type) == 'dbdate']
    # Convertir columnas dbdate a string
    for col in dbdate_cols:
        table = table.set_column(table.schema.get_field_index(col), col, table.column(col).cast('string'))
    # Convertir a pandas DataFrame
    df_result = table.to_pandas(ignore_metadata=True)
    return df_result

#-----------------------
# Validar datos 
# -----------------------
def validar_fechas_completas(

    df,
    fecha_col='fecha',
    fecha_inicio=None,
    fecha_fin=None
):
    """
    Valida que todas las fechas diarias entre fecha_inicio y fecha_fin estén presentes en el DataFrame.
    Devuelve un diccionario con:
        - 'missing_dates': lista de fechas faltantes
        - 'rango_usado': (fecha_inicio_real, fecha_fin_real)
    Los argumentos fecha_inicio y fecha_fin pueden ser None, string o datetime.
    Si son None, se usan los valores mínimo y máximo del DataFrame.
    """
    # Convertir fechas a datetime
    fechas = pd.to_datetime(df[fecha_col])
    if fecha_inicio is None:
        fecha_inicio_real = fechas.min()
    else:
        fecha_inicio_real = pd.to_datetime(fecha_inicio)
    if fecha_fin is None:
        fecha_fin_real = fechas.max()
    else:
        fecha_fin_real = pd.to_datetime(fecha_fin)

    fechas_completas = pd.date_range(start=fecha_inicio_real, end=fecha_fin_real, freq='D')
    fechas_presentes = pd.to_datetime(df[fecha_col].unique())
    missing_dates = np.setdiff1d(fechas_completas, fechas_presentes)
    print(f"Validando fechas entre {fecha_inicio_real.date()} y {fecha_fin_real.date()} ({len(fechas_completas)} días)")
    print(f"Total de fechas faltantes: {len(missing_dates)}")
    if len(missing_dates) > 0:
        print("Fechas faltantes:", missing_dates)
    else:
        print("No faltan fechas en el rango especificado.")
    return {
        'missing_dates': missing_dates,
        'rango_usado': (fecha_inicio_real, fecha_fin_real)
    }


# -----------------------
# Cargar datos raw y validar
# -----------------------   

def load_raw_data(
    parquet_path=None,
    fecha_inicio=None,
    fecha_fin=None,
    descargar_bq=False
):
    """
    Descarga, carga y valida los datos raw antes de la agregación semanal.
    Devuelve el DataFrame limpio y listo para agregación.
    """

    # Obtener los paths centralizados
    PATHS = get_paths()
    raw_bq_parquet = PATHS.get('RAW_BQ_PARQUET', None)

    # Si no se especifica parquet_path, usar el path por defecto
    if parquet_path is None:
        parquet_path = raw_bq_parquet

    # Si no se pasa fecha_fin, usar el último domingo completo
    if fecha_fin is None:
        fecha_fin = get_last_sunday()


    # 1. Descargar datos desde BigQuery si se indica o si el archivo no existe
    parquet_path_obj = Path(parquet_path)
    if descargar_bq or not parquet_path_obj.exists():
        print(f"Descargando datos desde BigQuery porque descargar_bq={descargar_bq} o no existe el archivo {parquet_path}")
        descargar_datos_bigquery()


    # 2. Cargar datos raw desde parquet
    df_raw = cargar_datos_raw(parquet_path)

    # Asegurar que la columna 'fecha' es datetime antes de cualquier filtrado
    if 'fecha' in df_raw.columns:
        df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])

    # 3. Filtrar por fechas solo si se especifican
    if fecha_inicio is not None:
        df_raw = df_raw[df_raw['fecha'] >= fecha_inicio]
    if fecha_fin is not None:
        df_raw = df_raw[df_raw['fecha'] <= fecha_fin]

    # Validar continuidad solo si ambos están definidos
    if fecha_inicio is not None and fecha_fin is not None:
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

# --- Lógica de Semana Santa ---
def get_easter_ranges():
    """
    Devuelve un diccionario con los rangos de fechas de Semana Santa por año.
    """
    return {
        2023: pd.date_range('2023-04-03', '2023-04-09'),
        2024: pd.date_range('2024-03-25', '2024-03-31'),
        2025: pd.date_range('2025-04-14', '2025-04-20'),
    }

def mark_easter(df):
    """
    Añade la columna is_easter al DataFrame y marca las fechas de Semana Santa.
    """
    easter_ranges = get_easter_ranges()
    df = df.copy()
    df['is_easter'] = 0
    for year, dates in easter_ranges.items():
        df.loc[df['fecha'].dt.date.isin(dates.date), 'is_easter'] = 1
    return df

def get_easter_weeks():
    """
    Devuelve una lista de tuplas (year, week) correspondientes a las semanas de Semana Santa.
    """
    easter_weeks = []
    for year, dates in get_easter_ranges().items():
        for date in dates:
            easter_weeks.append((date.isocalendar().year, date.isocalendar().week))
    return list(set(easter_weeks))

# --- Agregación semanal ---
def aggregate_weekly(df, min_dias_semana=7):
    """
    Realiza la agregación semanal sobre el DataFrame, filtrando semanas incompletas.
    """
    iso = df['fecha'].dt.isocalendar()
    df['year_iso'] = iso['year']
    df['week_iso'] = iso['week']
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
    df_semanal.rename(columns={'year_iso':'year','week_iso':'week'}, inplace=True)
    # Añadir columna week_start: primer lunes de cada semana ISO
    df_semanal['week_start'] = pd.to_datetime(
        df_semanal['year'].astype(str) + '-W' + df_semanal['week'].astype(str) + '-1',
        format='%G-W%V-%u'
    )
    df_semanal = df_semanal.sort_values('week_start').reset_index(drop=True)
    return df_semanal

## ---------------
# Transformación a series temporales semanales  
## ----------------- 

def transformar_a_series_temporales(
    df_raw,
    fecha_inicio=None,
    fecha_fin=None,
    familia=None,
    output_path=None,
    min_dias_semana=7,
    guardar_interim=False
):
    """
    Limpia, homogeneiza y agrega los datos diarios a series semanales completas para la familia indicada.

    Parámetros:
    - df_raw: DataFrame con datos crudos
    - fecha_inicio: Fecha de inicio (str o datetime, opcional)
    - fecha_fin: Fecha fin (str o datetime, opcional)
    - familia: Familia de productos a filtrar (str, opcional)
    - output_path: Path opcional para guardar el resultado (str o Path)
    - min_dias_semana: Mínimo de días para considerar una semana (int, por defecto 7)
    - guardar_interim: Si True, guarda el resultado en la carpeta interim (bool)

    Retorna:
    - DataFrame con series temporales semanales

    Esta versión utiliza funciones modulares para:
    - Homogeneización de datos (homogenization)
    - Imputación avanzada de nulos (impute_null_values)
    - Lógica de Semana Santa (mark_easter, get_easter_weeks)
    - Agregación semanal (aggregate_weekly)
    """
    df = df_raw.copy()

    # Homogeneización y filtrado
    df = homogenization(df)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Si no se pasan fechas, usar el rango del DataFrame
    if fecha_inicio is None:
        fecha_inicio = df['fecha'].min()
    if fecha_fin is None:
        fecha_fin = df['fecha'].max()
    df = df[(df['fecha'] >= fecha_inicio) & (df['fecha'] <= fecha_fin)]

    # Filtrar familia si se proporciona
    if familia is not None:
        df = df[df['familia'] == familia]

    # Imputar valores nulos avanzados
    df = impute_null_values(df, cols_nulos=['base_imponible', 'total'], verbose=False)

    # Asegurar columnas exógenas
    if 'is_summer_peak' not in df.columns:
        df['is_summer_peak'] = df['fecha'].dt.month.isin([7, 8]).astype(int)

    # Marcar Semana Santa
    df = mark_easter(df)

    # Agregación semanal
    df_semanal = aggregate_weekly(df, min_dias_semana=min_dias_semana)

    # Filtrar familia solo si se especifica
    if familia is not None:
        df_familia_semanal = df_semanal.query(f"familia=='{familia}'").copy()
    else:
        df_familia_semanal = df_semanal.copy()

    # Asegurar que las semanas de Semana Santa estén marcadas correctamente
    easter_weeks = get_easter_weeks()
    for year, week in easter_weeks:
        mask = (df_familia_semanal['year'] == year) & (df_familia_semanal['week'] == week)
        if len(df_familia_semanal[mask]) > 0:
            df_familia_semanal.loc[mask, 'is_easter'] = 1

    # Guardar el resultado si se proporciona un path de salida
    if output_path:
        df_familia_semanal.to_parquet(str(output_path), index=False)
        print(f"Series temporales guardadas en: {output_path}")
        
    # Guardar en interim si se solicita
    if guardar_interim:
        guardar_time_series_interim(df_familia_semanal, familia)
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
    
    # Obtener la ruta absoluta a la carpeta data/interim del proyecto
    if interim_dir is None:
        # Intentar encontrar la carpeta data/interim relativa al módulo
        try:
            # Primero intentamos usar la ruta relativa al archivo actual
            module_path = Path(__file__).resolve().parent  # src/
            project_root = module_path.parent  # raíz del proyecto
            interim_dir = project_root / 'data' / 'interim'
        except:
            print("AVISO: No se pudo determinar la ruta del proyecto, usando 'data/interim'")
            interim_dir = 'data/interim'
    
    os.makedirs(interim_dir, exist_ok=True)
    # Usar solo la fecha (YYYYMMDD) sin horas, minutos ni segundos
    date_only = datetime.datetime.now().strftime('%Y%m%d')
    filename = f"time_series_{familia}_weekly_{date_only}.parquet"
    filepath = os.path.join(str(interim_dir), filename)
    df.to_parquet(filepath, index=False)
    print(f"Archivo guardado en: {filepath}")
    return filepath

# ------------------------------
# Generación de lags y target
# ------------------------------

def generar_lags(df, lags_list, columna='base_imponible'):
    """
    Genera variables de lag a partir de una serie temporal.
    
    Parámetros:
    - df: DataFrame con datos de serie temporal
    - lags_list: Lista de periodos de lag a generar
    - columna: Columna para la que se generan los lags
    
    Retorna:
    - DataFrame con columnas de lag añadidas
    """
    df_lags = df.copy()
    
    # Ordenar por week_start si existe, si no por año y semana
    if 'week_start' in df_lags.columns:
        df_lags = df_lags.sort_values('week_start')
    elif 'year' in df_lags.columns and 'week' in df_lags.columns:
        df_lags = df_lags.sort_values(['year', 'week'])
    
    # Generar lags
    for lag in lags_list:
        df_lags[f'{columna}_lag{lag}'] = df_lags[columna].shift(lag)
    
    return df_lags

def generar_target(df, columna='base_imponible', periodos_adelante=1):
    """
    Genera variable target para forecasting.
    
    Parámetros:
    - df: DataFrame con datos de serie temporal
    - columna: Columna que se usará como target
    - periodos_adelante: Número de periodos a predecir (por defecto 1)
    
    Retorna:
    - DataFrame con columna target añadida
    """
    df_target = df.copy()
    
    # Ordenar por week_start si existe, si no por año y semana
    if 'week_start' in df_target.columns:
        df_target = df_target.sort_values('week_start')
    elif 'year' in df_target.columns and 'week' in df_target.columns:
        df_target = df_target.sort_values(['year', 'week'])
    
    # Generar target
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
    """
    Prepara features (lags) y target para modelado de forecasting.
    
    Parámetros:
    - df: DataFrame con serie temporal
    - lags_list: Lista de lags a generar
    - columna_target: Columna que se usará como target
    - cols_exogenas: Lista de columnas exógenas a incluir en features
    - periodos_adelante: Número de periodos a predecir
    - eliminar_nulos: Si True, elimina filas con valores nulos
    
    Retorna:
    - X: DataFrame con features
    - y: Series con target
    - df_completo: DataFrame con features y target
    """
    if cols_exogenas is None:
        cols_exogenas = []
    
    # Generar lags
    df_features = generar_lags(df, lags_list, columna_target)
    
    # Generar target
    df_completo, target_name = generar_target(
        df_features, 
        columna_target, 
        periodos_adelante
    )
    
    # Preparar features y target
    cols_lags = [f'{columna_target}_lag{lag}' for lag in lags_list]
    # Si existe week_start, mantenerla en X y df_completo para splits posteriores
    extra_cols = []
    if 'week_start' in df_completo.columns:
        extra_cols.append('week_start')

    # Eliminar columnas que no se usarán nunca como features
    drop_cols = ['familia', 'year', 'week', 'dias_semana']
    df_completo = df_completo.drop(columns=[col for col in drop_cols if col in df_completo.columns])

    X = df_completo[cols_lags + cols_exogenas + extra_cols]
    y = df_completo[target_name]

    # Eliminar filas con valores nulos si se solicita
    if eliminar_nulos:
        mask_completos = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask_completos]
        y = y[mask_completos]
        df_completo = df_completo[mask_completos]

    return X, y, df_completo

def guardar_datos_procesados(X, y, df_completo, familia='BOLLERIA', processed_dir=None):
    """
    Guarda los datasets finales listos para modelado (X, y, df_completo) en la carpeta processed.
    
    Parámetros:
    - X: DataFrame con las features
    - y: Series con el target
    - df_completo: DataFrame completo con features y target
    - familia: Nombre de la familia de productos
    - processed_dir: Directorio donde guardar los archivos. Si es None, usa la ruta del proyecto
    
    Retorna:
    - Diccionario con las rutas donde se guardaron los archivos
    """
    import os
    import datetime
    from pathlib import Path
    
    # Obtener la ruta absoluta a la carpeta data/processed del proyecto
    if processed_dir is None:
        try:
            # Primero intentamos usar la ruta relativa al archivo actual
            module_path = Path(__file__).resolve().parent  # src/
            project_root = module_path.parent  # raíz del proyecto
            processed_dir = project_root / 'data' / 'processed'
        except:
            print("AVISO: No se pudo determinar la ruta del proyecto, usando 'data/processed'")
            processed_dir = 'data/processed'
    
    os.makedirs(processed_dir, exist_ok=True)
    date_only = datetime.datetime.now().strftime('%Y%m%d')
    
    # Guardar los tres datasets
    files = {}
    
    # Guardar X (features)
    x_filename = f"ts_X_{familia.lower()}_{date_only}.parquet"
    x_filepath = os.path.join(str(processed_dir), x_filename)
    X.to_parquet(x_filepath, index=False)
    files['X'] = x_filepath
    
    # Guardar y (target)
    y_filename = f"ts_y_{familia.lower()}_{date_only}.parquet"
    y_filepath = os.path.join(str(processed_dir), y_filename)
    y.to_frame().to_parquet(y_filepath, index=False)
    files['y'] = y_filepath
    
    # Guardar df_completo (dataset completo)
    df_filename = f"ts_df_{familia.lower()}_{date_only}.parquet"
    df_filepath = os.path.join(str(processed_dir), df_filename)
    df_completo.to_parquet(df_filepath, index=False)
    files['df_completo'] = df_filepath
    
    print(f"Datos procesados guardados en la carpeta: {processed_dir}")
    print(f"- Features (X): {x_filename}")
    print(f"- Target (y): {y_filename}")
    print(f"- Dataset completo: {df_filename}")
    
    return files


# --- Subida a BigQuery ---
def subir_df_a_bigquery(X=None, y=None, df_completo=None, dataset_base="features", familia="BOLLERIA", project_id=None, if_exists="replace"):
    """
    Sube los DataFrames procesados (X, y, df_completo) a BigQuery como tablas separadas.
    - dataset_base: str, nombre del dataset de BigQuery (por ejemplo, 'features')
    - familia: str, nombre de la familia para sufijo de tabla
    - project_id: str, ID del proyecto de GCP (opcional si está configurado por entorno)
    - if_exists: 'replace' (sobrescribe) o 'append' (añade filas)
    """
    from google.cloud import bigquery
    import pandas as pd
    # Asegura que y es DataFrame si es Series
    if y is not None and isinstance(y, pd.Series):
        y = y.to_frame()
    import datetime
    date_only = datetime.datetime.now().strftime('%Y%m%d')
    tablas = {
        f"ts_X_{familia.lower()}_{date_only}": X,
        f"ts_y_{familia.lower()}_{date_only}": y,
        f"ts_df_{familia.lower()}_{date_only}": df_completo
    }
    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE" if if_exists=="replace" else "WRITE_APPEND")
    resultados = {}
    for nombre, df in tablas.items():
        if df is not None:
            table_name = f"{dataset_base}.{nombre}"
            print(f"Subiendo DataFrame a BigQuery: {table_name} ...")
            job = client.load_table_from_dataframe(df, table_name, job_config=job_config)
            job.result()  # Espera a que termine
            print(f"Subida completada a {table_name} ({df.shape[0]} filas, {df.shape[1]} columnas)")
            resultados[nombre] = True
        else:
            resultados[nombre] = False
    return resultados
