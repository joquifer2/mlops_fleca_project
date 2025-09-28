# -------------------------
# LIBRERÍAS Y CONFIGURACIÓN INICIAL
# -------------------------
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import joblib
import logging
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

# Añade src al path para importar los módulos propios
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / 'src'))

import config 
from pathlib import Path
from model import transformar_features_target
from config import COLUMNA_TARGET, COLS_EXOGENAS, PERIODOS_ADELANTE, ELIMINAR_NULOS
from config import PRED_FEATURE_GROUP_METADATA, PRED_FEATURE_VIEW_METADATA

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('inference')

# -------------------------
# NOTA SOBRE METADATOS Y ENFOQUE
# -------------------------
# En esta implementación usamos diccionarios de metadatos definidos en config.py para
# configurar los feature groups y feature views. Este enfoque tiene varias ventajas:
# 1. Mayor mantenibilidad: Todos los parámetros están centralizados
# 2. Mayor flexibilidad: Podemos pasar los metadatos directamente a las funciones de la API
# 3. Mejor legibilidad: Los parámetros tienen nombres descriptivos en el diccionario
# 4. Reutilización: Podemos usar los mismos metadatos en diferentes partes del código

# -------------------------
# CONEXIÓN Y ACCESO A HOPSWORKS
# -------------------------
def conectar_hopsworks_feature_store():
    """
    Conecta con Hopsworks y retorna el proyecto y el feature store usando las credenciales de config.py
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    feature_store = project.get_feature_store()
    return project, feature_store

# -------------------------
# CARGA Y TRANSFORMACIÓN DE DATOS DESDE FEATURE VIEW
# -------------------------
def cargar_y_transformar_feature_view(feature_store, modelo, columna_target, cols_exogenas, 
                                 periodos_adelante=1, eliminar_nulos=True, metadata=None, name=None, version=None):
    """
    Carga una feature view, ordena por la columna de fecha, transforma las features y añade el target.
    Si modelo es None, devuelve el DataFrame histórico completo y None para features.
    
    Args:
        feature_store: Feature store de Hopsworks
        modelo: Modelo cargado desde el registry
        columna_target: Nombre de la columna target
        cols_exogenas: Lista de columnas exógenas
        periodos_adelante: Número de periodos a predecir
        eliminar_nulos: Si se eliminan filas con valores nulos
        metadata: Diccionario con metadatos de feature view (nombre y versión)
        name: Nombre de la feature view (si no se pasa metadata)
        version: Versión de la feature view (si no se pasa metadata)
    """
    # Determinar name y version según lo que se proporcione
    if metadata is not None:
        name = metadata.get('name')
        version = metadata.get('version')
    elif name is None or version is None:
        raise ValueError("Debe proporcionar metadata o name y version")

    fv = feature_store.get_feature_view(name=name, version=version)
    df_original = fv.get_batch_data().sort_values('week_start').reset_index(drop=True)

    # Si no se pasa modelo, solo devolvemos el histórico completo
    if modelo is None:
        return df_original, None, None

    # Extraer lags del modelo (nombres de columnas que contienen 'lag')
    lags_list = [int(col.split('lag')[-1]) for col in modelo.feature_names_in_ if 'lag' in col]
    print(f"Lags detectados en el modelo: {lags_list}")

    df = transformar_features_target(
        df_original,
        lags_list=lags_list,
        columna_target=columna_target,
        cols_exogenas=cols_exogenas,
        periodos_adelante=periodos_adelante,
        eliminar_nulos=eliminar_nulos,
        return_format='dataframe'
    )
    df = df.reset_index(drop=True)
    features = df.drop(columns=['week_start'])
    if 'target' in features.columns:
        features = features.drop(columns=['target'])
    return df_original, df, features

# -------------------------
# CARGA DE MODELO DESDE EL REGISTRY
# -------------------------
def cargar_modelo_desde_registry(project, name, version, model_file):
    """
    Carga un modelo desde el model registry de Hopsworks.
    Retorna el modelo cargado con joblib.
    """
    model_registry = project.get_model_registry()
    model = model_registry.get_model(name=name, version=version)
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / model_file)
    return model

# -------------------------
# PREDICCIÓN
# -------------------------
def predecir(model, features, solo_ultima=True):
    """
    Realiza la predicción usando el modelo y el DataFrame de features.
    Si solo_ultima=True, predice solo la última fila (próxima semana).
    """
    if solo_ultima:
        features = features.iloc[[-1]]
    pred = model.predict(features)
    return pred

# -------------------------
# GUARDADO DE PREDICCIONES EN HOPSWORKS
# -------------------------
def guardar_predicciones_en_hopsworks(feature_store, df_predicciones, 
                              fg_metadata=None, fv_metadata=None):
    """
    Guarda las predicciones en un feature group y crea un feature view en Hopsworks
    usando diccionarios de metadatos.
    
    Args:
        feature_store: Feature store de Hopsworks
        df_predicciones: DataFrame con las predicciones, debe tener 'week_start' y 'predicted_base_imponible'
        fg_metadata: Diccionario con los metadatos del feature group (si es None, usa los del config)
        fv_metadata: Diccionario con los metadatos del feature view (si es None, usa los del config)
        
    Returns:
        tuple: (feature_group, feature_view) creados o recuperados
    
    Nota sobre la API de Hopsworks:
    La API de Hopsworks puede comportarse de manera diferente según la versión y la configuración.
    En particular, el método `insert` puede devolver diferentes tipos de objetos:
    - Con wait_for_job=True: Bloquea hasta que el job termine y devuelve el resultado
    - Con wait_for_job=False: Puede devolver un objeto Job o una tupla, dependiendo de la versión
    Por seguridad, utilizamos wait_for_job=True para evitar problemas con el monitoreo asíncrono.
    """
    logger = logging.getLogger('guardar_predicciones')
    
    # Si no se pasan metadatos, usar los definidos en config.py
    if fg_metadata is None:
        fg_metadata = config.PRED_FEATURE_GROUP_METADATA
    
    if fv_metadata is None:
        fv_metadata = config.PRED_FEATURE_VIEW_METADATA
    
    def obtener_o_crear_feature_group(fs, metadata):
        """Obtiene un feature group existente o crea uno nuevo si no existe usando los metadatos"""
        # Aseguramos que version sea entero y los metadatos correctos
        metadata = metadata.copy()
        if 'version' in metadata:
            try:
                metadata['version'] = int(float(metadata['version']))
            except Exception:
                raise ValueError(f"El campo 'version' debe ser convertible a entero, valor recibido: {metadata['version']}")
        try:
            fg = fs.get_feature_group(name=metadata['name'], version=metadata['version'])
            logger.info(f"Feature group existente: {metadata['name']} v{metadata['version']}")
        except Exception as e:
            logger.info(f"Feature group no encontrado, intentando crear uno nuevo: {str(e)}")
            try:
                fg = fs.create_feature_group(**metadata)
                logger.info(f"Feature group creado: {metadata['name']} v{metadata['version']}")
            except Exception as e:
                logger.error(f"Error al crear feature group: {str(e)}")
                try:
                    fg = fs.get_feature_group(name=metadata['name'], version=metadata['version'])
                    logger.info(f"Feature group encontrado después del error: {metadata['name']} v{metadata['version']}")
                except:
                    logger.error(f"No se pudo recuperar el feature group después del error")
                    raise
        return fg
    
    def obtener_o_crear_feature_view(fs, fg, metadata):
        """
        Obtiene un feature view existente o crea uno nuevo si no existe. Si existe, verifica que la consulta sea select_all().
        Si la consulta no es select_all(), lo recrea para asegurar que siempre esté actualizado.
        """
        metadata = metadata.copy()
        if 'version' in metadata:
            try:
                metadata['version'] = int(float(metadata['version']))
            except Exception:
                raise ValueError(f"El campo 'version' debe ser convertible a entero, valor recibido: {metadata['version']}")
        try:
            fv = fs.get_feature_view(name=metadata['name'], version=metadata['version'])
            # Verificar si la consulta es select_all()
            query_actual = getattr(fv, 'query', None)
            query_select_all = fg.select_all()
            # Si la consulta no es select_all(), recrear el feature view
            if query_actual is not None and str(query_actual) != str(query_select_all):
                logger.info(f"La consulta del feature view no es select_all(). Se recrea el feature view.")
                fv.delete()
                metadata_copy = metadata.copy()
                metadata_copy['query'] = query_select_all
                fv = fs.create_feature_view(**metadata_copy)
                logger.info(f"Feature view recreado: {metadata['name']} v{metadata['version']}")
            else:
                logger.info(f"Feature view existente y actualizado: {metadata['name']} v{metadata['version']}")
        except Exception as e:
            logger.info(f"Feature view no encontrado, intentando crear uno nuevo: {str(e)}")
            try:
                query = fg.select_all()
                metadata_copy = metadata.copy()
                metadata_copy['query'] = query
                fv = fs.create_feature_view(**metadata_copy)
                logger.info(f"Feature view creado: {metadata['name']} v{metadata['version']}")
            except Exception as e:
                logger.error(f"Error al crear feature view: {str(e)}")
                try:
                    fv = fs.get_feature_view(name=metadata['name'], version=metadata['version'])
                    logger.info(f"Feature view encontrado después del error: {metadata['name']} v{metadata['version']}")
                except:
                    logger.error(f"No se pudo recuperar el feature view después del error")
                    raise
        return fv
    
    # 1. Obtener o crear el feature group
    fg_pred = obtener_o_crear_feature_group(feature_store, fg_metadata)
    
    # 2. Insertar datos en el feature group
    if fg_pred:
        try:
            # Insertar datos con wait_for_job=True para asegurar que se completa
            # Esto es más simple y seguro que manejar jobs asíncronos
            logger.info("Insertando datos en el feature group...")
            fg_pred.insert(df_predicciones, write_options={'wait_for_job': True})
            logger.info("✅ Predicciones insertadas en el feature group exitosamente")
            
            # 3. Obtener o crear el feature view
            fv_pred = obtener_o_crear_feature_view(feature_store, fg_pred, fv_metadata)
                
        except Exception as e:
            logger.error(f"❌ Error al insertar predicciones: {str(e)}")
            fv_pred = None
    else:
        fv_pred = None
        logger.error("❌ No se pudo obtener ni crear el feature group")
    
    return fg_pred, fv_pred

# -------------------------
# OBTENER PREDICCIONES DESDE FEATURE VIEW DE HOPSWORKS
# -------------------------
def obtener_predicciones_feature_view(feature_store, metadata=None, name=None, version=None):
    """
    Obtiene el DataFrame de la feature view de predicciones desde Hopsworks.
    Args:
        feature_store: objeto feature_store de Hopsworks
        metadata: diccionario con metadatos (opcional)
        name: nombre de la feature view (opcional)
        version: versión de la feature view (opcional)
    Returns:
        DataFrame con las predicciones
    """
    if metadata is not None:
        name = metadata.get('name')
        version = metadata.get('version')
    elif name is None or version is None:
        raise ValueError("Debe proporcionar metadata o name y version")
    fv = feature_store.get_feature_view(name=name, version=version)
    df_predicciones = fv.get_batch_data()
    df_predicciones = df_predicciones.sort_values('week_start').reset_index(drop=True)
    return df_predicciones

# -------------------------
# VISUALIZACIÓN DE HISTÓRICO Y PREDICCIÓN
# -------------------------
def visualizar_historico_predicciones(df_historico, df_prediccion, columna_target='base_imponible', columna_fecha='week_start', ax=None):
    """
    Grafica los valores históricos y todas las predicciones en un gráfico interactivo Plotly.
    Args:
        df_historico: DataFrame con los datos históricos (debe incluir columna de fecha y target)
        df_prediccion: DataFrame con las predicciones (puede incluir varias fechas y valores)
        columna_target: Nombre de la columna objetivo en el histórico
        columna_fecha: Nombre de la columna de fechas
    Returns:
        fig: objeto Plotly Figure
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    # Histórico
    fig.add_trace(go.Scatter(
        x=df_historico[columna_fecha],
        y=df_historico[columna_target],
        mode='lines+markers',
        name='Histórico',
        hovertemplate='Fecha: %{x}<br>Histórico: %{y}<extra></extra>'
    ))
    # Predicciones (pueden ser varias)
    fig.add_trace(go.Scatter(
        x=df_prediccion[columna_fecha],
        y=df_prediccion['predicted_base_imponible'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='red', dash='dash'),
        marker=dict(symbol='x', size=10),
        hovertemplate='Fecha: %{x}<br>Predicción: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title='Histórico y Predicciones de Ventas de Bollería',
        xaxis_title='Fecha',
        yaxis_title='Base Imponible',
        legend=dict(x=0, y=1),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# -------------------------
# BLOQUE PRINCIPAL DE EJEMPLO Y TEST
# -------------------------
if __name__ == "__main__":

    project, feature_store = conectar_hopsworks_feature_store()
    print("Conexión exitosa a Hopsworks.")
    print("Nombre del proyecto:", project.name)
    print("Nombre del feature store:", feature_store.name)

    # Cargar el modelo antes de procesar las features
    modelo = cargar_modelo_desde_registry(
        project,
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
        model_file=config.MODEL_FILE
    )
    print("Modelo cargado correctamente.")

    # 1. Cargar el histórico original (sin transformar)
    fv = feature_store.get_feature_view(name=config.HISTORICAL_FEATURE_VIEW_METADATA['name'], version=config.HISTORICAL_FEATURE_VIEW_METADATA['version'])
    df_historico = fv.get_batch_data().sort_values('week_start').reset_index(drop=True)
    print("Histórico original (últimas filas):")
    print(df_historico.tail())
    ultima_semana_real = df_historico['week_start'].max()
    print(f"Última semana real disponible en el histórico: {ultima_semana_real}")

    # 2. Generar los features para la semana futura (15/09/2025)
    fecha_futura = ultima_semana_real + timedelta(days=7)
    print(f"Generando features para la semana: {fecha_futura}")

    # Obtener los valores de lags usando las semanas correctas
    nueva_fila = {}
    for lag in [int(col.split('lag')[-1]) for col in modelo.feature_names_in_ if 'lag' in col]:
        semana_lag = fecha_futura - timedelta(days=7*lag)
        valor_lag = df_historico.loc[df_historico['week_start'] == semana_lag, COLUMNA_TARGET].values
        nueva_fila[f'{COLUMNA_TARGET}_lag{lag}'] = valor_lag[0] if len(valor_lag) > 0 else np.nan
    # Añadir exógenas (usamos el valor más reciente)
    for col in COLS_EXOGENAS:
        nueva_fila[col] = df_historico[col].iloc[-1] if col in df_historico.columns else np.nan
    nueva_fila['week_start'] = fecha_futura

    # Convertir a DataFrame y reordenar columnas
    df_features_futuro = pd.DataFrame([nueva_fila])[list(modelo.feature_names_in_) + ['week_start']]

    print("Features generados para la semana futura:")
    print(df_features_futuro)

    # 3. Realizar la predicción sobre la semana futura
    resultado = predecir(modelo, df_features_futuro[modelo.feature_names_in_], solo_ultima=False)

    print(f"Predicción para la semana {fecha_futura.date()}: {resultado[0]:.2f}")

    # 4. Crear DataFrame con la predicción
    df_predicciones = pd.DataFrame({
        'week_start': [fecha_futura],
        'predicted_base_imponible': [resultado[0]]
    })

    # 5. Guardar la predicción en Hopsworks usando los metadatos
    print("\nGuardando predicción en Hopsworks...")
    fg_pred, fv_pred = guardar_predicciones_en_hopsworks(
        feature_store=feature_store,
        df_predicciones=df_predicciones,
        fg_metadata=config.PRED_FEATURE_GROUP_METADATA,
        fv_metadata=config.PRED_FEATURE_VIEW_METADATA
    )

    if fg_pred is not None:
        print(f"✅ Predicción guardada en Feature Group: {fg_pred.name} (v{fg_pred.version})")

    if fv_pred is not None:
        print(f"✅ Feature View disponible: {fv_pred.name} (v{fv_pred.version})")

    # 6. Resumen de la ejecución
    print("\n=== Resumen de Ejecución ===")
    print(f"• Fecha de predicción: {fecha_futura}")
    print(f"• Valor predicho: {resultado[0]:.2f}")
    print(f"• Feature Group: {config.PRED_FEATURE_GROUP_METADATA['name']} (v{config.PRED_FEATURE_GROUP_METADATA['version']})")
    print(f"• Feature View: {config.PRED_FEATURE_VIEW_METADATA['name']} (v{config.PRED_FEATURE_VIEW_METADATA['version']})")
    print("===========================")

# Este script ha sido mejorado para utilizar un enfoque basado en diccionarios de metadatos
# que facilita la configuración y el mantenimiento. Las principales mejoras son:
# 1. Uso de diccionarios de metadatos centralizados en config.py
# 2. Funciones que aceptan estos metadatos como parámetros
# 3. Mejor manejo de errores y tiempos de espera para jobs en Hopsworks
# 4. Mayor flexibilidad al poder pasar metadatos diferentes según necesidades
