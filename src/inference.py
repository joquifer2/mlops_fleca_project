from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import joblib
import logging
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

# Añade src al path para importar los módulos
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / 'src'))

import src.config as config 
from pathlib import Path
from src.model import transformar_features_target
from config import COLUMNA_TARGET, COLS_EXOGENAS, PERIODOS_ADELANTE, ELIMINAR_NULOS
from config import PRED_FEATURE_GROUP_METADATA, PRED_FEATURE_VIEW_METADATA

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('inference')

# Nota sobre el enfoque de metadatos:
# En esta implementación usamos diccionarios de metadatos definidos en config.py para
# configurar los feature groups y feature views. Este enfoque tiene varias ventajas:
# 1. Mayor mantenibilidad: Todos los parámetros están centralizados
# 2. Mayor flexibilidad: Podemos pasar los metadatos directamente a las funciones de la API
# 3. Mejor legibilidad: Los parámetros tienen nombres descriptivos en el diccionario
# 4. Reutilización: Podemos usar los mismos metadatos en diferentes partes del código



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

def cargar_y_transformar_feature_view(feature_store, modelo, columna_target, cols_exogenas, 
                                 periodos_adelante=1, eliminar_nulos=True, metadata=None, name=None, version=None):
    """
    Carga una feature view, ordena por la columna de fecha, transforma las features y añade el target.
    Los lags se extraen automáticamente de las features del modelo.
    Retorna el DataFrame procesado y el DataFrame de features sin 'week_start' ni 'target'.
    
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
    df = fv.get_batch_data()
    df = df.sort_values('week_start').reset_index(drop=True)
    # Extraer lags del modelo (nombres de columnas que contienen 'lag')
    lags_list = [int(col.split('lag')[-1]) for col in modelo.feature_names_in_ if 'lag' in col]
    print(f"Lags detectados en el modelo: {lags_list}")
    
    df = transformar_features_target(
        df,
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
    return df, features

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

def predecir(model, features, solo_ultima=True):
    """
    Realiza la predicción usando el modelo y el DataFrame de features.
    Si solo_ultima=True, predice solo la última fila (próxima semana).
    """
    if solo_ultima:
        features = features.iloc[[-1]]
    pred = model.predict(features)
    return pred

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
        try:
            # Intentar recuperar el feature group existente
            fg = fs.get_feature_group(name=metadata['name'], version=metadata['version'])
            logger.info(f"Feature group existente: {metadata['name']} v{metadata['version']}")
        except Exception as e:
            logger.info(f"Feature group no encontrado, intentando crear uno nuevo: {str(e)}")
            try:
                # Si no existe, crearlo usando los metadatos del diccionario
                fg = fs.create_feature_group(**metadata)
                logger.info(f"Feature group creado: {metadata['name']} v{metadata['version']}")
            except Exception as e:
                logger.error(f"Error al crear feature group: {str(e)}")
                # Intentar una última vez obtener el feature group, por si fue creado pero con un error
                try:
                    fg = fs.get_feature_group(name=metadata['name'], version=metadata['version'])
                    logger.info(f"Feature group encontrado después del error: {metadata['name']} v{metadata['version']}")
                except:
                    logger.error(f"No se pudo recuperar el feature group después del error")
                    raise
        return fg
    
    def obtener_o_crear_feature_view(fs, fg, metadata):
        """Obtiene un feature view existente o crea uno nuevo si no existe usando los metadatos"""
        try:
            # Intentar recuperar el feature view existente
            fv = fs.get_feature_view(name=metadata['name'], version=metadata['version'])
            logger.info(f"Feature view existente: {metadata['name']} v{metadata['version']}")
        except Exception as e:
            logger.info(f"Feature view no encontrado, intentando crear uno nuevo: {str(e)}")
            try:
                # Si no existe, crearlo basado en el feature group
                query = fg.select_all()
                # Crear copia del diccionario y agregar la query
                metadata_copy = metadata.copy()
                metadata_copy['query'] = query
                fv = fs.create_feature_view(**metadata_copy)
                logger.info(f"Feature view creado: {metadata['name']} v{metadata['version']}")
            except Exception as e:
                logger.error(f"Error al crear feature view: {str(e)}")
                # Intentar una última vez obtener el feature view, por si fue creado pero con un error
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

    # Ejemplo de uso de cargar_y_transformar_feature_view con metadatos
    df, features = cargar_y_transformar_feature_view(
        feature_store=feature_store,
        modelo=modelo,
        columna_target=COLUMNA_TARGET,
        cols_exogenas=COLS_EXOGENAS,
        periodos_adelante=PERIODOS_ADELANTE,
        eliminar_nulos=ELIMINAR_NULOS,
        metadata=config.HISTORICAL_FEATURE_VIEW_METADATA
    )
    print("DataFrame transformado (primeras filas):")
    print(df.head())
    print("Features para el modelo (primeras filas):")
    print(features.head())

    # Filtrar las columnas de features para que coincidan con las del modelo
    features = features[modelo.feature_names_in_]

    # Realizar la predicción para la próxima semana
    resultado = predecir(modelo, features, solo_ultima=True)
    
    # Obtener la fecha de la última semana de datos
    ultimo_lunes = df.iloc[-1]['week_start']
    fecha_siguiente = ultimo_lunes + timedelta(days=7)
    
    print("Predicción próxima semana:")
    print(f"Fecha: {df.iloc[-1]['week_start']}")
    print(f"Predicción base_imponible: {resultado[0]:.2f}")
    
    # Crear DataFrame con la predicción
    df_predicciones = pd.DataFrame({
        'week_start': [fecha_siguiente],
        'predicted_base_imponible': [resultado[0]]
    })
    
    # Guardar la predicción en Hopsworks usando los metadatos
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

    # Resumen de la ejecución
    print("\n=== Resumen de Ejecución ===")
    print(f"• Fecha de predicción: {fecha_siguiente}")
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
