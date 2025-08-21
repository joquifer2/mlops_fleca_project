from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks
import joblib

# Añade src al path para importar los módulos
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / 'src'))

import src.config as config 
from pathlib import Path
from src.model import transformar_features_target
from config import COLUMNA_TARGET, COLS_EXOGENAS, PERIODOS_ADELANTE, ELIMINAR_NULOS 



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

def cargar_y_transformar_feature_view(feature_store, name, version, modelo, columna_target, cols_exogenas, periodos_adelante=1, eliminar_nulos=True):
    """
    Carga una feature view, ordena por la columna de fecha, transforma las features y añade el target.
    Los lags se extraen automáticamente de las features del modelo.
    Retorna el DataFrame procesado y el DataFrame de features sin 'week_start' ni 'target'.
    """
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

    # Ejemplo de uso de cargar_y_transformar_feature_view
    df, features = cargar_y_transformar_feature_view(
        feature_store,
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION,
        modelo=modelo,
        columna_target=COLUMNA_TARGET,
        cols_exogenas=COLS_EXOGENAS,
        periodos_adelante=1,
        eliminar_nulos=True
    )
    print("DataFrame transformado (primeras filas):")
    print(df.head())
    print("Features para el modelo (primeras filas):")
    print(features.head())

    # Filtrar las columnas de features para que coincidan con las del modelo
    features = features[modelo.feature_names_in_]

    # Realizar la predicción para la próxima semana
    resultado = predecir(modelo, features, solo_ultima=True)
    print("Predicción próxima semana:")
    print(f"Fecha: {df.iloc[-1]['week_start']}")
    print(f"Predicción base_imponible: {resultado[0]:.2f}")

