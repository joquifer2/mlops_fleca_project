
# -------------------------
# CONFIGURACIÓN INICIAL Y LIBRERÍAS
# -------------------------


# Librerías estándar y de análisis
import zipfile
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

# Librerías de visualización y Streamlit
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# Añade src al path para importar los módulos propios
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / 'src'))


# Importación de funciones principales del pipeline
from src.inference import (
    conectar_hopsworks_feature_store, 
    cargar_y_transformar_feature_view,
    cargar_modelo_desde_registry,
    predecir,
    guardar_predicciones_en_hopsworks
)
from src.paths import ROOT_DIR
import src.config as config



# -------------------------
# CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# -------------------------

# Configuración de la página
st.set_page_config(layout="wide")

# Título y encabezado
st.title('Predicción semanal de ventas de bollería')
st.header('by Jordi Quiroga')

# Barra de progreso
progress_bar = st.sidebar.header('Progreso')
progress_bar = st.sidebar.progress(0)

# Definir el número de pasos de la barra de progreso
N_STEPS = 5 


# -------------------------
# PASO 1: Conexión con Hopsworks
# -------------------------
with st.spinner('Conectando con Hopsworks...'):
    proyecto, feature_store = conectar_hopsworks_feature_store()
    st.sidebar.write("Paso 1. Conexión establecida.")
    progress_bar.progress(1 / N_STEPS)




# -------------------------
# PASO 2: Cargar modelo desde el Model Registry
# -------------------------
with st.spinner('Cargando modelo desde el Model Registry...'):
    modelo = cargar_modelo_desde_registry(
        proyecto,
        config.MODEL_NAME,
        config.MODEL_VERSION,
        config.MODEL_FILE
    )
    st.sidebar.write("Paso 2. Modelo cargado desde el Model Registry.")
    progress_bar.progress(2 / N_STEPS)



# -------------------------
# PASO 3: Cargar y transformar datos desde Feature View
# -------------------------
with st.spinner('Datos cargados y transformados...'):
    df, features = cargar_y_transformar_feature_view(
        feature_store,
        modelo,
        config.COLUMNA_TARGET,
        config.COLS_EXOGENAS,
        config.PERIODOS_ADELANTE,
        config.ELIMINAR_NULOS,
        metadata=config.HISTORICAL_FEATURE_VIEW_METADATA
    )
    st.sidebar.write("Paso 3.Datos cargados y transformados.")
    progress_bar.progress(3 / N_STEPS)
    print(features)


# -------------------------
# PASO 4: Realizar predicción y preparar DataFrame para guardar
# -------------------------
with st.spinner('Realizando predicción...'):
    # Seleccionamos las features que requiere el modelo
    features_modelo = features[modelo.feature_names_in_]
    # Realizamos la predicción para la próxima semana
    resultado = predecir(modelo, features_modelo, solo_ultima=True)
    st.sidebar.write("Paso 4. Predicción realizada.")
    progress_bar.progress(4 / N_STEPS)

    # Calculamos la fecha de la próxima semana y mostramos resultados
    ultimo_lunes = df.iloc[-1]['week_start']
    fecha_siguiente = ultimo_lunes + timedelta(days=7)
    st.subheader('Predicción próxima semana:')
    st.write(f"Fecha de la última semana con datos: {ultimo_lunes}")
    st.write(f"Fecha de la predicción: {fecha_siguiente}")
    st.write(f"Valor predicho: {resultado[0]:.2f}")

    # Creamos un DataFrame con la fecha y el valor predicho para guardar en Hopsworks
    df_pred = pd.DataFrame({
        'week_start': [fecha_siguiente],
        'predicted_base_imponible': [resultado[0]]
    })



# -------------------------
# PASO 5: Guardar predicción en Hopsworks (bajo petición)
# -------------------------
guardar = st.sidebar.checkbox("¿Guardar predicción en Hopsworks?", value=False)
if guardar:
    with st.spinner('Guardando predicciones en Hopsworks...'):
        guardar_predicciones_en_hopsworks(feature_store, df_pred)
        st.sidebar.write("Paso 5. Predicciones guardadas en Hopsworks.")
        progress_bar.progress(5 / N_STEPS)
else:
    st.sidebar.write("Paso 5. Predicción NO guardada en Hopsworks.")
    progress_bar.progress(5 / N_STEPS)


# -------------------------
# PASO 6: Visualizar gráfico de histórico y predicción
# -------------------------

# Importar función para obtener predicciones guardadas
from src.inference import visualizar_historico_predicciones, obtener_predicciones_feature_view

with st.spinner('Visualizando gráfico de histórico y predicción...'):
    # Obtener todas las predicciones guardadas en Hopsworks (feature view de predicciones)
    df_predicciones = obtener_predicciones_feature_view(
        feature_store,
        metadata=config.PRED_FEATURE_VIEW_METADATA
    )
    fig = visualizar_historico_predicciones(df, df_predicciones, columna_target='target')
    st.plotly_chart(fig, use_container_width=True)
    st.sidebar.write("Paso 6. Gráfico interactivo mostrado.")


