
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
# sys.path.append(str(Path().resolve().parent / 'src'))
# Ruta absoluta del directorio raíz del proyecto (un nivel arriba de src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Importación de funciones principales del pipeline
from src.inference import (
    conectar_hopsworks_feature_store, 
    cargar_y_transformar_feature_view,
    cargar_modelo_desde_registry,
    predecir,
    guardar_predicciones_en_hopsworks,
    visualizar_historico_predicciones,
    obtener_predicciones_feature_view
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
    _, df, features = cargar_y_transformar_feature_view(
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
    # Obtener la predicción más reciente desde el Feature View de predicción
    df_predicciones = obtener_predicciones_feature_view(
        feature_store,
        metadata=config.PRED_FEATURE_VIEW_METADATA
    )
    pred_ultima = df_predicciones.sort_values('week_start').tail(1)
    fecha_pred = pred_ultima['week_start'].values[0]
    valor_pred = pred_ultima['predicted_base_imponible'].values[0]
    st.subheader('Predicción más reciente en Feature Store:')
    st.write(f"Fecha de la predicción: {fecha_pred}")
    st.write(f"Valor predicho: {valor_pred:.2f}")
    # Creamos un DataFrame con la fecha y el valor predicho para guardar en Hopsworks (opcional)
    df_pred = pd.DataFrame({
        'week_start': [fecha_pred],
        'predicted_base_imponible': [valor_pred]
    })



# -------------------------
# PASO 5: Guardar predicción en Hopsworks (bajo petición)
# -------------------------

# Selector para guardar predicción y mostrar gráfico

opciones = ["Guardar predicción en Hopsworks y mostrar gráfico", "Solo mostrar gráfico (no guardar)"]
accion = st.sidebar.radio(
    "¿Qué quieres hacer en el paso 5?",
    opciones,
    index=None
)

if accion is None:
    st.sidebar.warning("Selecciona una opción para continuar con el proceso.")
    mostrar_grafico = False
elif accion == opciones[0]:
    with st.spinner('Guardando predicciones en Hopsworks...'):
        guardar_predicciones_en_hopsworks(feature_store, df_pred)
        st.sidebar.write("Paso 5. Predicciones guardadas en Hopsworks.")
        progress_bar.progress(5 / N_STEPS)
    mostrar_grafico = True
elif accion == opciones[1]:
    st.sidebar.write("Paso 5. Predicción NO guardada en Hopsworks.")
    progress_bar.progress(5 / N_STEPS)
    mostrar_grafico = True


# -------------------------
# PASO 6: Visualizar gráfico de histórico y predicción
# -------------------------

# Importar función para obtener predicciones guardadas
from src.inference import visualizar_historico_predicciones, obtener_predicciones_feature_view


# Paso 6 solo se ejecuta si el usuario lo decide en el selector
if mostrar_grafico:
    with st.spinner('Visualizando gráfico de histórico y predicción...'):
        df_predicciones = obtener_predicciones_feature_view(
            feature_store,
            metadata=config.PRED_FEATURE_VIEW_METADATA
        )
        fig = visualizar_historico_predicciones(df, df_predicciones, columna_target='target')
        st.plotly_chart(fig, use_container_width=True)
        st.sidebar.write("Paso 6. Gráfico interactivo mostrado.")


