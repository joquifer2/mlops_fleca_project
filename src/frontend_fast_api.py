# -------------------------
# CONFIGURACIÓN INICIAL Y LIBRERÍAS
# -------------------------
import streamlit as st
from datetime import datetime
import pandas as pd
from src.api_client import llamar_api_prediccion
from src.inference import (
    conectar_hopsworks_feature_store,
    guardar_predicciones_en_hopsworks,
    visualizar_historico_predicciones,
    obtener_predicciones_feature_view,
    cargar_y_transformar_feature_view
)
import src.config as config

# -------------------------
# CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# -------------------------
# Configuración de la página
st.set_page_config(layout="wide")

st.title('Predicción semanal de ventas de bollería')
st.header('by Jordi Quiroga')

progress_bar = st.sidebar.header('Progreso')
progress_bar = st.sidebar.progress(0)
N_STEPS = 2

# -------------------------
# PASO 1: Llamar a la API y mostrar resultado
# -------------------------
with st.spinner('Obteniendo predicción desde la API...'):
    timestamp = datetime.now()
    resultado_api = llamar_api_prediccion(timestamp)
    st.sidebar.write("Paso 1. Predicción obtenida desde la API.")
    progress_bar.progress(1 / N_STEPS)

    st.subheader('Predicción próxima semana:')
    st.write(f"Fecha de la predicción: {resultado_api['week_start'][0]}")
    st.write(f"Valor predicho: {resultado_api['prediction'][0]:.2f}")

    # DataFrame para guardar en Hopsworks si se desea
    df_pred = pd.DataFrame({
        'week_start': [resultado_api['week_start'][0]],
        'predicted_base_imponible': [resultado_api['prediction'][0]]
    })

# -------------------------
# PASO 2: Guardar predicción en Hopsworks y mostrar gráfico
# -------------------------
opciones = ["Guardar predicción en Hopsworks y mostrar gráfico", "Solo mostrar gráfico (no guardar)"]
accion = st.sidebar.radio(
    "¿Qué quieres hacer en el paso 2?",
    opciones,
    index=None
)

feature_store = None
if accion is None:
    st.sidebar.warning("Selecciona una opción para continuar con el proceso.")
    mostrar_grafico = False
else:
    # Conectar a Hopsworks solo si se va a guardar o visualizar
    with st.spinner('Conectando con Hopsworks para operaciones de guardado/visualización...'):
        _, feature_store = conectar_hopsworks_feature_store()

    if accion == opciones[0]:
        with st.spinner('Guardando predicciones en Hopsworks...'):
            guardar_predicciones_en_hopsworks(feature_store, df_pred)
            st.sidebar.write("Paso 2. Predicciones guardadas en Hopsworks.")
            progress_bar.progress(2 / N_STEPS)
        mostrar_grafico = True
    elif accion == opciones[1]:
        st.sidebar.write("Paso 2. Predicción NO guardada en Hopsworks.")
        progress_bar.progress(2 / N_STEPS)
        mostrar_grafico = True

# -------------------------
# VISUALIZACIÓN: Histórico y predicción
# -------------------------
# Solo se ejecuta si el usuario lo decide en el selector y hay conexión a Hopsworks
if mostrar_grafico and feature_store is not None:
    with st.spinner('Visualizando gráfico de histórico y predicción...'):
        # Cargar histórico real desde Hopsworks
        df_historico, _ = cargar_y_transformar_feature_view(
            feature_store,
            modelo=None,  # Solo visualización, no requiere modelo
            columna_target=config.COLUMNA_TARGET,
            cols_exogenas=config.COLS_EXOGENAS,
            periodos_adelante=config.PERIODOS_ADELANTE,
            eliminar_nulos=config.ELIMINAR_NULOS,
            metadata=config.HISTORICAL_FEATURE_VIEW_METADATA
        )
        # Cargar predicciones guardadas
        df_predicciones = obtener_predicciones_feature_view(
            feature_store,
            metadata=config.PRED_FEATURE_VIEW_METADATA
        )
        # Visualizar ambos en el gráfico interactivo
        fig = visualizar_historico_predicciones(df_historico, df_predicciones, columna_target='base_imponible')
        st.plotly_chart(fig, use_container_width=True)
        st.sidebar.write("Paso 2. Gráfico interactivo mostrado.")


