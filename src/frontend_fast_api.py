# -------------------------
# CONFIGURACIÓN INICIAL Y LIBRERÍAS
# -------------------------


import os
import streamlit as st
from datetime import datetime
import pandas as pd
from api_client import llamar_api_prediccion


# -------------------------
# CONFIGURACIÓN DE LA INTERFAZ STREAMLIT
# -------------------------
# Configuración de la página
st.set_page_config(layout="wide")

st.title('Predicción semanal de ventas de bollería')
st.header('by Jordi Quiroga')


# Debug opcional: ver qué API_URL está leyendo la UI
st.caption(f"API_URL = {os.getenv('API_URL', '(sin definir; usando fallback)')}")
st.caption(f"HTTP_PROXY={os.getenv('HTTP_PROXY')}  HTTPS_PROXY={os.getenv('HTTPS_PROXY')}  ALL_PROXY={os.getenv('ALL_PROXY')}")
st.caption(f"http_proxy={os.getenv('http_proxy')}  https_proxy={os.getenv('https_proxy')}  all_proxy={os.getenv('all_proxy')}")


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

    # Debug: mostrar la respuesta cruda de la API
    st.write('DEBUG respuesta API:', resultado_api)
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
accion = st.sidebar.radio("¿Qué quieres hacer en el paso 2?", opciones, index=None)

feature_store = None
mostrar_grafico = False

if accion is None:
    st.sidebar.warning("Selecciona una opción para continuar con el proceso.")
else:
    # 👉 Importar Hopsworks SOLO si se va a usar
    try:
        import src.config as config
        from src.inference import (
            conectar_hopsworks_feature_store,
            guardar_predicciones_en_hopsworks,
            visualizar_historico_predicciones,
            obtener_predicciones_feature_view,
            cargar_y_transformar_feature_view
        )
    except Exception as e:
        st.error(
            "La UI no tiene configuradas las credenciales de Hopsworks o falta configuración.\n\n"
            f"Detalle: {e}\n\n"
            "Soluciones: añade HOPSWORKS_* en este servicio o mueve toda la lógica de Hopsworks a la API."
        )
        st.stop()

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
# VISUALIZACIÓN
# -------------------------
if mostrar_grafico and feature_store is not None:
    with st.spinner('Visualizando gráfico de histórico y predicción...'):
        df_historico, _ = cargar_y_transformar_feature_view(
            feature_store,
            modelo=None,
            columna_target=config.COLUMNA_TARGET,
            cols_exogenas=config.COLS_EXOGENAS,
            periodos_adelante=config.PERIODOS_ADELANTE,
            eliminar_nulos=config.ELIMINAR_NULOS,
            metadata=config.HISTORICAL_FEATURE_VIEW_METADATA
        )
        df_predicciones = obtener_predicciones_feature_view(
            feature_store,
            metadata=config.PRED_FEATURE_VIEW_METADATA
        )
        fig = visualizar_historico_predicciones(
            df_historico, df_predicciones, columna_target='base_imponible'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.sidebar.write("Paso 2. Gráfico interactivo mostrado.")


