# Copia de seguridad antes de refactorización con session_state

import streamlit as st

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

st.set_page_config(page_title="App Fleca", layout="wide")

# Crear dos pestañas: Predicción y Dashboard evaluación
tab_prediccion, tab_dashboard = st.tabs(["Predicción", "Dashboard evaluación"])

with tab_prediccion:
    st.title("Predicción semanal de ventas de bollería")
    st.header("by Jordi Quiroga")
    col_sidebar, col_sep, col_main = st.columns([0.5, 0.25, 3.45])
    with col_sidebar:
        st.header("Progreso")
        progress_bar = st.progress(0)
        N_STEPS = 5
        # PASO 1: Conexión con Hopsworks
        with st.spinner('Conectando con Hopsworks...'):
            proyecto, feature_store = conectar_hopsworks_feature_store()
            st.write("Paso 1. Conexión establecida.")
            progress_bar.progress(1 / N_STEPS)
        # PASO 2: Cargar modelo desde el Model Registry
        with st.spinner('Cargando modelo desde el Model Registry...'):
            modelo = cargar_modelo_desde_registry(
                proyecto,
                config.MODEL_NAME,
                config.MODEL_VERSION,
                config.MODEL_FILE
            )
            st.write("Paso 2. Modelo cargado desde el Model Registry.")
            progress_bar.progress(2 / N_STEPS)
        # PASO 3: Cargar y transformar datos desde Feature View
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
            st.write("Paso 3.Datos cargados y transformados.")
            progress_bar.progress(3 / N_STEPS)
            print(features)
        # PASO 4: Realizar predicción y preparar DataFrame para guardar
        with st.spinner('Realizando predicción...'):
            features_modelo = features[modelo.feature_names_in_]
            resultado = predecir(modelo, features_modelo, solo_ultima=True)
            st.write("Paso 4. Predicción realizada.")
            progress_bar.progress(4 / N_STEPS)
        # PASO 5: Guardar predicción en Hopsworks (bajo petición)
        opciones = ["Guardar predicción en Hopsworks y mostrar gráfico", "Solo mostrar gráfico (no guardar)"]
        accion = st.radio(
            "¿Qué quieres hacer en el paso 5?",
            opciones,
            index=None
        )
        if accion is None:
            st.warning("Selecciona una opción para continuar con el proceso.")
            mostrar_grafico = False
        elif accion == opciones[0]:
            with st.spinner('Guardando predicciones en Hopsworks...'):
                ultimo_lunes = df.iloc[-1]['week_start']
                fecha_siguiente = ultimo_lunes + timedelta(days=7)
                df_pred = pd.DataFrame({
                    'week_start': [fecha_siguiente],
                    'predicted_base_imponible': [resultado[0]]
                })
                guardar_predicciones_en_hopsworks(feature_store, df_pred)
            st.write("Paso 5. Predicciones guardadas en Hopsworks.")
            progress_bar.progress(5 / N_STEPS)
            mostrar_grafico = True
        elif accion == opciones[1]:
            st.write("Paso 5. Predicción NO guardada en Hopsworks.")
            progress_bar.progress(5 / N_STEPS)
            mostrar_grafico = True
    with col_sep:
        st.markdown('<div style="border-left:2px solid #bbb;height:100vh;"></div>', unsafe_allow_html=True)
    with col_main:
        ultimo_lunes = df.iloc[-1]['week_start']
        fecha_siguiente = ultimo_lunes + timedelta(days=7)
        st.header('Predicción próxima semana:')
        st.write(f"Fecha de la última semana con datos: {ultimo_lunes}")
        st.write(f"Fecha de la predicción: {fecha_siguiente}")
        st.write(f"Valor predicho: {resultado[0]:.2f}")
        from src.inference import visualizar_historico_predicciones, obtener_predicciones_feature_view
        if mostrar_grafico:
            with st.spinner('Visualizando gráfico de histórico y predicción...'):
                df_predicciones = obtener_predicciones_feature_view(
                    feature_store,
                    metadata=config.PRED_FEATURE_VIEW_METADATA
                )
                fig = visualizar_historico_predicciones(df, df_predicciones, columna_target='target')
                st.plotly_chart(fig, use_container_width=True)
            st.write("Paso 6. Gráfico interactivo mostrado.")

with tab_dashboard:
    st.title("Dashboard evaluación")
    st.info("Esta pestaña está vacía y lista para añadir el dashboard de evaluación.")
