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
    st.write("Aquí irá la lógica de predicción.")

with tab_dashboard:
    st.title("Dashboard evaluación")
    st.info("Esta pestaña está vacía y lista para añadir el dashboard de evaluación.")
