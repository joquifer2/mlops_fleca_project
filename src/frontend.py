
import zipfile
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st

import pydeck as pdk
import matplotlib.pyplot as plt

# Añade src al path para importar los módulos
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / 'src'))



from src.paths import ROOT_DIR


st.set_page_config(layout="wide")
st.title('Predicción semanal de ventas de bollería')
progress_bar = st.sidebar.header('Progreso')
progress_bar = st.sidebar.progress(0)
N_STEPS = 5