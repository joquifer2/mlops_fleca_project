import requests
import pandas as pd
import os

# API_URL = "http://127.0.0.1:8000/predict"  # Puedes parametrizar esto si lo necesitas
DEFAULT_API_URL = "http://fastapi:8000/predict"  # Usar el nombre del servicio Docker
API_URL = os.getenv("API_URL", DEFAULT_API_URL)

def llamar_api_prediccion(timestamp, api_url=API_URL):
    """
    Llama al endpoint de predicción de la API FastAPI y devuelve la predicción como DataFrame.
    Args:
        timestamp (datetime): Fecha/hora en formato datetime o date
        api_url (str): URL del endpoint de la API
    Returns:
        pd.DataFrame: DataFrame con la predicción recibida
    """
    payload = {"timestamp": timestamp.isoformat()}
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        # Ajusta según la estructura real de la respuesta
        return pd.DataFrame([response.json()])
    else:
        raise RuntimeError(f"Error al llamar a la API: {response.status_code} - {response.text}")
