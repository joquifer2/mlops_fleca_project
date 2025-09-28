# src/api_client.py
import os
from urllib.parse import urljoin
import requests
import pandas as pd

# URL base por defecto para la API 
# En Docker Compose usa el nombre del servicio, fuera de Docker usa localhost
DEFAULT_API_BASE = os.getenv("API_URL", "http://localhost:8000")

# Lee la URL de la API de la variable de entorno API_URL, si existe, o usa la base por defecto
API_URL_ENV = DEFAULT_API_BASE.strip()

def _normalize_predict_url(api_url_env: str) -> str:
    """
    Normaliza la URL para el endpoint /predict.
    Si la URL ya termina en /predict, la devuelve tal cual.
    Si no, la concatena correctamente.
    """
    base = api_url_env.rstrip("/")
    if base.endswith("/predict"):
        return base
    return urljoin(base + "/", "predict")

def llamar_api_prediccion(timestamp, api_url: str | None = None) -> pd.DataFrame:
    """
    Llama al endpoint de predicción de la API FastAPI y devuelve la predicción como DataFrame.
    - timestamp: datetime a enviar en la petición (en formato ISO)
    - api_url: URL del endpoint de la API (opcional, por defecto usa API_URL_ENV)
    - Desactiva el uso de proxies del sistema para evitar problemas en entornos corporativos.
    """
    url = _normalize_predict_url(api_url or API_URL_ENV)
    payload = {"timestamp": timestamp.isoformat()}

    # 🔒 Desactivar proxies sí o sí (mayúsculas/minúsculas)
    session = requests.Session()
    session.trust_env = False                 # ignora HTTP_PROXY/HTTPS_PROXY/ALL_PROXY
    session.proxies = {"http": None, "https": None}  # cinturón y tirantes

    r = session.post(url, json=payload, timeout=30)
    r.raise_for_status()

    data = r.json()
    # Devuelve un DataFrame con la predicción (soporta respuesta dict o lista de dicts)
    return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)


def obtener_datos_grafico(api_url: str | None = None) -> dict:
    """
    Llama al endpoint /chart-data de la API para obtener datos históricos y predicciones.
    - api_url: URL base de la API (opcional, por defecto usa API_URL_ENV)
    - Devuelve un diccionario con 'historical' y 'predictions'
    """
    base_url = (api_url or API_URL_ENV).rstrip("/")
    
    # Asegurarse de que la URL base no termine en /predict
    if base_url.endswith("/predict"):
        base_url = base_url[:-8]  # Remover "/predict"
    
    url = urljoin(base_url + "/", "chart-data")

    # 🔒 Desactivar proxies sí o sí
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}

    r = session.get(url, timeout=30)
    r.raise_for_status()

    return r.json()

