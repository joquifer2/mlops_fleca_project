# src/api_client.py
import os
from urllib.parse import urljoin
import requests
import pandas as pd

DEFAULT_API_BASE = "http://fastapi:8000"  # útil en local con docker-compose
API_URL_ENV = os.getenv("API_URL", DEFAULT_API_BASE).strip()

def _normalize_predict_url(api_url_env: str) -> str:
    base = api_url_env.rstrip("/")
    if base.endswith("/predict"):
        return base
    return urljoin(base + "/", "predict")

def llamar_api_prediccion(timestamp, api_url: str | None = None) -> pd.DataFrame:
    url = _normalize_predict_url(api_url or API_URL_ENV)
    payload = {"timestamp": timestamp.isoformat()}

    # 🔒 Desactivar proxies sí o sí (mayúsculas/minúsculas)
    session = requests.Session()
    session.trust_env = False                 # ignora HTTP_PROXY/HTTPS_PROXY/ALL_PROXY
    session.proxies = {"http": None, "https": None}  # cinturón y tirantes

    r = session.post(url, json=payload, timeout=30)
    r.raise_for_status()

    data = r.json()
    return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

