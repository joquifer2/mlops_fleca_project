# Su única misión es definir la "forma" de tus datos
# Contiene todos tus modelos Pydantic para las solicitudes y respuestas de la API
# Si necesitas saber qué datos entran o salen de la API, este es el archivo a revisar


from pydantic import BaseModel # Importa BaseModel de Pydantic para definir modelos de datos
from datetime import date


# Define los modelos de datos que se utilizarán en la aAPI
class PredictionRequest(BaseModel):
    """
    Modelo de datos para la solicitud de predicción.
    Actualmente no se requieren parámetros en la solicitud.
    """
    timestamp: str  # formato ISO, e.g., "2023-10-01T00:00:00"


# Define un modelo para la predicción
class PredictionResponse(BaseModel):
    """
    Modelo de datos para la respuesta de predicción.
    """
    week_start: date
    prediction: float
