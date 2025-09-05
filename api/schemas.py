from pydantic import BaseModel
from datetime import date

class PredictionRequest(BaseModel):
    # Define aquí los campos de entrada necesarios para la predicción
    pass

class PredictionResponse(BaseModel):
    week_start: date
    prediction: float
