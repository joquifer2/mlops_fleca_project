# Su misión es definir la lógica de los "endpoints" o rutas de la API.
# Contiene toda la lógica de que hacer cuando un usuario llama a /predict
# Importa APIRouter y HTTPException para manejar rutas y errores HTTP
from fastapi import APIRouter, HTTPException 

# Importa los esquemas de solicitud y respuesta
from api.schemas import PredictionRequest, PredictionResponse 

# Importa funciones de inferencia
from src.inference import (
    conectar_hopsworks_feature_store,
    cargar_y_transformar_feature_view,
    cargar_modelo_desde_registry,
    predecir
)

import pandas as pd
import src.config as config
from datetime import timedelta



# Crea una instancia de APIRouter para definir las rutas de la API
router = APIRouter()

# Define la ruta /predict que acepta solicitudes POST y devuelve un modelo de respuesta PredictionResponse
@router.post("/predict", response_model=PredictionResponse)
async def predecir_ventas(request: PredictionRequest):
    """
    Endpoint para predecir las ventas de bollería usando la lógica y metadatos centralizados.
    """
    try:
        # 1. Conectar a Hopsworks y obtener feature_store y project
        project, feature_store = conectar_hopsworks_feature_store()

        # 2. Cargar el modelo desde el registry usando metadatos de config
        modelo = cargar_modelo_desde_registry(
            project=project,
            name=config.MODEL_NAME,
            version=config.MODEL_VERSION,
            model_file=config.MODEL_FILE
        )

        # 3. Cargar y transformar los datos de la feature view
        # Usamos los metadatos centralizados en config
        df, features = cargar_y_transformar_feature_view(
            feature_store=feature_store,
            modelo=modelo,
            columna_target=config.COLUMNA_TARGET,
            cols_exogenas=config.COLS_EXOGENAS,
            periodos_adelante=config.PERIODOS_ADELANTE,
            eliminar_nulos=config.ELIMINAR_NULOS,
            metadata=config.HISTORICAL_FEATURE_VIEW_METADATA
        )

        # 4. Filtrar las columnas de features para que coincidan con las del modelo
        features = features[modelo.feature_names_in_]

        # 5. Realizar la predicción para la próxima semana
        prediction = predecir(modelo, features, solo_ultima=True)
        prediction_value = float(prediction[0])

        # Calcular la fecha de la próxima semana
        last_week = df.iloc[-1]['week_start']
        next_week = last_week + timedelta(days=7)
        # 6. Devolver la predicción como diccionario compatible con Pydantic
        response = PredictionResponse(week_start=next_week, prediction=prediction_value)
        return response.dict()

    except Exception as e:
        # Manejar errores y devolver una excepción HTTP 500
        raise HTTPException(status_code=500, detail=str(e))