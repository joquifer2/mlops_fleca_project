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
        historico_df, _, _ = cargar_y_transformar_feature_view(
            feature_store=feature_store,
            modelo=None,
            columna_target=config.COLUMNA_TARGET,
            cols_exogenas=config.COLS_EXOGENAS,
            periodos_adelante=config.PERIODOS_ADELANTE,
            eliminar_nulos=config.ELIMINAR_NULOS,
            metadata=config.HISTORICAL_FEATURE_VIEW_METADATA
        )
        # 4. Generar los features para la próxima semana con logs detallados
        ultima_semana_real = historico_df['week_start'].max()
        next_week = ultima_semana_real + timedelta(days=7)
        nueva_fila = {}
        lags = [int(col.split('lag')[-1]) for col in modelo.feature_names_in_ if 'lag' in col]
        print(f"Intentando generar features para la semana: {next_week} con lags: {lags}")
        for lag in lags:
            semana_lag = next_week - timedelta(days=7*lag)
            valor_lag = historico_df.loc[historico_df['week_start'] == semana_lag, config.COLUMNA_TARGET].values
            print(f"Lag {lag}: semana {semana_lag}, valor encontrado: {valor_lag}")
            if len(valor_lag) == 0:
                raise HTTPException(status_code=422, detail=f"No hay valor para el lag {lag} (semana {semana_lag}) en el histórico. No se puede predecir la semana {next_week}.")
            nueva_fila[f'{config.COLUMNA_TARGET}_lag{lag}'] = valor_lag[0]
        for col in config.COLS_EXOGENAS:
            nueva_fila[col] = historico_df[col].iloc[-1] if col in historico_df.columns else None
        nueva_fila['week_start'] = next_week
        df_features_futuro = pd.DataFrame([nueva_fila])[list(modelo.feature_names_in_) + ['week_start']]
        print("Features generados para la semana futura:")
        print(df_features_futuro)
        # 5. Realizar la predicción para la próxima semana
        prediction = predecir(modelo, df_features_futuro[modelo.feature_names_in_], solo_ultima=False)
        prediction_value = float(prediction[0])
        # 6. Devolver la predicción como diccionario compatible con Pydantic
        response = PredictionResponse(week_start=next_week, prediction=prediction_value)
        return response.dict()
        

    except Exception as e:
        # Manejar errores y devolver una excepción HTTP 500
        raise HTTPException(status_code=500, detail=str(e))