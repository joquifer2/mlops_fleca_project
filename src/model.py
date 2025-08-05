
# ----------------------------------
# Pipeline de preparación de datos y entrenamiento de modelo XGBoost
# ----------------------------------

import pandas as pd
from datetime import datetime
from src.data_utils import cargar_datos_raw, transformar_a_series_temporales, generar_lags, generar_target, transformar_features_target, guardar_datos_procesados
from src.paths import PROCESSED_DIR
from src.data_split import train_test_split

def preparar_datos_pipeline(
    parquet_file: str = 'ts_bolleria_20250803.parquet',
    split_date: str = '2025-03-03',
    target: str = 'base_imponible'
) -> tuple:
    """
    Carga el DataFrame procesado, realiza el split train/test y devuelve X_train, y_train, X_test, y_test.
    Parámetros:
        parquet_file: nombre del archivo parquet en PROCESSED_DIR
        split_date: fecha de corte para el split
        target: columna objetivo
    """
    # Cargar el DataFrame procesado
    df = pd.read_parquet(PROCESSED_DIR / parquet_file)
    # Split train/test
    X_train, y_train, X_test, y_test = train_test_split(
        df,
        split_date=split_date,
        target=target
    )
    return X_train, y_train, X_test, y_test
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

#------------------------------------
# Entrenamiento y evaluación del modelo XGBoost
#------------------------------------


def train_evaluate_xgboost(X_train, y_train, X_test, y_test, params):
    """
    Entrena y evalúa un modelo XGBoost con los hiperparámetros dados.
    Elimina la columna 'week_start' si existe en los datos.
    Devuelve un diccionario con las métricas MAE, RMSE, MAPE y R2.
    """
    # Elimina la columna 'week_start' si existe
    if 'week_start' in X_train.columns:
        X_train = X_train.drop(columns=['week_start'])
    if 'week_start' in X_test.columns:
        X_test = X_test.drop(columns=['week_start'])

    # Instancia y entrena el modelo
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'model': model
    }
