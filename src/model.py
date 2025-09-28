
# ----------------------------------
# Repositorio de funciones de preparación de datos y entrenamiento de modelo XGBoost
# ----------------------------------

import pandas as pd
from datetime import datetime
from src.data_utils import cargar_datos_raw, transformar_a_series_temporales, generar_lags, generar_target, transformar_features_target, guardar_datos_procesados
from src.paths import PROCESSED_DIR
from src.data_split import train_test_split

def preparar_datos_pipeline(
    parquet_file: str = None,
    target: str = None,
    split_date: str = None
) -> tuple:
    """
    Carga el DataFrame procesado más reciente (ts_df_bolleria_*.parquet) de PROCESSED_DIR, selecciona automáticamente el target ('base_imponible_next1' si existe, si no 'base_imponible'), realiza un split temporal 80/20 (20% test) usando la columna 'week_start', y devuelve X_train, y_train, X_test, y_test.

    Parámetros:
        parquet_file: nombre del archivo parquet en PROCESSED_DIR (opcional, por defecto usa el más reciente)
        target: columna objetivo (opcional, por defecto selecciona automáticamente)

    Returns:
        X_train, y_train, X_test, y_test: conjuntos de entrenamiento y test listos para modelado
    """
    # Selección automática del archivo parquet más reciente si no se especifica
    import os
    if parquet_file is None:
        archivos = sorted([
            f for f in os.listdir(PROCESSED_DIR)
            if f.startswith('ts_df_bolleria_') and f.endswith('.parquet')
        ])
        if not archivos:
            raise FileNotFoundError('No se encontró ningún archivo ts_df_bolleria_*.parquet en processed.')
        parquet_file = archivos[-1]
    df = pd.read_parquet(PROCESSED_DIR / parquet_file)

    # Selección automática del target si no se especifica
    target = 'base_imponible'

    # Ordenar por fecha para asegurar el split temporal
    df = df.sort_values('week_start').reset_index(drop=True)

    # Determinar split_date: si no se pasa, usar el 80% automático
    if split_date is None:
        split_idx = int(len(df) * 0.8)
        split_date = df.loc[split_idx, 'week_start']

    # Split temporal train/test usando la función personalizada
    X_train, y_train, X_test, y_test = train_test_split(
        df,
        split_date=split_date,
        target=target
    )
    # Devolver también el nombre de archivo, la fecha de corte y el target usado para trazabilidad
    return X_train, y_train, X_test, y_test, parquet_file, split_date, target



#------------------------------------
# Entrenamiento y evaluación del modelo XGBoost
#------------------------------------

# Importamos librerias
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def train_evaluate_xgboost(X_train, y_train, X_test, y_test, params=None):
    """
    Entrena y evalúa un modelo XGBoost con los hiperparámetros dados (por defecto los mejores de Optuna).
    Elimina la columna 'week_start' si existe en los datos.
    Devuelve un diccionario con las métricas MAE, RMSE, MAPE y R2.
    """
    # Mejores hiperparámetros Optuna (extraídos del notebook 10_03):
    best_params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'lambda': 1.5217227415395248e-07,
        'alpha': 3.805804544409099e-06,
        'subsample': 0.2068525521593345,
        'colsample_bytree': 0.8857508899692184,
        'n_estimators': 51,
        'max_depth': 7,
        'learning_rate': 0.08962161965694515,
        'random_state': 42
    }
    if params is None:
        params = best_params
    # Elimina la columna 'week_start' si existe
    if 'week_start' in X_train.columns:
        X_train = X_train.drop(columns=['week_start'])
    if 'week_start' in X_test.columns:
        X_test = X_test.drop(columns=['week_start'])

    # Instancia y entrena el modelo
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)

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
