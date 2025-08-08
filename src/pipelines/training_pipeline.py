


"""
Pipeline de entrenamiento automático para forecasting de ventas de bollería semanal.

Características:
- Carga automáticamente el archivo de features más reciente generado por el feature pipeline.
- Usa split temporal 80/20 automático (por defecto, recomendado para producción y continual learning).
- El target es siempre 'base_imponible' para máxima consistencia y reproducibilidad.
- Entrena un modelo XGBoost optimizado y guarda el resultado.
- Imprime información de trazabilidad: archivo, fecha de corte y target usados.
"""

import os
import joblib
import pandas as pd
from pathlib import Path
from src.paths import ROOT_DIR
from src.model import preparar_datos_pipeline, train_evaluate_xgboost
from src.paths import MODELS_DIR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


# Configuración de paths
PROCESSED_DIR = Path(ROOT_DIR) / 'data' / 'processed'
MODELS_DIR = Path(ROOT_DIR) / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def main():


    # 1. Preparar datos y obtener info de trazabilidad
    #   - Usa el archivo parquet más reciente
    #   - Split temporal 80/20 automático
    #   - Target siempre 'base_imponible'
    X_train, y_train, X_test, y_test, parquet_file, split_date, target = preparar_datos_pipeline()

    print(f"Archivo parquet usado: {parquet_file}")
    print(f"Fecha de corte usada: {split_date}")
    print(f"Target usado: {target}")

    # 2. Entrenar y evaluar modelo XGBoost con mejores hiperparámetros (Optuna, notebook 10_03)
    resultados = train_evaluate_xgboost(X_train, y_train, X_test, y_test)
    model = resultados['model']
    print(f"Métricas test: MAE={resultados['mae']:.2f} | RMSE={resultados['rmse']:.2f} | MAPE={resultados['mape']:.2%} | R2={resultados['r2']:.3f}")

    # Guardar modelo entrenado
    model_path = MODELS_DIR / 'xgboost_optimized_bolleria.pkl'
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")



# Punto de entrada principal
if __name__ == "__main__":
    main()

