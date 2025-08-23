
import pandas as pd
from datetime import datetime
from typing import Tuple


def train_test_split(
        df: pd.DataFrame,
        split_date=None,
        target: str = 'base_imponible',
        split_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    """
    Divide un DataFrame de series temporales en train y test según fecha de corte usando 'week_start'.

    Parámetros:
    - df: pd.DataFrame - DataFrame completo, debe incluir la columna 'week_start' y el target.
    - split_date: str o datetime, opcional - Fecha de corte (primer lunes de la semana de split, formato 'YYYY-MM-DD' o datetime). Si no se pasa, se calcula automáticamente usando split_ratio.
    - target: str - Nombre de la columna objetivo (target) en el DataFrame.
    - split_ratio: float - Proporción de datos para entrenamiento (por defecto 0.8).

    Returns:
    - X_train: pd.DataFrame - Features de entrenamiento.
    - y_train: pd.Series - Target de entrenamiento.
    - X_test: pd.DataFrame - Features de prueba.
    - y_test: pd.Series - Target de prueba.
    """

    
    df_copy = df.copy()
    df_copy['week_start'] = pd.to_datetime(df_copy['week_start'])
    
    # Ordenar por la columna temporal antes de dividir
    df_copy = df_copy.sort_values('week_start').reset_index(drop=True)

    # Calcular split_date automáticamente si no se pasa
    if split_date is None:
        split_idx = int(len(df_copy) * split_ratio)
        split_date = df_copy.loc[split_idx, 'week_start']
    else:
        split_date = pd.to_datetime(split_date)

    # División temporal
    train_data = df_copy[df_copy['week_start'] <= split_date].reset_index(drop=True)
    test_data = df_copy[df_copy['week_start'] > split_date].reset_index(drop=True)

    # Separar características (X) y objetivo (y)
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    return X_train, y_train, X_test, y_test