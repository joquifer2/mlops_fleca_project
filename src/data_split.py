
import pandas as pd
from datetime import datetime
from typing import Tuple


def train_test_split(
        df: pd.DataFrame,
        split_date,
        target: str = 'base_imponible',
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    """
    Divide un DataFrame de series temporales en train y test según fecha de corte usando 'week_start'.

    Parámetros:
    - df: pd.DataFrame - DataFrame completo, debe incluir la columna 'week_start' y el target.
    - split_date: str o datetime - Fecha de corte (primer lunes de la semana de split, formato 'YYYY-MM-DD' o datetime).
    - target: str - Nombre de la columna objetivo (target) en el DataFrame.

    Returns:
    - X_train: pd.DataFrame - Features de entrenamiento.
    - y_train: pd.Series - Target de entrenamiento.
    - X_test: pd.DataFrame - Features de prueba.
    - y_test: pd.Series - Target de prueba.
    """

    
    # Conversión robusta de split_date y columna week_start a datetime
    split_date = pd.to_datetime(split_date)
    df_copy = df.copy()
    df_copy['week_start'] = pd.to_datetime(df_copy['week_start'])
    
    # División temporal
    train_data = df_copy[df_copy['week_start'] <= split_date].reset_index(drop=True)
    test_data = df_copy[df_copy['week_start'] > split_date].reset_index(drop=True)

    # Separar características (X) y objetivo (y)
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]  
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]  
    
    return X_train, y_train, X_test, y_test