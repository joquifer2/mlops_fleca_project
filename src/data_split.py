from datetime import datetime
from typing import Tuple
import pandas as pd

def train_test_split(
        df: pd.DataFrame, 
        split_date,
        target: str = 'base_imponible',
        date_column: str = 'week'  # Assuming 'week' is the date column
        ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    """
    Divide un dataframe de series en train y test según fecha de corte.

    Parameters:
    - df: pd.DataFrame - DataFrame completo, debe incluir las columnas semana, lags, is_summer_pick, is_easter y el target.
    - split_date: int - Número de semana (ej: 202413) para dividir el DataFrame.
    - target: str - Nombre de la columna objetivo (target) en el DataFrame.
    - date_column: str - Nombre de la columna de fecha en el DataFrame.

    Returns:
    - X_train: pd.DataFrame - Features de entrenamiento.
    - y_train: pd.Series - Target de entrenamiento.
    - X_test: pd.DataFrame - Features de prueba.
    - y_test: pd.Series - Target de prueba.
    """

    # Convertir split_date al mismo tipo que la columna date_column para asegurar una comparación correcta
    split_date = df[date_column].dtype.type(split_date)
    
    # División temporal (asegurando compatibilidad de tipos)
    train_data = df[df[date_column] <= split_date]
    test_data = df[df[date_column] > split_date]

    # Separar características (X) y objetivo (y)
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]  
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]  
    
    return X_train, y_train, X_test, y_test