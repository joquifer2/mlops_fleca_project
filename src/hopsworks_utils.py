# -------------------------
# HOPSWORKS UTILS: Funciones de acceso y consulta a Hopsworks
# -------------------------
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

# Ejemplo de función para conectar al feature store
from src.inference import conectar_hopsworks_feature_store

# Función para obtener predicciones desde una feature view

def obtener_predicciones_feature_view(feature_store, metadata=None, name=None, version=None):
    """
    Obtiene el DataFrame de la feature view de predicciones desde Hopsworks.
    Args:
        feature_store: objeto feature_store de Hopsworks
        metadata: diccionario con metadatos (opcional)
        name: nombre de la feature view (opcional)
        version: versión de la feature view (opcional)
    Returns:
        DataFrame con las predicciones
    """
    if metadata is not None:
        name = metadata.get('name')
        version = metadata.get('version')
    elif name is None or version is None:
        raise ValueError("Debe proporcionar metadata o name y version")
    fv = feature_store.get_feature_view(name=name, version=version)
    df_predicciones = fv.get_batch_data()
    df_predicciones = df_predicciones.sort_values('week_start').reset_index(drop=True)
    return df_predicciones

# Aquí puedes añadir más funciones de acceso/consulta a Hopsworks según crezca el proyecto.
