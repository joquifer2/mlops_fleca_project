

"""
Feature pipeline para ingestión y guardado local de features procesados.
"""

import os
from src.data_utils import load_raw_data, transformar_a_series_temporales, transformar_features_target, guardar_time_series_interim, guardar_datos_procesados
from src.paths import ROOT_DIR


def run_feature_pipeline(
    fecha_inicio="2023-01-02",
    fecha_fin=None,
    familia="BOLLERIA",
    descargar_bq=True):
    """
    Ejecuta el pipeline de ingestión de features:
    - Carga datos raw
    - Transforma a series temporales para la familia indicada
    - Genera features y target
    - Guarda los datos procesados localmente
    """
    # 1. Cargar datos raw
    df_raw = load_raw_data(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        descargar_bq=descargar_bq
    )
    # 2. Transformar a series temporales para la familia indicada
    df_ts = transformar_a_series_temporales(df_raw, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, familia=familia)

    # 3. Generar features y target
    X, y, df_completo = transformar_features_target(
        df_ts,
        lags_list=[1, 2, 3, 52],
        columna_target='base_imponible',
        cols_exogenas=['is_summer_peak', 'is_easter'],
        eliminar_nulos=True
    )

    # 4. Guardar localmente en interim y processed
    filepath = guardar_time_series_interim(df_ts, familia)
    archivos_guardados = guardar_datos_procesados(X, y, df_completo, familia=familia)

    print(f"Features semanales guardados en: {filepath}")
    print("Archivos de features y target guardados:")
    for tipo, ruta in archivos_guardados.items():
        print(f"- {tipo}: {ruta}")

    # Printar la última semana cargada
    if not df_ts.empty:
        print("\nÚltima semana cargada:")
        print(df_ts.iloc[[-1]])
    else:
        print("No hay datos en df_ts para mostrar la última semana.")

if __name__ == "__main__":
    run_feature_pipeline()

# Printar la última semana cargada
