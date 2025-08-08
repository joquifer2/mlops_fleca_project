

"""
Feature pipeline para ingestión y guardado local de features procesados.
Siempre guarda los datos procesados usando el archivo más reciente generado.
"""

import os
from src.data_utils import load_raw_data, transformar_a_series_temporales, transformar_features_target, guardar_time_series_interim, guardar_datos_procesados
from src.paths import ROOT_DIR


def run_feature_pipeline():
	"""
	Ejecuta el pipeline de ingestión de features:
	- Carga datos raw
	- Transforma a series temporales
	- Genera features y target
	- Guarda los datos procesados localmente
	- El archivo generado será el más reciente y se usará automáticamente en el pipeline de entrenamiento
	"""
	# 1. Cargar datos raw
	df_raw = load_raw_data(
		fecha_inicio="2023-01-02",
		fecha_fin="2025-06-30",
		descargar_bq=False
	)
	# 2. Transformar a series temporales
	df_ts = transformar_a_series_temporales(df_raw)

	# 3. Generar features y target
	X, y, df_completo = transformar_features_target(
		df_ts,
		lags_list=[1, 2, 3, 52],
		columna_target='base_imponible',
		cols_exogenas=['is_summer_peak', 'is_easter'],
		eliminar_nulos=True
	)

	# 4. Guardar localmente en interim y processed
	familia = 'BOLLERIA'
	filepath = guardar_time_series_interim(df_ts, familia)
	archivos_guardados = guardar_datos_procesados(X, y, df_completo, familia=familia)
	print(f"Features semanales guardados en: {filepath}")
	print("Archivos de features y target guardados:")
	for tipo, ruta in archivos_guardados.items():
		print(f"- {tipo}: {ruta}")

if __name__ == "__main__":
	run_feature_pipeline()