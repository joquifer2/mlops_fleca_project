# Guía: Configuración de Feast para Batch Inference con BigQuery (sin GCS ni Redis)

Este documento describe, paso a paso, cómo subir tu Parquet a BigQuery y configurar Feast como Feature Store únicamente con BigQuery como offline store y un registry local.

---

## 1. Subir tu Parquet a BigQuery

### 1.1 (Opcional) Copiar a GCS para carga más fiable

```bash
gsutil cp ./data/my_features.parquet gs://mi-temporal-bucket/data/my_features.parquet
```

> **Qué y por qué**: Transfiere el Parquet a GCS para una carga más estable. Si cargas directo desde local, puedes omitir este paso.

### 1.2 Cargar a BigQuery

#### a) Desde GCS

```bash
bq load \
  --source_format=PARQUET \
  mi_proyecto.mi_dataset.raw_features \
  gs://mi-temporal-bucket/data/my_features.parquet
```

#### b) Directo desde local

```bash
bq load \
  --source_format=PARQUET \
  mi_proyecto.mi_dataset.raw_features \
  ./data/my_features.parquet
```

> **Qué y por qué**: Crea la tabla `raw_features` en BigQuery para que Feast pueda leer los datos.

---

## 2. Preparar el entorno Python

```bash
# 1. Crear y activar virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install feast google-cloud-bigquery pandas
```

> **Qué y por qué**: Aísla dependencias e instala Feast, cliente BigQuery y pandas para manipular DataFrames.

---

## 3. Inicializar el repo de Feast

```bash
feast init forecasting_repo
cd forecasting_repo
```

> **Qué y por qué**: Genera la estructura básica para tu Feature Repository:
>
> ```
> forecasting_repo/
> ├── feature_repo.yaml
> ├── data_source.py
> ├── feature_views.py
> └── README.md
> ```

---

## 4. Configurar `feature_repo.yaml` para registry local

```yaml
project: forecasting_project
provider: local

registry: ./registry.db   # Registry local en SQLite

offline_store:
  type: bigquery
  dataset: mi_dataset     # Dataset en BQ donde Feast creará tablas de features
```

> **Qué y por qué**: Usa un registry local (`registry.db`) y BigQuery como único offline store.

---

## 5. Definir la fuente de datos

En `data_source.py`:

```python
from feast import BigQuerySource

raw_features = BigQuerySource(
    table_ref="mi_proyecto.mi_dataset.raw_features",  # tu tabla en BQ
    event_timestamp_column="event_ts",                # marca temporal
)
```

> **Qué y por qué**: Configura Feast para leer la tabla raw en BigQuery.

---

## 6. Definir Entidad y FeatureView

En `feature_views.py`:

```python
from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from data_source import raw_features

producto = Entity(
    name="producto_id",
    value_type=Int64,
    description="ID único del producto"
)

forecast_fv = FeatureView(
    name="forecast_features",
    entities=["producto_id"],
    ttl=None,            # sin caducidad
    batch_source=raw_features,
    schema=[
        Field(name="venta_estimada", dtype=Float32),
        Field(name="stock_nivel",   dtype=Float32),
    ],
)
```

> **Qué y por qué**: Define la entidad `producto_id` y cómo agrupar los campos de features.

---

## 7. Aplicar la configuración

```bash
feast apply
```

> **Qué y por qué**: Crea/actualiza `registry.db` y genera la tabla `mi_dataset.forecast_features` en BigQuery.

---

## 8. Recuperar features históricas (batch inference)

En `batch_inference.py`:

```python
import pandas as pd
from feast import FeatureStore

# 1) Carga el repo local de Feast
store = FeatureStore(repo_path=".")

# 2) Define el DataFrame de entidades + timestamp
entity_df = pd.DataFrame({
    "producto_id": [101, 102, 103],
    "event_ts": [
        "2025-07-30T00:00:00",
        "2025-07-30T00:00:00",
        "2025-07-30T00:00:00",
    ]
})

# 3) Obtener features históricas
training_df = store.get_historical_features(
    entity_df=entity_df,
    feature_refs=[
        "forecast_features:venta_estimada",
        "forecast_features:stock_nivel",
    ]
).to_df()

print(training_df)
```

> **Qué y por qué**: usa `get_historical_features` para generar una consulta SQL que une `entity_df` con tu tabla de features en BigQuery y devuelve un DataFrame listo para modelado.

---

## 9. Resumen del flujo completo

1. Carga tu Parquet a BigQuery con `bq load`.
2. Crea y activa un virtualenv, instala `feast`, `google-cloud-bigquery` y `pandas`.
3. `feast init forecasting_repo` → estructura básica del repo.
4. Configura `feature_repo.yaml` con registry local y offline store en BigQuery.
5. Define `BigQuerySource` en `data_source.py`.
6. Define `Entity` + `FeatureView` en `feature_views.py`.
7. Ejecuta `feast apply` para crear `registry.db` y tablas en BQ.
8. En tu script de batch, llama a `get_historical_features()` para recuperar los features desde BigQuery.



## 9. Resumen del flujo completo

1. Carga tu Parquet a BigQuery con `bq load`.
2. Crea y activa un virtualenv, instala `feast`, `google-cloud-bigquery` y `pandas`.
3. `feast init forecasting_repo` → estructura básica del repo.
4. Configura `feature_repo.yaml` con registry local y offline store en BigQuery.
5. Define `BigQuerySource` en `data_source.py`.
6. Define `Entity` + `FeatureView` en `feature_views.py`.
7. Ejecuta `feast apply` para crear `registry.db` y tablas en BQ.
8. En tu script de batch, llama a `get_historical_features()` para recuperar los features desde BigQuery.