# Fleca MLOps Forecasting Project
**Trabajo final del Máster ML Engineer - Nodd3r**

## Objetivo académico
Este proyecto ha sido desarrollado como trabajo final del Máster ML Engineer en Nodd3r, con el objetivo de demostrar competencias en:

1. Diseño y despliegue de sistemas MLOps
2. Automatización de pipelines de Machine Learning
3. Integración de modelos en producción
4. Buenas prácticas de ingeniería de datos y software


## Descripción
Este proyecto consiste en el desarrollo y despliegue de una API de predicción de ventas semanales de productos bollería de un negocio de hostelería, utilizando técnicas avanzadas de Machine Learning y MLOps. El objetivo principal es demostrar la capacidad de construir, versionar y poner en producción un modelo de forecasting real, integrando buenas prácticas de ingeniería de datos y automatización.

## Características principales
- API RESTful desarrollada con FastAPI y Docker.
- Frontend interactivo con Streamlit para visualización y consumo de la API.
- Pipeline MLOps completo: entrenamiento, registro y despliegue de modelos.
- Integración con Hopsworks para gestión de feature store y model registry.
- Automatización con GitHub Actions para CI/CD y ejecución de pipelines.
- Configuración reproducible con Poetry y Docker Compose.

## Estructura del proyecto

mlops_fleca_project/
│
├── api/                # Código de la API FastAPI
├── src/                # Código fuente y utilidades
├── notebooks/          # Jupyter Notebooks de experimentación y análisis
├── models/             # Modelos entrenados y serializados
├── data/               # Datos en diferentes estados (raw, processed, etc.)
├── .github/workflows/  # Workflows de CI/CD (GitHub Actions)
├── Dockerfile          # Imagen Docker para despliegue
├── docker-compose.yml  # Orquestación de servicios (API + Streamlit)
├── pyproject.toml      # Configuración de dependencias (Poetry)
├── README.md           # Este archivo
└── ...                 # Otros archivos y utilidades


## Despliegue en Render
El modelo y la API han sido desplegados en Render y están disponibles públicamente:

API FastAPI: https://mlops-fleca-project-api.onrender.com
Frontend Streamlit: https://mlops-fleca-project.onrender.com
Puedes acceder a la documentación interactiva de la API en:

https://mlops-fleca-project-api.onrender.com/docs

### Resumen operativo
Este repositorio incluye una solución completa lista para demostración y despliegue. A modo de resumen operativo:

- API de predicción: genera predicciones semanales utilizando el modelo XGBoost entrenado y versionado en el proyecto.
- Gráfico comparativo: el frontend muestra datos históricos reales frente a las predicciones para facilitar la interpretación.
- Frontend claro y ligero: interfaz construida con Streamlit que actúa únicamente como consumidor de la API.
- Arquitectura separada: el frontend no accede directamente a Hopsworks; toda la lógica de features/modelos está en la API.
- Despliegue en la nube: API y frontend están desplegados en Render y accesibles públicamente (URLs arriba).

Consideraciones finales (opcional):

- Las credenciales y secretos no están incluidos en el repositorio; se han configurado como variables secretas en la plataforma de Render.

- Para reproducibilidad en despliegues de producción se recomienda usar la imagen Docker incluida y variables de entorno controladas.


## Despliegue en Cloud Run de Google Cloud
El proyecto está preparado para despliegue automático en Google Cloud Run, aprovechando la integración nativa con GitHub y los flujos CI/CD modernos. El objetivo es garantizar una arquitectura robusta, escalable y reproducible, alineada con los estándares de MLOps en producción.

**Arquitectura y componentes**
* API FastAPI: Servicio backend que expone los endpoints de predicción y gestión de modelos. Desplegado como contenedor en Cloud Run, accesible vía HTTPS y escalable bajo demanda.
* Frontend Streamlit: Interfaz ligera para visualización y consumo de la API, también desplegada como servicio independiente en Cloud Run.
* GitHub Actions: Automatiza el ciclo de build, test y despliegue. Cada push a main dispara la construcción de la imagen Docker y su despliegue en Cloud Run.
* Docker: Ambos servicios (API y frontend) cuentan con Dockerfiles optimizados para Cloud Run, asegurando portabilidad y tiempos de build mínimos.

**Descripción del despliegue**
1. Preparación del código: El repositorio incluye los Dockerfiles y .dockerignore necesarios para construir imágenes ligeras y reproducibles. 2. Los directorios opcionales (como models) se gestionan automáticamente en el build para evitar errores.
3. Integración con GitHub: Al hacer push a la rama principal, GitHub Actions ejecuta los workflows definidos, construyendo y testeando los contenedores.
4. Despliegue automático: Cloud Build recibe las imágenes y las publica en Cloud Run, exponiendo los servicios en URLs públicas y seguras.
Variables de entorno: Las credenciales sensibles (API keys, nombres de proyecto, etc.) se configuran como variables secretas en Cloud Run, nunca en el código fuente.

**URLs**
API FastAPI: https://mlops-fleca-project-api-142425263805.europe-west1.run.app
Frontend Streamlit: https://mlops-fleca-project-streamlit-142425263805.europe-west1.run.app/predict

Puedes acceder a la documentación interactiva de la API en: https://mlops-fleca-project-api-142425263805.europe-west1.run.app/docs



## Despliegue básico en Streamlit Cloud

Además del despliegue en Render y en google cloud, el frontend interactivo ha sido publicado en Streamlit Cloud. Esta plataforma permite visualizar y consumir el modelo de predicción directamente desde el navegador, sin necesidad de instalar dependencias ni ejecutar código localmente.

- **Streamlit Cloud:** [https://mlopsflecaprojectv0.streamlit.app/](https://mlopsflecaprojectv0.streamlit.app/)

En esta interfaz puedes:
- Realizar predicciones de ventas semanales de bollería usando el modelo desplegado.
- Visualizar los resultados y gráficos generados por el modelo.
- Probar la API de forma sencilla y rápida, ideal para demostraciones y validación del funcionamiento en producción.

> El frontend de Streamlit Cloud ejecuta el modelo directamente con el código disponible en este repositorio de GitHub, por lo que las predicciones se realizan en tiempo real sobre la versión publicada en Streamlit Cloud, de forma independiente a la API pública de Render.
