@echo off
REM Script para arrancar el servidor MLflow en local
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
