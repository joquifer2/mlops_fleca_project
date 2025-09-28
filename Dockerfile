# -------------------------
# Dockerfile para MLOps Fleca Project con Poetry
# -------------------------

FROM mcr.microsoft.com/devcontainers/python:3.11

# Establece el directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Configura Poetry para no crear un entorno virtual (ya estamos en Docker)
RUN poetry config virtualenvs.create false

# Copia los archivos de configuración de Poetry
COPY pyproject.toml poetry.lock ./

# Instala las dependencias usando Poetry (solo las de producción, sin instalar el proyecto local)
RUN poetry install --only=main --no-root

# Copia todo el código
COPY . .

# Expone los puertos
EXPOSE 8000 8501

# Comando por defecto
CMD ["bash"]
