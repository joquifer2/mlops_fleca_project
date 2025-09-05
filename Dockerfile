# -------------------------
# Dockerfile para MLOps Fleca Project
# -------------------------

FROM python:3.10-slim



# Establece el directorio de trabajo
WORKDIR /app

# Instala compiladores y dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential gcc \
	&& rm -rf /var/lib/apt/lists/*


# Copia todo el código (incluyendo README.md y src)
COPY . .

# Instala poetry
RUN pip install poetry

# Instala las dependencias del proyecto
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Expone los puertos para FastAPI (8000) y Streamlit (8501)
EXPOSE 8000 8501

# Comando por defecto (no inicia nada, lo hace docker-compose)
CMD ["bash"]
