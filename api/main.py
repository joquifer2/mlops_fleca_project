# Su misión es ser el punto de entrada principal para la aplicación API.
# Crea la instancia principal de FastAPI y une todas las piezas (como los diferentes routers).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import router
import logging
logging.basicConfig(level=logging.DEBUG)

# Crea una instancia de FastAPI
app = FastAPI(
    title="Bolleria Predicction API", 
    description="API para predecir las ventas de bollería usando modelo entrenado", 
    version="1.0.0"
)

# Configurar CORS para permitir conexiones desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluye el router de la aplicación FastAPI
app.include_router(router)
