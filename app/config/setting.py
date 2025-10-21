from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings

settings = get_settings()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,      # Orígenes permitidos
    allow_credentials=True,                   # Permite cookies/auth headers
    allow_methods=["*"],                      # Todos los métodos HTTP
    allow_headers=["*"],                      # Todos los headers
)