# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os  # ✅ IMPORTANTE: Import agregado

from app.config.settings import get_settings
from app.routes import upload, clean, train
from app.schemas.models import HealthCheckResponse

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    description="API Backend para procesamiento de datos y entrenamiento de modelos ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ✅ CORS CONFIGURADO - Permitir Vercel y localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://predict-prep-port.vercel.app",
        "https://*.vercel.app",
        "*"  # ✅ Temporal para debugging - puedes quitarlo después
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Incluir routers
app.include_router(upload.router, prefix=settings.api_prefix, tags=["Upload"])
app.include_router(clean.router, prefix=settings.api_prefix, tags=["Clean"])
app.include_router(train.router, prefix=settings.api_prefix, tags=["Train"])

@app.get("/")
async def root():
    return {
        "message": "Backend ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "port": os.getenv("PORT", "unknown"),  # ✅ Muestra el puerto
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/health")
async def health_check():
    try:
        from app.config.settings import supabase_client
        buckets = supabase_client.storage.list_buckets()
        supabase_status = "connected"
        storage_status = "accessible"
    except Exception as e:
        supabase_status = f"error: {str(e)}"
        storage_status = "inaccessible"
    
    return {
        "status": "healthy" if supabase_status == "connected" else "degraded",
        "message": "Backend ML API funcionando",
        "timestamp": datetime.utcnow().isoformat(),
        "port": os.getenv("PORT", "unknown"),
        "services": {
            "supabase": supabase_status,
            "storage": storage_status
        }
    }

@app.on_event("startup")
async def startup_event():
    port = os.getenv("PORT", "8000")
    print("🚀 Backend ML API iniciado")
    print(f"🌐 Puerto: {port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"✅ Endpoints disponibles:")
    print(f"   - POST /api/upload")
    print(f"   - GET  /api/datasets/{{user_id}}")
    print(f"   - POST /api/analyze")
    print(f"   - POST /api/clean")
    print(f"   - POST /api/train")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Backend detenido")