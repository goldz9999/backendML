# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.config.settings import get_settings
from app.routes import upload, clean, train  # ✅ clean ya está importado
from app.schemas.models import HealthCheckResponse

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    description="API Backend para procesamiento de datos y entrenamiento de modelos ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Incluir routers - clean ya incluye los endpoints /analyze y /clean
app.include_router(upload.router, prefix=settings.api_prefix, tags=["Upload"])
app.include_router(clean.router, prefix=settings.api_prefix, tags=["Clean"])  # ✅ Aquí están /analyze y /clean
app.include_router(train.router, prefix=settings.api_prefix, tags=["Train"])

@app.get("/")
async def root():
    return {
        "message": "Backend ML API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    from app.config.settings import supabase_client
    
    try:
        buckets = supabase_client.storage.list_buckets()
        supabase_status = "connected"
        storage_status = "accessible"
    except Exception as e:
        supabase_status = f"error: {str(e)}"
        storage_status = "inaccessible"
    
    return {
        "status": "healthy" if supabase_status == "connected" else "unhealthy",
        "message": "Backend ML API funcionando",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "supabase": supabase_status,
            "storage": storage_status
        }
    }

@app.on_event("startup")
async def startup_event():
    print("🚀 Backend ML API iniciado")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"✅ Endpoints disponibles:")
    print(f"   - POST /api/upload")
    print(f"   - GET  /api/datasets/{{user_id}}")
    print(f"   - POST /api/analyze")  # ✅ Este endpoint ahora existe
    print(f"   - POST /api/clean")
    print(f"   - POST /api/train")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Backend detenido")