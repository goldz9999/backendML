# app/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from supabase import create_client, Client
from functools import lru_cache


class Settings(BaseSettings):
    """
    Configuración de la aplicación usando Pydantic Settings.
    Las variables se cargan automáticamente desde el archivo .env
    """
    
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    
    # Storage Buckets
    bucket_datasets: str = Field(default="datasets", env="BUCKET_DATASETS")
    bucket_cleaned: str = Field(default="cleaned_datasets", env="BUCKET_CLEANED")
    bucket_models: str = Field(default="models", env="BUCKET_MODELS")
    
    # File Upload Limits (en bytes)
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    max_file_size: int = Field(default=52428800)  # 50 MB en bytes
    
    # Allowed File Types
    allowed_file_types: list[str] = Field(
        default=["text/csv", "application/vnd.ms-excel", 
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
    )
    
    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "https://predict-prep-port.vercel.app"],
        env="CORS_ORIGINS"
    )
    
    # API Configuration
    api_prefix: str = Field(default="/api", env="API_PREFIX")
    project_name: str = Field(default="Backend ML API", env="PROJECT_NAME")
    debug: bool = Field(default=True, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Función cacheada para obtener la configuración.
    Se ejecuta solo una vez y reutiliza la instancia.
    """
    return Settings()


def get_supabase_client() -> Client:
    """
    Crea y retorna el cliente de Supabase con service_role_key.
    Este cliente tiene acceso total a Storage y Database sin políticas RLS.
    """
    settings = get_settings()
    
    supabase: Client = create_client(
        settings.supabase_url,
        settings.supabase_service_role_key
    )
    
    return supabase


# Instancia global del cliente de Supabase
supabase_client = get_supabase_client()