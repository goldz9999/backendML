# app/schemas/models.py
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List
import uuid


# ============================================
# 6.4.2. Schemas para Datasets
# ============================================

class DatasetUploadRequest(BaseModel):
    """Request para subir un dataset original"""
    user_id: str = Field(..., description="UUID del usuario generado en el frontend")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('user_id debe ser un UUID válido')
        return v


class DatasetUploadResponse(BaseModel):
    """Response después de subir un dataset"""
    message: str
    dataset_id: str
    user_id: str
    file_name: str
    file_path: str
    file_size: int
    file_type: str
    uploaded_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Dataset subido exitosamente",
                "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "660f9511-f30c-52e5-b827-557766551111",
                "file_name": "ventas_2024.csv",
                "file_path": "660f9511-f30c-52e5-b827-557766551111/ventas_2024.csv",
                "file_size": 1024000,
                "file_type": "text/csv",
                "uploaded_at": "2024-01-15T10:30:00"
            }
        }


class DatasetListResponse(BaseModel):
    """Response para listar datasets de un usuario"""
    datasets: List[dict]
    total: int


# ============================================
# 6.4.3. Schemas para Limpieza de Datos
# ============================================

class CleanDatasetRequest(BaseModel):
    """Request para limpiar un dataset"""
    user_id: str = Field(..., description="UUID del usuario")
    dataset_id: str = Field(..., description="ID del dataset original a limpiar")
    
    # Opciones de limpieza
    remove_nulls: bool = Field(default=True, description="Eliminar filas con valores nulos")
    remove_duplicates: bool = Field(default=True, description="Eliminar filas duplicadas")
    fill_nulls_method: Optional[str] = Field(
        default=None, 
        description="Método para rellenar nulos: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'"
    )
    
    @validator('user_id', 'dataset_id')
    def validate_uuids(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('Debe ser un UUID válido')
        return v
    
    @validator('fill_nulls_method')
    def validate_fill_method(cls, v):
        if v is not None:
            allowed = ['mean', 'median', 'mode', 'forward_fill', 'backward_fill']
            if v not in allowed:
                raise ValueError(f'fill_nulls_method debe ser uno de: {", ".join(allowed)}')
        return v


class CleanDatasetResponse(BaseModel):
    """Response después de limpiar un dataset"""
    message: str
    cleaned_dataset_id: str
    user_id: str
    original_dataset_id: str
    file_path: str
    original_rows: int
    cleaned_rows: int
    rows_removed: int
    original_columns: int
    cleaned_columns: int
    cleaning_summary: dict
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Dataset limpio creado exitosamente",
                "cleaned_dataset_id": "770a1722-g41d-63f9-c938-668877662222",
                "user_id": "660f9511-f30c-52e5-b827-557766551111",
                "original_dataset_id": "550e8400-e29b-41d4-a716-446655440000",
                "file_path": "660f9511-f30c-52e5-b827-557766551111/770a1722-g41d-63f9-c938-668877662222.csv",
                "original_rows": 1000,
                "cleaned_rows": 950,
                "rows_removed": 50,
                "original_columns": 10,
                "cleaned_columns": 10,
                "cleaning_summary": {
                    "nulls_removed": 30,
                    "duplicates_removed": 20
                },
                "created_at": "2024-01-15T10:35:00"
            }
        }


# ============================================
# 6.4.4. Schemas para Entrenamiento de Modelos
# ============================================

class TrainModelRequest(BaseModel):
    """Request para entrenar un modelo de ML"""
    user_id: str = Field(..., description="UUID del usuario")
    cleaned_dataset_id: str = Field(..., description="ID del dataset limpio")
    model_type: str = Field(..., description="Tipo de modelo: 'random_forest', 'linear_regression', 'logistic_regression', 'svm', 'neural_network'")
    target_column: str = Field(..., description="Nombre de la columna objetivo (target)")
    
    # Configuración del modelo
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Porcentaje de datos para testing (0.1 - 0.5)")
    random_state: int = Field(default=42, description="Semilla aleatoria para reproducibilidad")
    
    # Hiperparámetros opcionales (serán específicos por modelo)
    hyperparameters: Optional[dict] = Field(default=None, description="Hiperparámetros específicos del modelo")
    
    @validator('user_id', 'cleaned_dataset_id')
    def validate_uuids(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('Debe ser un UUID válido')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed = ['random_forest', 'linear_regression', 'logistic_regression', 'svm', 'neural_network']
        if v not in allowed:
            raise ValueError(f'model_type debe ser uno de: {", ".join(allowed)}')
        return v


class TrainModelResponse(BaseModel):
    """Response después de entrenar un modelo"""
    message: str
    model_id: str
    user_id: str
    cleaned_dataset_id: str
    model_type: str
    model_path: str
    metrics: dict
    training_time_seconds: float
    trained_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Modelo entrenado y guardado exitosamente",
                "model_id": "880b2833-h52e-74g0-d049-779988773333",
                "user_id": "660f9511-f30c-52e5-b827-557766551111",
                "cleaned_dataset_id": "770a1722-g41d-63f9-c938-668877662222",
                "model_type": "random_forest",
                "model_path": "660f9511-f30c-52e5-b827-557766551111/880b2833-h52e-74g0-d049-779988773333/model.joblib",
                "metrics": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.94,
                    "f1_score": 0.935
                },
                "training_time_seconds": 12.5,
                "trained_at": "2024-01-15T10:40:00"
            }
        }


class ModelListResponse(BaseModel):
    """Response para listar modelos de un usuario"""
    models: List[dict]
    total: int


# ============================================
# 6.4.5. Schemas Genéricos (Error, Success)
# ============================================

class ErrorResponse(BaseModel):
    """Response estándar para errores"""
    error: str
    detail: Optional[str] = None
    status_code: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Dataset no encontrado",
                "detail": "No existe un dataset con el ID proporcionado",
                "status_code": 404
            }
        }


class SuccessResponse(BaseModel):
    """Response genérica de éxito"""
    message: str
    data: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operación completada exitosamente",
                "data": {}
            }
        }


class HealthCheckResponse(BaseModel):
    """Response para el health check"""
    status: str
    message: str
    timestamp: datetime
    services: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Backend ML API funcionando correctamente",
                "timestamp": "2024-01-15T10:00:00",
                "services": {
                    "supabase": "connected",
                    "storage": "accessible"
                }
            }
        }