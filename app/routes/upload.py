# app/routes/upload.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import uuid
from datetime import datetime
from io import BytesIO
import pandas as pd

from app.config.settings import supabase_client, get_settings

router = APIRouter()
settings = get_settings()

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Sube un dataset original al bucket 'datasets'
    """
    try:
        # Validar UUID del usuario
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "user_id debe ser un UUID válido")
        
        # Validar tipo de archivo
        if file.content_type not in settings.allowed_file_types:
            raise HTTPException(400, f"Tipo de archivo no permitido: {file.content_type}")
        
        # Leer contenido
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validar tamaño
        if file_size > settings.max_file_size:
            raise HTTPException(413, f"Archivo demasiado grande. Máximo: {settings.max_file_size_mb}MB")
        
        # Generar ruta
        file_path_storage = f"{user_id}/{file.filename}"
        
        # Subir a Supabase Storage
        supabase_client.storage.from_(settings.bucket_datasets).upload(
            path=file_path_storage,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        
        # Leer CSV para obtener info
        df = pd.read_csv(BytesIO(file_content))
        
        # Registrar en BD
        dataset_id = str(uuid.uuid4())
        supabase_client.table("datasets").insert({
            "id": dataset_id,
            "user_id": user_id,
            "name": file.filename,
            "file_path": file_path_storage,
            "file_type": file.content_type,
            "file_size": file_size,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()
        
        return {
            "message": "Dataset subido exitosamente",
            "dataset_id": dataset_id,
            "user_id": user_id,
            "file_name": file.filename,
            "file_path": file_path_storage,
            "file_size": file_size,
            "rows": len(df),
            "columns": len(df.columns)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error al subir dataset: {str(e)}")