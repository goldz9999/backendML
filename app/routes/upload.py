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
        
        # 🔹 Detectar y leer el archivo según su extensión
        try:
            if file.filename.endswith(".csv"):
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(BytesIO(file_content), encoding="latin1")

            elif file.filename.endswith(".xlsx"):
                df = pd.read_excel(BytesIO(file_content))

            elif file.filename.endswith(".json"):
                df = pd.read_json(BytesIO(file_content))

            else:
                raise HTTPException(400, f"Formato no soportado: {file.filename}")

        except Exception as e:
            raise HTTPException(400, f"Error al leer el archivo: {str(e)}")
        
        # Registrar en BD
        dataset_id = str(uuid.uuid4())
        supabase_client.table("datasets").insert({
            "user_id": user_id,
            "name": file.filename,
            "file_path": file_path_storage,
            "file_type": file.filename.split(".")[-1].lower(),
            "file_size": file_size,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns_info": df.columns.tolist(),
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


@router.get("/datasets/{user_id}")
async def get_user_datasets(user_id: str):
    """
    Obtiene todos los datasets de un usuario específico
    """
    try:
        # Validar UUID
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "user_id debe ser un UUID válido")
        
        # Obtener datasets del usuario desde Supabase
        response = supabase_client.table("datasets")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("uploaded_at", desc=True)\
            .execute()
        
        if not response.data:
            return {
                "datasets": [],
                "total": 0,
                "user_id": user_id
            }
        
        # Formatear datos para el frontend
        datasets = []
        for dataset in response.data:
            datasets.append({
                "id": dataset["id"],
                "name": dataset["name"],
                "file_type": dataset["file_type"],
                "file_size": dataset["file_size"],
                "file_size_mb": round(dataset["file_size"] / (1024 * 1024), 2),
                "num_rows": dataset.get("num_rows", 0),
                "num_columns": dataset.get("num_columns", 0),
                "uploaded_at": dataset["uploaded_at"],
                "file_path": dataset["file_path"]
            })
        
        return {
            "datasets": datasets,
            "total": len(datasets),
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error al obtener datasets: {str(e)}")
