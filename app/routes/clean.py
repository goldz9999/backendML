# app/routes/clean.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from io import BytesIO
from app.config.settings import supabase_client

router = APIRouter()


class AnalyzeDatasetRequest(BaseModel):
    user_id: str
    dataset_id: str


class AnalyzeDatasetResponse(BaseModel):
    dataset_id: str
    total_rows: int
    total_columns: int
    columns_info: dict  # {"columna": {"dtype": "int64", "nulls": 5, "null_percentage": 10.5}}
    total_nulls: int


@router.post("/analyze", response_model=AnalyzeDatasetResponse)
async def analyze_dataset(request: AnalyzeDatasetRequest):
    """
    Analiza un dataset y retorna información sobre valores nulos.
    Este endpoint se llama ANTES de limpiar para mostrar al usuario qué se encontró.
    """
    try:
        # 1. Obtener información del dataset desde la BD
        dataset = supabase_client.table("datasets")\
            .select("*")\
            .eq("id", request.dataset_id)\
            .eq("user_id", request.user_id)\
            .single()\
            .execute()
        
        if not dataset.data:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        file_path = dataset.data["file_path"]
        
        # 2. Descargar CSV desde Supabase Storage
        file_bytes = supabase_client.storage.from_("datasets").download(file_path)
        df = pd.read_csv(BytesIO(file_bytes))
        
        # 3. Analizar cada columna
        columns_info = {}
        total_nulls = 0
        
        for column in df.columns:
            null_count = df[column].isna().sum()
            total_nulls += null_count
            
            columns_info[column] = {
                "dtype": str(df[column].dtype),
                "nulls": int(null_count),
                "null_percentage": round((null_count / len(df)) * 100, 2),
                "is_numeric": pd.api.types.is_numeric_dtype(df[column])
            }
        
        return AnalyzeDatasetResponse(
            dataset_id=request.dataset_id,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns_info=columns_info,
            total_nulls=int(total_nulls)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar dataset: {str(e)}")


class CleanDatasetRequest(BaseModel):
    user_id: str
    dataset_id: str
    replace_nulls: bool = True  # Si True, reemplaza NULL por "N/A"


class CleanDatasetResponse(BaseModel):
    message: str
    cleaned_dataset_id: str
    original_dataset_id: str
    file_path: str
    original_rows: int
    cleaned_rows: int
    columns_with_nulls: dict
    status_changes: dict  # {"row_2": "inactive", "row_4": "active"}


@router.post("/clean", response_model=CleanDatasetResponse)
async def clean_dataset(request: CleanDatasetRequest):
    """
    Limpia el dataset reemplazando NULL por "N/A" y calcula el status de cada fila.
    
    Reglas:
    - NULL en columna numérica (int/float) → status = "inactive"
    - NULL en columna de texto (string) → status = "active"
    """
    try:
        # 1. Obtener dataset original
        dataset = supabase_client.table("datasets")\
            .select("*")\
            .eq("id", request.dataset_id)\
            .eq("user_id", request.user_id)\
            .single()\
            .execute()
        
        if not dataset.data:
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        file_path = dataset.data["file_path"]
        
        # 2. Descargar y leer CSV
        file_bytes = supabase_client.storage.from_("datasets").download(file_path)
        df = pd.read_csv(BytesIO(file_bytes))
        
        # 3. Guardar información de tipos de datos ANTES de limpiar
        original_dtypes = df.dtypes.to_dict()
        columns_with_nulls = {}
        
        # Identificar columnas con nulls
        for column in df.columns:
            null_count = df[column].isna().sum()
            if null_count > 0:
                columns_with_nulls[column] = {
                    "nulls": int(null_count),
                    "original_dtype": str(original_dtypes[column]),
                    "is_numeric": pd.api.types.is_numeric_dtype(df[column])
                }
        
        # 4. Crear columna "status" si no existe
        if "status" not in df.columns:
            df["status"] = "active"  # Default: active
        
        # 5. Calcular status para cada fila
        status_changes = {}
        
        for index, row in df.iterrows():
            has_null_in_numeric = False
            
            # Verificar si la fila tiene NULL en alguna columna numérica
            for column, info in columns_with_nulls.items():
                if pd.isna(row[column]) and info["is_numeric"]:
                    has_null_in_numeric = True
                    break
            
            # Asignar status
            if has_null_in_numeric:
                df.at[index, "status"] = "inactive"
                status_changes[f"row_{index}"] = "inactive"
            else:
                df.at[index, "status"] = "active"
                if f"row_{index}" not in status_changes:
                    status_changes[f"row_{index}"] = "active"
        
        # 6. Reemplazar NULL por "N/A" (DESPUÉS de calcular status)
        if request.replace_nulls:
            df = df.fillna("N/A")
        
        # 7. Generar ID para dataset limpio
        import uuid
        cleaned_dataset_id = str(uuid.uuid4())
        
        # 8. Convertir DataFrame a CSV
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # 9. Subir a bucket "cleaned_datasets"
        cleaned_file_path = f"{request.user_id}/{cleaned_dataset_id}.csv"
        
        supabase_client.storage.from_("cleaned_datasets").upload(
            path=cleaned_file_path,
            file=csv_buffer.getvalue(),
            file_options={"content-type": "text/csv"}
        )
        
        # 10. Registrar en tabla "cleaned_datasets"
        supabase_client.table("cleaned_datasets").insert({
            "id": cleaned_dataset_id,
            "user_id": request.user_id,
            "original_dataset_id": request.dataset_id,
            "file_path": cleaned_file_path,
            "original_rows": len(df),
            "cleaned_rows": len(df),
            "rows_removed": 0,  # No removemos filas, solo reemplazamos valores
            "columns_with_nulls": columns_with_nulls,
            "status_changes": status_changes
        }).execute()
        
        return CleanDatasetResponse(
            message="Dataset limpio exitosamente",
            cleaned_dataset_id=cleaned_dataset_id,
            original_dataset_id=request.dataset_id,
            file_path=cleaned_file_path,
            original_rows=len(df),
            cleaned_rows=len(df),
            columns_with_nulls=columns_with_nulls,
            status_changes=status_changes
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar dataset: {str(e)}")