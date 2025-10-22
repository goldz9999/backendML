# app/routes/clean.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from io import BytesIO
from typing import List, Dict, Any
from app.config.settings import supabase_client

router = APIRouter()


class AnalyzeDatasetRequest(BaseModel):
    user_id: str
    dataset_id: int


class AnalyzeDatasetResponse(BaseModel):
    dataset_id: int
    total_rows: int
    total_columns: int
    columns_info: dict  # {"columna": {"dtype": "int64", "nulls": 5, "null_percentage": 10.5}}
    total_nulls: int
    preview_data: List[Dict[str, Any]]  # 🆕 NUEVO: Lista de filas para preview


@router.post("/analyze", response_model=AnalyzeDatasetResponse)
async def analyze_dataset(request: AnalyzeDatasetRequest):
    """
    Analiza un dataset y retorna información sobre valores nulos + preview de datos.
    Soporta CSV y XLSX.
    """
    print("=" * 50)
    print("🔍 ANALYZE ENDPOINT CALLED")
    print(f"📦 Request received:")
    print(f"   user_id: {request.user_id}")
    print(f"   dataset_id: {request.dataset_id}")
    print("=" * 50)
    
    try:
        # 1. Obtener información del dataset desde la BD
        print(f"🔎 Buscando dataset en BD...")
        dataset = supabase_client.table("datasets")\
            .select("*")\
            .eq("id", request.dataset_id)\
            .eq("user_id", request.user_id)\
            .single()\
            .execute()
        
        print(f"📊 Dataset encontrado: {dataset.data}")
        
        if not dataset.data:
            print("❌ Dataset no encontrado en BD")
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        file_path = dataset.data["file_path"]
        file_type = dataset.data.get("file_type", "csv").lower()
        print(f"📁 File path: {file_path}")
        print(f"📄 File type: {file_type}")
        
        # 2. Descargar archivo desde Storage
        print(f"⬇️  Descargando archivo desde Storage...")
        file_bytes = supabase_client.storage.from_("datasets").download(file_path)
        print(f"✅ Archivo descargado: {len(file_bytes)} bytes")
        
        # 🔹 Leer el archivo según su tipo
        try:
            if file_type == "csv":
                try:
                    df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(BytesIO(file_bytes), encoding="latin1")
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(BytesIO(file_bytes))
            else:
                raise HTTPException(400, f"Tipo de archivo no soportado: {file_type}")
        except Exception as e:
            raise HTTPException(400, f"Error al leer el archivo: {str(e)}")
        
        print(f"📊 Archivo leído: {len(df)} filas, {len(df.columns)} columnas")
        print(f"📋 Columnas: {list(df.columns)}")
        
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
        
        print(f"📈 Análisis completado:")
        print(f"   Total nulls: {total_nulls}")
        print(f"   Columnas analizadas: {len(columns_info)}")
        
        # 4. Generar preview de las primeras 20 filas
        print(f"🖼️  Generando preview...")
        preview_rows = []
        for idx, row in df.iterrows():
            row_dict = {"_id": int(idx) + 1}
            
            for column in df.columns:
                value = row[column]
                
                if pd.isna(value):
                    row_dict[column] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    row_dict[column] = str(value)
                elif isinstance(value, (int, float)):
                    row_dict[column] = float(value) if isinstance(value, float) else int(value)
                else:
                    row_dict[column] = str(value)
            
            preview_rows.append(row_dict)
        
        print(f"✅ Preview generado: {len(preview_rows)} filas")
        
        response_data = AnalyzeDatasetResponse(
            dataset_id=request.dataset_id,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns_info=columns_info,
            total_nulls=int(total_nulls),
            preview_data=preview_rows
        )
        
        print(f"🎉 Response preparado exitosamente")
        print("=" * 50)
        
        return response_data
    
    except HTTPException as he:
        print(f"❌ HTTPException: {he.detail}")
        raise
    except Exception as e:
        print(f"💥 Error inesperado: {type(e).__name__}")
        print(f"💥 Mensaje: {str(e)}")
        import traceback
        print(f"💥 Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error al analizar dataset: {str(e)}")
