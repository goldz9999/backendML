# app/routes/clean.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.config.settings import supabase_client
import os

router = APIRouter()


class AnalyzeDatasetRequest(BaseModel):
    user_id: str
    dataset_id: int


class AnalyzeDatasetResponse(BaseModel):
    dataset_id: int
    total_rows: int
    total_columns: int
    columns_info: dict
    total_nulls: int
    preview_data: List[Dict[str, Any]]


class CleanDatasetRequest(BaseModel):
    user_id: str
    dataset_id: int
    operation: List[str]
    options: Optional[Dict[str, Any]] = None


class CleanDatasetResponse(BaseModel):
    message: str
    cleaned_dataset_id: str
    original_dataset_id: str
    file_path: str
    original_rows: int
    cleaned_rows: int
    columns_with_nulls: Dict[str, Any]
    status_changes: Dict[str, int]
    operations_applied: List[str]


class CleanedDatasetInfo(BaseModel):
    id: int
    name: str
    num_rows: int
    num_columns: int
    created_at: str
    cleaning_operations: dict
    file_path: str


@router.get("/cleaned-datasets/{user_id}")
async def get_cleaned_datasets(user_id: str):
    """
    Obtiene todos los datasets limpios de un usuario.
    Necesario para el frontend de Train.
    """
    try:
        # Validar UUID
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "user_id debe ser un UUID válido")
        
        # Obtener datasets limpios
        response = supabase_client.table("cleaned_datasets")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        
        if not response.data:
            return {
                "datasets": [],
                "total": 0,
                "user_id": user_id
            }
        
        # Formatear respuesta
        datasets = []
        for dataset in response.data:
            datasets.append({
                "id": dataset["id"],
                "name": dataset["name"],
                "num_rows": dataset["num_rows"],
                "num_columns": dataset["num_columns"],
                "created_at": dataset["created_at"],
                "cleaning_operations": dataset.get("cleaning_operations", {}),
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
        raise HTTPException(500, f"Error al obtener datasets limpios: {str(e)}")


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
        
        # 3. Leer el archivo según su tipo
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
        
        # 4. Analizar cada columna
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
        
        # 5. Generar preview de las primeras 20 filas
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
            dataset_id=str(request.dataset_id),
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


@router.post("/clean", response_model=CleanDatasetResponse)
async def clean_dataset(request: CleanDatasetRequest):
    """
    Aplica TODAS las operaciones de limpieza a un dataset y lo guarda UNA SOLA VEZ.
    """
    print("=" * 50)
    print("🧹 CLEAN ENDPOINT CALLED")
    print(f"📦 User ID: {request.user_id}")
    print(f"📦 Dataset ID: {request.dataset_id}")
    print(f"📦 Operations: {request.operation}")
    print(f"📦 Options: {request.options}")
    print("=" * 50)

    try:
        # 🔹 1. Obtener dataset original
        dataset = supabase_client.table("datasets")\
            .select("*")\
            .eq("id", request.dataset_id)\
            .eq("user_id", request.user_id)\
            .single()\
            .execute()

        if not dataset.data:
            raise HTTPException(404, "Dataset no encontrado")

        file_path = dataset.data["file_path"]
        file_type = dataset.data.get("file_type", "csv").lower()

        # 🔹 2. Descargar y leer archivo
        file_bytes = supabase_client.storage.from_("datasets").download(file_path)

        if file_type == "csv":
            df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8")
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(BytesIO(file_bytes))
        else:
            raise HTTPException(400, f"Tipo de archivo no soportado: {file_type}")

        original_rows = len(df)
        operations_applied = []
        columns_with_nulls = {}
        status_changes = {}

        # 🔹 3. Asegurar que operation sea lista
        operations = request.operation if isinstance(request.operation, list) else [request.operation]

        # 🔹 4. Aplicar TODAS las operaciones en secuencia
        for op in operations:
            print(f"🔄 Aplicando operación: {op}")

            if op == "replace_nulls":
                for column in df.columns:
                    null_count = df[column].isna().sum()
                    if null_count > 0:
                        columns_with_nulls[column] = {
                            "nulls": int(null_count),
                            "is_numeric": pd.api.types.is_numeric_dtype(df[column])
                        }
                        df[column] = df[column].fillna("N/A")
                
                if "status" not in df.columns:
                    df["status"] = "active"
                
                for column, info in columns_with_nulls.items():
                    if info["is_numeric"]:
                        df.loc[df[column] == "N/A", "status"] = "inactive"

                status_changes = {
                    "active": int((df["status"] == "active").sum()),
                    "inactive": int((df["status"] == "inactive").sum())
                }
                operations_applied.append("replace_nulls")

            elif op == "impute":
                method = request.options.get("method", "mean") if request.options else "mean"
                for column in df.columns:
                    if df[column].isna().sum() > 0:
                        if pd.api.types.is_numeric_dtype(df[column]):
                            if method == "mean":
                                df[column] = df[column].fillna(df[column].mean())
                            elif method == "median":
                                df[column] = df[column].fillna(df[column].median())
                            elif method == "mode":
                                mode_value = df[column].mode()[0] if not df[column].mode().empty else 0
                                df[column] = df[column].fillna(mode_value)
                        else:
                            mode_value = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                            df[column] = df[column].fillna(mode_value)

                if "status" not in df.columns:
                    df["status"] = "active"
                operations_applied.append(f"impute_{method}")

            elif op == "normalize":
                scaler = StandardScaler()
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if "status" in numeric_columns:
                    numeric_columns.remove("status")
                if numeric_columns:
                    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                operations_applied.append("normalize")

            elif op == "encode":
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                if "status" in categorical_columns:
                    categorical_columns.remove("status")
                for column in categorical_columns:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column].astype(str))
                operations_applied.append("encode")

            else:
                raise HTTPException(400, f"Operación no soportada: {op}")

        # 🔹 5. Guardar resultado final UNA SOLA VEZ
        cleaned_file_name = f"{dataset.data['name'].split('.')[0]}_cleaned.csv"
        cleaned_file_path = f"{request.user_id}/{cleaned_file_name}"
        cleaned_dataset_id = str(uuid.uuid4())

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8")
        csv_buffer.seek(0)

        # ✅ Sobrescribir si ya existe
        try:
            supabase_client.storage.from_("cleaned_datasets").remove([cleaned_file_path])
        except:
            pass

        supabase_client.storage.from_("cleaned_datasets").upload(
            path=cleaned_file_path,
            file=csv_buffer.getvalue(),
            file_options={"content-type": "text/csv"}
        )

        # ✅ Insertar o actualizar en BD
        existing = supabase_client.table("cleaned_datasets")\
            .select("id")\
            .eq("user_id", request.user_id)\
            .eq("original_dataset_id", str(request.dataset_id))\
            .execute()

        if existing.data:
            # Actualizar registro existente
            supabase_client.table("cleaned_datasets")\
                .update({
                    "name": cleaned_file_name,
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "cleaning_operations": {
                        "operations": operations_applied,
                        "columns_affected": list(columns_with_nulls.keys())
                    },
                    "file_path": cleaned_file_path,
                    "columns_with_nulls": columns_with_nulls,
                    "created_at": datetime.utcnow().isoformat()
                })\
                .eq("id", existing.data[0]["id"])\
                .execute()
        else:
            # Crear nuevo registro
            supabase_client.table("cleaned_datasets").insert({
                "user_id": request.user_id,
                "original_dataset_id": str(request.dataset_id),
                "name": cleaned_file_name,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "cleaning_operations": {
                    "operations": operations_applied,
                    "columns_affected": list(columns_with_nulls.keys())
                },
                "file_path": cleaned_file_path,
                "columns_with_nulls": columns_with_nulls,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

        return CleanDatasetResponse(
            message="Dataset limpio creado exitosamente",
            cleaned_dataset_id=cleaned_dataset_id,
            original_dataset_id=str(request.dataset_id),
            file_path=cleaned_file_path,
            original_rows=original_rows,
            cleaned_rows=len(df),
            columns_with_nulls=columns_with_nulls,
            status_changes=status_changes,
            operations_applied=operations_applied
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(500, f"Error al limpiar dataset: {str(e)}")

    
@router.post("/analyze-cleaned")
async def analyze_cleaned_dataset(request: AnalyzeDatasetRequest):
        """
        Analiza un CLEANED dataset y retorna columnas + preview.
        Idéntico a /analyze pero busca en cleaned_datasets.
        """
        try:
            # 1. Buscar en tabla "cleaned_datasets" (no "datasets")
            dataset = supabase_client.table("cleaned_datasets")\
                .select("*")\
                .eq("id", request.dataset_id)\
                .eq("user_id", request.user_id)\
                .single()\
                .execute()
            
            if not dataset.data:
                raise HTTPException(404, "Cleaned dataset no encontrado")
            
            file_path = dataset.data["file_path"]
            
            # 2. Descargar desde bucket "cleaned_datasets"
            file_bytes = supabase_client.storage.from_("cleaned_datasets").download(file_path)
            
            # 3. Leer CSV (siempre es CSV porque clean.py guarda como CSV)
            df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8")
            
            # 4. Excluir columna "status" si existe
            if "status" in df.columns:
                df = df.drop(columns=["status"])
            
            # 5. Retornar columnas y preview
            return {
                "dataset_id": str(request.dataset_id),
                "columns": df.columns.tolist(),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "preview_data": df.head(10).to_dict('records')
            }
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Error: {str(e)}")