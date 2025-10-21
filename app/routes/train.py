# app/routes/train.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import BytesIO
import joblib
import uuid
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.config.settings import supabase_client

router = APIRouter()


class TrainModelRequest(BaseModel):
    user_id: str
    cleaned_dataset_id: str
    model_type: str  # "random_forest", "linear_regression", "logistic_regression", "svm"
    target_column: str
    test_size: float = 0.2
    random_state: int = 42


class TrainModelResponse(BaseModel):
    message: str
    model_id: str
    cleaned_dataset_id: str
    model_type: str
    model_path: str
    excluded_columns: list[str]
    metrics: dict
    training_time_seconds: float
    trained_at: str


@router.post("/train", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest):
    """
    Entrena un modelo de ML excluyendo columnas que tienen "N/A" 
    (que originalmente eran numéricas y tenían NULL).
    
    Proceso:
    1. Descargar dataset limpio
    2. Identificar columnas con "N/A" que eran numéricas
    3. Excluir esas columnas del entrenamiento
    4. Excluir filas con status="inactive" (opcional)
    5. Entrenar modelo solo con datos válidos
    6. Guardar modelo en Storage
    """
    start_time = datetime.utcnow()
    
    try:
        # 1. Obtener información del dataset limpio
        cleaned_dataset = supabase_client.table("cleaned_datasets")\
            .select("*")\
            .eq("id", request.cleaned_dataset_id)\
            .eq("user_id", request.user_id)\
            .single()\
            .execute()
        
        if not cleaned_dataset.data:
            raise HTTPException(status_code=404, detail="Dataset limpio no encontrado")
        
        file_path = cleaned_dataset.data["file_path"]
        columns_with_nulls = cleaned_dataset.data.get("columns_with_nulls", {})
        
        # 2. Descargar CSV limpio
        file_bytes = supabase_client.storage.from_("cleaned_datasets").download(file_path)
        df = pd.read_csv(BytesIO(file_bytes))
        
        # 3. Identificar columnas a EXCLUIR
        # Excluir columnas que:
        # - Tienen "N/A"
        # - Originalmente eran numéricas (int/float)
        excluded_columns = []
        
        for column, info in columns_with_nulls.items():
            if info.get("is_numeric", False):
                # Esta columna era numérica y tenía nulls → EXCLUIR
                excluded_columns.append(column)
        
        print(f"🚫 Columnas excluidas del entrenamiento: {excluded_columns}")
        
        # 4. Filtrar filas con status="inactive" (opcional)
        # Esto elimina filas que tenían NULL en columnas numéricas
        if "status" in df.columns:
            df_active = df[df["status"] == "active"].copy()
            print(f"📊 Filas activas: {len(df_active)} de {len(df)}")
        else:
            df_active = df.copy()
        
        # 5. Eliminar columnas a excluir y columnas no necesarias
        columns_to_drop = excluded_columns + ["status"]  # También eliminar "status"
        columns_to_drop = [col for col in columns_to_drop if col in df_active.columns]
        
        df_train = df_active.drop(columns=columns_to_drop, errors='ignore')
        
        # 6. Verificar que la columna target existe
        if request.target_column not in df_train.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columna target '{request.target_column}' no encontrada"
            )
        
        # 7. Separar features (X) y target (y)
        X = df_train.drop(columns=[request.target_column])
        y = df_train[request.target_column]
        
        # 8. Convertir columnas categóricas restantes a numéricas (One-Hot Encoding)
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # 9. Eliminar filas con "N/A" en features (por si quedó alguna)
        mask = ~X_encoded.isin(["N/A", "nan", "NaN"]).any(axis=1)
        X_encoded = X_encoded[mask]
        y = y[mask]
        
        print(f"✅ Datos para entrenamiento: {X_encoded.shape}")
        print(f"✅ Features: {list(X_encoded.columns)}")
        
        # 10. Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, 
            test_size=request.test_size, 
            random_state=request.random_state
        )
        
        # 11. Entrenar modelo según el tipo
        if request.model_type == "random_forest":
            # Detectar si es clasificación o regresión
            if y.dtype == 'object' or y.nunique() < 20:
                model = RandomForestClassifier(random_state=request.random_state)
                task_type = "classification"
            else:
                model = RandomForestRegressor(random_state=request.random_state)
                task_type = "regression"
        
        elif request.model_type == "linear_regression":
            model = LinearRegression()
            task_type = "regression"
        
        elif request.model_type == "logistic_regression":
            model = LogisticRegression(random_state=request.random_state, max_iter=1000)
            task_type = "classification"
        
        elif request.model_type == "svm":
            if y.dtype == 'object' or y.nunique() < 20:
                model = SVC(random_state=request.random_state)
                task_type = "classification"
            else:
                model = SVR()
                task_type = "regression"
        
        else:
            raise HTTPException(status_code=400, detail=f"Tipo de modelo no soportado: {request.model_type}")
        
        # 12. Entrenar
        model.fit(X_train, y_train)
        
        # 13. Calcular métricas
        y_pred = model.predict(X_test)
        
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                "task_type": "classification"
            }
        else:
            metrics = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2_score": float(r2_score(y_test, y_pred)),
                "task_type": "regression"
            }
        
        # 14. Serializar modelo
        model_buffer = BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        
        # 15. Generar ID del modelo
        model_id = str(uuid.uuid4())
        
        # 16. Subir modelo a Storage
        model_path = f"{request.user_id}/{model_id}/model.joblib"
        
        supabase_client.storage.from_("models").upload(
            path=model_path,
            file=model_buffer.getvalue(),
            file_options={"content-type": "application/octet-stream"}
        )
        
        # 17. Guardar métricas como JSON
        import json
        metrics_buffer = json.dumps(metrics, indent=2).encode()
        metrics_path = f"{request.user_id}/{model_id}/metrics.json"
        
        supabase_client.storage.from_("models").upload(
            path=metrics_path,
            file=metrics_buffer,
            file_options={"content-type": "application/json"}
        )
        
        # 18. Registrar en tabla "models"
        end_time = datetime.utcnow()
        training_time = (end_time - start_time).total_seconds()
        
        supabase_client.table("models").insert({
            "id": model_id,
            "user_id": request.user_id,
            "cleaned_dataset_id": request.cleaned_dataset_id,
            "model_type": request.model_type,
            "model_path": model_path,
            "excluded_columns": excluded_columns,
            "metrics": metrics
        }).execute()
        
        return TrainModelResponse(
            message="Modelo entrenado exitosamente",
            model_id=model_id,
            cleaned_dataset_id=request.cleaned_dataset_id,
            model_type=request.model_type,
            model_path=model_path,
            excluded_columns=excluded_columns,
            metrics=metrics,
            training_time_seconds=training_time,
            trained_at=end_time.isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar modelo: {str(e)}")