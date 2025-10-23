# app/routes/train.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from io import BytesIO
import joblib
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import time

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from app.config.settings import supabase_client

router = APIRouter()


# ============================================
# MODELOS PYTORCH
# ============================================

class MLPClassifier(nn.Module):
    """Neural Network (Perceptrón Multicapa) para clasificación"""
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPRegressor(nn.Module):
    """Neural Network para regresión"""
    def __init__(self, input_size, hidden_layers, dropout=0.2):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    """CNN para datos tabulares"""
    def __init__(self, input_size, conv_layers, fc_layers, output_size, dropout=0.3):
        super(CNNClassifier, self).__init__()
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        
        for out_channels in conv_layers:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # Calculate flattened size
        self.conv_output_size = input_size // (2 ** len(conv_layers)) * conv_layers[-1]
        
        # Fully connected layers
        fc_layer_list = []
        prev_size = self.conv_output_size
        
        for fc_size in fc_layers:
            fc_layer_list.append(nn.Linear(prev_size, fc_size))
            fc_layer_list.append(nn.ReLU())
            fc_layer_list.append(nn.Dropout(dropout))
            prev_size = fc_size
        
        fc_layer_list.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*fc_layer_list)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.flatten(1)
        return self.fc(x)


class LSTMClassifier(nn.Module):
    """LSTM para series temporales o secuencias"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])  # Last hidden state
        return out


# ============================================
# SCHEMAS
# ============================================

class TrainModelRequest(BaseModel):
    user_id: str
    dataset_id: int  # ID de cleaned_datasets
    name: str  # Nombre del modelo
    algorithm: str  # "random_forest", "linear_regression", "logistic_regression", "svm", "neural_network", "cnn", "lstm"
    target_variable: str
    hyperparameters: Optional[Dict[str, Any]] = None
    test_size: float = 0.2
    random_state: int = 42


class TrainModelResponse(BaseModel):
    message: str
    model_id: int
    name: str
    algorithm: str
    model_path: str
    metrics: dict
    training_time: int
    trained_at: str


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def load_cleaned_dataset(user_id: str, dataset_id: int):
    """Carga un dataset limpio desde Supabase"""
    
    # Obtener metadata
    dataset = supabase_client.table("cleaned_datasets")\
        .select("*")\
        .eq("id", dataset_id)\
        .eq("user_id", user_id)\
        .single()\
        .execute()
    
    if not dataset.data:
        raise HTTPException(404, "Dataset limpio no encontrado")
    
    file_path = dataset.data["file_path"]
    
    # Descargar CSV
    file_bytes = supabase_client.storage.from_("cleaned_datasets").download(file_path)
    df = pd.read_csv(BytesIO(file_bytes))
    
    return df, dataset.data


def prepare_data_for_training(df: pd.DataFrame, target_column: str, columns_with_nulls: dict):
    """
    Prepara datos para entrenamiento:
    1. Excluye columnas con "N/A" que eran numéricas
    2. Filtra filas con status="active"
    3. Convierte categóricas a numéricas
    """
    
    # Identificar columnas a excluir
    excluded_columns = []
    for column, info in columns_with_nulls.items():
        if info.get("is_numeric", False):
            excluded_columns.append(column)
    
    print(f"🚫 Columnas excluidas: {excluded_columns}")
    
    # Filtrar filas activas
    if "status" in df.columns:
        df = df[df["status"] == "active"].copy()
        print(f"✅ Filas activas: {len(df)}")
    
    # Eliminar columnas excluidas y status
    columns_to_drop = excluded_columns + ["status"]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Verificar target
    if target_column not in df.columns:
        raise HTTPException(400, f"Columna target '{target_column}' no encontrada")
    
    # Separar X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # One-Hot Encoding para categóricas
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Eliminar filas con "N/A" residual
    mask = ~X_encoded.isin(["N/A", "nan", "NaN"]).any(axis=1)
    X_encoded = X_encoded[mask]
    y = y[mask]
    
    return X_encoded, y, excluded_columns


def train_sklearn_model(X_train, X_test, y_train, y_test, algorithm, hyperparameters):
    """Entrena modelo de Scikit-learn"""
    
    start_time = time.time()
    
    # Detectar tipo de tarea
    is_classification = y_train.dtype == 'object' or y_train.nunique() < 20
    
    # Seleccionar modelo
    if algorithm == "random_forest":
        if is_classification:
            model = RandomForestClassifier(**(hyperparameters or {}))
        else:
            model = RandomForestRegressor(**(hyperparameters or {}))
    
    elif algorithm == "linear_regression":
        model = LinearRegression()
    
    elif algorithm == "logistic_regression":
        model = LogisticRegression(**(hyperparameters or {}))
    
    elif algorithm == "svm":
        if is_classification:
            model = SVC(**(hyperparameters or {}))
        else:
            model = SVR(**(hyperparameters or {}))
    
    else:
        raise HTTPException(400, f"Algoritmo no soportado: {algorithm}")
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predecir
    y_pred = model.predict(X_test)
    
    training_time = int(time.time() - start_time)
    
    # Calcular métricas
    if is_classification:
        metrics = {
            "task_type": "classification",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Feature importance para Random Forest
        if algorithm == "random_forest" and hasattr(model, 'feature_importances_'):
            feature_names = X_train.columns.tolist()
            importances = model.feature_importances_
            metrics["feature_importance"] = dict(zip(feature_names, importances.tolist()))
    
    else:
        metrics = {
            "task_type": "regression",
            "mse": float(mean_squared_error(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2_score": float(r2_score(y_test, y_pred))
        }
    
    return model, metrics, training_time


def train_pytorch_model(X_train, X_test, y_train, y_test, algorithm, hyperparameters):
    """Entrena modelo de PyTorch"""
    
    start_time = time.time()
    
    # Detectar tipo de tarea
    is_classification = y_train.dtype == 'object' or y_train.nunique() < 20
    
    # Preparar datos para PyTorch
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    input_size = X_train.shape[1]
    
    if is_classification:
        # Encodear labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        y_train_tensor = torch.LongTensor(y_train_encoded)
        y_test_tensor = torch.LongTensor(y_test_encoded)
        
        num_classes = len(le.classes_)
        criterion = nn.CrossEntropyLoss()
    else:
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
        criterion = nn.MSELoss()
    
    # Crear DataLoaders
    batch_size = hyperparameters.get("batch_size", 32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    if algorithm == "neural_network":
        hidden_layers = hyperparameters.get("hidden_layers", [128, 64, 32])
        dropout = hyperparameters.get("dropout", 0.2)
        
        if is_classification:
            model = MLPClassifier(input_size, hidden_layers, num_classes, dropout)
        else:
            model = MLPRegressor(input_size, hidden_layers, dropout)
    
    elif algorithm == "cnn":
        conv_layers = hyperparameters.get("conv_layers", [64, 128])
        fc_layers = hyperparameters.get("fc_layers", [256, 128])
        dropout = hyperparameters.get("dropout", 0.3)
        
        model = CNNClassifier(input_size, conv_layers, fc_layers, num_classes if is_classification else 1, dropout)
    
    elif algorithm == "lstm":
        hidden_size = hyperparameters.get("hidden_size", 128)
        num_layers = hyperparameters.get("num_layers", 2)
        dropout = hyperparameters.get("dropout", 0.2)
        
        # Reshape para LSTM (batch, seq_len, features)
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_test_tensor = X_test_tensor.unsqueeze(1)
        
        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes if is_classification else 1, dropout)
    
    else:
        raise HTTPException(400, f"Algoritmo PyTorch no soportado: {algorithm}")
    
    # Optimizer
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    epochs = hyperparameters.get("epochs", 100)
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [] if is_classification else [],
        "val_acc": [] if is_classification else []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            if is_classification:
                loss = criterion(outputs, y_batch)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == y_batch).sum().item()
            else:
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_total += y_batch.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        training_history["train_loss"].append(avg_train_loss)
        
        if is_classification:
            train_acc = train_correct / train_total
            training_history["train_acc"].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                
                if is_classification:
                    loss = criterion(outputs, y_batch)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == y_batch).sum().item()
                else:
                    loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                val_total += y_batch.size(0)
        
        avg_val_loss = val_loss / len(test_loader)
        training_history["val_loss"].append(avg_val_loss)
        
        if is_classification:
            val_acc = val_correct / val_total
            training_history["val_acc"].append(val_acc)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            if is_classification:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Calcular métricas finales
    model.eval()
    with torch.no_grad():
        if algorithm == "lstm":
            y_pred = model(X_test_tensor.unsqueeze(1))
        else:
            y_pred = model(X_test_tensor)
        
        if is_classification:
            _, y_pred_classes = torch.max(y_pred, 1)
            y_pred_np = y_pred_classes.numpy()
            y_test_np = y_test_encoded
            
            metrics = {
                "task_type": "classification",
                "accuracy": float(accuracy_score(y_test_np, y_pred_np)),
                "precision": float(precision_score(y_test_np, y_pred_np, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test_np, y_pred_np, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test_np, y_pred_np, average='weighted', zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test_np, y_pred_np).tolist(),
                "training_history": training_history,
                "best_epoch": best_epoch,
                "total_epochs": epochs
            }
        else:
            y_pred_np = y_pred.numpy()
            y_test_np = y_test_tensor.numpy()
            
            metrics = {
                "task_type": "regression",
                "mse": float(mean_squared_error(y_test_np, y_pred_np)),
                "mae": float(mean_absolute_error(y_test_np, y_pred_np)),
                "r2_score": float(r2_score(y_test_np, y_pred_np)),
                "training_history": training_history,
                "best_epoch": best_epoch,
                "total_epochs": epochs
            }
    
    training_time = int(time.time() - start_time)
    
    return model, metrics, training_time, scaler


# ============================================
# ENDPOINT PRINCIPAL
# ============================================

@router.post("/train", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest):
    """
    Entrena un modelo de ML (Scikit-learn o PyTorch)
    """
    print("=" * 50)
    print("🎯 TRAIN ENDPOINT CALLED")
    print(f"📦 User ID: {request.user_id}")
    print(f"📦 Dataset ID: {request.dataset_id}")
    print(f"📦 Model Name: {request.name}")
    print(f"📦 Algorithm: {request.algorithm}")
    print(f"📦 Target: {request.target_variable}")
    print("=" * 50)
    
    try:
        # 1. Cargar dataset limpio
        df, dataset_metadata = load_cleaned_dataset(request.user_id, request.dataset_id)
        print(f"✅ Dataset cargado: {len(df)} filas")
        
        # 2. Preparar datos
        columns_with_nulls = dataset_metadata.get("columns_with_nulls", {})
        X, y, excluded_columns = prepare_data_for_training(df, request.target_variable, columns_with_nulls)
        print(f"✅ Datos preparados: X shape {X.shape}")
        
        # 3. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=request.test_size,
            random_state=request.random_state
        )
        
        # 4. Entrenar según algoritmo
        sklearn_algos = ["random_forest", "linear_regression", "logistic_regression", "svm"]
        pytorch_algos = ["neural_network", "cnn", "lstm"]
        
        if request.algorithm in sklearn_algos:
            print(f"🔧 Entrenando con Scikit-learn...")
            model, metrics, training_time = train_sklearn_model(
                X_train, X_test, y_train, y_test,
                request.algorithm,
                request.hyperparameters
            )
            
            # Serializar modelo
            model_buffer = BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)
            model_extension = "joblib"
        
        elif request.algorithm in pytorch_algos:
            print(f"🔥 Entrenando con PyTorch...")
            model, metrics, training_time, scaler = train_pytorch_model(
                X_train, X_test, y_train, y_test,
                request.algorithm,
                request.hyperparameters or {}
            )
            
            # Serializar modelo PyTorch
            model_buffer = BytesIO()
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture': {
                    'algorithm': request.algorithm,
                    'input_size': X.shape[1],
                    'hyperparameters': request.hyperparameters
                },
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist()
            }, model_buffer)
            model_buffer.seek(0)
            model_extension = "pth"
        
        else:
            raise HTTPException(400, f"Algoritmo no soportado: {request.algorithm}")
        
        print(f"✅ Modelo entrenado en {training_time}s")
        print(f"📊 Métricas: {metrics}")
        
        # 5. Generar ID y subir a Storage
        model_uuid = str(uuid.uuid4())
        model_path = f"{request.user_id}/{model_uuid}/model.{model_extension}"
        
        print(f"💾 Subiendo modelo a Storage...")
        supabase_client.storage.from_("models").upload(
            path=model_path,
            file=model_buffer.getvalue(),
            file_options={"content-type": "application/octet-stream"}
        )
        
        # 6. Guardar métricas como JSON
        import json
        metrics_buffer = json.dumps(metrics, indent=2).encode()
        metrics_path = f"{request.user_id}/{model_uuid}/metrics.json"
        
        supabase_client.storage.from_("models").upload(
            path=metrics_path,
            file=metrics_buffer,
            file_options={"content-type": "application/json"}
        )
        
        # 7. Registrar en BD
        print(f"💾 Registrando en BD...")
        insert_response = supabase_client.table("models").insert({
            "user_id": request.user_id,
            "dataset_id": request.dataset_id,
            "name": request.name,
            "algorithm": request.algorithm,
            "hyperparameters": request.hyperparameters or {},
            "target_variable": request.target_variable,
            "feature_columns": X.columns.tolist(),
            "accuracy": metrics.get("accuracy") or metrics.get("r2_score"),
            "metrics": metrics,
            "model_path": model_path,
            "training_time": training_time,
            "status": "ready"
        }).execute()
        
        model_id = insert_response.data[0]["id"]
        
        print(f"✅ Modelo guardado con ID: {model_id}")
        print("=" * 50)
        
        return TrainModelResponse(
            message="Modelo entrenado exitosamente",
            model_id=model_id,
            name=request.name,
            algorithm=request.algorithm,
            model_path=model_path,
            metrics=metrics,
            training_time=training_time,
            trained_at=datetime.utcnow().isoformat()
        )
    
    except HTTPException as he:
        raise
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(500, f"Error al entrenar modelo: {str(e)}")


@router.get("/models/{user_id}")
async def get_user_models(user_id: str):
    """
    Obtiene todos los modelos entrenados de un usuario
    """
    try:
        # Validar UUID
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(400, "user_id debe ser un UUID válido")
        
        # Obtener modelos
        response = supabase_client.table("models")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("is_active", True)\
            .order("trained_at", desc=True)\
            .execute()
        
        return {
            "models": response.data,
            "total": len(response.data),
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error al obtener modelos: {str(e)}")