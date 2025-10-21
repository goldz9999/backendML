# app/utils/helpers.py
import hashlib
import uuid
from datetime import datetime

def validate_file_type(file_content_type: str, allowed_types: list[str]) -> bool:
    """Valida tipo MIME del archivo"""
    return file_content_type in allowed_types

def validate_file_size(file_size: int, max_size: int) -> bool:
    """Valida tamaño del archivo"""
    return file_size <= max_size

def generate_unique_filename(original_filename: str, user_id: str) -> str:
    """Genera nombre único para archivo"""
    timestamp = int(datetime.utcnow().timestamp())
    unique_id = str(uuid.uuid4())[:8]
    return f"{user_id}/{timestamp}_{unique_id}_{original_filename}"

def calculate_file_hash(file_content: bytes) -> str:
    """Calcula SHA-256 hash del archivo"""
    return hashlib.sha256(file_content).hexdigest()