"""Utility helpers for JSON-safe MongoDB data."""

from typing import Any
from bson import ObjectId
import base64
from datetime import datetime


def to_jsonable(data: Any) -> Any:
    """Recursively convert MongoDB documents to JSON-serializable objects."""
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]
    if isinstance(data, dict):
        return {key: to_jsonable(value) for key, value in data.items()}
    if isinstance(data, ObjectId):
        return str(data)
    if isinstance(data, bytes):
        # Convert raw bytes (e.g., embeddings or images) to base64 strings
        return base64.b64encode(data).decode("utf-8")
    if isinstance(data, datetime):
        # Convert datetime to ISO string
        return data.isoformat()
    return data
