"""Photo data model for the RAG system."""

from typing import List
from pydantic import BaseModel


class Photo(BaseModel):
    """Schema for photo metadata and embeddings."""

    id: str
    path: str
    embedding: List[float]
    timestamp: str
    known_person: str | None = None
