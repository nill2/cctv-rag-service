"""Lightweight local Face RAG Engine (no external APIs)."""

from __future__ import annotations

import os
import numpy as np
from typing import Any
from pymongo import MongoClient

# Type alias for clarity
NDArrayFloat = np.ndarray[Any, np.dtype[np.float32]]


class RAGEngine:
    """Face Retrieval and Analysis Engine (RAG-like)."""

    def __init__(self) -> None:
        """Initialize an empty in-memory vector store."""
        self.texts: list[str] = []
        self.vectors: NDArrayFloat | None = None
        self.metadata: list[dict[str, Any]] = []

    # ==============================
    # Data Loading
    # ==============================

    def load_vectors(self, face_records: list[dict[str, Any]]) -> None:
        """Load known face embeddings into memory.

        Args:
            face_records (list[dict[str, Any]]): List of face records, each containing:
                - name: Person name or "unknown"
                - embedding: Face embedding vector (list[float])
                - metadata: Optional photo info (location, time, etc.)
        """
        if not face_records:
            raise ValueError("No face records provided to load.")

        self.texts = [
            f"Photo of {r.get('name', 'unknown')} at {r.get('camera_location', 'N/A')} "
            f"time {r.get('timestamp', 'N/A')}"
            for r in face_records
        ]

        embeddings: list[NDArrayFloat] = [
            np.array(r["embedding"], dtype=np.float32)
            for r in face_records
            if "embedding" in r
        ]

        if not embeddings:
            raise ValueError("No valid embeddings found in provided records.")

        self.vectors = np.vstack(embeddings)
        self.metadata = face_records

    def load_from_mongo(self) -> None:
        """Load face embeddings from MongoDB into memory."""
        mongo_host = os.getenv("MONGO_HOST")
        mongo_port = int(os.getenv("MONGO_PORT", "27017"))
        mongo_db = os.getenv("MONGO_DB")
        face_collection = os.getenv("FACE_COLLECTION")

        if not all([mongo_host, mongo_db, face_collection]):
            raise ValueError("Missing MongoDB configuration environment variables.")

        print(f"🔗 Connecting to MongoDB: {mongo_host}")
        client = MongoClient(mongo_host, mongo_port)
        db = client[mongo_db]
        collection = db[face_collection]

        face_records = list(collection.find({}))
        if not face_records:
            raise ValueError("No records found in MongoDB face collection.")

        self.load_vectors(face_records)
        print(f"✅ Loaded {len(face_records)} face embeddings from MongoDB.")

    # ==============================
    # Core Search Functions
    # ==============================

    def _cosine_similarity(self, query_vec: NDArrayFloat) -> NDArrayFloat:
        """Compute cosine similarity between query vector and stored vectors."""
        if self.vectors is None:
            raise ValueError("Vector store is empty. Load vectors first.")

        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        similarity: NDArrayFloat = np.dot(self.vectors, query_vec) / norms
        return similarity

    def query(self, query_vec: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top matching faces for a given embedding vector."""
        if self.vectors is None:
            raise ValueError("Vector store not initialized. Call load_vectors() first.")

        query_array: NDArrayFloat = np.array(query_vec, dtype=np.float32)
        similarity = self._cosine_similarity(query_array)

        top_indices = np.argsort(similarity)[::-1][:top_k]
        results: list[dict[str, Any]] = [
            {
                "text": self.texts[idx],
                "score": float(similarity[idx]),
                "metadata": self.metadata[idx],
            }
            for idx in top_indices
        ]
        return results

    def find_known_person(self, name: str) -> list[dict[str, Any]]:
        """Find all photos containing a known person."""
        if not self.metadata:
            raise ValueError("No data loaded. Load vectors first.")

        return [r for r in self.metadata if r.get("name") == name]

    def find_unknown_faces(self) -> list[dict[str, Any]]:
        """Find all photos containing unidentified faces."""
        if not self.metadata:
            raise ValueError("No data loaded. Load vectors first.")

        return [r for r in self.metadata if r.get("name") == "unknown"]
