"""Lightweight local Face RAG Engine (no external APIs)."""

from __future__ import annotations

import numpy as np
from typing import Any

# Type alias for clarity
NDArrayFloat = np.ndarray[Any, np.dtype[np.float32]]


class RAGEngine:
    """Face Retrieval and Analysis Engine (RAG-like)."""

    def __init__(self) -> None:
        """Initialize an empty in-memory vector store."""
        self.texts: list[str] = []
        self.vectors: NDArrayFloat | None = None
        self.metadata: list[dict[str, Any]] = []

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
            np.array(r["embedding"], dtype=np.float32) for r in face_records
        ]
        self.vectors = np.vstack(embeddings)
        self.metadata = face_records

    def _cosine_similarity(self, query_vec: NDArrayFloat) -> NDArrayFloat:
        """Compute cosine similarity between query vector and stored vectors."""
        if self.vectors is None:
            raise ValueError("Vector store is empty. Load vectors first.")

        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        similarity: NDArrayFloat = np.dot(self.vectors, query_vec) / norms
        return similarity

    def query(self, query_vec: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top matching faces for a given embedding vector.

        Args:
            query_vec (list[float]): The embedding vector to search for.
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            list[dict[str, Any]]: Ranked search results with similarity scores.
        """
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
