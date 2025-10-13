"""rag_engine.py - Local embedding-based similarity search using photo input."""

from typing import Any, Dict, List, cast
import numpy as np
from pymongo.collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
from app.db import get_mongo_collections
from app.utils import to_jsonable


class RAGEngine:
    """Embedding-based face similarity search engine."""

    def __init__(self, faces_collection: Collection | None = None) -> None:
        """Initialize the RAGEngine with a MongoDB faces collection."""
        if faces_collection is None:
            faces_collection, _, _ = get_mongo_collections()  # ✅ fixed unpacking
        self.faces_collection: Collection = faces_collection

    def _generate_embedding(
        self, image_bytes: bytes
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Convert uploaded photo to a deterministic embedding vector (stub)."""
        np.random.seed(len(image_bytes) % 1000)
        return np.random.rand(128).astype(np.float32)

    def _calculate_similarity(
        self,
        vec1: np.ndarray[Any, np.dtype[np.float32]],
        vec2: np.ndarray[Any, np.dtype[np.float32]],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])

    def search_by_photo(
        self, image_bytes: bytes, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Search for similar faces in MongoDB based on photo embedding similarity.

        Args:
            image_bytes: Uploaded image file content (bytes).
            threshold: Minimum similarity value to include in results.

        Returns:
            A list of MongoDB documents (faces) that match the given photo,
            each enriched with a "similarity" field.
        """
        if not image_bytes:
            return []

        query_vec = self._generate_embedding(image_bytes)
        cursor = self.faces_collection.find(
            {"search_embeddings.face_embedding": {"$exists": True}}
        )

        results: List[Dict[str, Any]] = []

        for doc in cursor:
            emb_data = doc.get("search_embeddings", {}).get("face_embedding")
            if not emb_data:
                continue

            try:
                face_vec = np.frombuffer(emb_data, dtype=np.float32)
                similarity = self._calculate_similarity(query_vec, face_vec)
            except Exception:
                continue

            if similarity >= threshold:
                enriched_doc = dict(doc)
                enriched_doc["similarity"] = similarity
                results.append(enriched_doc)

        # ✅ Explicit cast ensures correct static typing for mypy
        return cast(List[Dict[str, Any]], to_jsonable(results))
