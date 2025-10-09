"""
face_search.py
---------------
Local embedding-based face search engine.
Supports queries for known and unknown faces, fully offline.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from pymongo.collection import Collection
from sklearn.metrics.pairwise import cosine_similarity

# Type alias for clarity and MyPy compatibility
NDArrayFloat = np.ndarray[Any, np.dtype[np.float32]]


class FaceSearcher:
    """Performs vector similarity search on face embeddings stored in MongoDB."""

    def __init__(
        self, face_collection: Collection, known_collection: Collection
    ) -> None:
        """Initialize the FaceSearcher."""
        self.face_collection: Collection = face_collection
        self.known_collection: Collection = known_collection

    def _get_known_faces(self) -> List[Dict[str, Any]]:
        """Fetch all known faces (reference embeddings)."""
        return list(self.known_collection.find({}))

    def _get_photo_faces(self) -> List[Dict[str, Any]]:
        """Fetch all processed photos containing faces."""
        return list(self.face_collection.find({"has_faces": True}))

    def _calculate_similarity(self, vec1: NDArrayFloat, vec2: NDArrayFloat) -> float:
        """Compute cosine similarity between two embeddings."""
        vec1 = np.array(vec1, dtype=np.float32).reshape(1, -1)
        vec2 = np.array(vec2, dtype=np.float32).reshape(1, -1)
        return float(cosine_similarity(vec1, vec2)[0][0])

    def find_photos_with_person(
        self, name: str, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find all photos where a known person appears.

        Args:
            name: Person's name as stored in known_faces.
            threshold: Minimum cosine similarity to count as a match.

        Returns:
            List of photo metadata where the person appears.
        """
        known_faces = self._get_known_faces()
        person = next((p for p in known_faces if p.get("name") == name), None)
        if not person:
            return []

        person_vec: NDArrayFloat = np.array(person["embedding"], dtype=np.float32)
        all_photos = self._get_photo_faces()
        matches: List[Dict[str, Any]] = []

        for photo in all_photos:
            emb_data = photo.get("search_embeddings", {}).get("face_embedding")
            if not emb_data:
                continue

            photo_vec: NDArrayFloat = np.frombuffer(emb_data, dtype=np.float32)
            similarity = self._calculate_similarity(person_vec, photo_vec)

            if similarity >= threshold:
                matches.append(
                    {
                        "filename": photo.get("filename"),
                        "similarity": similarity,
                        "date": photo.get("date"),
                        "camera_location": photo.get("camera_location"),
                    }
                )

        return matches

    def find_unknown_faces(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """
        Find all photos with unknown people (not matching any known face).

        Args:
            threshold: Maximum similarity to still count as 'unknown'.

        Returns:
            List of photo metadata with unknown faces.
        """
        known_faces = self._get_known_faces()
        all_photos = self._get_photo_faces()
        unknown_photos: List[Dict[str, Any]] = []

        for photo in all_photos:
            emb_data = photo.get("search_embeddings", {}).get("face_embedding")
            if not emb_data:
                continue

            photo_vec: NDArrayFloat = np.frombuffer(emb_data, dtype=np.float32)

            max_sim = 0.0
            for known in known_faces:
                known_vec: NDArrayFloat = np.array(known["embedding"], dtype=np.float32)
                sim = self._calculate_similarity(known_vec, photo_vec)
                max_sim = max(max_sim, sim)

            if max_sim < threshold:
                unknown_photos.append(
                    {
                        "filename": photo.get("filename"),
                        "max_similarity": max_sim,
                        "date": photo.get("date"),
                        "camera_location": photo.get("camera_location"),
                    }
                )

        return unknown_photos
