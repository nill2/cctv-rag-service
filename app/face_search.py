"""face_search.py - MongoDB-based search for known and unknown faces."""

from typing import Any, Dict, List
from pymongo.collection import Collection
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FaceSearcher:
    """Utility class to find known and unknown faces in MongoDB."""

    def __init__(
        self, faces_collection: Collection, known_faces_collection: Collection
    ) -> None:
        """Initialize FaceSearcher with MongoDB collections."""
        self.faces_collection = faces_collection
        self.known_faces_collection = known_faces_collection
        logger.info(
            f"FaceSearcher initialized with collections: faces={faces_collection.name}, known_faces={known_faces_collection.name}"
        )

    def find_known_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Find all documents where a person with this name appears."""
        logger.info(
            f"Searching for known faces by name='{name}' in collection: {self.faces_collection.name}"
        )
        query = {"matched_persons": {"$in": [name]}}
        cursor = self.faces_collection.find(query)
        results = list(cursor)
        logger.info(f"Found {len(results)} documents for name '{name}'")
        return results

    def find_unknown_faces(self) -> List[Dict[str, Any]]:
        """
        Find documents where some faces are still unknown.

        A document is considered 'unknown' if face_count > number of matched_persons.
        """
        logger.info(
            f"Searching for unknown faces in collection: {self.faces_collection.name}"
        )

        cursor = self.faces_collection.find({"has_faces": True})
        results = list(cursor)

        unknowns = []
        for doc in results:
            face_count = doc.get("face_count", 0)
            matched = doc.get("matched_persons", [])
            matched_count = len(matched) if isinstance(matched, list) else 0

            if face_count > matched_count:
                unknowns.append(doc)

        logger.info(
            f"Found {len(unknowns)} documents where face_count > matched_persons"
        )
        return unknowns

    def find_known_persons(self, names: List[str]) -> List[Dict[str, Any]]:
        """Find documents containing any known person from the given list."""
        logger.info(
            f"Searching for documents containing known persons: {names} in collection: {self.faces_collection.name}"
        )
        query = {"matched_persons": {"$in": names}}
        cursor = self.faces_collection.find(query)
        results = list(cursor)
        logger.info(f"Found {len(results)} documents containing any of {names}")
        return results
