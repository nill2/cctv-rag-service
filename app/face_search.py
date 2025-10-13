"""face_search.py - MongoDB-based search for known, unknown, and CCTV faces."""

from typing import Any, Dict, List, Optional
from pymongo.collection import Collection
import logging
from typing import cast

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FaceSearcher:
    """Utility class to find known, unknown, and latest CCTV faces in MongoDB."""

    def __init__(
        self,
        faces_collection: Collection,
        known_faces_collection: Collection,
        photos_collection: Collection,
    ) -> None:
        """Initialize FaceSearcher with MongoDB collections."""
        self.faces_collection = faces_collection
        self.known_faces_collection = known_faces_collection
        self.photos_collection = photos_collection
        logger.info(
            f"FaceSearcher initialized with collections: faces={faces_collection.name}, "
            f"known_faces={known_faces_collection.name}, photos={photos_collection.name}"
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
        """Find documents where some faces are still unknown."""
        logger.info(f"Searching for unknown faces in {self.faces_collection.name}")
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
        logger.info(f"Searching for documents containing {names}")
        query = {"matched_persons": {"$in": names}}
        cursor = self.faces_collection.find(query)
        results = list(cursor)
        logger.info(f"Found {len(results)} documents containing any of {names}")
        return results

    # ✅ Return all known faces
    def get_all_known_faces(self) -> List[Dict[str, Any]]:
        """Return all documents from the known_faces_collection."""
        logger.info("Fetching all known faces")
        cursor = self.known_faces_collection.find()
        results = list(cursor)
        logger.info(f"Found {len(results)} known face entries")
        return results

    # ✅ Return all photos with detected faces there
    def photos_detected_faces(self) -> List[Dict[str, Any]]:
        """Return all documents from the nill-home-faces."""
        logger.info("Fetching all known faces")
        cursor = self.faces_collection.find()
        results = list(cursor)
        logger.info(f"Found {len(results)} known face entries")
        return results

    # ✅ Return the latest CCTV entry from the photos collection
    def get_latest_cctv_entry(self) -> Optional[Dict[str, Any]]:
        """Return the most recent CCTV entry from the photos_collection."""
        logger.info("Fetching latest CCTV entry from photos collection based on 'date'")
        latest_doc = cast(
            Optional[Dict[str, Any]],
            self.photos_collection.find_one(sort=[("date", -1)]),
        )
        if latest_doc:
            logger.info(f"Latest CCTV entry found with date={latest_doc.get('date')}")
        else:
            logger.warning("No CCTV entries found in photos collection")
        return latest_doc
