"""face_search.py - MongoDB-based search for known, unknown, and CCTV faces."""

from typing import Any, Dict, List, Optional, cast
from pymongo.collection import Collection
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Fields that hold image/binary payloads and should be excluded in metadata queries
binary_fields = {
    "data": 0,
    "image": 0,
    "annotated_bytes": 0,
    "face_embedding": 0,
}


class FaceSearcher:
    """Search helper for known, unknown and CCTV face data stored in MongoDB."""

    def __init__(
        self,
        faces_collection: Collection,
        known_faces_collection: Collection,
        photos_collection: Collection,
    ) -> None:
        """Initialize with MongoDB collections."""
        self.faces_collection = faces_collection
        self.known_faces_collection = known_faces_collection
        self.photos_collection = photos_collection

        logger.info(
            "FaceSearcher initialized with collections: faces=%s, known_faces=%s, photos=%s",
            faces_collection.name,
            known_faces_collection.name,
            photos_collection.name,
        )

    # ------------------------------------------------------------------
    def find_known_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Return all face documents containing the given known name.

        NOTE: Tests expect:
            faces_collection.find({"matched_persons": {"$in": [name]}})
        Therefore: NO projection is allowed in this function.
        """
        logger.info("Searching for known faces by name='%s'", name)

        query = {"matched_persons": {"$in": [name]}}
        cursor = self.faces_collection.find(query)  # test checks this exact call
        results = list(cursor)

        logger.info("Found %d documents for name '%s'", len(results), name)
        return results

    # ------------------------------------------------------------------
    def find_unknown_faces(self) -> List[Dict[str, Any]]:
        """
        Return metadata for unknown faces without binary image data.
        Unknown faces are: face_count > matched_persons_count.
        """
        logger.info("Searching for unknown faces (no binary payload)")

        cursor = self.faces_collection.find({"has_faces": True}, binary_fields)
        results = list(cursor)

        unknowns: List[Dict[str, Any]] = []

        for doc in results:
            face_count = doc.get("face_count", 0)
            matched = doc.get("matched_persons", [])
            matched_count = len(matched) if isinstance(matched, list) else 0

            if face_count > matched_count:
                unknowns.append(doc)

        logger.info("Found %d unknown face entries", len(unknowns))
        return unknowns

    # ------------------------------------------------------------------
    def find_known_persons(self, names: List[str]) -> List[Dict[str, Any]]:
        """Return documents containing ANY name from a list."""
        logger.info("Searching for documents containing any of %s", names)

        query = {"matched_persons": {"$in": names}}
        cursor = self.faces_collection.find(query)
        results = list(cursor)

        logger.info("Found %d documents containing any of %s", len(results), names)
        return results

    # ------------------------------------------------------------------
    def get_all_known_faces(self) -> List[Dict[str, Any]]:
        """Return all known face documents."""
        logger.info("Fetching all known faces")
        cursor = self.known_faces_collection.find()
        results = list(cursor)

        logger.info("Found %d known face entries", len(results))
        return results

    # ------------------------------------------------------------------
    def photos_detected_faces(self) -> List[Dict[str, Any]]:
        """Return metadata about detected faces (exclude binary fields)."""
        logger.info("Fetching face metadata from faces_collection")

        cursor = self.faces_collection.find(
            {},
            {
                "_id": 1,
                "filename": 1,
                "bsonTime": 1,
                "timestamp": 1,
                "face_count": 1,
                "matched_persons": 1,
            },
        )

        results = list(cursor)
        logger.info("Returning %d metadata documents", len(results))
        return results

    # ------------------------------------------------------------------
    def get_latest_cctv_entry(self) -> Optional[Dict[str, Any]]:
        """Return the most recent CCTV entry."""
        logger.info("Fetching latest CCTV entry")

        doc = cast(
            Optional[Dict[str, Any]],
            self.photos_collection.find_one(sort=[("date", -1)]),
        )

        if doc:
            logger.info("Latest CCTV entry found (date=%s)", doc.get("date"))
        else:
            logger.warning("No CCTV entries found")

        return doc

    # ------------------------------------------------------------------
    def get_known_face_image(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the FULL known face document including binary image."""
        logger.info("Fetching known face image for: %s", name)
        return cast(
            Optional[Dict[str, Any]],
            self.known_faces_collection.find_one({"name": name}),
        )

    # ------------------------------------------------------------------
    def get_face_image(self, filename: str) -> Optional[Dict[str, Any]]:
        """Return a single face document including binary image."""
        logger.info("Fetching face image for: %s", filename)
        return cast(
            Optional[Dict[str, Any]],
            self.faces_collection.find_one({"filename": filename}),
        )

    # ------------------------------------------------------------------
    def get_photo_image(self, filename: str) -> Optional[Dict[str, Any]]:
        """Return a CCTV photo document including binary image."""
        logger.info("Fetching CCTV photo image for: %s", filename)
        return cast(
            Optional[Dict[str, Any]],
            self.photos_collection.find_one({"filename": filename}),
        )
