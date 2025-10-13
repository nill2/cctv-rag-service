"""Database connection helpers for MongoDB."""

from typing import Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
from app.config import (
    MONGO_URI,
    MONGO_DB,
    FACES_COLLECTION,
    KNOWN_FACES_COLLECTION,
    MONGO_COLLECTION,  # ✅ The photos collection name
)


def get_mongo_collections() -> Tuple[Collection, Collection, Collection]:
    """Initialize MongoDB connection and return face, known face, and photos collections."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    faces = db[FACES_COLLECTION]
    known_faces = db[KNOWN_FACES_COLLECTION]
    photos = db[MONGO_COLLECTION]
    return faces, known_faces, photos
