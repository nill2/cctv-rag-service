"""Database connection helpers for MongoDB."""

from typing import Tuple
from pymongo import MongoClient
from pymongo.collection import Collection
from app.config import MONGO_URI, MONGO_DB, FACES_COLLECTION, KNOWN_FACES_COLLECTION


def get_mongo_collections() -> Tuple[Collection, Collection]:
    """Initialize MongoDB connection and return face and known face collections."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    faces = db[FACES_COLLECTION]
    known_faces = db[KNOWN_FACES_COLLECTION]
    return faces, known_faces
