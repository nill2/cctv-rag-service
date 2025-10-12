"""Configuration module for environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_HOST")
MONGO_DB = os.getenv("MONGO_DB", "nill-home")
FACES_COLLECTION = os.getenv("FACE_COLLECTION", "nill-home-faces")
KNOWN_FACES_COLLECTION = os.getenv("KNOWN_FACES_COLLECTION", "nill-known-faces")
