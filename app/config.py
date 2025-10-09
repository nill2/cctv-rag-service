"""config.py"""

import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "nill-home")
FACES_COLLECTION = os.getenv("MONGO_COLLECTION", "nill-faces")
KNOWN_FACES_COLLECTION = os.getenv("KNOWN_FACES_COLLECTION", "nill-known-faces")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
