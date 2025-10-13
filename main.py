"""Main FastAPI application for local face recognition and search."""

import base64
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # ✅ Added
from app.face_search import FaceSearcher
from app.rag_engine import RAGEngine
from app.db import get_mongo_collections
from app.utils import to_jsonable

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# App initialization
# ---------------------------------------------------------
app = FastAPI(title="CCTV Face Recognition API", version="1.0.0")

# ✅ Allow frontend (Render + local dev) via CORS
origins = [
    "https://nill-spa.onrender.com",  # production frontend on Render
    "http://127.0.0.1:5000",  # local Flask dev
    "http://localhost:5000",  # alternate local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Initialize MongoDB + app components
# ---------------------------------------------------------
faces_collection, known_collection, photos_collection = get_mongo_collections()
searcher = FaceSearcher(faces_collection, known_collection, photos_collection)
rag_engine = RAGEngine(faces_collection)

UPLOAD_FILE_DEP = File(...)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/faces/known")
def find_known_faces(names: List[str]) -> JSONResponse:
    """Find all photos containing one or more known persons."""
    if not names:
        raise HTTPException(status_code=400, detail="Name list cannot be empty")

    found_results = []
    seen_ids = set()
    skipped_names = []

    for name in names:
        results = searcher.find_known_faces_by_name(name)
        if results:
            for doc in results:
                doc_id = str(doc.get("_id"))
                if doc_id not in seen_ids:
                    found_results.append(doc)
                    seen_ids.add(doc_id)
        else:
            skipped_names.append(name)

    return JSONResponse(
        {"results": to_jsonable(found_results), "skipped_names": skipped_names}
    )


@app.get("/faces/unknown")
def find_unknown_faces() -> JSONResponse:
    """Find photos where some faces remain unidentified."""
    results = searcher.find_unknown_faces()
    return JSONResponse({"results": to_jsonable(results)})


@app.post("/faces/search")
async def find_person_from_photo(
    file: UploadFile = UPLOAD_FILE_DEP,
    threshold: float = 0.8,
) -> JSONResponse:
    """Upload a photo and search for similar faces."""
    try:
        contents = await file.read()
        results = rag_engine.search_by_photo(contents, threshold)
        return JSONResponse({"results": results, "threshold": threshold})
    except Exception as e:
        logger.error("Error processing photo search: %s", e)
        raise HTTPException(
            status_code=500, detail=str(e)
        ) from e  # pylint: disable=raise-missing-from


@app.get("/saved_known_faces")
def get_saved_known_faces() -> JSONResponse:
    """Return all known faces."""
    results = searcher.get_all_known_faces()
    return JSONResponse({"results": to_jsonable(results)})


@app.get("/photos_detected_faces")
def photos_detected_faces() -> JSONResponse:
    """Return all faces with detected faces."""
    results = searcher.photos_detected_faces()
    return JSONResponse({"results": to_jsonable(results)})


@app.get("/current_cctv")
def get_current_cctv() -> JSONResponse:
    """Return the most recent CCTV entry."""
    latest = searcher.get_latest_cctv_entry()
    if not latest:
        raise HTTPException(status_code=404, detail="No CCTV entries found")
    return JSONResponse({"result": to_jsonable(latest)})


# ---------------------------------------------------------
# Image fetch endpoints
# ---------------------------------------------------------


@app.get("/known_face_image/{name}")
def get_known_face_image(name: str) -> Response:
    """Return the image bytes for a known face by name."""
    doc = searcher.get_known_face_image(name)
    if not doc or "image" not in doc:
        raise HTTPException(status_code=404, detail=f"No image found for {name}")
    try:
        img_bytes = base64.b64decode(doc["image"])
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error("Error decoding known face image for %s: %s", name, e)
        raise HTTPException(
            status_code=500, detail=str(e)
        ) from e  # pylint: disable=raise-missing-from


@app.get("/face_image/{filename}")
def get_face_image(filename: str) -> Response:
    """Return the image bytes for a detected face by filename."""
    doc = searcher.get_face_image(filename)
    if not doc or "image" not in doc:
        raise HTTPException(status_code=404, detail=f"No image found for {filename}")
    try:
        img_bytes = base64.b64decode(doc["image"])
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error("Error decoding face image for %s: %s", filename, e)
        raise HTTPException(
            status_code=500, detail=str(e)
        ) from e  # pylint: disable=raise-missing-from


@app.get("/photo_image/{filename}")
def get_photo_image(filename: str) -> Response:
    """Return the image bytes for a photo by filename."""
    doc = searcher.get_photo_image(filename)
    if not doc or "image" not in doc:
        raise HTTPException(status_code=404, detail=f"No image found for {filename}")
    try:
        img_bytes = base64.b64decode(doc["image"])
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error("Error decoding photo image for %s: %s", filename, e)
        raise HTTPException(
            status_code=500, detail=str(e)
        ) from e  # pylint: disable=raise-missing-from
