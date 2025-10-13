"""Main FastAPI application for local face recognition and search."""

import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.face_search import FaceSearcher
from app.rag_engine import RAGEngine
from app.db import get_mongo_collections
from app.utils import to_jsonable

# ✅ Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CCTV Face Recognition API", version="1.0.0")

# ✅ Initialize MongoDB collections once
faces_collection, known_collection, photos_collection = get_mongo_collections()
logger.info(
    f"MongoDB collections initialized: faces={faces_collection.name}, "
    f"known={known_collection.name}, photos={photos_collection.name}"
)

searcher = FaceSearcher(faces_collection, known_collection, photos_collection)
rag_engine = RAGEngine(faces_collection)

# ✅ Move this outside the function to avoid flake8 B008 warning
UPLOAD_FILE_DEP = File(...)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "ok"}


@app.post("/faces/known")
def find_known_faces(names: List[str]) -> JSONResponse:
    """Find all photos containing one or more known persons."""
    logger.info(f"Received known faces search for names: {names}")

    if not names:
        logger.warning("Empty name list provided to /faces/known")
        raise HTTPException(status_code=400, detail="Name list cannot be empty")

    found_results: List[dict] = []
    skipped_names: List[str] = []
    seen_ids = set()

    for name in names:
        person_results = searcher.find_known_faces_by_name(name)
        logger.info(f"Search for '{name}' returned {len(person_results)} results")

        if person_results:
            for doc in person_results:
                doc_id = str(doc.get("_id"))
                if doc_id not in seen_ids:
                    found_results.append(doc)
                    seen_ids.add(doc_id)
        else:
            skipped_names.append(name)

    logger.info(f"Total unique results: {len(found_results)}, skipped: {skipped_names}")
    return JSONResponse(
        {"results": to_jsonable(found_results), "skipped_names": skipped_names}
    )


@app.get("/faces/unknown")
def find_unknown_faces() -> JSONResponse:
    """Find photos where some faces remain unidentified."""
    logger.info("Searching for unknown faces")
    results = searcher.find_unknown_faces()
    logger.info(f"Found {len(results)} unknown face entries")
    return JSONResponse({"results": to_jsonable(results)})


@app.post("/faces/search")
async def find_person_from_photo(
    file: UploadFile = UPLOAD_FILE_DEP,  # ✅ avoids B008 warning
    threshold: float = 0.8,
) -> JSONResponse:
    """Upload a photo and search for similar faces among known persons."""
    try:
        logger.info(f"Photo upload received: {file.filename}, threshold={threshold}")
        contents = await file.read()
        results = rag_engine.search_by_photo(contents, threshold)
        logger.info(f"Search returned {len(results)} results")
        return JSONResponse({"results": results, "threshold": threshold})
    except Exception as e:
        logger.exception("Error while processing photo search")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/saved_known_faces")
def get_saved_known_faces() -> JSONResponse:
    """Return all known faces from the known_faces_collection."""
    logger.info("Fetching all saved known faces")
    results = searcher.get_all_known_faces()
    return JSONResponse({"results": to_jsonable(results)})


@app.get("/photos_detected_faces")
def photos_detected_faces() -> JSONResponse:
    """Return all known faces from the known_faces_collection."""
    logger.info("Fetching all photos with detected faces")
    results = searcher.photos_detected_faces()
    return JSONResponse({"results": to_jsonable(results)})


@app.get("/current_cctv")
def get_current_cctv() -> JSONResponse:
    """Return the most recent CCTV entry."""
    logger.info("Fetching latest CCTV entry from photos collection")
    latest = searcher.get_latest_cctv_entry()
    if not latest:
        raise HTTPException(status_code=404, detail="No CCTV entries found")
    return JSONResponse({"result": to_jsonable(latest)})
