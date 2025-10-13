"""Unit tests for FaceSearcher and RAGEngine."""

import pytest
from unittest.mock import MagicMock
import numpy as np
from app.face_search import FaceSearcher
from app.rag_engine import RAGEngine


@pytest.fixture
def mock_collections():
    """Create mock MongoDB collections for testing."""
    face_collection = MagicMock()
    known_collection = MagicMock()
    photos_collection = MagicMock()

    face_collection.find.return_value = [
        {"filename": "photo1.jpg", "matched_persons": ["Alice"], "face_count": 1},
        {"filename": "photo2.jpg", "matched_persons": [], "face_count": 2},
    ]
    known_collection.find.return_value = [
        {"name": "Alice", "embedding": np.ones(128).tolist()},
        {"name": "Bob", "embedding": np.zeros(128).tolist()},
    ]
    photos_collection.find_one.return_value = {
        "filename": "latest_cctv.jpg",
        "date": "2025-01-01T12:00:00Z",
    }

    return face_collection, known_collection, photos_collection


def test_find_known_faces_by_name(mock_collections):
    """Should return list of photos where the given name appears."""
    face_collection, known_collection, photos_collection = mock_collections
    searcher = FaceSearcher(face_collection, known_collection, photos_collection)
    results = searcher.find_known_faces_by_name("Alice")

    assert isinstance(results, list)
    face_collection.find.assert_called_once_with(
        {"matched_persons": {"$in": ["Alice"]}}
    )
    assert results[0]["filename"] == "photo1.jpg"


def test_find_unknown_faces(mock_collections):
    """Should return only documents with no matched persons."""
    face_collection, known_collection, photos_collection = mock_collections
    searcher = FaceSearcher(face_collection, known_collection, photos_collection)
    results = searcher.find_unknown_faces()

    assert len(results) == 1
    assert results[0]["filename"] == "photo2.jpg"
    assert results[0]["matched_persons"] == []


def test_rag_search_by_photo(mock_collections):
    """Should return list of documents enriched with similarity scores."""
    face_collection, _, _ = mock_collections
    engine = RAGEngine(face_collection)

    results = engine.search_by_photo(b"fake_image_data", threshold=0.5)

    assert isinstance(results, list)
    if results:
        doc = results[0]
        assert "similarity" in doc
        assert isinstance(doc["similarity"], float)
