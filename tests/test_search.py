"""Unit tests for FaceSearcher functionality in face_search.py."""

from unittest.mock import MagicMock
import numpy as np
import pytest
from app.face_search import FaceSearcher


@pytest.fixture
def mock_collections():
    """Create mock MongoDB collections with fake data."""
    face_collection = MagicMock()
    known_collection = MagicMock()

    # Mock known faces
    known_collection.find.return_value = [
        {"name": "Alice", "embedding": np.ones(128, dtype=np.float32).tolist()},
        {"name": "Bob", "embedding": np.zeros(128, dtype=np.float32).tolist()},
    ]

    # Mock photo faces
    photo_embeddings = [
        np.ones(128, dtype=np.float32).tobytes(),
        (np.ones(128, dtype=np.float32) * 0.7).tobytes(),
        np.zeros(128, dtype=np.float32).tobytes(),
    ]

    face_collection.find.return_value = [
        {
            "filename": f"photo_{i}.jpg",
            "search_embeddings": {"face_embedding": emb},
            "has_faces": True,
            "date": f"2025-01-{i + 1:02d}",
            "camera_location": "TestCam",
        }
        for i, emb in enumerate(photo_embeddings)
    ]

    return face_collection, known_collection


def test_find_photos_with_person(mock_collections):
    """Ensure known person search returns matching photos."""
    face_collection, known_collection = mock_collections
    searcher = FaceSearcher(face_collection, known_collection)

    results = searcher.find_photos_with_person("Alice", threshold=0.8)

    assert isinstance(results, list)
    assert results, "Expected at least one match for Alice"
    assert all("filename" in r and "similarity" in r for r in results)
    assert all(r["similarity"] >= 0.8 for r in results)


def test_find_photos_with_unknown_person(mock_collections):
    """Ensure searching for a non-existing person returns empty list."""
    face_collection, known_collection = mock_collections
    searcher = FaceSearcher(face_collection, known_collection)

    results = searcher.find_photos_with_person("Charlie")
    assert not results


def test_find_unknown_faces(mock_collections):
    """Ensure unknown faces are correctly detected below threshold."""
    face_collection, known_collection = mock_collections
    searcher = FaceSearcher(face_collection, known_collection)

    results = searcher.find_unknown_faces(threshold=0.75)

    assert isinstance(results, list)
    assert all("filename" in r for r in results)
    assert all(r["max_similarity"] < 0.75 for r in results)
