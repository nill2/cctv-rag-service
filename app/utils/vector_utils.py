"""Utility functions for vector math operations."""

from __future__ import annotations
from typing import Any
import numpy as np

# Typed alias for float32 arrays
NDArrayFloat = np.ndarray[Any, np.dtype[np.float32]]


def cosine_similarity(vec1: NDArrayFloat, vec2: NDArrayFloat) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def normalize_vector(vector: NDArrayFloat) -> NDArrayFloat:
    """Normalize a numpy vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
