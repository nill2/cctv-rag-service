"""Unit tests for the /search endpoint in the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def example_query():
    """Provide an example search query payload."""
    return {"query": "detect person in frame"}


def test_search_endpoint_status(example_query):
    """Verify that the /search endpoint returns HTTP 200."""
    response = client.post("/search", json=example_query)
    assert response.status_code == 200


def test_search_response_format(example_query):
    """Check that the /search endpoint returns JSON with the expected structure."""
    response = client.post("/search", json=example_query)
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()

    # Ensure 'results' key is present and properly formatted
    assert "results" in data
    assert isinstance(data["results"], list)

    # Validate result structure if data is present
    if data["results"]:
        first = data["results"][0]
        assert isinstance(first, dict)
        assert "text" in first
        assert "score" in first


def test_search_empty_query():
    """Ensure that an empty query returns HTTP 400 or a validation error."""
    response = client.post("/search", json={"query": ""})
    assert response.status_code in (400, 422)
