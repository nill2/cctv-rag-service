"""Main module for the Face Insight RAG API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from mcp.server.fastmcp import FastMCP
from app.rag_engine import RAGEngine


# ==============================
# Lifespan & Application setup
# ==============================

rag_engine = RAGEngine()
mcp: FastMCP | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # pylint: disable=unused-argument
    """Initialize and shutdown the MCP server."""
    global mcp  # pylint: disable=global-statement
    mcp = FastMCP(name="face-insight-rag")

    # Register MCP tools
    @mcp.tool()
    async def find_known_person(name: str) -> dict[str, Any]:
        """Find photos with a known person by name."""
        results = rag_engine.find_known_person(name)
        return {"count": len(results), "matches": results}

    @mcp.tool()
    async def find_unknown_faces() -> dict[str, Any]:
        """Find photos containing unknown or stranger faces."""
        results = rag_engine.find_unknown_faces()
        return {"count": len(results), "unknown_faces": results}

    # Run MCP server in the background
    task = asyncio.create_task(mcp.run_sse_async())

    yield  # App runs here

    # Graceful shutdown
    task.cancel()


app = FastAPI(title="Face Insight RAG API", lifespan=lifespan)

# ==============================
# Routes
# ==============================

NAME_QUERY = Query(..., description="Name of the person to search for")


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.get("/search/known")
async def search_known(name: str = NAME_QUERY) -> dict[str, Any]:
    """Search for known faces by name."""
    results = rag_engine.find_known_person(name)
    return {"count": len(results), "matches": results}


@app.get("/search/unknown")
async def search_unknown() -> dict[str, Any]:
    """Search for unknown (unidentified) faces."""
    results = rag_engine.find_unknown_faces()
    return {"count": len(results), "unknown_faces": results}


# ==============================
# Entrypoint
# ==============================


def run() -> None:
    """Keep the container alive while MCP runs its internal server."""
    import time

    print("✅ Face Insight RAG API with MCP is running on port 8000...")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")


if __name__ == "__main__":
    run()
