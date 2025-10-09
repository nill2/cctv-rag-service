"""Main module for the Face Insight RAG API."""

from __future__ import annotations

import asyncio
from typing import Any
from fastapi import FastAPI, Query
from modelcontext import MCPServer, MCPResponse
from .rag_engine import RAGEngine

# ==============================
# Application setup
# ==============================

app: FastAPI = FastAPI(title="Face Insight RAG API")
rag_engine: RAGEngine = RAGEngine()
mcp: MCPServer | None = None

# Define Query parameters outside functions (to satisfy flake8-bugbear B008)
NAME_QUERY = Query(..., description="Name of the person to search for")

# ==============================
# FastAPI routes
# ==============================


@app.on_event("startup")
async def setup_mcp() -> None:
    """Initialize MCP server on application startup."""
    global mcp  # pylint: disable=global-statement
    mcp = MCPServer(name="face-insight-rag")


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.get("/search/known")
async def search_known(name: str = NAME_QUERY) -> dict[str, Any]:
    """Search for known faces by name.

    Args:
        name (str): The name of the person to search for.

    Returns:
        dict[str, Any]: Match count and corresponding results.
    """
    results: list[dict[str, Any]] = rag_engine.find_known_person(name)
    return {"count": len(results), "matches": results}


@app.get("/search/unknown")
async def search_unknown() -> dict[str, Any]:
    """Search for unknown (unidentified) faces."""
    results: list[dict[str, Any]] = rag_engine.find_unknown_faces()
    return {"count": len(results), "unknown_faces": results}


# ==============================
# MCP-compatible async functions
# ==============================


async def mcp_find_known_person(name: str) -> MCPResponse:
    """Find photos with a known person by name."""
    results: list[dict[str, Any]] = rag_engine.find_known_person(name)
    return MCPResponse(content={"count": len(results), "matches": results})


async def mcp_find_unknown_faces() -> MCPResponse:
    """Find photos containing unknown or stranger faces."""
    results: list[dict[str, Any]] = rag_engine.find_unknown_faces()
    return MCPResponse(content={"count": len(results), "unknown_faces": results})


# ==============================
# Run both servers
# ==============================


def run() -> None:
    """Run FastAPI and MCP servers concurrently."""
    import uvicorn

    global mcp  # pylint: disable=global-statement
    loop = asyncio.get_event_loop()

    if mcp is None:
        mcp = MCPServer(name="face-insight-rag")

    loop.create_task(mcp.run(port=7001))
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
