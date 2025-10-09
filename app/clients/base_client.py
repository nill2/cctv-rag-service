"""Base HTTP client with common functionality."""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, cast
import httpx

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    """Base HTTP client providing GET, POST, and health check methods with error handling."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize the base HTTP client.

        Args:
            base_url (str): The base URL for the HTTP client.
            timeout (float, optional): Timeout in seconds for HTTP requests. Defaults to 30.0.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "CCTV-MCP-Bridge/1.0"},
        )

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform a GET request with built-in error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error %s for GET %s: %s", exc.response.status_code, endpoint, exc
            )
            raise
        except httpx.RequestError as exc:
            logger.error("Request error for GET %s: %s", endpoint, exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error for GET %s: %s", endpoint, exc)
            raise

    async def post(
        self, endpoint: str, json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform a POST request with built-in error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = await self.client.post(url, json=json_data)
            response.raise_for_status()
            return cast(Dict[str, Any], response.json())
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error %s for POST %s: %s", exc.response.status_code, endpoint, exc
            )
            raise
        except httpx.RequestError as exc:
            logger.error("Request error for POST %s: %s", endpoint, exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error for POST %s: %s", endpoint, exc)
            raise

    async def health_check(self) -> bool:
        """Check if the remote service is healthy."""
        try:
            response = await self.get("/health")
            return response.get("status") == "healthy"
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self.client.aclose()
