"""HTTP client for macaw-asr server.

Equivalent to Ollama's api/client.go. The CLI uses this same client
that external consumers use (eat your own dogfood).

Usage:
    client = ASRClient("http://localhost:8766")
    response = await client.transcribe(audio_bytes)
    print(response.text)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable
from urllib.parse import urljoin

from macaw_asr.api.types import (
    ModelInfo,
    PullResponse,
    StreamFinishResponse,
    TranscribeResponse,
)

logger = logging.getLogger("macaw-asr.api.client")

_DEFAULT_BASE_URL = "http://localhost:8766"


class ASRClient:
    """HTTP client for the macaw-asr server.

    Provides typed methods for all API endpoints.
    Uses urllib to avoid aiohttp/httpx dependency — the client
    is lightweight and synchronous (CLI use case).
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    def transcribe(
        self,
        audio: bytes,
        model: str = "",
        language: str = "",
    ) -> TranscribeResponse:
        """Batch transcription.

        Args:
            audio: PCM 16-bit audio bytes.
            model: Model ID (optional, uses server default).
            language: Language code (optional).

        Returns:
            TranscribeResponse with text and timing info.
        """
        import base64
        import urllib.request

        payload = {
            "audio_b64": base64.b64encode(audio).decode(),
        }
        if model:
            payload["model"] = model
        if language:
            payload["language"] = language

        data = self._post("/api/transcribe", payload)
        return TranscribeResponse(
            text=data.get("text", ""),
            model=data.get("model", ""),
            total_duration_ms=data.get("total_duration_ms", 0),
            load_duration_ms=data.get("load_duration_ms", 0),
            inference_duration_ms=data.get("inference_duration_ms", 0),
            tokens=data.get("tokens", 0),
        )

    def pull(
        self,
        model_id: str,
        progress_fn: Callable[[PullResponse], None] | None = None,
    ) -> None:
        """Pull a model from HuggingFace via the server."""
        data = self._post("/api/pull", {"model": model_id})
        if progress_fn and isinstance(data, list):
            for item in data:
                progress_fn(PullResponse(
                    status=item.get("status", ""),
                    completed=item.get("completed", 0),
                    total=item.get("total", 0),
                ))

    def list_models(self) -> list[ModelInfo]:
        """List locally available models."""
        data = self._get("/api/models")
        return [
            ModelInfo(
                name=m.get("name", ""),
                model_id=m.get("model_id", ""),
                size_bytes=m.get("size_bytes", 0),
            )
            for m in data
        ]

    def health(self) -> dict[str, Any]:
        """Check server health."""
        return self._get("/health")

    # ==================== Internal ====================

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        import urllib.request

        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to macaw-asr server at {self._base_url}: {e}"
            ) from e

    def _get(self, path: str) -> Any:
        import urllib.request

        url = f"{self._base_url}{path}"
        try:
            with urllib.request.urlopen(url) as resp:
                return json.loads(resp.read())
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to macaw-asr server at {self._base_url}: {e}"
            ) from e
