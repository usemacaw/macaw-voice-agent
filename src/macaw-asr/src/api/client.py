"""HTTP client for macaw-asr server.

Equivalent to Ollama's api/client.go.
Supports both sync and streaming (NDJSON) responses.
CLI uses this same client (dogfooding).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Callable

logger = logging.getLogger("macaw-asr.api.client")

_DEFAULT_BASE_URL = "http://localhost:8766"


class ASRClient:
    """HTTP client for the macaw-asr server."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    # ==================== Transcribe ====================

    def transcribe(
        self, audio: bytes, model: str = "", language: str = "",
    ) -> dict[str, Any]:
        """Batch transcription (non-streaming)."""
        return self._post("/api/transcribe", {
            "model": model,
            "audio": base64.b64encode(audio).decode(),
            "language": language,
            "stream": False,
        })

    def transcribe_stream(
        self, audio: bytes, model: str = "", language: str = "",
        fn: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        """Streaming transcription. Calls fn for each NDJSON line."""
        return self._stream("/api/transcribe", {
            "model": model,
            "audio": base64.b64encode(audio).decode(),
            "language": language,
            "stream": True,
        }, fn)

    # ==================== Model Management ====================

    def pull(self, model: str, fn: Callable[[dict], None] | None = None) -> list[dict]:
        """Pull a model (streaming progress)."""
        return self._stream("/api/pull", {"model": model, "stream": True}, fn)

    def show(self, model: str) -> dict[str, Any]:
        return self._post("/api/show", {"model": model})

    def list(self) -> dict[str, Any]:
        return self._get("/api/tags")

    def ps(self) -> dict[str, Any]:
        return self._get("/api/ps")

    def delete(self, model: str) -> dict[str, Any]:
        return self._request("DELETE", "/api/delete", {"model": model})

    def version(self) -> dict[str, Any]:
        return self._get("/api/version")

    def health(self) -> str:
        import urllib.request
        url = f"{self._base_url}/"
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode()

    # ==================== Internal ====================

    def _post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, payload)

    def _get(self, path: str) -> dict:
        import urllib.request
        url = f"{self._base_url}{path}"
        try:
            with urllib.request.urlopen(url) as resp:
                return json.loads(resp.read())
        except Exception as e:
            raise ConnectionError(f"Failed: {url}: {e}") from e

    def _request(self, method: str, path: str, payload: dict) -> dict:
        import urllib.request
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(data["error"])
                return data
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err = json.loads(body)
                raise RuntimeError(err.get("error", body))
            except (json.JSONDecodeError, RuntimeError):
                raise
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise ConnectionError(f"Failed: {url}: {e}") from e

    def _stream(
        self, path: str, payload: dict,
        fn: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        """Read NDJSON streaming response line by line (Ollama pattern)."""
        import urllib.request
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/x-ndjson",
            },
            method="POST",
        )
        results = []
        try:
            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    results.append(data)
                    if fn:
                        fn(data)
                    if data.get("error"):
                        raise RuntimeError(data["error"])
        except RuntimeError:
            raise
        except Exception as e:
            raise ConnectionError(f"Failed: {url}: {e}") from e
        return results
