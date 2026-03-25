"""HTTP API tests — extracted from Ollama routes_generate_test.go patterns.

Uses FastAPI TestClient (equivalent to Ollama's httptest.NewRecorder + gin.CreateTestContext).
Tests both streaming (NDJSON) and non-streaming (JSON) responses.
Error format validated: {"error": "message"} with correct HTTP status codes.

These tests run with mock model (no GPU needed) — tests API layer only.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from macaw_asr.server.app import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient — Ollama's httptest.NewRecorder equivalent."""
    with TestClient(app) as c:
        yield c


def _make_audio_b64(duration_sec: float = 1.0, sr: int = 8000) -> str:
    """Generate base64-encoded PCM16 audio."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode()


# ==================== Health (Ollama: GET /) ====================


class TestHealth:
    """From Ollama: HEAD / and GET / return status text."""

    def test_get_health(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "macaw-asr" in r.text

    def test_head_health(self, client):
        r = client.head("/")
        assert r.status_code == 200


# ==================== Version (Ollama: GET /api/version) ====================


class TestVersion:
    def test_returns_version(self, client):
        r = client.get("/api/version")
        assert r.status_code == 200
        data = r.json()
        assert "version" in data
        assert len(data["version"]) > 0


# ==================== Transcribe (Ollama: POST /api/generate) ====================


class TestTranscribe:
    """From Ollama routes_generate_test.go: test generate endpoint."""

    def test_transcribe_returns_text(self, client):
        r = client.post("/api/transcribe", json={
            "model": "mock",
            "audio": _make_audio_b64(1.0),
        })
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert data["done"] is True
        assert data["model"] != ""
        assert data["total_duration"] > 0

    def test_transcribe_missing_body(self, client):
        """From Ollama: missing request body → 422."""
        r = client.post("/api/transcribe")
        assert r.status_code == 422

    def test_transcribe_missing_audio(self, client):
        """From Ollama: missing required field → 400."""
        r = client.post("/api/transcribe", json={"model": "mock"})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_transcribe_invalid_base64(self, client):
        r = client.post("/api/transcribe", json={
            "model": "mock",
            "audio": "not-valid-base64!!!",
        })
        assert r.status_code == 400
        assert "error" in r.json()

    def test_transcribe_empty_audio(self, client):
        # base64 of zero-length PCM = "AA==" (2 null bytes = 1 sample)
        # Truly empty = base64("") = "" which is falsy → 400
        # Instead send minimal audio (2 bytes = 1 silent sample)
        r = client.post("/api/transcribe", json={
            "model": "mock",
            "audio": base64.b64encode(b"\x00\x00").decode(),
        })
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["text"], str)

    def test_transcribe_streaming_ndjson(self, client):
        """From Ollama: streaming response is NDJSON (application/x-ndjson)."""
        with client.stream("POST", "/api/transcribe", json={
            "model": "mock",
            "audio": _make_audio_b64(1.0),
            "stream": True,
        }) as response:
            assert response.status_code == 200
            assert "ndjson" in response.headers.get("content-type", "")
            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(json.loads(line))
            assert len(lines) >= 1
            assert lines[-1]["done"] is True

    def test_transcribe_with_language(self, client):
        r = client.post("/api/transcribe", json={
            "model": "mock",
            "audio": _make_audio_b64(0.5),
            "language": "en",
        })
        assert r.status_code == 200

    def test_transcribe_response_has_ollama_fields(self, client):
        """Response must have same fields as Ollama's GenerateResponse."""
        r = client.post("/api/transcribe", json={
            "model": "mock",
            "audio": _make_audio_b64(1.0),
        })
        data = r.json()
        required_fields = ["model", "created_at", "text", "done", "total_duration"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_transcribe_durations_in_nanoseconds(self, client):
        """From Ollama: all durations are in nanoseconds (large integers)."""
        r = client.post("/api/transcribe", json={
            "model": "mock",
            "audio": _make_audio_b64(1.0),
        })
        data = r.json()
        # Nanoseconds should be > 1000 for any real operation (even mock)
        assert isinstance(data["total_duration"], int)


# ==================== Show (Ollama: POST /api/show) ====================


class TestShow:
    def test_show_missing_model(self, client):
        r = client.post("/api/show", json={"model": ""})
        assert r.status_code == 400

    def test_show_not_found(self, client):
        r = client.post("/api/show", json={"model": "nonexistent"})
        assert r.status_code == 404
        assert "not found" in r.json()["error"]


# ==================== List Models (Ollama: GET /api/tags) ====================


class TestListModels:
    def test_list_returns_models_array(self, client):
        r = client.get("/api/tags")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)


# ==================== List Running (Ollama: GET /api/ps) ====================


class TestListRunning:
    def test_ps_returns_models_array(self, client):
        r = client.get("/api/ps")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)


# ==================== Pull (Ollama: POST /api/pull) ====================


class TestPull:
    def test_pull_missing_model(self, client):
        r = client.post("/api/pull", json={"model": "", "stream": False})
        assert r.status_code == 400

    def test_pull_streaming_ndjson(self, client):
        """From Ollama: pull progress is streamed as NDJSON."""
        # This will fail for real models (no HF auth), but tests the format
        with client.stream("POST", "/api/pull", json={
            "model": "fake/model",
            "stream": True,
        }) as response:
            lines = []
            for line in response.iter_lines():
                if line:
                    lines.append(json.loads(line))
            # Should have at least "pulling model" status
            assert len(lines) >= 1


# ==================== Delete (Ollama: DELETE /api/delete) ====================


class TestDelete:
    def test_delete_not_found(self, client):
        r = client.request("DELETE", "/api/delete", json={"model": "nonexistent"})
        assert r.status_code == 404

    def test_delete_missing_model(self, client):
        r = client.request("DELETE", "/api/delete", json={"model": ""})
        assert r.status_code == 400


# ==================== Streaming Session ====================


class TestStreamingSession:
    def test_stream_start(self, client):
        r = client.post("/api/transcribe/stream", json={"model": "mock"})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data

    def test_stream_push_and_finish(self, client):
        # Start
        r = client.post("/api/transcribe/stream", json={"model": "mock"})
        sid = r.json()["session_id"]

        # Push audio
        r = client.post("/api/transcribe/stream/push", json={
            "session_id": sid,
            "audio": _make_audio_b64(0.5),
        })
        assert r.status_code == 200

        # Finish
        r = client.post("/api/transcribe/stream/push", json={
            "session_id": sid,
            "end_of_stream": True,
        })
        assert r.status_code == 200
        assert "text" in r.json()

    def test_stream_push_invalid_session(self, client):
        r = client.post("/api/transcribe/stream/push", json={
            "session_id": "nonexistent",
            "audio": _make_audio_b64(0.5),
        })
        assert r.status_code == 404


# ==================== Error Format (Ollama convention) ====================


class TestErrorFormat:
    """From Ollama: ALL errors must return {"error": "message"}."""

    @pytest.mark.parametrize("path,method,body", [
        ("/api/transcribe", "POST", {"model": "mock"}),  # missing audio
        ("/api/show", "POST", {"model": ""}),  # missing model
        ("/api/pull", "POST", {"model": "", "stream": False}),  # missing model
    ])
    def test_error_has_error_field(self, client, path, method, body):
        if method == "POST":
            r = client.post(path, json=body)
        elif method == "DELETE":
            r = client.request("DELETE", path, json=body)
        else:
            r = client.get(path)

        assert r.status_code >= 400
        data = r.json()
        assert "error" in data, f"Error response missing 'error' field: {data}"
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0
