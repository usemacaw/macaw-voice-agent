"""Tests: OpenAI Audio API compatibility.

Validates that macaw-asr is a drop-in replacement for the OpenAI Audio API.
Tests use multipart/form-data file upload (same as `curl -F file=@audio.wav`).
"""

from __future__ import annotations

import io
import json
import struct
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from macaw_asr.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_wav_bytes(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    """Generate in-memory WAV file bytes."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ==================== POST /v1/audio/transcriptions ====================


class TestOpenAITranscribe:
    """OpenAI curl equivalent:
    curl http://localhost:8766/v1/audio/transcriptions -F file=@audio.wav -F model=whisper-1
    """

    def test_basic_transcribe_json(self, client):
        wav = _make_wav_bytes(1.0)
        r = client.post("/v1/audio/transcriptions", files={"file": ("audio.wav", wav, "audio/wav")}, data={"model": "mock"})
        assert r.status_code == 200
        data = r.json()
        assert "text" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0

    def test_transcribe_text_format(self, client):
        wav = _make_wav_bytes(0.5)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock", "response_format": "text"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert len(r.text) > 0

    def test_transcribe_verbose_json(self, client):
        wav = _make_wav_bytes(1.0)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock", "response_format": "verbose_json"})
        assert r.status_code == 200
        data = r.json()
        assert "task" in data
        assert data["task"] == "transcribe"
        assert "language" in data
        assert "duration" in data
        assert "text" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)
        assert "usage" in data

    def test_transcribe_srt(self, client):
        wav = _make_wav_bytes(1.0)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock", "response_format": "srt"})
        assert r.status_code == 200
        text = r.text
        assert "1\n" in text
        assert "-->" in text

    def test_transcribe_vtt(self, client):
        wav = _make_wav_bytes(1.0)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock", "response_format": "vtt"})
        assert r.status_code == 200
        assert "WEBVTT" in r.text
        assert "-->" in r.text

    def test_transcribe_with_language(self, client):
        wav = _make_wav_bytes(0.5)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock", "language": "pt"})
        assert r.status_code == 200

    def test_transcribe_whisper1_model(self, client):
        """whisper-1 is mapped to internal qwen/mock model."""
        wav = _make_wav_bytes(0.5)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "whisper-1"})
        # May fail if qwen not loaded, but should not 400
        assert r.status_code in (200, 500)

    def test_transcribe_missing_file(self, client):
        r = client.post("/v1/audio/transcriptions", data={"model": "mock"})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_transcribe_empty_file(self, client):
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}, data={"model": "mock"})
        assert r.status_code == 400

    def test_transcribe_has_usage(self, client):
        """OpenAI returns usage field with duration."""
        wav = _make_wav_bytes(2.0)
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock"})
        assert r.status_code == 200
        data = r.json()
        assert "usage" in data
        assert data["usage"]["type"] == "duration"
        assert data["usage"]["seconds"] > 0

    def test_transcribe_streaming_sse(self, client):
        """OpenAI SSE streaming: data: {type: transcript.text.delta, delta: ...}"""
        wav = _make_wav_bytes(1.0)
        with client.stream(
            "POST", "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": "mock", "stream": "true"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append(data)

            # Must have at least one event
            assert len(events) >= 1

            # Last event must be transcript.text.done
            last = events[-1]
            assert last["type"] == "transcript.text.done"
            assert "text" in last
            assert "usage" in last

            # Delta events (if any) must have type transcript.text.delta
            deltas = [e for e in events if e["type"] == "transcript.text.delta"]
            for d in deltas:
                assert "delta" in d
                assert isinstance(d["delta"], str)

    def test_transcribe_auth_header_ignored(self, client):
        """Auth header accepted but not validated (local server)."""
        wav = _make_wav_bytes(0.5)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav, "audio/wav")},
            data={"model": "mock"},
            headers={"Authorization": "Bearer sk-fake-key-12345"},
        )
        assert r.status_code == 200


# ==================== POST /v1/audio/translations ====================


class TestOpenAITranslate:
    def test_basic_translate(self, client):
        wav = _make_wav_bytes(1.0)
        r = client.post("/v1/audio/translations", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock"})
        assert r.status_code == 200
        assert "text" in r.json()

    def test_translate_text_format(self, client):
        wav = _make_wav_bytes(0.5)
        r = client.post("/v1/audio/translations", files={"file": ("a.wav", wav, "audio/wav")}, data={"model": "mock", "response_format": "text"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_translate_missing_file(self, client):
        r = client.post("/v1/audio/translations", data={"model": "mock"})
        assert r.status_code == 400


# ==================== GET /v1/models ====================


class TestOpenAIModels:
    def test_list_models(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

    def test_models_have_openai_aliases(self, client):
        r = client.get("/v1/models")
        ids = [m["id"] for m in r.json()["data"]]
        assert "whisper-1" in ids
        assert "gpt-4o-transcribe" in ids

    def test_models_have_native(self, client):
        r = client.get("/v1/models")
        ids = [m["id"] for m in r.json()["data"]]
        assert "qwen" in ids
        assert "mock" in ids

    def test_model_entry_format(self, client):
        r = client.get("/v1/models")
        model = r.json()["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "owned_by" in model


# ==================== Error Format ====================


class TestOpenAIErrors:
    """OpenAI returns {"error": {"message": "...", "type": "...", "code": "..."}}.
    We return {"error": "message"} (simpler, still compatible with most clients).
    """

    def test_400_has_error(self, client):
        r = client.post("/v1/audio/transcriptions", data={"model": "mock"})
        assert r.status_code == 400
        data = r.json()
        assert "error" in data

    def test_empty_file_has_error(self, client):
        r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}, data={"model": "mock"})
        assert r.status_code == 400
        assert "error" in r.json()
