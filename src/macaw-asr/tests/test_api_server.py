"""HTTP API tests — consolidated endpoints.

Tests OpenAI-compatible endpoints (/v1/*) and operational endpoints (/api/*).
Uses FastAPI TestClient (mock model, no GPU needed).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from macaw_asr.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ==================== Health ====================


class TestHealth:
    def test_get(self, client):
        assert client.get("/").status_code == 200

    def test_head(self, client):
        assert client.head("/").status_code == 200


# ==================== GET /api/version ====================


class TestVersion:
    def test_returns_version(self, client):
        r = client.get("/api/version")
        assert r.status_code == 200
        assert "version" in r.json()


# ==================== POST /api/show ====================


class TestShow:
    def test_show_known_model(self, client):
        r = client.post("/api/show", json={"model": "qwen"})
        assert r.status_code == 200
        data = r.json()
        assert "details" in data
        assert data["details"]["family"] == "qwen"
        assert data["details"]["parameter_size"] == "0.6B"

    def test_show_by_registry_name(self, client):
        r = client.post("/api/show", json={"model": "whisper-tiny"})
        assert r.status_code == 200

    def test_show_missing_model(self, client):
        r = client.post("/api/show", json={"model": ""})
        assert r.status_code == 400

    def test_show_not_found(self, client):
        r = client.post("/api/show", json={"model": "nonexistent-xyz"})
        assert r.status_code == 404


# ==================== GET /api/ps ====================


class TestPs:
    def test_ps_returns_models(self, client):
        r = client.get("/api/ps")
        assert r.status_code == 200
        assert "models" in r.json()


# ==================== POST /api/pull ====================


class TestPull:
    def test_pull_missing_model(self, client):
        r = client.post("/api/pull", json={"model": "", "stream": False})
        assert r.status_code == 400

    def test_pull_streaming_format(self, client):
        with client.stream("POST", "/api/pull", json={"model": "fake/model", "stream": True}) as response:
            lines = [json.loads(l) for l in response.iter_lines() if l]
            assert len(lines) >= 1


# ==================== DELETE /api/delete ====================


class TestDelete:
    def test_delete_not_found(self, client):
        r = client.request("DELETE", "/api/delete", json={"model": "nonexistent"})
        assert r.status_code == 404

    def test_delete_missing_model(self, client):
        r = client.request("DELETE", "/api/delete", json={"model": ""})
        assert r.status_code == 400


# ==================== Error Format ====================


class TestErrorFormat:
    @pytest.mark.parametrize("path,method,body", [
        ("/api/show", "POST", {"model": ""}),
        ("/api/pull", "POST", {"model": "", "stream": False}),
        ("/api/delete", "DELETE", {"model": ""}),
    ])
    def test_error_has_error_field(self, client, path, method, body):
        r = client.request(method, path, json=body)
        assert r.status_code >= 400
        assert "error" in r.json()
