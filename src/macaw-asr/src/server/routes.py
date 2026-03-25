"""HTTP API routes for macaw-asr server.

Equivalent to Ollama's server/routes.go. Thin HTTP layer that
delegates to the Scheduler for model management and inference.

Endpoints:
    POST /api/transcribe     — Batch transcription
    POST /api/stream/start   — Start streaming session
    POST /api/stream/push    — Push audio chunk
    POST /api/stream/finish  — Finish session, get final text
    POST /api/pull           — Download a model
    GET  /api/models          — List local models
    GET  /api/models/loaded   — List loaded models
    DELETE /api/models/{id}   — Remove a model
    GET  /health              — Health check
"""

from __future__ import annotations

import json
import logging
import time as _time
import uuid
from typing import Any

from macaw_asr.api.types import (
    PullResponse,
    StreamFinishResponse,
    TranscribeResponse,
)
from macaw_asr.config import EngineConfig
from macaw_asr.server.scheduler import Scheduler

logger = logging.getLogger("macaw-asr.server.routes")


class ASRServer:
    """HTTP server for macaw-asr.

    Framework-agnostic route handlers. The create_app() function
    wires these to an actual HTTP framework.
    """

    def __init__(self, scheduler: Scheduler, default_config: EngineConfig) -> None:
        self._scheduler = scheduler
        self._default_config = default_config

    async def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded_models": self._scheduler.list_loaded(),
        }

    async def transcribe(self, audio: bytes, model: str = "", language: str = "") -> TranscribeResponse:
        """Batch transcription endpoint."""
        config = self._resolve_config(model, language)

        t_total = _time.perf_counter()
        engine = await self._scheduler.get_runner(config)
        load_ms = (_time.perf_counter() - t_total) * 1000

        t0 = _time.perf_counter()
        text = await engine.transcribe(audio)
        inference_ms = (_time.perf_counter() - t0) * 1000
        total_ms = (_time.perf_counter() - t_total) * 1000

        return TranscribeResponse(
            text=text,
            model=config.model_id,
            total_duration_ms=total_ms,
            load_duration_ms=load_ms,
            inference_duration_ms=inference_ms,
        )

    async def stream_start(self, model: str = "", language: str = "", session_id: str = "") -> dict[str, str]:
        """Start a streaming session."""
        config = self._resolve_config(model, language)
        engine = await self._scheduler.get_runner(config)

        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        await engine.create_session(session_id)
        return {"session_id": session_id, "model": config.model_id}

    async def stream_push(self, session_id: str, audio: bytes) -> dict[str, str]:
        """Push audio to a streaming session."""
        engine = self._find_engine_for_session(session_id)
        text = await engine.push_audio(session_id, audio)
        return {"session_id": session_id, "text": text}

    async def stream_finish(self, session_id: str) -> StreamFinishResponse:
        """Finish a streaming session."""
        engine = self._find_engine_for_session(session_id)
        text = await engine.finish_session(session_id)
        return StreamFinishResponse(
            text=text,
            session_id=session_id,
        )

    async def pull_model(self, model_id: str) -> list[PullResponse]:
        """Pull a model from HuggingFace."""
        responses: list[PullResponse] = []
        self._scheduler.registry.pull(
            model_id,
            progress_fn=lambda r: responses.append(r),
        )
        return responses

    async def list_models(self) -> list[dict[str, Any]]:
        """List locally available models."""
        models = self._scheduler.registry.list()
        return [
            {
                "name": m.name,
                "model_id": m.model_id,
                "size_bytes": m.size_bytes,
            }
            for m in models
        ]

    async def list_loaded(self) -> list[str]:
        return self._scheduler.list_loaded()

    async def remove_model(self, model_id: str) -> bool:
        await self._scheduler.unload(model_id)
        return self._scheduler.registry.remove(model_id)

    # ==================== Internal ====================

    def _resolve_config(self, model: str = "", language: str = "") -> EngineConfig:
        """Build config from request params, falling back to defaults."""
        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model_id"] = model
            # Infer model_name from model_id
            name = model.split("/")[-1].lower().replace("-", "_")
            for key in ("qwen", "whisper", "parakeet"):
                if key in name:
                    kwargs["model_name"] = key
                    break
        if language:
            kwargs["language"] = language

        if not kwargs:
            return self._default_config

        # Create new config with overrides
        from dataclasses import asdict
        base = asdict(self._default_config)
        # Flatten nested configs
        audio = base.pop("audio")
        streaming = base.pop("streaming")
        base.update(kwargs)
        from macaw_asr.config import AudioConfig, StreamingConfig
        return EngineConfig(
            **{k: v for k, v in base.items() if k in EngineConfig.__dataclass_fields__},
            audio=AudioConfig(**audio),
            streaming=StreamingConfig(**streaming),
        )

    def _find_engine_for_session(self, session_id: str) -> Any:
        """Find which loaded engine owns a session."""
        for ref in self._scheduler._loaded.values():
            if session_id in ref.engine._sessions:
                return ref.engine
        raise RuntimeError(f"Session '{session_id}' not found in any loaded model")


def create_app(config: EngineConfig | None = None) -> tuple[ASRServer, Scheduler]:
    """Create server and scheduler instances.

    Returns (server, scheduler) — caller is responsible for
    starting the scheduler and serving HTTP.

    Usage:
        server, scheduler = create_app()
        await scheduler.start()
        # Wire server methods to your HTTP framework
        # ...
        await scheduler.stop()
    """
    if config is None:
        config = EngineConfig.from_env()

    scheduler = Scheduler()
    server = ASRServer(scheduler, config)
    return server, scheduler
