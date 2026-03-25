"""FastAPI application for macaw-asr.

OpenAI Audio API compatible. Drop-in replacement:
    client = OpenAI(base_url="http://localhost:8766/v1", api_key="unused")

Endpoints:
    POST /v1/audio/transcriptions  — Transcribe audio (multipart, OpenAI format)
    POST /v1/audio/translations    — Translate audio to English (multipart)
    GET  /v1/models                — List available models

    POST /api/show                 — Show model details (operational)
    GET  /api/ps                   — List loaded/running models (operational)
    POST /api/pull                 — Download a model (operational)
    DELETE /api/delete             — Remove a model (operational)
    GET  /api/version              — Server version (operational)
    GET  /                         — Health check
"""

from __future__ import annotations

import json
import logging
import re
import time as _time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

import macaw_asr
from macaw_asr.api.types import (
    DeleteRequest,
    ModelDetails,
    ModelEntry,
    ListResponse,
    PsResponse,
    PullRequest,
    RunningModel,
    ShowRequest,
    ShowResponse,
    VersionResponse,
)
from macaw_asr.audio.decode import decode_audio
from macaw_asr.audio.preprocessing import resample
from macaw_asr.config import EngineConfig
from macaw_asr.server.scheduler import Scheduler

logger = logging.getLogger("macaw-asr.server")

_scheduler: Scheduler | None = None
_default_config: EngineConfig | None = None

# OpenAI model name → internal model name
_MODEL_ALIASES = {
    "whisper-1": "qwen",
    "gpt-4o-transcribe": "qwen",
    "gpt-4o-mini-transcribe": "qwen",
    "gpt-4o-mini-transcribe-2025-12-15": "qwen",
    "gpt-4o-transcribe-diarize": "qwen",
}

_SHORT_NAMES = {"qwen", "whisper", "parakeet", "mock"}


# ==================== Lifespan ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler, _default_config
    _default_config = EngineConfig.from_env()
    _scheduler = Scheduler()
    await _scheduler.start()
    logger.info("macaw-asr server started")
    yield
    await _scheduler.stop()
    logger.info("macaw-asr server stopped")


# ==================== App ====================


app = FastAPI(title="macaw-asr", version=macaw_asr.__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = _time.perf_counter()
    response = await call_next(request)
    ms = (_time.perf_counter() - t0) * 1000
    logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, ms)
    return response


# ==================== Health ====================


@app.get("/")
@app.head("/")
async def health():
    return "macaw-asr is running"


# ====================================================================
# OpenAI Audio API  (/v1/*)
# ====================================================================


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    """OpenAI-compatible: POST /v1/audio/transcriptions (multipart/form-data)."""
    audio_f32, file_sr, audio_duration, config, response_format, temperature, stream = await _parse_audio_request(request)

    input_sr = config.audio.input_sample_rate
    if file_sr != input_sr:
        audio_f32 = resample(audio_f32, file_sr, input_sr)
    pcm_bytes = (audio_f32 * 32767).astype(np.int16).tobytes()

    t0 = _time.perf_counter()
    engine = await _get_engine(config)

    # SSE Streaming
    if stream:
        return await _handle_stream(engine, config, audio_f32, file_sr, audio_duration)

    # Non-streaming
    text = await engine.transcribe(pcm_bytes)
    return _format_response(text, config, audio_duration, response_format, temperature)


@app.post("/v1/audio/translations")
async def translate(request: Request):
    """OpenAI-compatible: POST /v1/audio/translations (multipart/form-data).
    Forces language=en for translation to English."""
    audio_f32, file_sr, audio_duration, config, response_format, temperature, stream = await _parse_audio_request(request, force_language="en")

    input_sr = config.audio.input_sample_rate
    if file_sr != input_sr:
        audio_f32 = resample(audio_f32, file_sr, input_sr)
    pcm_bytes = (audio_f32 * 32767).astype(np.int16).tobytes()

    engine = await _get_engine(config)
    text = await engine.transcribe(pcm_bytes)

    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}


@app.get("/v1/models")
async def list_models_openai():
    """OpenAI-compatible: GET /v1/models."""
    from macaw_asr.models.base import list_models as _list

    data = []
    seen = set()
    for name in _list():
        data.append({"id": name, "object": "model", "created": 0, "owned_by": "macaw-asr"})
        seen.add(name)
    for alias in _MODEL_ALIASES:
        if alias not in seen:
            data.append({"id": alias, "object": "model", "created": 0, "owned_by": "macaw-asr"})
    return {"object": "list", "data": data}


# ====================================================================
# Operational API  (/api/*)
# ====================================================================


@app.get("/api/version")
async def version():
    return VersionResponse(version=macaw_asr.__version__)


@app.post("/api/show")
async def show(req: ShowRequest):
    if not req.model:
        raise HTTPException(400, detail="missing model field")

    # Check manifest registry
    for m in _scheduler.registry.list():
        if m.name == req.model or m.model_id == req.model:
            return ShowResponse(
                model_info={"general.architecture": m.family or "unknown", "general.model_id": m.model_id},
                details={"family": m.family or "unknown", "parameter_size": m.parameters or "unknown", "quantization_level": "BF16"},
            )

    # Check loaded models
    for model_id, ref in _scheduler._loaded.items():
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        if req.model in (model_id, short, ref.model_id, ref.engine.config.model_name):
            cfg = ref.engine.config
            return _show_from_config(cfg)

    # Check default config / aliases
    if _default_config:
        cfg = _default_config
        if req.model in (cfg.model_name, cfg.model_id) or req.model in _MODEL_ALIASES or req.model in _SHORT_NAMES:
            return _show_from_config(cfg, loaded=False)

    raise HTTPException(404, detail=f"model '{req.model}' not found")


@app.get("/api/ps")
async def list_running():
    models = []
    for model_id, ref in _scheduler._loaded.items():
        models.append(RunningModel(
            name=ref.model_id.split("/")[-1] if "/" in ref.model_id else ref.model_id,
            model=ref.model_id, size=0, size_vram=0,
        ))
    return PsResponse(models=models)


@app.post("/api/pull")
async def pull(req: PullRequest):
    if not req.model:
        raise HTTPException(400, detail="missing model field")
    if req.stream:
        async def stream():
            yield json.dumps({"status": "pulling model"}) + "\n"
            try:
                _scheduler.registry.pull(req.model)
                yield json.dumps({"status": "success"}) + "\n"
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson")
    try:
        _scheduler.registry.pull(req.model)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.delete("/api/delete")
async def delete(req: DeleteRequest):
    if not req.model:
        raise HTTPException(400, detail="missing model field")
    await _scheduler.unload(req.model)
    if not _scheduler.registry.remove(req.model):
        raise HTTPException(404, detail=f"model '{req.model}' not found")
    return {"status": "success"}


# ====================================================================
# Error Handlers
# ====================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ====================================================================
# Internal Helpers
# ====================================================================


async def _parse_audio_request(
    request: Request, force_language: str = "",
) -> tuple[np.ndarray, int, float, EngineConfig, str, float, bool]:
    """Parse multipart form from OpenAI-style request. Returns (audio_f32, sr, duration, config, format, temp, stream)."""
    form = await request.form()

    file_field = form.get("file")
    if file_field is None:
        raise HTTPException(400, detail="missing 'file' field")
    if not hasattr(file_field, "read"):
        raise HTTPException(400, detail="'file' must be a file upload")

    file_bytes = await file_field.read()
    filename = getattr(file_field, "filename", "audio.wav")
    if len(file_bytes) == 0:
        raise HTTPException(400, detail="empty audio file")

    model_name = str(form.get("model", "whisper-1"))
    language = force_language or str(form.get("language", ""))
    response_format = str(form.get("response_format", "json"))
    temperature = float(form.get("temperature", 0))
    stream = str(form.get("stream", "false")).lower() == "true"

    internal_model = _MODEL_ALIASES.get(model_name, model_name)

    try:
        audio_f32, file_sr = decode_audio(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    audio_duration = len(audio_f32) / file_sr if file_sr > 0 else 0
    config = _resolve_config(internal_model, language)

    return audio_f32, file_sr, audio_duration, config, response_format, temperature, stream


async def _get_engine(config: EngineConfig):
    try:
        return await _scheduler.get_runner(config)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


def _format_response(text: str, config: EngineConfig, duration: float, fmt: str, temperature: float):
    """Format transcription response based on response_format."""
    if fmt == "text":
        return PlainTextResponse(text)

    if fmt == "verbose_json":
        return {
            "task": "transcribe",
            "language": config.language,
            "duration": round(duration, 2),
            "text": text,
            "segments": [{
                "id": 0, "seek": 0, "start": 0.0, "end": round(duration, 2),
                "text": text, "tokens": [], "temperature": temperature,
                "avg_logprob": 0.0, "compression_ratio": 1.0, "no_speech_prob": 0.0,
            }],
            "usage": {"type": "duration", "seconds": int(duration) + 1},
        }

    if fmt == "srt":
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), duration % 60
        end_ts = f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
        return PlainTextResponse(f"1\n00:00:00,000 --> {end_ts}\n{text}\n")

    if fmt == "vtt":
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), duration % 60
        end_ts = f"{h:02d}:{m:02d}:{s:06.3f}"
        return PlainTextResponse(f"WEBVTT\n\n00:00:00.000 --> {end_ts}\n{text}\n", media_type="text/vtt")

    # Default: json
    return {"text": text, "usage": {"type": "duration", "seconds": int(duration) + 1}}


async def _handle_stream(engine, config, audio_f32_input_sr, file_sr, audio_duration):
    """SSE streaming: yield transcript.text.delta + transcript.text.done events."""
    import asyncio
    import queue
    import threading

    from macaw_asr._executor import run_in_executor
    from macaw_asr.decode.postprocess import clean_asr_text

    model_rate = config.audio.model_sample_rate
    input_sr = config.audio.input_sample_rate
    if input_sr != model_rate:
        float_audio_model = resample(audio_f32_input_sr, input_sr, model_rate)
    else:
        float_audio_model = audio_f32_input_sr

    async def sse_stream():
        model = engine._model
        executor = engine._executor

        inputs = await run_in_executor(executor, model.prepare_inputs, float_audio_model, "")
        strategy = engine._create_strategy()

        q = queue.Queue()

        def _run():
            try:
                for delta, is_done, output in model.generate_stream(inputs, strategy):
                    q.put((delta, is_done, output))
            except Exception as e:
                q.put(("__ERROR__", True, str(e)))

        threading.Thread(target=_run, daemon=True).start()

        full_text = ""
        while True:
            try:
                delta, is_done, output = q.get(timeout=30)
            except queue.Empty:
                break

            if delta == "__ERROR__":
                yield f"data: {json.dumps({'error': output})}\n\n"
                break

            if delta:
                full_text += delta
                yield f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': delta})}\n\n"

            if is_done:
                final = clean_asr_text(full_text)
                yield f"data: {json.dumps({'type': 'transcript.text.done', 'text': final, 'usage': {'type': 'duration', 'seconds': int(audio_duration) + 1}})}\n\n"
                break

            await asyncio.sleep(0)

    return StreamingResponse(sse_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _resolve_config(model: str = "", language: str = "") -> EngineConfig:
    from dataclasses import asdict
    from macaw_asr.config import AudioConfig, StreamingConfig

    if not model and _default_config:
        return _default_config if not language else _resolve_config(_default_config.model_name, language)

    kwargs: dict[str, Any] = {}
    if model:
        if model in _SHORT_NAMES:
            kwargs["model_name"] = model
            if _default_config and _default_config.model_name == model:
                kwargs["model_id"] = _default_config.model_id
        elif "/" in model:
            kwargs["model_id"] = model
            name = model.split("/")[-1].lower().replace("-", "_")
            for key in _SHORT_NAMES:
                if key in name:
                    kwargs["model_name"] = key
                    break
            if "model_name" not in kwargs:
                kwargs["model_name"] = model
        else:
            kwargs["model_name"] = model
            kwargs["model_id"] = model
    if language:
        kwargs["language"] = language

    base = asdict(_default_config)
    audio = base.pop("audio")
    streaming = base.pop("streaming")
    base.update(kwargs)
    return EngineConfig(
        **{k: v for k, v in base.items() if k in EngineConfig.__dataclass_fields__},
        audio=AudioConfig(**audio), streaming=StreamingConfig(**streaming),
    )


def _show_from_config(cfg: EngineConfig, loaded: bool = True) -> ShowResponse:
    param_match = re.search(r'(\d+\.?\d*[BbMm])', cfg.model_id)
    param_size = param_match.group(1).upper() if param_match else "unknown"
    info = {
        "general.architecture": cfg.model_name,
        "general.model_id": cfg.model_id,
        "general.device": cfg.device,
        "general.dtype": cfg.dtype,
        "general.language": cfg.language,
        "general.max_new_tokens": cfg.streaming.max_new_tokens,
        "general.input_sample_rate": cfg.audio.input_sample_rate,
        "general.model_sample_rate": cfg.audio.model_sample_rate,
    }
    if not loaded:
        info["status"] = "not loaded (will load on first request)"
    return ShowResponse(
        model_info=info,
        details={"family": cfg.model_name, "parameter_size": param_size, "quantization_level": cfg.dtype.upper()},
    )
