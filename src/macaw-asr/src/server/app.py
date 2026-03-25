"""FastAPI application for macaw-asr.

Equivalent to Ollama's server/routes.go + Gin wiring.
Each endpoint follows Ollama's patterns:
- NDJSON streaming (application/x-ndjson)
- Error format: {"error": "message"}
- Durations in nanoseconds
"""

from __future__ import annotations

import base64
import json
import logging
import time as _time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import macaw_asr
from macaw_asr.api.types import (
    DeleteRequest,
    ListResponse,
    ModelDetails,
    ModelEntry,
    PsResponse,
    PullRequest,
    PullResponse,
    RunningModel,
    ShowRequest,
    ShowResponse,
    StreamFinishResponse,
    StreamPushRequest,
    StreamStartRequest,
    TranscribeRequest,
    TranscribeResponse,
    VersionResponse,
)
from macaw_asr.config import EngineConfig
from macaw_asr.server.scheduler import Scheduler

logger = logging.getLogger("macaw-asr.server")

_scheduler: Scheduler | None = None
_default_config: EngineConfig | None = None


def _ms_to_ns(ms: float) -> int:
    """Convert milliseconds to nanoseconds (Ollama convention)."""
    return int(ms * 1_000_000)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


app = FastAPI(
    title="macaw-asr",
    version=macaw_asr.__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request Logging Middleware ====================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = _time.perf_counter()
    response = await call_next(request)
    ms = (_time.perf_counter() - t0) * 1000
    logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, ms)
    return response


# ==================== Health (Ollama GET /) ====================


@app.get("/")
@app.head("/")
async def health():
    return "macaw-asr is running"


# ==================== Version (Ollama GET /api/version) ====================


@app.get("/api/version")
async def version():
    return VersionResponse(version=macaw_asr.__version__)


# ==================== Transcribe (Ollama POST /api/generate) ====================


@app.post("/api/transcribe")
async def transcribe(req: TranscribeRequest):
    if not req.model and _default_config:
        req.model = _default_config.model_name

    if not req.audio:
        raise HTTPException(400, detail="missing audio field")

    try:
        audio_bytes = base64.b64decode(req.audio)
    except Exception:
        raise HTTPException(400, detail="invalid base64 audio")

    if len(audio_bytes) == 0:
        return TranscribeResponse(
            model=req.model or (_default_config.model_id if _default_config else ""),
            created_at=_now_iso(),
            text="",
            done=True,
        )

    config = _resolve_config(req.model, req.language)

    t_total = _time.perf_counter()
    try:
        engine = await _scheduler.get_runner(config)
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    load_ns = _ms_to_ns((_time.perf_counter() - t_total) * 1000)

    t0 = _time.perf_counter()
    text = await engine.transcribe(audio_bytes)
    eval_ns = _ms_to_ns((_time.perf_counter() - t0) * 1000)
    total_ns = _ms_to_ns((_time.perf_counter() - t_total) * 1000)

    resp = TranscribeResponse(
        model=config.model_id,
        created_at=_now_iso(),
        text=text,
        done=True,
        total_duration=total_ns,
        load_duration=load_ns,
        eval_duration=eval_ns,
    )

    if req.stream:
        # NDJSON: single response line
        async def stream():
            yield json.dumps(resp.model_dump()) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return resp


# ==================== Show (Ollama POST /api/show) ====================


@app.post("/api/show")
async def show(req: ShowRequest):
    if not req.model:
        raise HTTPException(400, detail="missing model field")

    models = _scheduler.registry.list()
    found = None
    for m in models:
        if m.name == req.model or m.model_id == req.model:
            found = m
            break

    if not found:
        raise HTTPException(404, detail=f"model '{req.model}' not found")

    return ShowResponse(
        model_info={
            "general.architecture": found.family or "unknown",
            "general.parameter_count": found.parameters or "unknown",
        },
        details={
            "family": found.family or "unknown",
            "parameter_size": found.parameters or "unknown",
            "quantization_level": "BF16",
        },
    )


# ==================== List Models (Ollama GET /api/tags) ====================


@app.get("/api/tags")
async def list_models():
    models = _scheduler.registry.list()
    entries = []
    for m in models:
        entries.append(ModelEntry(
            name=m.name,
            model=m.model_id,
            size=m.size_bytes,
            details=ModelDetails(family=m.family or "unknown"),
        ))
    return ListResponse(models=entries)


# ==================== List Running (Ollama GET /api/ps) ====================


@app.get("/api/ps")
async def list_running():
    loaded = _scheduler._loaded
    models = []
    for model_id, ref in loaded.items():
        models.append(RunningModel(
            name=ref.model_id.split("/")[-1] if "/" in ref.model_id else ref.model_id,
            model=ref.model_id,
            size=0,
            size_vram=0,
        ))
    return PsResponse(models=models)


# ==================== Pull (Ollama POST /api/pull) ====================


@app.post("/api/pull")
async def pull(req: PullRequest):
    if not req.model:
        raise HTTPException(400, detail="missing model field")

    if req.stream:
        async def stream():
            yield json.dumps({"status": "pulling model"}) + "\n"
            try:
                def progress(pr):
                    pass  # sync callback, can't yield from here
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


# ==================== Delete (Ollama DELETE /api/delete) ====================


@app.delete("/api/delete")
async def delete(req: DeleteRequest):
    if not req.model:
        raise HTTPException(400, detail="missing model field")

    await _scheduler.unload(req.model)
    removed = _scheduler.registry.remove(req.model)
    if not removed:
        raise HTTPException(404, detail=f"model '{req.model}' not found")

    return {"status": "success"}


# ==================== Streaming Session Endpoints ====================


@app.post("/api/transcribe/stream")
async def stream_start(req: StreamStartRequest):
    config = _resolve_config(req.model, req.language)
    engine = await _scheduler.get_runner(config)

    session_id = req.session_id or str(uuid.uuid4())[:8]
    await engine.create_session(session_id)
    return {"session_id": session_id, "model": config.model_id}


@app.post("/api/transcribe/stream/push")
async def stream_push(req: StreamPushRequest):
    if not req.session_id:
        raise HTTPException(400, detail="missing session_id")

    engine = _find_engine_for_session(req.session_id)
    audio_bytes = base64.b64decode(req.audio) if req.audio else b""

    if req.end_of_stream:
        text = await engine.finish_session(req.session_id)
        return StreamFinishResponse(text=text, session_id=req.session_id)

    text = await engine.push_audio(req.session_id, audio_bytes)
    return {"session_id": req.session_id, "text": text}


# ==================== Error Handler (Ollama format) ====================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )


# ==================== Internal ====================


def _resolve_config(model: str = "", language: str = "") -> EngineConfig:
    from dataclasses import asdict
    from macaw_asr.config import AudioConfig, StreamingConfig

    if not model and _default_config:
        if not language:
            return _default_config
        model = _default_config.model_name

    # Known short names → use default config's model_id
    _SHORT_NAMES = {"qwen", "whisper", "parakeet", "mock"}

    kwargs: dict[str, Any] = {}
    if model:
        if model in _SHORT_NAMES:
            # Short name: use as model_name, keep default model_id
            kwargs["model_name"] = model
            if _default_config and _default_config.model_name == model:
                kwargs["model_id"] = _default_config.model_id
        elif "/" in model:
            # HuggingFace model ID (e.g. Qwen/Qwen3-ASR-0.6B)
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
        audio=AudioConfig(**audio),
        streaming=StreamingConfig(**streaming),
    )


def _find_engine_for_session(session_id: str):
    for ref in _scheduler._loaded.values():
        if session_id in ref.engine._sessions:
            return ref.engine
    raise HTTPException(404, detail=f"session '{session_id}' not found")


# ====================================================================
# OpenAI-Compatible API (/v1/audio/*)
# Drop-in replacement for OpenAI Audio API.
# Any OpenAI SDK works: client = OpenAI(base_url="http://localhost:8766/v1")
# ====================================================================

# Model name mapping: OpenAI model names → internal model names
_OPENAI_MODEL_MAP = {
    "whisper-1": "qwen",
    "gpt-4o-transcribe": "qwen",
    "gpt-4o-mini-transcribe": "qwen",
    "gpt-4o-mini-transcribe-2025-12-15": "qwen",
    "gpt-4o-transcribe-diarize": "qwen",
}


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(request: Request):
    """OpenAI-compatible transcription endpoint.

    Accepts multipart/form-data with file upload (same as OpenAI).
    Returns json, text, verbose_json, srt, or vtt.
    """
    from fastapi import UploadFile, Form, File
    from macaw_asr.audio.decode import decode_audio
    from macaw_asr.audio.preprocessing import resample

    # Parse multipart form
    form = await request.form()

    # Required: file
    file_field = form.get("file")
    if file_field is None:
        raise HTTPException(400, detail="missing 'file' field")

    if hasattr(file_field, "read"):
        file_bytes = await file_field.read()
        filename = getattr(file_field, "filename", "audio.wav")
    else:
        raise HTTPException(400, detail="'file' must be a file upload")

    if len(file_bytes) == 0:
        raise HTTPException(400, detail="empty audio file")

    # Optional fields
    model_name = str(form.get("model", "whisper-1"))
    language = str(form.get("language", ""))
    response_format = str(form.get("response_format", "json"))
    prompt = str(form.get("prompt", ""))
    temperature = float(form.get("temperature", 0))
    stream = str(form.get("stream", "false")).lower() == "true"

    # Map OpenAI model name to internal
    internal_model = _OPENAI_MODEL_MAP.get(model_name, model_name)

    # Decode audio file to float32
    try:
        audio_f32, file_sr = decode_audio(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    audio_duration = len(audio_f32) / file_sr if file_sr > 0 else 0

    # Resample to input_sample_rate (8kHz) then to PCM16 bytes for engine
    config = _resolve_config(internal_model, language)
    input_sr = config.audio.input_sample_rate
    if file_sr != input_sr:
        audio_f32 = resample(audio_f32, file_sr, input_sr)

    # Convert to PCM16 bytes (engine expects this)
    pcm_bytes = (audio_f32 * 32767).astype(np.int16).tobytes()

    # Transcribe
    t0 = _time.perf_counter()
    try:
        engine = await _scheduler.get_runner(config)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    text = await engine.transcribe(pcm_bytes)
    total_sec = _time.perf_counter() - t0

    # Build response based on format
    if response_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text)

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or config.language,
            "duration": round(audio_duration, 2),
            "text": text,
            "segments": [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": round(audio_duration, 2),
                    "text": text,
                    "tokens": [],
                    "temperature": temperature,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                }
            ],
            "usage": {
                "type": "duration",
                "seconds": int(audio_duration) + 1,
            },
        }

    if response_format == "srt":
        from fastapi.responses import PlainTextResponse
        dur = audio_duration
        h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
        end_ts = f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
        srt = f"1\n00:00:00,000 --> {end_ts}\n{text}\n"
        return PlainTextResponse(srt, media_type="text/plain")

    if response_format == "vtt":
        from fastapi.responses import PlainTextResponse
        dur = audio_duration
        h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
        end_ts = f"{h:02d}:{m:02d}:{s:06.3f}"
        vtt = f"WEBVTT\n\n00:00:00.000 --> {end_ts}\n{text}\n"
        return PlainTextResponse(vtt, media_type="text/vtt")

    # Default: json
    resp = {"text": text}

    # Add usage if available
    resp["usage"] = {
        "type": "duration",
        "seconds": int(audio_duration) + 1,
    }

    return resp


@app.post("/v1/audio/translations")
async def openai_translate(request: Request):
    """OpenAI-compatible translation endpoint.

    Same as transcriptions but sets language=en for translation to English.
    """
    from macaw_asr.audio.decode import decode_audio
    from macaw_asr.audio.preprocessing import resample

    form = await request.form()

    file_field = form.get("file")
    if file_field is None:
        raise HTTPException(400, detail="missing 'file' field")

    if hasattr(file_field, "read"):
        file_bytes = await file_field.read()
        filename = getattr(file_field, "filename", "audio.wav")
    else:
        raise HTTPException(400, detail="'file' must be a file upload")

    if len(file_bytes) == 0:
        raise HTTPException(400, detail="empty audio file")

    model_name = str(form.get("model", "whisper-1"))
    internal_model = _OPENAI_MODEL_MAP.get(model_name, model_name)

    try:
        audio_f32, file_sr = decode_audio(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # Translation: force English output
    config = _resolve_config(internal_model, "en")
    input_sr = config.audio.input_sample_rate
    if file_sr != input_sr:
        audio_f32 = resample(audio_f32, file_sr, input_sr)

    pcm_bytes = (audio_f32 * 32767).astype(np.int16).tobytes()

    try:
        engine = await _scheduler.get_runner(config)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    text = await engine.transcribe(pcm_bytes)

    response_format = str(form.get("response_format", "json"))
    if response_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text)

    return {"text": text}


@app.get("/v1/models")
async def openai_list_models():
    """OpenAI-compatible list models endpoint."""
    from macaw_asr.models.base import list_models

    models = list_models()
    model_list = []
    for name in models:
        model_list.append({
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "macaw-asr",
        })
    # Add OpenAI aliases
    for alias in _OPENAI_MODEL_MAP:
        model_list.append({
            "id": alias,
            "object": "model",
            "created": 0,
            "owned_by": "macaw-asr",
        })

    return {"object": "list", "data": model_list}


# Need numpy for PCM conversion in openai endpoints
import numpy as np
