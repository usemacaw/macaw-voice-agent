"""OpenAI Audio API routes: /v1/audio/transcriptions, /v1/audio/translations.

Fixes from audit:
- #1/#2: Uses engine.transcribe_stream() — no access to engine._model or raw threads
- No globals imported — uses request.app.state
- Errors logged with exc_info=True
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from macaw_asr.audio.decode import decode_audio
from macaw_asr.audio.preprocessing import resample
from macaw_asr._executor import run_in_executor
from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.models.registry import get as get_meta, list_names

logger = logging.getLogger("macaw-asr.server.routes.audio")

router = APIRouter(prefix="/v1/audio", tags=["audio"])

# OpenAI model aliases → maps to server's default model
_OPENAI_ALIASES = {"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}


@router.post("/transcriptions")
async def transcribe(request: Request):
    """OpenAI-compatible: POST /v1/audio/transcriptions."""
    scheduler = request.app.state.scheduler
    default_config = request.app.state.default_config

    audio_f32, file_sr, duration, config, fmt, temp, stream = await _parse_audio(request, default_config)

    input_sr = config.audio.input_sample_rate
    if file_sr != input_sr:
        audio_f32 = resample(audio_f32, file_sr, input_sr)
    pcm = (audio_f32 * 32767).astype(np.int16).tobytes()

    try:
        engine = await scheduler.get_runner(config)
    except Exception as e:
        logger.error("Failed to get runner: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))

    if stream:
        return await _handle_stream(engine, config, audio_f32, file_sr, duration)

    text = await engine.transcribe(pcm)
    return _format(text, config, duration, fmt, temp)


@router.post("/translations")
async def translate(request: Request):
    """OpenAI-compatible: POST /v1/audio/translations (forces language=en)."""
    scheduler = request.app.state.scheduler
    default_config = request.app.state.default_config

    audio_f32, file_sr, duration, config, fmt, temp, stream = await _parse_audio(request, default_config, force_language="en")

    input_sr = config.audio.input_sample_rate
    if file_sr != input_sr:
        audio_f32 = resample(audio_f32, file_sr, input_sr)
    pcm = (audio_f32 * 32767).astype(np.int16).tobytes()

    try:
        engine = await scheduler.get_runner(config)
    except Exception as e:
        logger.error("Failed to get runner: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))

    text = await engine.transcribe(pcm)
    if fmt == "text":
        return PlainTextResponse(text)
    return {"text": text}


# ==================== Helpers ====================


async def _parse_audio(request, default_config, force_language=""):
    from macaw_asr.server.app import resolve_config

    form = await request.form()

    file_field = form.get("file")
    if file_field is None:
        raise HTTPException(400, detail="missing 'file' field")
    if not hasattr(file_field, "read"):
        raise HTTPException(400, detail="'file' must be a file upload")

    file_bytes = await file_field.read()
    filename = getattr(file_field, "filename", "audio.wav")
    if not file_bytes:
        raise HTTPException(400, detail="empty audio file")

    model = str(form.get("model", "whisper-1"))
    language = force_language or str(form.get("language", ""))
    fmt = str(form.get("response_format", "json"))
    temp = float(form.get("temperature", 0))
    stream = str(form.get("stream", "false")).lower() == "true"

    # Map OpenAI aliases to server's default model
    if model in _OPENAI_ALIASES:
        model = default_config.model_name

    try:
        audio_f32, sr = decode_audio(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    duration = len(audio_f32) / sr if sr > 0 else 0
    config = resolve_config(model, language, default_config)
    return audio_f32, sr, duration, config, fmt, temp, stream


def _format(text, config, duration, fmt, temp):
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        return {
            "task": "transcribe", "language": config.language,
            "duration": round(duration, 2), "text": text,
            "segments": [{"id": 0, "seek": 0, "start": 0.0, "end": round(duration, 2),
                          "text": text, "tokens": [], "temperature": temp,
                          "avg_logprob": 0.0, "compression_ratio": 1.0, "no_speech_prob": 0.0}],
            "usage": {"type": "duration", "seconds": int(duration) + 1},
        }
    if fmt == "srt":
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), duration % 60
        ts = f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
        return PlainTextResponse(f"1\n00:00:00,000 --> {ts}\n{text}\n")
    if fmt == "vtt":
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), duration % 60
        ts = f"{h:02d}:{m:02d}:{s:06.3f}"
        return PlainTextResponse(f"WEBVTT\n\n00:00:00.000 --> {ts}\n{text}\n", media_type="text/vtt")
    return {"text": text, "usage": {"type": "duration", "seconds": int(duration) + 1}}


async def _handle_stream(engine, config, audio_f32_input, file_sr, duration):
    """SSE streaming via asyncio.Queue — no raw threads, no event loop blocking."""
    model_rate = config.audio.model_sample_rate
    input_sr = config.audio.input_sample_rate
    audio_model = resample(audio_f32_input, input_sr, model_rate) if input_sr != model_rate else audio_f32_input

    async def sse():
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _run():
            """Runs in thread pool — feeds results into async queue."""
            try:
                gen = engine.transcribe_stream(audio_model)
                for delta, done, out in gen:
                    loop.call_soon_threadsafe(q.put_nowait, (delta, done, out))
            except Exception as e:
                logger.error("Stream decode failed", exc_info=True)
                loop.call_soon_threadsafe(q.put_nowait, ("__ERROR__", True, str(e)))

        loop.run_in_executor(None, _run)

        full = ""
        while True:
            try:
                delta, done, out = await asyncio.wait_for(q.get(), timeout=60)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'inference timeout'})}\n\n"
                break
            if delta == "__ERROR__":
                yield f"data: {json.dumps({'error': out})}\n\n"
                break
            if delta:
                full += delta
                yield f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': delta})}\n\n"
            if done:
                yield f"data: {json.dumps({'type': 'transcript.text.done', 'text': clean_asr_text(full), 'usage': {'type': 'duration', 'seconds': int(duration) + 1}})}\n\n"
                break

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
