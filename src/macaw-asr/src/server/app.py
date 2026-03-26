"""FastAPI application — thin composition layer.

Fixes from audit:
- #3: No global mutable state. Uses app.state for dependency injection.
- No circular imports between app.py and routes.
- _resolve_config guards against None default_config.
- Removed "parakeet" from SHORT_NAMES (no implementation).
"""

from __future__ import annotations

import logging
import time as _time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import macaw_asr
from macaw_asr.config import EngineConfig
from macaw_asr.models.registry import list_names, is_known
from macaw_asr.server.scheduler import Scheduler

logger = logging.getLogger("macaw-asr.server")


# ==================== Lifespan ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = EngineConfig.from_env()
    scheduler = Scheduler()
    await scheduler.start()

    # Store in app.state — no globals (fix #3)
    app.state.scheduler = scheduler
    app.state.default_config = config

    logger.info("macaw-asr server started (model=%s)", config.model_name)
    yield
    await scheduler.stop()
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


# Error handlers — Ollama format: {"error": "message"}
@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exc(request: Request, exc: Exception):
    logger.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ==================== Helpers (used by route modules via request.app.state) ====================


def get_scheduler(request: Request) -> Scheduler:
    """Get scheduler from app.state. No globals."""
    return request.app.state.scheduler


def get_default_config(request: Request) -> EngineConfig:
    """Get default config from app.state."""
    return request.app.state.default_config


def resolve_config(model: str, language: str, default_config: EngineConfig) -> EngineConfig:
    """Build config from request params + defaults. Guards against None."""
    if not model:
        if not language:
            return default_config
        model = default_config.model_name

    kwargs: dict[str, Any] = {}
    if model:
        if is_known(model):
            from macaw_asr.models.registry import get as get_meta
            meta = get_meta(model)
            kwargs["model_name"] = model
            kwargs["model_id"] = meta.model_id if meta else model
        elif "/" in model:
            kwargs["model_id"] = model
            name = model.split("/")[-1].lower().replace("-", "_")
            for known in list_names():
                if known.replace("-", "_") in name:
                    kwargs["model_name"] = known
                    break
            if "model_name" not in kwargs:
                kwargs["model_name"] = model
        else:
            kwargs["model_name"] = model
            kwargs["model_id"] = model
    if language:
        kwargs["language"] = language

    return default_config.replace(**kwargs)


# ==================== Wire Routers ====================

from macaw_asr.server.routes.system import router as system_router
from macaw_asr.server.routes.audio import router as audio_router
from macaw_asr.server.routes.models import router as models_router

app.include_router(system_router)
app.include_router(audio_router)
app.include_router(models_router)
