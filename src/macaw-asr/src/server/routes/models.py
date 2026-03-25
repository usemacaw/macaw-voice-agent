"""Model management routes. Uses request.app.state — no globals."""

from __future__ import annotations

import json
import re
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from macaw_asr.api.types import (
    DeleteRequest, PsResponse, PullRequest,
    RunningModel, ShowRequest, ShowResponse,
)
from macaw_asr.config import EngineConfig
from macaw_asr.models.registry import list_all, get as get_meta, is_known

logger = logging.getLogger("macaw-asr.server.routes.models")

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models():
    """List real implemented models from centralized registry."""
    data = []
    for meta in list_all():
        if meta.name == "mock":
            continue
        data.append({
            "id": meta.name, "object": "model", "created": 0,
            "owned_by": "macaw-asr", "model_id": meta.model_id,
            "family": meta.family, "parameters": meta.param_size,
        })
    return {"object": "list", "data": data}


@router.post("/api/show")
async def show(req: ShowRequest, request: Request):
    scheduler = request.app.state.scheduler
    default_config = request.app.state.default_config

    if not req.model:
        raise HTTPException(400, detail="missing model field")

    # Check manifest
    for m in scheduler.registry.list():
        if m.name == req.model or m.model_id == req.model:
            return ShowResponse(
                model_info={"general.architecture": m.family or "unknown"},
                details={"family": m.family or "unknown", "parameter_size": m.parameters or "unknown", "quantization_level": "BF16"},
            )

    # Check loaded models (via encapsulated method)
    ref = scheduler.get_loaded_ref(req.model)
    if ref:
        _, cfg = ref
        return _show_config(cfg)

    # Check centralized registry
    meta = get_meta(req.model)
    if meta:
        return ShowResponse(
            model_info={
                "general.architecture": meta.family,
                "general.model_id": meta.model_id,
                "general.param_size": meta.param_size,
                "general.dtype": meta.dtype,
                "status": "not loaded",
            },
            details={"family": meta.family, "parameter_size": meta.param_size, "quantization_level": meta.dtype.upper()},
        )

    # Check default config
    if default_config and req.model in (default_config.model_name, default_config.model_id):
        return _show_config(default_config, loaded=False)

    raise HTTPException(404, detail=f"model '{req.model}' not found")


@router.get("/api/ps")
async def list_running(request: Request):
    scheduler = request.app.state.scheduler
    models = []
    for model_id, cfg in scheduler.iter_loaded():
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        models.append(RunningModel(name=short, model=model_id, size=0, size_vram=0))
    return PsResponse(models=models)


@router.post("/api/pull")
async def pull(req: PullRequest, request: Request):
    scheduler = request.app.state.scheduler
    if not req.model:
        raise HTTPException(400, detail="missing model field")
    if req.stream:
        async def stream():
            yield json.dumps({"status": "pulling model"}) + "\n"
            try:
                scheduler.registry.pull(req.model)
                yield json.dumps({"status": "success"}) + "\n"
            except Exception as e:
                logger.error("Pull failed: %s", e, exc_info=True)
                yield json.dumps({"error": str(e)}) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson")
    try:
        scheduler.registry.pull(req.model)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.delete("/api/delete")
async def delete(req: DeleteRequest, request: Request):
    scheduler = request.app.state.scheduler
    if not req.model:
        raise HTTPException(400, detail="missing model field")
    await scheduler.unload(req.model)
    if not scheduler.registry.remove(req.model):
        raise HTTPException(404, detail=f"model '{req.model}' not found")
    return {"status": "success"}


def _show_config(cfg: EngineConfig, loaded: bool = True) -> ShowResponse:
    match = re.search(r'(\d+\.?\d*[BbMm])', cfg.model_id)
    param = match.group(1).upper() if match else "unknown"
    info = {
        "general.architecture": cfg.model_name,
        "general.model_id": cfg.model_id,
        "general.device": cfg.device,
        "general.dtype": cfg.dtype,
        "general.language": cfg.language,
        "general.max_new_tokens": cfg.streaming.max_new_tokens,
    }
    if not loaded:
        info["status"] = "not loaded (will load on first request)"
    return ShowResponse(model_info=info, details={"family": cfg.model_name, "parameter_size": param, "quantization_level": cfg.dtype.upper()})
