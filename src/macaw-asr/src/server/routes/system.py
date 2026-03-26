"""System routes: /, /api/version, /api/metrics.

SRP: Health check, version, and system metrics.
"""

from __future__ import annotations

import time as _time

from fastapi import APIRouter, Request

import macaw_asr
from macaw_asr.api.types import VersionResponse

router = APIRouter(tags=["system"])


@router.get("/")
@router.head("/")
async def health():
    return "macaw-asr is running"


@router.get("/api/version")
async def version():
    return VersionResponse(version=macaw_asr.__version__)


@router.get("/api/metrics")
async def metrics(request: Request):
    """Aggregated performance metrics for all loaded models."""
    start_time = getattr(request.app.state, "start_time", _time.time())
    scheduler = getattr(request.app.state, "scheduler", None)
    collector = getattr(request.app.state, "metrics", None)

    # Models loaded
    models_loaded = []
    if scheduler:
        for model_id, config in scheduler.iter_loaded():
            ref = scheduler.get_loaded_ref(model_id)
            ref_data = {
                "model_id": model_id,
                "model_name": config.model_name,
                "device": config.device,
            }
            # Add startup timings and request count from _RunnerRef
            loaded = scheduler._loaded.get(model_id)
            if loaded:
                ref_data["replicas"] = len(loaded.engines)
                ref_data["request_count"] = loaded.request_count
                ref_data["startup_ms"] = loaded.engine.startup_timings
            models_loaded.append(ref_data)

    # Request stats
    request_stats = collector.get_stats() if collector else {"total": 0, "by_model": {}}

    # System info
    system = {"version": macaw_asr.__version__}
    try:
        import torch
        if torch.cuda.is_available():
            system["gpu_count"] = torch.cuda.device_count()
            system["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass

    return {
        "uptime_s": round(_time.time() - start_time, 1),
        "models_loaded": models_loaded,
        "requests": request_stats,
        "system": system,
    }
