"""Model scheduler — load, cache, and evict models.

Equivalent to Ollama's server/sched.go. The scheduler is the single
coordination point for all inference requests:
- Loads models on demand
- Caches loaded models for reuse
- Evicts models under memory pressure or TTL

The server routes delegate to the scheduler, which provides
a running ASREngine for the requested model.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field

from macaw_asr.config import EngineConfig
from macaw_asr.manifest.registry import ModelRegistry
from macaw_asr.runner.engine import ASREngine

logger = logging.getLogger("macaw-asr.server.scheduler")

_DEFAULT_KEEP_ALIVE_SEC = 300  # 5 minutes


@dataclass
class _RunnerRef:
    """Reference to a loaded model runner."""

    engine: ASREngine
    model_id: str
    last_used: float = field(default_factory=_time.time)
    request_count: int = 0


class Scheduler:
    """Manages loaded models and schedules inference requests.

    Only one model loads at a time (GPU serialization).
    Multiple requests can use a loaded model concurrently.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        keep_alive_sec: float = _DEFAULT_KEEP_ALIVE_SEC,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._keep_alive_sec = keep_alive_sec
        self._loaded: dict[str, _RunnerRef] = {}
        self._load_lock = asyncio.Lock()
        self._eviction_task: asyncio.Task | None = None

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    async def start(self) -> None:
        """Start the scheduler (begins eviction loop)."""
        self._eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info("Scheduler started (keep_alive=%ds)", self._keep_alive_sec)

    async def stop(self) -> None:
        """Stop scheduler and unload all models."""
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass

        for model_id in list(self._loaded.keys()):
            await self._unload(model_id)

        logger.info("Scheduler stopped")

    async def get_runner(self, config: EngineConfig) -> ASREngine:
        """Get a running engine for the given config.

        Loads the model if not already cached.
        Updates last_used timestamp for eviction.
        """
        model_id = config.model_id

        # Fast path: already loaded
        ref = self._loaded.get(model_id)
        if ref is not None and ref.engine.is_started:
            ref.last_used = _time.time()
            ref.request_count += 1
            return ref.engine

        # Slow path: load model (one at a time)
        async with self._load_lock:
            # Double-check after acquiring lock
            ref = self._loaded.get(model_id)
            if ref is not None and ref.engine.is_started:
                ref.last_used = _time.time()
                ref.request_count += 1
                return ref.engine

            # Resolve model path (download if needed)
            # Skip resolve/pull for internal models (mock, etc.)
            from macaw_asr.models.base import _MODEL_REGISTRY, _KNOWN_MODULES
            is_internal = config.model_name in _MODEL_REGISTRY or config.model_name in _KNOWN_MODULES
            if not is_internal:
                try:
                    self._registry.resolve(config.model_id)
                except FileNotFoundError:
                    logger.info("Model not found locally, pulling: %s", config.model_id)
                    self._registry.pull(config.model_id)

            # Create and start engine
            engine = ASREngine(config)
            await engine.start()

            self._loaded[model_id] = _RunnerRef(
                engine=engine,
                model_id=model_id,
                request_count=1,
            )
            logger.info("Model loaded: %s", model_id)
            return engine

    def list_loaded(self) -> list[str]:
        """List currently loaded model IDs."""
        return [ref.model_id for ref in self._loaded.values()]

    async def unload(self, model_id: str) -> bool:
        """Explicitly unload a model. Returns True if was loaded."""
        return await self._unload(model_id)

    # ==================== Internal ====================

    async def _unload(self, model_id: str) -> bool:
        ref = self._loaded.pop(model_id, None)
        if ref is None:
            return False
        try:
            await ref.engine.stop()
            logger.info("Model unloaded: %s (served %d requests)", model_id, ref.request_count)
        except Exception as e:
            logger.warning("Error unloading model %s: %s", model_id, e)
        return True

    async def _eviction_loop(self) -> None:
        """Periodically evict models past their keep-alive TTL."""
        while True:
            await asyncio.sleep(30)  # Check every 30s
            now = _time.time()
            to_evict = [
                model_id
                for model_id, ref in self._loaded.items()
                if (now - ref.last_used) > self._keep_alive_sec
            ]
            for model_id in to_evict:
                logger.info("Evicting model (TTL expired): %s", model_id)
                await self._unload(model_id)
