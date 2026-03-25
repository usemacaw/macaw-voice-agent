"""Model scheduler — implements IScheduler contract.

SRP: model loading, caching, eviction. Does NOT serve HTTP.
Encapsulates loaded models — external access via methods only.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field

from macaw_asr.config import EngineConfig
from macaw_asr.manifest.contracts import IModelRegistry
from macaw_asr.manifest.registry import ModelRegistry
from macaw_asr.runner.contracts import IEngine
from macaw_asr.runner.engine import ASREngine
from macaw_asr.server.contracts import IScheduler

logger = logging.getLogger("macaw-asr.server.scheduler")

_DEFAULT_KEEP_ALIVE_SEC = 300


@dataclass
class _RunnerRef:
    engine: ASREngine
    model_id: str
    last_used: float = field(default_factory=_time.time)
    request_count: int = 0


class Scheduler(IScheduler):
    """Manages loaded models. Implements IScheduler.

    Encapsulation: _loaded is private. Access via methods only.
    """

    def __init__(
        self, registry: IModelRegistry | None = None,
        keep_alive_sec: float = _DEFAULT_KEEP_ALIVE_SEC,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._keep_alive_sec = keep_alive_sec
        self._loaded: dict[str, _RunnerRef] = {}
        self._load_lock = asyncio.Lock()
        self._eviction_task: asyncio.Task | None = None

    @property
    def registry(self) -> IModelRegistry:
        return self._registry

    async def start(self) -> None:
        self._eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info("Scheduler started (keep_alive=%ds)", self._keep_alive_sec)

    async def stop(self) -> None:
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
        for mid in list(self._loaded):
            await self._unload(mid)
        logger.info("Scheduler stopped")

    async def get_runner(self, config: EngineConfig) -> IEngine:
        model_id = config.model_id

        # Fast path: cached
        ref = self._loaded.get(model_id)
        if ref and ref.engine.is_started:
            ref.last_used = _time.time()
            ref.request_count += 1
            return ref.engine

        # Slow path: load
        async with self._load_lock:
            ref = self._loaded.get(model_id)
            if ref and ref.engine.is_started:
                ref.last_used = _time.time()
                ref.request_count += 1
                return ref.engine

            # Resolve/pull if not internal model
            from macaw_asr.models.registry import is_known
            is_internal = is_known(config.model_name)
            if not is_internal:
                try:
                    self._registry.resolve(config.model_id)
                except FileNotFoundError:
                    logger.info("Pulling: %s", config.model_id)
                    self._registry.pull(config.model_id)

            engine = ASREngine(config)
            await engine.start()
            self._loaded[model_id] = _RunnerRef(engine=engine, model_id=model_id, request_count=1)
            logger.info("Model loaded: %s", model_id)
            return engine

    def list_loaded(self) -> list[str]:
        return [ref.model_id for ref in self._loaded.values()]

    async def unload(self, model_id: str) -> bool:
        return await self._unload(model_id)

    # ==================== Encapsulated access for server ====================

    def get_loaded_ref(self, model_id: str) -> tuple[str, EngineConfig] | None:
        """Safe read access to loaded model info. No internal state exposed."""
        ref = self._loaded.get(model_id)
        if ref:
            return ref.model_id, ref.engine.config
        # Search by short name
        for mid, ref in self._loaded.items():
            short = mid.split("/")[-1] if "/" in mid else mid
            if model_id in (mid, short, ref.engine.config.model_name):
                return ref.model_id, ref.engine.config
        return None

    def iter_loaded(self):
        """Iterate loaded models safely. Yields (model_id, engine_config)."""
        for ref in self._loaded.values():
            yield ref.model_id, ref.engine.config

    def find_engine_for_session(self, session_id: str) -> IEngine | None:
        """Find which engine owns a session. Uses public API, no internal access."""
        for ref in self._loaded.values():
            if ref.engine.session_exists(session_id):
                return ref.engine
        return None

    # ==================== Internal ====================

    async def _unload(self, model_id: str) -> bool:
        ref = self._loaded.pop(model_id, None)
        if ref is None:
            return False
        try:
            await ref.engine.stop()
            logger.info("Unloaded: %s (served %d requests)", model_id, ref.request_count)
        except Exception as e:
            logger.warning("Error unloading %s: %s", model_id, e)
        return True

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = _time.time()
            to_evict = [mid for mid, ref in self._loaded.items() if (now - ref.last_used) > self._keep_alive_sec]
            for mid in to_evict:
                logger.info("Evicting (TTL): %s", mid)
                await self._unload(mid)
