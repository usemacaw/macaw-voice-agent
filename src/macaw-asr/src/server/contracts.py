"""Rigid contracts for the server/scheduler layer.

IScheduler — model loading, caching, eviction
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from macaw_asr.config import EngineConfig
from macaw_asr.runner.contracts import IEngine


class IScheduler(ABC):
    """Contract: model scheduling and lifecycle."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def get_runner(self, config: EngineConfig) -> IEngine: ...

    @abstractmethod
    def list_loaded(self) -> list[str]: ...

    @abstractmethod
    async def unload(self, model_id: str) -> bool: ...
