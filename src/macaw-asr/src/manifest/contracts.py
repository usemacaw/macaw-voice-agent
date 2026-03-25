"""Rigid contracts for model management layer.

IModelPaths    — filesystem layout for model storage
IModelRegistry — download, cache, resolve, list, remove models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from macaw_asr.api.types import ModelInfo


class IModelPaths(ABC):
    """Contract: model storage filesystem layout."""

    @property
    @abstractmethod
    def home(self) -> Path: ...

    @property
    @abstractmethod
    def models_dir(self) -> Path: ...

    @abstractmethod
    def model_dir(self, model_id: str) -> Path: ...

    @abstractmethod
    def model_exists(self, model_id: str) -> bool: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...


class IModelRegistry(ABC):
    """Contract: model download, cache, resolution."""

    @abstractmethod
    def resolve(self, model_id: str) -> str:
        """Resolve model ID to local path. Raises FileNotFoundError if not found."""

    @abstractmethod
    def pull(self, model_id: str, progress_fn: Callable | None = None) -> str:
        """Download model. Returns local path."""

    @abstractmethod
    def remove(self, model_id: str) -> bool:
        """Remove model. Returns True if removed."""

    @abstractmethod
    def list(self) -> list[ModelInfo]:
        """List all locally available models."""
