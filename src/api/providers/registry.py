"""
Generic provider registry with lazy auto-import.

Centralizes the duplicated register/create pattern used by ASR, LLM, and TTS
providers. Each provider type creates a ProviderRegistry instance and exposes
its .register and .create methods.

DESIGN NOTE — Auto-discovery pattern:
    Provider implementations call register() at module level (e.g.,
    register_asr_provider("remote", RemoteASR) at bottom of asr_remote.py).
    The registry only imports the module when create() is called for that name.
    This means:
    1. Importing the ABC module (e.g., providers.asr) is cheap — no heavy deps.
    2. Provider modules are only loaded when actually requested via config.
    3. Registration happens as a side-effect of import — the KNOWN_MODULES dict
       maps provider names to module paths for this lazy import.
"""

from __future__ import annotations

import importlib
import logging
from typing import Generic, TypeVar, Type

logger = logging.getLogger("open-voice-api.registry")

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    """Generic registry for provider ABC implementations."""

    def __init__(self, provider_type: str, known_modules: dict[str, str]):
        self._providers: dict[str, Type[T]] = {}
        self._known_modules = known_modules
        self._provider_type = provider_type

    def register(self, name: str, cls: Type[T]) -> None:
        """Register a provider class by name."""
        self._providers[name] = cls
        logger.debug(f"Registered {self._provider_type} provider: {name}")

    def create(self, name: str) -> T:
        """Create a provider instance by name, auto-importing if needed."""
        if name not in self._providers and name in self._known_modules:
            importlib.import_module(self._known_modules[name])
        if name not in self._providers:
            raise ValueError(
                f"Unknown {self._provider_type} provider: {name}. "
                f"Available: {list(self._providers.keys())}"
            )
        return self._providers[name]()
