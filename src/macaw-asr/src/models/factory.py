"""Model Factory — creates IASRModel instances from the centralized registry.

No hardcoded model names. All model info comes from models/registry.py.
"""

from __future__ import annotations

import importlib
import logging
from typing import Type

from macaw_asr.models.contracts import IASRModel
from macaw_asr.models.registry import get as get_meta, list_names, list_all

logger = logging.getLogger("macaw-asr.models.factory")

# Class registry: populated by each model's __init__.py via register_model()
_CLASS_REGISTRY: dict[str, Type[IASRModel]] = {}


def register_model(name: str, cls: Type[IASRModel]) -> None:
    """Register a model class. Called by model module __init__.py."""
    _CLASS_REGISTRY[name] = cls
    logger.debug("Model class registered: %s -> %s", name, cls.__name__)


class ModelFactory:
    """Factory Pattern: creates IASRModel instances by name.

    Uses centralized registry for metadata, lazy-imports modules.
    """

    @staticmethod
    def create(name: str) -> IASRModel:
        """Create model instance by name. Lazy-imports the module."""
        # Lazy import if not yet registered
        if name not in _CLASS_REGISTRY:
            meta = get_meta(name)
            if meta is None:
                available = ", ".join(list_names())
                raise ValueError(f"Model '{name}' not registered. Available: [{available}].")
            try:
                importlib.import_module(meta.module)
            except ImportError as e:
                from macaw_asr.models.registry import get_family_deps
                deps_info = get_family_deps(meta.family)
                if deps_info and deps_info.install_cmd:
                    raise ValueError(
                        f"Model '{name}' requires additional dependencies.\n"
                        f"Install with: {deps_info.install_cmd}"
                    ) from e
                raise ValueError(f"Model '{name}' requires dependencies: {e}") from e

        if name not in _CLASS_REGISTRY:
            raise ValueError(f"Model '{name}' module loaded but class not registered.")

        return _CLASS_REGISTRY[name]()

    @staticmethod
    def list_models() -> list[str]:
        return list_names()
