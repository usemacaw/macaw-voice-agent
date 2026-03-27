"""Centralized model registry — single source of truth.

Fixes shotgun surgery: adding a model requires editing ONLY this file
+ creating the model module. No other files need modification.

Every model registers its metadata here. Server, CLI, routes, and factory
all consult this registry — they never hardcode model names.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Type

logger = logging.getLogger("macaw-asr.models.registry")


@dataclass(frozen=True)
class ModelMeta:
    """Metadata for a registered model. Single source of truth."""
    name: str              # Short name: "qwen", "whisper-tiny"
    model_id: str          # HuggingFace ID: "openai/whisper-tiny"
    module: str            # Python module: "macaw_asr.models.whisper"
    family: str            # Model family: "whisper", "qwen"
    param_size: str        # Parameter count: "39M", "0.6B"
    dtype: str             # Default dtype: "float16", "bfloat16"
    supports_streaming: bool = False
    supports_cuda_graphs: bool = False


@dataclass(frozen=True)
class FamilyDeps:
    """Dependency info for a model family."""
    pip_extra: str                   # e.g. "whisper"
    probe_packages: tuple[str, ...]  # importable names to check via find_spec
    install_cmd: str                 # e.g. 'pip install "macaw-asr[whisper]"'


# ==================== Central Registry ====================

_MODELS: dict[str, ModelMeta] = {}

_FAMILY_DEPS: dict[str, FamilyDeps] = {
    "whisper": FamilyDeps(
        "whisper", ("torch", "transformers"),
        'pip install "macaw-asr[whisper]"',
    ),
    "qwen": FamilyDeps(
        "qwen", ("torch", "transformers", "qwen_asr"),
        'pip install "macaw-asr[qwen]"',
    ),
    "faster-whisper": FamilyDeps(
        "faster-whisper", ("faster_whisper",),
        'pip install "macaw-asr[faster-whisper]"',
    ),
    "parakeet": FamilyDeps(
        "parakeet", ("torch", "nemo"),
        'pip install "macaw-asr[parakeet]"',
    ),
    "mock": FamilyDeps("", (), ""),
}


def register(meta: ModelMeta) -> None:
    """Register a model's metadata."""
    _MODELS[meta.name] = meta
    logger.debug("Model registered: %s (%s)", meta.name, meta.model_id)


def get(name: str) -> ModelMeta | None:
    """Get model metadata by name."""
    return _MODELS.get(name)


def list_all() -> list[ModelMeta]:
    """List all registered models."""
    return list(_MODELS.values())


def list_names() -> list[str]:
    """List all registered model names."""
    return sorted(_MODELS.keys())


def is_known(name: str) -> bool:
    """Check if a model name is registered."""
    return name in _MODELS


# ==================== Dependency Management ====================


def get_family_deps(family: str) -> FamilyDeps | None:
    """Get dependency info for a model family."""
    return _FAMILY_DEPS.get(family)


def check_deps(family: str) -> tuple[bool, list[str]]:
    """Check if dependencies for a model family are installed.

    Uses importlib.util.find_spec — no heavy imports (torch, etc.).
    Returns (all_ok, list_of_missing_package_names).
    """
    deps = _FAMILY_DEPS.get(family)
    if deps is None or not deps.probe_packages:
        return True, []
    missing = [p for p in deps.probe_packages if importlib.util.find_spec(p) is None]
    return len(missing) == 0, missing


def resolve_name(name: str) -> ModelMeta | None:
    """Resolve a short name or HuggingFace ID to ModelMeta.

    Tries: exact registry match → model_id match.
    """
    if name in _MODELS:
        return _MODELS[name]
    for meta in _MODELS.values():
        if meta.model_id == name:
            return meta
    return None


# ==================== Built-in Model Registrations ====================

# Qwen
register(ModelMeta(
    name="qwen", model_id="Qwen/Qwen3-ASR-0.6B",
    module="macaw_asr.models.qwen", family="qwen",
    param_size="0.6B", dtype="bfloat16",
    supports_streaming=True, supports_cuda_graphs=True,
))

# Whisper variants
for _name, _id, _size in [
    ("whisper-tiny", "openai/whisper-tiny", "39M"),
    ("whisper-small", "openai/whisper-small", "244M"),
    ("whisper-medium", "openai/whisper-medium", "769M"),
    ("whisper-large", "openai/whisper-large-v3", "1.5B"),
]:
    register(ModelMeta(
        name=_name, model_id=_id,
        module="macaw_asr.models.whisper", family="whisper",
        param_size=_size, dtype="float16",
    ))

# Faster-Whisper variants (CTranslate2 backend — no PyTorch dependency)
for _name, _id, _size in [
    ("faster-whisper-tiny", "openai/whisper-tiny", "39M"),
    ("faster-whisper-small", "openai/whisper-small", "244M"),
    ("faster-whisper-medium", "openai/whisper-medium", "769M"),
    ("faster-whisper-large", "openai/whisper-large-v3", "1.5B"),
]:
    register(ModelMeta(
        name=_name, model_id=_id,
        module="macaw_asr.models.faster_whisper", family="faster-whisper",
        param_size=_size, dtype="float16",
    ))

# NeMo variants (Parakeet, FastConformer — all use same backend)
register(ModelMeta(
    name="parakeet", model_id="nvidia/parakeet-tdt-0.6b-v3",
    module="macaw_asr.models.parakeet", family="parakeet",
    param_size="0.6B", dtype="float32",
))
register(ModelMeta(
    name="parakeet-tdt", model_id="nvidia/parakeet-tdt-0.6b-v3",
    module="macaw_asr.models.parakeet", family="parakeet",
    param_size="0.6B", dtype="float32",
))
register(ModelMeta(
    name="parakeet-ctc", model_id="nvidia/parakeet-ctc-1.1b",
    module="macaw_asr.models.parakeet", family="parakeet",
    param_size="1.1B", dtype="float32",
))
register(ModelMeta(
    name="fastconformer-pt", model_id="nvidia/stt_pt_fastconformer_hybrid_large_pc",
    module="macaw_asr.models.parakeet", family="parakeet",
    param_size="115M", dtype="float32",
))
register(ModelMeta(
    name="canary", model_id="nvidia/canary-1b-v2",
    module="macaw_asr.models.parakeet", family="parakeet",
    param_size="1.0B", dtype="float32",
))

# Mock (for testing)
register(ModelMeta(
    name="mock", model_id="mock",
    module="macaw_asr.models.mock", family="mock",
    param_size="0", dtype="float32",
    supports_streaming=True,
))
