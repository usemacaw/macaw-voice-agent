"""Rigid contracts for ASR model implementations.

Every model MUST implement these interfaces. No optional methods.
No defaults that silently do nothing. If a model doesn't support
a capability, it must explicitly raise NotImplementedError.

Design:
    IModelLoader    — load/unload weights (SRP: lifecycle only)
    IPreprocessor   — audio → model-ready tensors (SRP: preprocessing only)
    IDecoder        — tensors → text (SRP: inference only)
    IStreamDecoder  — tensors → token-by-token generator (SRP: streaming only)
    IASRModel       — composed facade that engine interacts with (Facade pattern)

SOLID:
    S — Each interface has one responsibility
    O — New models extend via new implementations, not modifying existing
    L — Any IASRModel can replace any other (engine doesn't know which model)
    I — Clients depend only on the interfaces they use
    D — Engine depends on abstractions (IASRModel), not concrete models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

import numpy as np

from macaw_asr.config import EngineConfig
from macaw_asr.decode.strategies import DecodeStrategy
from macaw_asr.models.types import ModelOutput


class IModelLoader(ABC):
    """Contract: model weight lifecycle management."""

    @abstractmethod
    def load(self, config: EngineConfig) -> None:
        """Load model weights into memory. Must be idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Free all model resources. Must be safe to call multiple times."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model weights are currently in memory."""


class IPreprocessor(ABC):
    """Contract: convert raw audio to model-ready input tensors."""

    @abstractmethod
    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        """Convert float32 audio at model_sample_rate to model-specific tensors.

        This is the expensive CPU/GPU step (mel spectrogram, feature extraction).
        Must be thread-safe for background precomputation.
        """

    def fast_finish_inputs(
        self, audio: np.ndarray, cached_inputs: Any,
        cached_audio_len: int, prefix: str = "",
    ) -> Any | None:
        """Fast-path input preparation from cached computation.

        Returns None if not supported. Engine falls back to full prepare_inputs().
        """
        return None


class IDecoder(ABC):
    """Contract: run inference on prepared inputs, return full text."""

    @abstractmethod
    def decode(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        """Run inference. Returns complete transcription."""


class IStreamDecoder(ABC):
    """Contract: run inference yielding tokens incrementally."""

    @abstractmethod
    def decode_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        """Yield (delta_text, is_done, final_output) per token.

        is_done=True on last yield, which includes ModelOutput.
        """


class IASRModel(ABC):
    """Facade contract: what the engine sees.

    Composes IModelLoader + IPreprocessor + IDecoder + IStreamDecoder.
    Every model implementation MUST satisfy this full interface.
    """

    # ==================== Lifecycle ====================

    @abstractmethod
    def load(self, config: EngineConfig) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    # ==================== Properties ====================

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """EOS token ID from tokenizer. Available after load()."""

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether model supports background precompute during audio push."""

    @property
    @abstractmethod
    def supports_cuda_graphs(self) -> bool:
        """Whether model can use CUDA graph capture."""

    # ==================== Inference ====================

    @abstractmethod
    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any: ...

    @abstractmethod
    def generate(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput: ...

    @abstractmethod
    def generate_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]: ...

    def fast_finish_inputs(
        self, audio: np.ndarray, cached_inputs: Any,
        cached_audio_len: int, prefix: str = "",
    ) -> Any | None:
        """Optional optimization. Default: not supported."""
        return None

    def compilable_module(self) -> Any:
        """Return nn.Module for torch.compile. Default: None (not supported)."""
        return None

    def apply_compiled_module(self, compiled_module: Any) -> None:
        """Replace internal module with compiled version. Default: no-op."""

    def warmup(self, config: EngineConfig | None = None) -> None:
        """Warmup to compile CUDA kernels. Default: no-op."""
