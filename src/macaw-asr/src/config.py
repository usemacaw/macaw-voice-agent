"""Frozen dataclass configuration for macaw-asr.

All configuration is explicit via dataclasses. No global state.
`from_env()` classmethod provided as convenience for server integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


LANGUAGE_MAP = {
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
}


def make_devices(n: int) -> tuple[str, ...]:
    """Generate N CUDA device strings: (cuda:0, cuda:1, ..., cuda:N-1)."""
    if n <= 0:
        return ()
    return tuple(f"cuda:{i}" for i in range(n))


def _parse_devices(value: str) -> tuple[str, ...]:
    """Parse MACAW_ASR_DEVICES value.

    Accepts:
        ""          → ()           (no multi-GPU)
        "2"         → ("cuda:0", "cuda:1")
        "4"         → ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
        "cuda:0,cuda:2" → ("cuda:0", "cuda:2")  (explicit list)
    """
    value = value.strip()
    if not value:
        return ()
    if value.isdigit():
        return make_devices(int(value))
    return tuple(d.strip() for d in value.split(",") if d.strip())


@dataclass(frozen=True)
class AudioConfig:
    """Audio format configuration."""

    input_sample_rate: int = 8000
    """Sample rate of incoming audio (Hz). Default: 8kHz (telephony)."""

    model_sample_rate: int = 16000
    """Sample rate the model expects (Hz). Default: 16kHz."""

    sample_width: int = 2
    """Bytes per sample. Default: 2 (PCM16)."""

    @classmethod
    def from_env(cls) -> AudioConfig:
        return cls(
            input_sample_rate=int(os.getenv("MACAW_ASR_INPUT_RATE", "8000")),
            model_sample_rate=int(os.getenv("MACAW_ASR_MODEL_RATE", "16000")),
        )


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming inference configuration."""

    chunk_trigger_sec: float = 1.0
    """Seconds of audio to accumulate before triggering background compute."""

    max_new_tokens: int = 32
    """Maximum tokens to generate per decode step."""

    enable_background_compute: bool = True
    """Run prepare_inputs() in background during speech."""

    repetition_window: int = 3
    """Stop decode if last N tokens are identical."""

    @classmethod
    def from_env(cls) -> StreamingConfig:
        return cls(
            chunk_trigger_sec=float(os.getenv("MACAW_ASR_CHUNK_SIZE_SEC", "1.0")),
            max_new_tokens=int(os.getenv("MACAW_ASR_MAX_NEW_TOKENS", "32")),
            enable_background_compute=os.getenv(
                "MACAW_ASR_BACKGROUND_COMPUTE", "true"
            ).lower() == "true",
        )


@dataclass(frozen=True)
class EngineConfig:
    """Root configuration for ASREngine.

    Composed of sub-configs. Immutable after creation.
    Thread-safe by design (frozen).
    """

    model_name: str = "qwen"
    """Model registry key (e.g. 'qwen', 'whisper', 'mock')."""

    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    """HuggingFace model ID or local path."""

    device: str = "cuda:0"
    """Device for inference (e.g. 'cuda:0', 'cpu')."""

    devices: tuple[str, ...] = ()
    """Multi-GPU: replicate model across these devices for higher throughput.
    Empty = use single `device`. Set via MACAW_ASR_DEVICES=2 (uses cuda:0,cuda:1)."""

    dtype: str = "bfloat16"
    """Model dtype (e.g. 'bfloat16', 'float16', 'float32')."""

    language: str = "pt"
    """ISO-639-1 language code."""

    max_inference_workers: int = 2
    """Max concurrent inference threads."""

    enable_compile: bool = False
    """Apply torch.compile to the model after loading (opt-in)."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)

    @property
    def language_name(self) -> str:
        """Full language name (e.g. 'Portuguese') for model prompts."""
        return LANGUAGE_MAP.get(self.language, self.language.title())

    def for_device(self, device: str) -> EngineConfig:
        """Create a copy pinned to a single device (used by scheduler for replicas)."""
        from dataclasses import asdict
        d = asdict(self)
        d["device"] = device
        d["devices"] = ()
        audio = d.pop("audio")
        streaming = d.pop("streaming")
        return EngineConfig(
            **{k: v for k, v in d.items() if k in EngineConfig.__dataclass_fields__},
            audio=AudioConfig(**audio),
            streaming=StreamingConfig(**streaming),
        )

    @classmethod
    def from_env(cls) -> EngineConfig:
        """Build config from environment variables.

        Convenience for gRPC server integration. Library users
        should prefer explicit config construction.
        """
        devices = _parse_devices(os.getenv("MACAW_ASR_DEVICES", ""))

        return cls(
            model_name=os.getenv("MACAW_ASR_MODEL", "qwen"),
            model_id=os.getenv(
                "QWEN_STT_MODEL",
                os.getenv("MACAW_ASR_MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
            ),
            device=os.getenv("QWEN_DEVICE", os.getenv("MACAW_ASR_DEVICE", "cuda:0")),
            devices=devices,
            dtype=os.getenv("MACAW_ASR_DTYPE", "bfloat16"),
            language=os.getenv("MACAW_ASR_LANGUAGE", "pt"),
            max_inference_workers=int(
                os.getenv("MACAW_ASR_INFERENCE_WORKERS", "2")
            ),
            audio=AudioConfig.from_env(),
            streaming=StreamingConfig.from_env(),
        )
