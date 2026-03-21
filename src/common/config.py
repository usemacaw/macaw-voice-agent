"""
Configuration for STT and TTS microservices.

Subset of ai-agent config — only what STT/TTS servers need.
All values from environment variables, exposed as frozen dataclasses.
"""

import os
from dataclasses import dataclass
from typing import Union

from dotenv import load_dotenv

load_dotenv()

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _validate_range(
    env_var: str,
    value: Union[int, float],
    min_val: Union[int, float, None] = None,
    max_val: Union[int, float, None] = None,
) -> Union[int, float]:
    if min_val is not None and value < min_val:
        raise ValueError(f"{env_var}={value} invalid: must be >= {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{env_var}={value} invalid: must be <= {max_val}")
    return value


def _env_int(env_var: str, default: str, min_val: int = None, max_val: int = None) -> int:
    raw = os.getenv(env_var, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{env_var}='{raw}' invalid: must be integer")
    return int(_validate_range(env_var, value, min_val, max_val))


def _env_float(env_var: str, default: str, min_val: float = None, max_val: float = None) -> float:
    raw = os.getenv(env_var, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{env_var}='{raw}' invalid: must be number")
    return float(_validate_range(env_var, value, min_val, max_val))


# =============================================================================
# LOGGING
# =============================================================================

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if _log_level not in _VALID_LOG_LEVELS:
    raise ValueError(
        f"LOG_LEVEL='{_log_level}' invalid. "
        f"Valid: {', '.join(sorted(_VALID_LOG_LEVELS))}"
    )

LOG_CONFIG = {
    "level": _log_level,
}


# =============================================================================
# Typed configuration (frozen dataclasses)
# =============================================================================


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 8000
    channels: int = 1
    sample_width: int = 2


@dataclass(frozen=True)
class STTConfig:
    provider: str = "mock"
    language: str = "pt"


@dataclass(frozen=True)
class TTSConfig:
    provider: str = "mock"
    language: str = "pt"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"


AUDIO = AudioConfig(
    sample_rate=_env_int("AUDIO_SAMPLE_RATE", "8000", min_val=1),
    channels=_env_int("AUDIO_CHANNELS", "1", min_val=1),
    sample_width=_env_int("AUDIO_SAMPLE_WIDTH", "2", min_val=1),
)

STT = STTConfig(
    provider=os.getenv("STT_PROVIDER", "mock"),
    language=os.getenv("STT_LANGUAGE", "pt"),
)

TTS = TTSConfig(
    provider=os.getenv("TTS_PROVIDER", "mock"),
    language=os.getenv("TTS_LANGUAGE", os.getenv("TTS_LANG", "pt")),
)


# Legacy dict access — kept for backward compatibility with providers
# that use AUDIO_CONFIG["sample_rate"] etc. Will be removed once all
# consumers migrate to AUDIO.sample_rate.
AUDIO_CONFIG = {
    "sample_rate": AUDIO.sample_rate,
    "channels": AUDIO.channels,
    "sample_width": AUDIO.sample_width,
}

STT_CONFIG = {
    "provider": STT.provider,
    "language": STT.language,
}

TTS_CONFIG = {
    "provider": TTS.provider,
    "language": TTS.language,
}

LLM = LLMConfig(
    provider=os.getenv("LLM_BACKEND_PROVIDER", "vllm"),
)

LLM_CONFIG = {
    "provider": LLM.provider,
}
