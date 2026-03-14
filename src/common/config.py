"""
Configuration for STT and TTS microservices.

Subset of ai-agent config — only what STT/TTS servers need.
All values from environment variables.
"""

import os
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
# AUDIO
# =============================================================================

AUDIO_CONFIG = {
    "sample_rate": _env_int("AUDIO_SAMPLE_RATE", "8000", min_val=1),
    "channels": _env_int("AUDIO_CHANNELS", "1", min_val=1),
    "sample_width": _env_int("AUDIO_SAMPLE_WIDTH", "2", min_val=1),
}


# =============================================================================
# STT (Speech-to-Text)
# =============================================================================

STT_CONFIG = {
    "provider": os.getenv("STT_PROVIDER", "mock"),
    "language": os.getenv("STT_LANGUAGE", "pt"),
}


# =============================================================================
# TTS (Text-to-Speech)
# =============================================================================

TTS_CONFIG = {
    "provider": os.getenv("TTS_PROVIDER", "mock"),
    "language": os.getenv("TTS_LANGUAGE", os.getenv("TTS_LANG", "pt")),
}
