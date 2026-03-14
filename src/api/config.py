"""
Environment-based configuration for OpenVoiceAPI server.

All settings loaded from env vars with sensible defaults.
Frozen dataclasses provide type-safe access; legacy dicts remain for
backward compatibility with providers not yet migrated.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(var: str, default: int, min_val: int = 0, max_val: int = 100000) -> int:
    raw = os.getenv(var, str(default))
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {var}={raw!r} is not a valid integer") from None
    if val < min_val or val > max_val:
        raise ValueError(f"{var}={val} out of range [{min_val}, {max_val}]")
    return val


def _env_float(var: str, default: float, min_val: float = 0.0, max_val: float = 100000.0) -> float:
    raw = os.getenv(var, str(default))
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {var}={raw!r} is not a valid number") from None
    if val < min_val or val > max_val:
        raise ValueError(f"{var}={val} out of range [{min_val}, {max_val}]")
    return val


WS_CONFIG = {
    "host": os.getenv("WS_HOST", "0.0.0.0"),
    "port": _env_int("WS_PORT", 8765, 0, 65535),
    "path": os.getenv("WS_PATH", "/v1/realtime"),
    "api_key": os.getenv("REALTIME_API_KEY", ""),
    "max_connections": _env_int("MAX_CONNECTIONS", 10, 1, 1000),
}

ASR_CONFIG = {
    "provider": os.getenv("ASR_PROVIDER", "remote"),
    "remote_target": os.getenv("ASR_REMOTE_TARGET", "localhost:50060"),
    "language": os.getenv("ASR_LANGUAGE", "pt"),
    "remote_timeout": _env_float("ASR_REMOTE_TIMEOUT", 30.0, 1.0, 300.0),
    # Streaming disabled by default: the STT proto AudioChunk has no language field,
    # so Qwen3-ASR defaults to Chinese. Batch TranscribeRequest has language field.
    "remote_streaming": os.getenv("ASR_REMOTE_STREAMING", "false").lower() == "true",
}

TTS_CONFIG = {
    "provider": os.getenv("TTS_PROVIDER", "remote"),
    "remote_target": os.getenv("TTS_REMOTE_TARGET", "localhost:50070"),
    "language": os.getenv("TTS_LANGUAGE", "pt"),
    "voice": os.getenv("TTS_VOICE", "alloy"),
    "remote_timeout": _env_float("TTS_REMOTE_TIMEOUT", 60.0, 1.0, 600.0),
}

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "anthropic"),
    "model": os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
    "max_tokens": _env_int("LLM_MAX_TOKENS", 1024, 1, 100000),
    "temperature": _env_float("LLM_TEMPERATURE", 0.8, 0.0, 2.0),
    "timeout": _env_float("LLM_TIMEOUT", 30.0, 1.0, 300.0),
    "system_prompt": os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant."),
}

VAD_CONFIG = {
    "aggressiveness": _env_int("VAD_AGGRESSIVENESS", 3, 0, 3),
    "silence_threshold_ms": _env_int("VAD_SILENCE_MS", 500, 50, 5000),
    "prefix_padding_ms": _env_int("VAD_PREFIX_PADDING_MS", 300, 0, 5000),
    "min_speech_ms": _env_int("VAD_MIN_SPEECH_MS", 250, 50, 5000),
    "min_speech_rms": _env_int("VAD_MIN_SPEECH_RMS", 500, 0, 50000),
}

AUDIO_CONFIG = {
    "sample_rate": 24000,
    "channels": 1,
    "sample_width": 2,
    "internal_sample_rate": 8000,
}

PIPELINE_CONFIG = {
    "sentence_queue_size": _env_int("PIPELINE_SENTENCE_QUEUE_SIZE", 6, 1, 100),
    "tts_prefetch_size": _env_int("PIPELINE_TTS_PREFETCH_SIZE", 4, 1, 50),
    "max_sentence_chars": _env_int("PIPELINE_MAX_SENTENCE_CHARS", 150, 50, 1000),
    "tts_timeout": _env_float("PIPELINE_TTS_TIMEOUT", 15.0, 1.0, 120.0),
    "sentence_timeout": _env_float("PIPELINE_SENTENCE_TIMEOUT", 15.0, 1.0, 120.0),
}

TOOL_CONFIG = {
    "enable_mock_tools": os.getenv("TOOL_ENABLE_MOCK", "false").lower() == "true",
    "enable_web_search": os.getenv("TOOL_ENABLE_WEB_SEARCH", "false").lower() == "true",
    "timeout": _env_float("TOOL_TIMEOUT", 10.0, 1.0, 60.0),
    "max_rounds": _env_int("TOOL_MAX_ROUNDS", 5, 1, 20),
    "default_filler": os.getenv("TOOL_DEFAULT_FILLER", "Um momento, por favor."),
}

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_raw_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if _raw_log_level not in _VALID_LOG_LEVELS:
    raise ValueError(
        f"LOG_LEVEL={_raw_log_level!r} is invalid. "
        f"Valid: {sorted(_VALID_LOG_LEVELS)}"
    )

LOG_CONFIG = {
    "level": _raw_log_level,
}


# ---------------------------------------------------------------------------
# Frozen dataclass policies (type-safe access with autocomplete)
# Legacy dicts above remain exported for backward compatibility.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VadPolicy:
    aggressiveness: int
    silence_threshold_ms: int
    prefix_padding_ms: int
    min_speech_ms: int
    min_speech_rms: int


@dataclass(frozen=True)
class PipelinePolicy:
    sentence_queue_size: int
    tts_prefetch_size: int
    max_sentence_chars: int
    tts_timeout: float
    sentence_timeout: float


@dataclass(frozen=True)
class LLMPolicy:
    provider: str
    model: str
    max_tokens: int
    temperature: float
    timeout: float
    system_prompt: str


@dataclass(frozen=True)
class ConnectionPolicy:
    host: str
    port: int
    path: str
    api_key: str
    max_connections: int


@dataclass(frozen=True)
class ToolPolicy:
    enable_mock_tools: bool
    enable_web_search: bool
    timeout: float
    max_rounds: int
    default_filler: str


# Instantiate from the dicts loaded above
VAD = VadPolicy(**VAD_CONFIG)
PIPELINE = PipelinePolicy(**PIPELINE_CONFIG)
LLM = LLMPolicy(**LLM_CONFIG)
CONNECTION = ConnectionPolicy(**WS_CONFIG)
TOOL = ToolPolicy(**TOOL_CONFIG)
