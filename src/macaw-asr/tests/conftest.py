"""Fixtures for macaw-asr real GPU tests.

Model selection follows Ollama's pattern (integration/utils_test.go):
    MACAW_ASR_TEST_MODEL=qwen pytest tests/         # default
    MACAW_ASR_TEST_MODEL=whisper pytest tests/       # future
    MACAW_ASR_TEST_MODEL=parakeet pytest tests/      # future

No mocks. All tests run real inference on GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from macaw_asr.config import AudioConfig, EngineConfig, StreamingConfig
from macaw_asr.models.base import create_model
from macaw_asr.runner.engine import ASREngine


# ==================== Model Registry (Ollama pattern) ====================
# Like Ollama's testModel = os.Getenv("OLLAMA_TEST_MODEL")
# Each model defines: registry key, HF model ID, default dtype, model sample rate


_MODEL_CONFIGS = {
    "qwen": {
        "model_name": "qwen",
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "dtype": "bfloat16",
        "model_sample_rate": 16000,
    },
    "whisper-tiny": {
        "model_name": "whisper-tiny",
        "model_id": "openai/whisper-tiny",
        "dtype": "float16",
        "model_sample_rate": 16000,
    },
    "whisper-small": {
        "model_name": "whisper-small",
        "model_id": "openai/whisper-small",
        "dtype": "float16",
        "model_sample_rate": 16000,
    },
    "whisper-medium": {
        "model_name": "whisper-medium",
        "model_id": "openai/whisper-medium",
        "dtype": "float16",
        "model_sample_rate": 16000,
    },
    "whisper-large": {
        "model_name": "whisper-large",
        "model_id": "openai/whisper-large-v3",
        "dtype": "float16",
        "model_sample_rate": 16000,
    },
    # Future:
    # "parakeet": {
    #     "model_name": "parakeet",
    #     "model_id": "nvidia/parakeet-tdt-0.6b-v2",
    #     "dtype": "float32",
    #     "model_sample_rate": 16000,
    # },
}

TEST_MODEL = os.getenv("MACAW_ASR_TEST_MODEL", "qwen")
DEVICE = os.getenv("MACAW_ASR_TEST_DEVICE", "cuda:0")
INPUT_SR = 8000


def get_test_model_config() -> dict:
    """Get config dict for the current test model."""
    if TEST_MODEL not in _MODEL_CONFIGS:
        available = ", ".join(_MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown test model '{TEST_MODEL}'. "
            f"Available: {available}. "
            f"Set MACAW_ASR_TEST_MODEL env var."
        )
    return _MODEL_CONFIGS[TEST_MODEL]


# ==================== Config Fixture ====================


@pytest.fixture(scope="session")
def engine_config() -> EngineConfig:
    """EngineConfig for the test model. Driven by MACAW_ASR_TEST_MODEL env var."""
    mc = get_test_model_config()
    return EngineConfig(
        model_name=mc["model_name"],
        model_id=mc["model_id"],
        device=DEVICE,
        dtype=mc["dtype"],
        language="pt",
        audio=AudioConfig(
            input_sample_rate=INPUT_SR,
            model_sample_rate=mc["model_sample_rate"],
        ),
        streaming=StreamingConfig(
            chunk_trigger_sec=1.0,
            max_new_tokens=32,
            enable_background_compute=True,
        ),
    )


# ==================== Engine Fixture ====================


@pytest.fixture(scope="session")
async def engine(engine_config) -> ASREngine:
    """Session-scoped engine. Starts once, reused across all tests."""
    eng = ASREngine(engine_config)
    await eng.start()
    yield eng
    await eng.stop()


# ==================== Model Fixture (for direct model tests) ====================


@pytest.fixture(scope="session")
def model(engine_config):
    """Session-scoped loaded ASRModel. Reused across all tests."""
    m = create_model(engine_config.model_name)
    m.load(engine_config)
    m.warmup()
    yield m
    m.unload()


# ==================== Audio Generators ====================


def make_pcm(duration_sec: float, freq: float = 440.0, sr: int = INPUT_SR) -> bytes:
    """Generate PCM16 tone at given freq/duration/sr."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return samples.tobytes()


def make_silence(duration_sec: float, sr: int = INPUT_SR) -> bytes:
    return np.zeros(int(duration_sec * sr), dtype=np.int16).tobytes()


def make_noise(duration_sec: float, amplitude: float = 0.1, sr: int = INPUT_SR, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    n = int(duration_sec * sr)
    samples = (rng.randn(n) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()
