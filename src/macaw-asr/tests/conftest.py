"""Fixtures for macaw-asr real GPU tests.

No mocks. All tests run against Qwen/Qwen3-ASR-0.6B on CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw_asr.config import AudioConfig, EngineConfig, StreamingConfig
from macaw_asr.runner.engine import ASREngine


QWEN_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
DEVICE = "cuda:0"
INPUT_SR = 8000
MODEL_SR = 16000


@pytest.fixture(scope="session")
def engine_config() -> EngineConfig:
    return EngineConfig(
        model_name="qwen",
        model_id=QWEN_MODEL_ID,
        device=DEVICE,
        dtype="bfloat16",
        language="pt",
        audio=AudioConfig(input_sample_rate=INPUT_SR, model_sample_rate=MODEL_SR),
        streaming=StreamingConfig(
            chunk_trigger_sec=1.0,
            max_new_tokens=32,
            enable_background_compute=True,
        ),
    )


_engine_instance: ASREngine | None = None


@pytest.fixture
async def engine(engine_config) -> ASREngine:
    """Shared engine. Starts once, reused across tests via module-level cache."""
    global _engine_instance
    if _engine_instance is None or not _engine_instance.is_started:
        _engine_instance = ASREngine(engine_config)
        await _engine_instance.start()
    return _engine_instance


# ==================== Audio generators ====================


def make_pcm(duration_sec: float, freq: float = 440.0, sr: int = INPUT_SR) -> bytes:
    """Generate PCM16 tone at given freq/duration/sr."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return wave.tobytes()


def make_silence(duration_sec: float, sr: int = INPUT_SR) -> bytes:
    return np.zeros(int(duration_sec * sr), dtype=np.int16).tobytes()


def make_noise(duration_sec: float, amplitude: float = 0.1, sr: int = INPUT_SR, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    n = int(duration_sec * sr)
    samples = (rng.randn(n) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()
