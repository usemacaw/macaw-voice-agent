"""
Faster-Whisper local ASR provider — CPU-optimized.

Config:
    ASR_PROVIDER=whisper
    WHISPER_MODEL=base                  # base, small, medium, large-v3-turbo
    WHISPER_COMPUTE_TYPE=int8           # int8 for CPU, float16 for GPU
    WHISPER_BEAM_SIZE=1                 # 1 = greedy (fastest)
    ASR_LANGUAGE=pt
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from config import ASR_CONFIG
from providers.asr import ASRProvider, register_asr_provider

logger = logging.getLogger("open-voice-api.asr.whisper")

_WHISPER_SAMPLE_RATE = 16000
_INTERNAL_SAMPLE_RATE = 8000


class WhisperASR(ASRProvider):
    """ASR provider using faster-whisper locally on CPU."""

    provider_name = "whisper"

    def __init__(self):
        self._model = None
        self._model_name = os.getenv("WHISPER_MODEL", "base")
        self._device = os.getenv("WHISPER_DEVICE", "cpu")
        self._compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self._beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
        self._language = ASR_CONFIG.get("language", "pt")

    async def connect(self) -> None:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Faster-Whisper: model={self._model_name}, "
            f"device={self._device}, compute={self._compute_type}"
        )
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            ),
        )
        logger.info("Faster-Whisper loaded and ready")

    async def transcribe(self, audio: bytes) -> str:
        if self._model is None:
            raise RuntimeError("WhisperASR not connected. Call connect() first.")

        from audio.utils import pcm_to_float32, resample

        t0 = time.perf_counter()

        # PCM 8kHz → float32 → resample to 16kHz (Whisper expects 16kHz)
        samples_8k = pcm_to_float32(audio)
        samples_16k = resample(samples_8k, _INTERNAL_SAMPLE_RATE, _WHISPER_SAMPLE_RATE)

        loop = asyncio.get_running_loop()

        def _transcribe():
            segments, info = self._model.transcribe(
                samples_16k,
                language=self._language,
                beam_size=self._beam_size,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=200,
                    speech_pad_ms=100,
                ),
            )
            texts = [s.text.strip() for s in segments]
            return " ".join(t for t in texts if t)

        result = await loop.run_in_executor(None, _transcribe)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"ASR (whisper): {elapsed_ms:.0f}ms → \"{result[:60]}\"")
        return result


register_asr_provider("whisper", WhisperASR)
