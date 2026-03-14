"""
Kokoro TTS Provider — Local synthesis via Kokoro-ONNX.

Lightweight model (~82M params) optimized for CPU via ONNX Runtime.
Supports native async streaming and Brazilian Portuguese.

Config:
    TTS_PROVIDER=kokoro
    KOKORO_MODEL_DIR=models/kokoro       # Directory with .onnx and .bin files
    KOKORO_VOICE=pf_dora                 # Voice (pf_dora=PT female, pm_alex, af_heart, etc)
    KOKORO_SPEED=1.0                     # Speed (0.5 to 2.0)
    KOKORO_LANG=pt-br                    # Language (pt-br, en-us, es, fr-fr, etc)
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, TYPE_CHECKING

from audio.utils import float32_to_pcm, resample
from config import TTS_CONFIG
from providers.tts import TTSProvider, register_tts_provider

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro

logger = logging.getLogger("open-voice-api.tts.kokoro")

_OUTPUT_SAMPLE_RATE = 24000  # Kokoro generates at 24kHz
_TARGET_SAMPLE_RATE = 8000   # Pipeline expects 8kHz

_LANG_MAP = {
    "pt": "pt-br",
    "en": "en-us",
    "es": "es",
    "fr": "fr-fr",
    "ja": "ja",
    "zh": "zh",
}


class KokoroTTS(TTSProvider):
    """TTS provider using Kokoro-ONNX for local voice synthesis.

    ~82M params, ONNX Runtime, runs well on CPU.
    Native async streaming via create_stream().
    """

    provider_name = "kokoro"

    def __init__(self):
        self._kokoro: Kokoro | None = None

        self._model_dir = os.getenv("KOKORO_MODEL_DIR", "models/kokoro")
        self._model_path = os.path.join(self._model_dir, "kokoro-v1.0.onnx")
        self._voices_path = os.path.join(self._model_dir, "voices-v1.0.bin")

        self._voice = os.getenv("KOKORO_VOICE", "pf_dora")
        self._speed = float(os.getenv("KOKORO_SPEED", "1.0"))

        lang_override = os.getenv("KOKORO_LANG")
        if lang_override:
            self._lang = lang_override
        else:
            tts_lang = TTS_CONFIG.get("language", "pt")
            self._lang = _LANG_MAP.get(tts_lang, "en-us")

    async def connect(self) -> None:
        from kokoro_onnx import Kokoro

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Kokoro model not found: {self._model_path}. "
                f"Download from https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
        if not os.path.isfile(self._voices_path):
            raise FileNotFoundError(
                f"Kokoro voices not found: {self._voices_path}. "
                f"Download from https://github.com/thewh1teagle/kokoro-onnx/releases"
            )

        logger.info(
            f"Loading Kokoro-ONNX: model={self._model_path}, "
            f"voice={self._voice}, lang={self._lang}"
        )

        self._kokoro = Kokoro(self._model_path, self._voices_path)

        voices = self._kokoro.get_voices()
        if self._voice not in voices:
            logger.warning(
                f"Voice '{self._voice}' not found. "
                f"Available: {voices[:10]}{'...' if len(voices) > 10 else ''}"
            )

        logger.info(f"Kokoro-ONNX loaded ({len(voices)} voices available)")

    async def disconnect(self) -> None:
        if self._kokoro is not None:
            del self._kokoro
            self._kokoro = None
            logger.info("Kokoro-ONNX unloaded")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to PCM 8kHz 16-bit mono."""
        if not text.strip():
            return b""

        if self._kokoro is None:
            raise RuntimeError("KokoroTTS not connected. Call connect() first.")

        audio_float, sr = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )

        if audio_float is None or len(audio_float) == 0:
            logger.warning(f"Kokoro returned no audio for: \"{text[:40]}\"")
            return b""

        audio_8k = resample(audio_float, sr, _TARGET_SAMPLE_RATE)
        pcm_data = float32_to_pcm(audio_8k)

        duration_s = len(pcm_data) / (_TARGET_SAMPLE_RATE * 2)
        logger.info(
            f"TTS (kokoro): {len(pcm_data)} bytes ({duration_s:.1f}s): \"{text[:40]}\""
        )
        return pcm_data

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streaming synthesis — yields audio chunks as generated."""
        if not text.strip():
            return

        if self._kokoro is None:
            raise RuntimeError("KokoroTTS not connected. Call connect() first.")

        chunks_sent = 0

        async for chunk_float, sr in self._kokoro.create_stream(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        ):
            if chunk_float is not None and len(chunk_float) > 0:
                audio_8k = resample(chunk_float, sr, _TARGET_SAMPLE_RATE)
                yield float32_to_pcm(audio_8k)
                chunks_sent += 1

        logger.info(f"TTS (kokoro) stream: {chunks_sent} chunks for \"{text[:40]}\"")

    @property
    def supports_streaming(self) -> bool:
        return True


register_tts_provider("kokoro", KokoroTTS)
