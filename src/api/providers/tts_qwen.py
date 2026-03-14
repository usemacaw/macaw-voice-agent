"""
Qwen3-TTS local provider (via faster-qwen3-tts).

Config:
    TTS_PROVIDER=qwen
"""

from __future__ import annotations

import asyncio
import logging
import os

from providers.tts import TTSProvider, register_tts_provider

logger = logging.getLogger("open-voice-api.tts.qwen")


class QwenTTS(TTSProvider):
    """TTS provider using Qwen3-TTS locally via faster-qwen3-tts."""

    provider_name = "qwen"

    def __init__(self):
        try:
            from faster_qwen3_tts import Qwen3TTS
        except ImportError:
            raise ImportError(
                "faster-qwen3-tts package required. "
                "Install with: pip install faster-qwen3-tts"
            )
        model_path = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS")
        self._model = Qwen3TTS(model_path)
        self._language = os.getenv("TTS_LANGUAGE", "pt")
        self._voice = os.getenv("TTS_VOICE", "alloy")
        logger.info(f"Qwen3-TTS loaded: model={model_path}")

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._model.synthesize, text, self._language
        )


register_tts_provider("qwen", QwenTTS)
