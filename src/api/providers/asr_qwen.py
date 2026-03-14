"""
Qwen3-ASR local provider.

Requires qwen-asr package with GPU support.

Config:
    ASR_PROVIDER=qwen
"""

from __future__ import annotations

import logging

from providers.asr import ASRProvider, register_asr_provider

logger = logging.getLogger("open-voice-api.asr.qwen")


class QwenASR(ASRProvider):
    """ASR provider using Qwen3-ASR locally."""

    provider_name = "qwen"

    def __init__(self):
        try:
            from qwen_asr import QwenASRModel
            self._model = QwenASRModel()
            logger.info("Qwen3-ASR model loaded")
        except ImportError:
            raise ImportError(
                "qwen-asr package required for QwenASR provider. "
                "Install with: pip install qwen-asr"
            )

    async def transcribe(self, audio: bytes) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._model.transcribe, audio)


register_asr_provider("qwen", QwenASR)
