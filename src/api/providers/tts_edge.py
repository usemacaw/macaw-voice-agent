"""
Edge TTS Provider — Microsoft Edge's free TTS API.

Excellent Brazilian Portuguese voices with no API key required.
Uses edge-tts package which communicates with Microsoft's Cognitive Services.

Config:
    TTS_PROVIDER=edge
    EDGE_TTS_VOICE=pt-BR-FranciscaNeural    # or: AntonioNeural, ThalitaNeural
    EDGE_TTS_RATE=+0%                        # Speech rate: -50% to +100%
    EDGE_TTS_PITCH=+0Hz                      # Pitch adjustment
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import edge_tts

from config import TTS_CONFIG
from providers.tts import TTSProvider, register_tts_provider

logger = logging.getLogger("open-voice-api.tts.edge")

# Shorthand → full voice name for PT-BR
_PT_BR_VOICES = {
    "francisca": "pt-BR-FranciscaNeural",
    "antonio": "pt-BR-AntonioNeural",
    "thalita": "pt-BR-ThalitaNeural",
    "brenda": "pt-BR-BrendaNeural",
}

# Auto-select voice based on language config
_LANG_DEFAULT_VOICE = {
    "pt": "pt-BR-FranciscaNeural",
    "en": "en-US-AriaNeural",
    "es": "es-MX-DaliaNeural",
    "fr": "fr-FR-DeniseNeural",
}


# Maximum MP3 payload before decode — prevents memory issues on abnormal responses
_MAX_MP3_BYTES = 10 * 1024 * 1024  # 10 MB


class EdgeTTS(TTSProvider):
    """TTS provider using Microsoft Edge's free TTS API.

    No API key required. Native PT-BR voices without accent.
    Audio received as MP3 and decoded to PCM via ffmpeg subprocess.
    """

    provider_name = "edge"

    def __init__(self):
        voice_env = os.getenv("EDGE_TTS_VOICE", "")
        if voice_env:
            self._voice = _PT_BR_VOICES.get(voice_env.lower(), voice_env)
        else:
            lang = TTS_CONFIG.get("language", "pt")
            self._voice = _LANG_DEFAULT_VOICE.get(lang, "pt-BR-FranciscaNeural")

        self._rate = os.getenv("EDGE_TTS_RATE", "+0%")
        self._pitch = os.getenv("EDGE_TTS_PITCH", "+0Hz")
        self._volume = os.getenv("EDGE_TTS_VOLUME", "+0%")

    async def connect(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found — required for Edge TTS MP3 decoding. "
                "Install with: apt-get install -y ffmpeg"
            )
        logger.info(
            f"Edge TTS ready: voice={self._voice}, rate={self._rate}"
        )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to PCM 8kHz 16-bit mono."""
        if not text.strip():
            return b""

        t0 = time.perf_counter()

        communicate = edge_tts.Communicate(
            text,
            voice=self._voice,
            rate=self._rate,
            pitch=self._pitch,
            volume=self._volume,
        )

        # Collect MP3 chunks
        mp3_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data.extend(chunk["data"])

        if not mp3_data:
            logger.warning(f"Edge TTS returned no audio for: \"{text[:40]}\"")
            return b""

        if len(mp3_data) > _MAX_MP3_BYTES:
            logger.error(
                f"Edge TTS MP3 too large ({len(mp3_data)} bytes, "
                f"limit {_MAX_MP3_BYTES}): \"{text[:40]}\""
            )
            return b""

        # Decode MP3 → PCM 8kHz 16-bit mono via ffmpeg
        pcm_data = await self._decode_mp3(bytes(mp3_data))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration_s = len(pcm_data) / (8000 * 2) if pcm_data else 0
        logger.info(
            f"TTS (edge): {elapsed_ms:.0f}ms → {len(pcm_data)} bytes "
            f"({duration_s:.1f}s): \"{text[:40]}\""
        )
        return pcm_data

    async def _decode_mp3(self, mp3_data: bytes) -> bytes:
        """Decode MP3 bytes to PCM 8kHz 16-bit mono using ffmpeg."""
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le",
            "-ar", "8000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=mp3_data)

        if proc.returncode != 0:
            logger.error(f"ffmpeg decode failed: {stderr.decode()[:200]}")
            return b""

        return stdout


register_tts_provider("edge", EdgeTTS)
