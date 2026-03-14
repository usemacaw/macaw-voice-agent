"""
Kokoro TTS Provider - Sintese de voz local via Kokoro-ONNX.

Modelo leve (~82M params) otimizado para CPU via ONNX Runtime.
Suporta streaming async nativo.

Requer: pip install kokoro-onnx

Modelos (download automatico ou manual):
- kokoro-v1.0.onnx (~300MB)
- voices-v1.0.bin (~50MB)

Configuracao via env vars:
    TTS_PROVIDER=kokoro
    KOKORO_MODEL_DIR=models/kokoro       # Diretorio com .onnx e .bin
    KOKORO_VOICE=af_heart                # Voz (af_heart, pf_dora, pm_alex, etc)
    KOKORO_SPEED=1.0                     # Velocidade (0.5 a 2.0)
    KOKORO_LANG=pt-br                    # Idioma (pt-br, en-us, es, fr-fr, etc)
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, TYPE_CHECKING

from common.config import AUDIO_CONFIG
from common.audio_utils import float32_to_pcm, resample
from tts.providers.base import TTSProvider, register_tts_provider

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro

logger = logging.getLogger("ai-agent.tts.kokoro")

_OUTPUT_SAMPLE_RATE = 24000  # Kokoro gera audio a 24kHz

_LANG_MAP = {
    "pt": "pt-br",
    "en": "en-us",
    "es": "es",
    "fr": "fr-fr",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "zh": "zh",
}


class KokoroTTS(TTSProvider):
    """TTS provider usando Kokoro-ONNX para sintese de voz local.

    Modelo leve (~82M params) com ONNX Runtime, roda bem em CPU.
    Suporta streaming async nativo via create_stream().
    """

    provider_name = "kokoro"

    def __init__(self):
        self._kokoro: Kokoro | None = None

        # Diretorio dos modelos
        self._model_dir = os.getenv("KOKORO_MODEL_DIR", "models/kokoro")
        self._model_path = os.path.join(self._model_dir, "kokoro-v1.0.onnx")
        self._voices_path = os.path.join(self._model_dir, "voices-v1.0.bin")

        # Voz e config
        self._voice = os.getenv("KOKORO_VOICE", "pf_dora")
        self._speed = float(os.getenv("KOKORO_SPEED", "1.0"))

        # Idioma: env override > mapeamento de TTS_LANG
        lang_override = os.getenv("KOKORO_LANG")
        if lang_override:
            self._lang = lang_override
        else:
            from common.config import TTS_CONFIG
            tts_lang = TTS_CONFIG.get("language", "pt")
            self._lang = _LANG_MAP.get(tts_lang, "en-us")

    async def connect(self) -> None:
        """Carrega modelo Kokoro-ONNX."""
        from kokoro_onnx import Kokoro

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Modelo Kokoro nao encontrado: {self._model_path}. "
                f"Baixe de https://github.com/thewh1teagle/kokoro-onnx/releases"
            )

        if not os.path.isfile(self._voices_path):
            raise FileNotFoundError(
                f"Arquivo de vozes nao encontrado: {self._voices_path}. "
                f"Baixe de https://github.com/thewh1teagle/kokoro-onnx/releases"
            )

        logger.info(
            f"Carregando Kokoro-ONNX: model={self._model_path}, "
            f"voice={self._voice}, lang={self._lang}"
        )

        self._kokoro = Kokoro(self._model_path, self._voices_path)

        voices = self._kokoro.get_voices()
        if self._voice not in voices:
            logger.warning(
                f"Voz '{self._voice}' nao encontrada. "
                f"Disponiveis: {voices[:10]}{'...' if len(voices) > 10 else ''}"
            )

        logger.info(
            f"Kokoro-ONNX carregado ({len(voices)} vozes disponiveis)"
        )

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._kokoro is not None:
            del self._kokoro
            self._kokoro = None
            logger.info("Kokoro-ONNX descarregado")

    async def synthesize(self, text: str) -> bytes:
        """Sintetiza texto em audio PCM 8kHz 16-bit.

        1. Kokoro create (24kHz float32)
        2. Resample 24kHz -> 8kHz
        3. Float32 -> PCM 16-bit LE
        """
        if not text.strip():
            return b""

        if self._kokoro is None:
            raise RuntimeError("KokoroTTS nao conectado. Chame connect() primeiro.")

        logger.info(f"TTS (kokoro): sintetizando {len(text)} chars...")

        audio_float, sr = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )

        if audio_float is None or len(audio_float) == 0:
            logger.warning("TTS (kokoro): modelo nao retornou audio")
            return b""

        # Resample para 8kHz (telefonia)
        target_rate = AUDIO_CONFIG["sample_rate"]
        audio_8k = resample(audio_float, sr, target_rate)

        # Float32 -> PCM 16-bit LE
        pcm_data = float32_to_pcm(audio_8k)

        logger.info(
            f"TTS (kokoro): {len(pcm_data)} bytes, "
            f"{len(pcm_data) / (target_rate * 2):.1f}s"
        )
        return pcm_data

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streaming real: yield chunks de audio conforme sao gerados.

        Usa Kokoro create_stream() que e async nativo.
        """
        if not text.strip():
            return

        if self._kokoro is None:
            raise RuntimeError("KokoroTTS nao conectado. Chame connect() primeiro.")

        logger.info(f"TTS (kokoro) stream: sintetizando {len(text)} chars...")

        target_rate = AUDIO_CONFIG["sample_rate"]
        chunks_sent = 0

        async for chunk_float, sr in self._kokoro.create_stream(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        ):
            if chunk_float is not None and len(chunk_float) > 0:
                audio_8k = resample(chunk_float, sr, target_rate)
                yield float32_to_pcm(audio_8k)
                chunks_sent += 1

        logger.info(f"TTS (kokoro) stream: {chunks_sent} chunks enviados")

    @property
    def supports_streaming(self) -> bool:
        """Kokoro-ONNX suporta streaming async nativo."""
        return True


# Auto-register quando o modulo e importado
register_tts_provider("kokoro", KokoroTTS)
