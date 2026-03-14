"""
FasterQwen3TTS Provider - Sintese de voz com streaming real e CUDA graphs.

Usa o pacote faster-qwen3-tts (MIT) para servir modelos Qwen3-TTS (Apache 2.0)
com streaming nativo e CUDA graph capture, atingindo TTFB ~156ms no RTX 4090.

Requer: pip install faster-qwen3-tts
GPU obrigatoria (CUDA, PyTorch >= 2.5.1).

Vantagens sobre vLLM-Omni:
- Streaming real (yield chunks enquanto gera)
- CUDA graphs funcionais (5-6x speedup)
- TTFB ~156ms vs ~2000ms do vLLM-Omni

Configuracao via env vars:
    TTS_PROVIDER=faster
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    QWEN_TTS_SPEAKER=Ryan
    QWEN_TTS_LANGUAGE=Portuguese
    FASTER_TTS_CHUNK_SIZE=4     # Tokens por chunk (menor = menor TTFB)
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, TYPE_CHECKING

import numpy as np

from common.config import TTS_CONFIG, AUDIO_CONFIG
from common.audio_utils import float32_to_pcm, resample, QWEN_LANGUAGE_MAP
from common.executor import run_inference
from tts.providers.base import TTSProvider, register_tts_provider

if TYPE_CHECKING:
    from faster_qwen3_tts import FasterQwen3TTS

logger = logging.getLogger("ai-agent.tts.faster")

_OUTPUT_SAMPLE_RATE = 12000  # faster-qwen3-tts gera audio a 12kHz


class FasterTTS(TTSProvider):
    """TTS provider usando faster-qwen3-tts com streaming e CUDA graphs.

    Carrega Qwen3-TTS na GPU com CUDA graph capture para inferencia
    otimizada. Suporta streaming real via generate_custom_voice_streaming().

    Requer GPU NVIDIA com CUDA e PyTorch >= 2.5.1.
    """

    provider_name = "faster"

    def __init__(self):
        self._model: FasterQwen3TTS | None = None
        self._model_name = os.getenv(
            "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        self._speaker = os.getenv("QWEN_TTS_SPEAKER", "Ryan").lower()
        self._chunk_size = int(os.getenv("FASTER_TTS_CHUNK_SIZE", "4"))

        # Idioma: env var override > TTS_CONFIG > default
        lang_override = os.getenv("QWEN_TTS_LANGUAGE")
        if lang_override:
            self._language = lang_override
        else:
            lang_code = TTS_CONFIG.get("language", "pt")
            self._language = QWEN_LANGUAGE_MAP.get(lang_code, "Portuguese")

    async def connect(self) -> None:
        """Carrega modelo e compila CUDA graphs."""
        from faster_qwen3_tts import FasterQwen3TTS

        logger.info(
            f"Carregando FasterQwen3TTS: model={self._model_name}, "
            f"speaker={self._speaker}, chunk_size={self._chunk_size}"
        )

        self._model = await run_inference(
            FasterQwen3TTS.from_pretrained,
            self._model_name,
        )

        # Warmup: primeira chamada compila CUDA graphs
        logger.info("FasterQwen3TTS: executando warmup (CUDA graph capture)...")
        await run_inference(self._warmup)

        logger.info("FasterQwen3TTS carregado e pronto")

    def _warmup(self) -> None:
        """Executa warmup para compilar CUDA graphs."""
        audio_list, sr = self._model.generate_custom_voice(
            text="Ola, tudo bem? Estou aqui para ajudar.",
            language=self._language,
            speaker=self._speaker,
        )
        logger.info(f"Warmup concluido: {len(audio_list)} chunks, sr={sr}")

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("FasterQwen3TTS descarregado")

    def _generate_and_convert(self, text: str) -> bytes:
        """Gera audio completo (batch) e converte para PCM 8kHz 16-bit.

        Roda no executor thread para nao bloquear o event loop.
        """
        audio_list, sr = self._model.generate_custom_voice(
            text=text,
            language=self._language,
            speaker=self._speaker,
        )

        if not audio_list:
            return b""

        # Concatena todos os chunks
        audio_float = np.concatenate(audio_list, axis=-1).flatten()
        target_rate = AUDIO_CONFIG["sample_rate"]
        audio_resampled = resample(audio_float.astype(np.float32), sr, target_rate)
        return float32_to_pcm(audio_resampled)

    async def synthesize(self, text: str) -> bytes:
        """Sintetiza texto em audio PCM 8kHz 16-bit (batch).

        Para streaming real, use synthesize_stream().
        """
        if not text.strip():
            return b""

        if self._model is None:
            raise RuntimeError("FasterTTS nao conectado. Chame connect() primeiro.")

        logger.info(
            f"TTS (faster): sintetizando {len(text)} chars "
            f"(speaker={self._speaker})..."
        )

        pcm_data = await run_inference(self._generate_and_convert, text)

        if not pcm_data:
            logger.warning("TTS (faster): modelo nao retornou audio")
            return b""

        target_rate = AUDIO_CONFIG["sample_rate"]
        logger.info(
            f"TTS (faster): {len(pcm_data)} bytes, "
            f"{len(pcm_data) / (target_rate * 2):.1f}s"
        )
        return pcm_data

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Sintetiza texto com streaming real - yield chunks conforme gerados.

        Cada chunk contem ~chunk_size decode steps de audio (ex: chunk_size=4 -> ~333ms).
        Audio e convertido de float32 12kHz para PCM 8kHz 16-bit.
        """
        if not text.strip():
            return

        if self._model is None:
            raise RuntimeError("FasterTTS nao conectado. Chame connect() primeiro.")

        logger.info(
            f"TTS (faster) stream: sintetizando {len(text)} chars "
            f"(speaker={self._speaker}, chunk_size={self._chunk_size})..."
        )

        target_rate = AUDIO_CONFIG["sample_rate"]
        chunk_count = 0
        total_bytes = 0

        # generate_custom_voice_streaming e sync generator, roda no executor
        import asyncio
        import queue as thread_queue

        audio_queue: thread_queue.Queue = thread_queue.Queue()
        loop = asyncio.get_running_loop()

        def _stream_to_queue():
            """Roda em thread: streaming generator -> queue."""
            try:
                for audio_chunk, sr, timing in self._model.generate_custom_voice_streaming(
                    text=text,
                    language=self._language,
                    speaker=self._speaker,
                    chunk_size=self._chunk_size,
                ):
                    # Converte chunk para PCM 8kHz
                    audio_float = audio_chunk.flatten().astype(np.float32)
                    audio_resampled = resample(audio_float, sr, target_rate)
                    pcm_chunk = float32_to_pcm(audio_resampled)
                    audio_queue.put(pcm_chunk)
            except Exception as e:
                audio_queue.put(e)
            finally:
                audio_queue.put(None)  # Sentinel

        # Inicia geracao em thread
        future = loop.run_in_executor(None, _stream_to_queue)

        try:
            while True:
                item = await loop.run_in_executor(None, audio_queue.get)

                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.error(f"TTS (faster) stream: erro na geracao: {item}")
                    raise item

                chunk_count += 1
                total_bytes += len(item)
                yield item
        finally:
            await asyncio.wrap_future(future)
            logger.info(
                f"TTS (faster) stream: {chunk_count} chunks, "
                f"{total_bytes} bytes, "
                f"{total_bytes / (target_rate * 2):.1f}s audio"
            )

    @property
    def supports_streaming(self) -> bool:
        """FasterTTS suporta streaming real."""
        return True


# Auto-register quando o modulo e importado
register_tts_provider("faster", FasterTTS)
