"""
Macaw Streaming TTS Provider — True streaming via macaw-qwen3-tts-streaming.

Streaming real: cada yield contém ~83ms de áudio (1 frame @ 12Hz).
TTFA: ~88ms no primeiro frame (com CUDA graphs quentes).

Requer GPU NVIDIA com CUDA e PyTorch >= 2.5.1.

Configuração via env vars:
    TTS_PROVIDER=macaw-streaming
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    QWEN_TTS_SPEAKER=Ryan
    QWEN_TTS_LANGUAGE=Portuguese
    MACAW_TTS_EMIT_PHASE1=1          # Frames before first emit (1 = ~88ms)
    MACAW_TTS_EMIT_PHASE2=4          # Frames per chunk in stable phase
    MACAW_TTS_DECODE_WINDOW=40       # Decoder context window
    MACAW_TTS_OVERLAP=512            # Crossfade overlap samples
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue as thread_queue
from typing import AsyncGenerator

import numpy as np

from common.config import TTS_CONFIG, AUDIO_CONFIG
from common.audio_utils import float32_to_pcm, resample
from common.executor import run_inference
from tts.providers.base import TTSProvider, register_tts_provider

logger = logging.getLogger("ai-agent.tts.macaw-streaming")

_OUTPUT_SAMPLE_RATE = 24000  # MacawTTS gera áudio a 24kHz


class MacawStreamingTTS(TTSProvider):
    """TTS provider usando macaw-qwen3-tts-streaming.

    Streaming real com CUDA graphs, two-phase latency, e Hann crossfade.
    Cada chunk contém ~83ms de áudio (Phase 1) ou ~333ms (Phase 2).

    Requer GPU NVIDIA com CUDA.
    """

    provider_name = "macaw-streaming"

    def __init__(self):
        self._model = None

        self._model_name = os.getenv(
            "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        )
        # Speaker: None for Base models, specific name for CustomVoice
        speaker_env = os.getenv("QWEN_TTS_SPEAKER", "")
        self._speaker = speaker_env if speaker_env else None
        self._emit_phase1 = int(os.getenv("MACAW_TTS_EMIT_PHASE1", "1"))
        self._emit_phase2 = int(os.getenv("MACAW_TTS_EMIT_PHASE2", "4"))
        self._decode_window = int(os.getenv("MACAW_TTS_DECODE_WINDOW", "40"))
        self._overlap = int(os.getenv("MACAW_TTS_OVERLAP", "512"))

        # Idioma
        lang_override = os.getenv("QWEN_TTS_LANGUAGE")
        if lang_override:
            self._language = lang_override
        else:
            from common.audio_utils import QWEN_LANGUAGE_MAP
            lang_code = TTS_CONFIG.get("language", "pt")
            self._language = QWEN_LANGUAGE_MAP.get(lang_code, "Portuguese")

    async def connect(self) -> None:
        """Carrega modelo e prepara CUDA graphs.

        🟢 GPU REQUIRED.
        """
        from macaw_tts.model import MacawTTS

        logger.info(
            f"Carregando MacawTTS: model={self._model_name}, "
            f"speaker={self._speaker}, decode_window={self._decode_window}"
        )

        self._model = await run_inference(
            MacawTTS.from_pretrained,
            self._model_name,
            decode_window=self._decode_window,
        )

        # Warmup com texto dummy para capturar CUDA graphs
        logger.info("MacawStreamingTTS: executando warmup (CUDA graph capture)...")
        await run_inference(self._warmup)

        logger.info("MacawStreamingTTS carregado e pronto")

    def _warmup(self) -> None:
        """Executa warmup para capturar CUDA graphs."""
        for chunk, sr, meta in self._model.stream(
            "Olá, tudo bem?",
            language=self._language,
            speaker=self._speaker,
        ):
            pass  # Consume todo o stream para garantir warmup completo
        logger.info("Warmup concluído")

    async def disconnect(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("MacawStreamingTTS descarregado")

    async def synthesize(self, text: str) -> bytes:
        """Síntese batch (fallback). Prefira synthesize_stream()."""
        if not text.strip():
            return b""

        if self._model is None:
            raise RuntimeError("MacawStreamingTTS não conectado. Chame connect().")

        target_rate = AUDIO_CONFIG["sample_rate"]
        all_audio = []

        def _generate():
            for chunk, sr, meta in self._model.stream(
                text,
                language=self._language,
                speaker=self._speaker,
                emit_every_phase1=self._emit_phase1,
                emit_every_phase2=self._emit_phase2,
                overlap_samples=self._overlap,
            ):
                audio_resampled = resample(chunk.astype(np.float32), sr, target_rate)
                all_audio.append(float32_to_pcm(audio_resampled))

        await run_inference(_generate)

        if not all_audio:
            return b""

        return b"".join(all_audio)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streaming real — yield PCM 8kHz chunks conforme gerados.

        🟢 GPU REQUIRED.

        Cada chunk contém áudio real gerado frame-a-frame pelo modelo.
        Phase 1: ~83ms por chunk (1 frame @ 12Hz)
        Phase 2: ~333ms por chunk (4 frames @ 12Hz)
        """
        if not text.strip():
            return

        if self._model is None:
            raise RuntimeError("MacawStreamingTTS não conectado. Chame connect().")

        target_rate = AUDIO_CONFIG["sample_rate"]
        audio_queue: thread_queue.Queue = thread_queue.Queue()
        loop = asyncio.get_running_loop()

        def _stream_to_queue():
            """Roda em thread: streaming generator → queue."""
            try:
                for chunk, sr, meta in self._model.stream(
                    text,
                    language=self._language,
                    speaker=self._speaker,
                    emit_every_phase1=self._emit_phase1,
                    emit_every_phase2=self._emit_phase2,
                    overlap_samples=self._overlap,
                ):
                    # Resample 24kHz → 8kHz e converter para PCM16
                    audio_resampled = resample(
                        chunk.astype(np.float32), sr, target_rate
                    )
                    pcm_chunk = float32_to_pcm(audio_resampled)
                    audio_queue.put(pcm_chunk)
            except Exception as e:
                audio_queue.put(e)
            finally:
                audio_queue.put(None)

        # Inicia geração em thread
        future = loop.run_in_executor(None, _stream_to_queue)

        try:
            while True:
                item = await loop.run_in_executor(None, audio_queue.get)

                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.error(f"TTS (macaw-streaming) erro: {item}")
                    raise item

                yield item
        finally:
            await asyncio.wrap_future(future)

    @property
    def supports_streaming(self) -> bool:
        """MacawStreamingTTS suporta streaming real."""
        return True


# Auto-register quando o módulo é importado
register_tts_provider("macaw-streaming", MacawStreamingTTS)
