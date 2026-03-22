"""
Macaw Streaming TTS Provider — True streaming via macaw-qwen3-tts-streaming.

Streaming real: cada yield contém ~83ms de áudio (1 frame @ 12Hz).
TTFA: ~88ms no primeiro frame (com CUDA graphs quentes).

Requer GPU NVIDIA com CUDA e PyTorch >= 2.5.1.

Configuração via env vars:
    TTS_PROVIDER=macaw-streaming
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
    QWEN_TTS_SPEAKER=              # Empty for Base models
    QWEN_TTS_LANGUAGE=Portuguese
    QWEN_TTS_REF_AUDIO=            # Path to reference audio for voice cloning
    QWEN_TTS_REF_TEXT=             # Transcript of reference audio
    MACAW_TTS_EMIT_PHASE1=1        # Frames before first emit (1 = ~88ms)
    MACAW_TTS_EMIT_PHASE2=4        # Frames per chunk in stable phase
    MACAW_TTS_DECODE_WINDOW=40     # Decoder context window
    MACAW_TTS_OVERLAP=512          # Crossfade overlap samples
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

    Voice consistency: Uses voice cloning when ref_audio is configured.
    A voice_clone_prompt is pre-computed once at connect() and reused
    for ALL synthesize calls, ensuring the same voice across sentences
    (critical for sentence pipeline which calls TTS per sentence).

    Requer GPU NVIDIA com CUDA.
    """

    provider_name = "macaw-streaming"

    def __init__(self):
        self._model = None
        self._voice_clone_prompt = None  # Pre-computed for consistent voice

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

        # Voice cloning reference audio (ensures consistent voice across calls)
        self._ref_audio = os.getenv("QWEN_TTS_REF_AUDIO", "")
        self._ref_text = os.getenv(
            "QWEN_TTS_REF_TEXT",
            "Olá, boa tarde! Meu nome é Sara e estou aqui para ajudar você. "
            "Posso verificar informações, fazer pesquisas e responder suas dúvidas.",
        )

        # Idioma
        lang_override = os.getenv("QWEN_TTS_LANGUAGE")
        if lang_override:
            self._language = lang_override
        else:
            from common.audio_utils import QWEN_LANGUAGE_MAP
            lang_code = TTS_CONFIG.get("language", "pt")
            self._language = QWEN_LANGUAGE_MAP.get(lang_code, "Portuguese")

    async def connect(self) -> None:
        """Carrega modelo, prepara CUDA graphs, e pré-computa voice clone prompt.

        🟢 GPU REQUIRED.
        """
        from macaw_tts.model import MacawTTS

        logger.info(
            f"Carregando MacawTTS: model={self._model_name}, "
            f"speaker={self._speaker}, ref_audio={self._ref_audio or 'none'}, "
            f"decode_window={self._decode_window}"
        )

        self._model = await run_inference(
            MacawTTS.from_pretrained,
            self._model_name,
            decode_window=self._decode_window,
        )

        # Pre-compute voice clone prompt if ref_audio is configured
        if self._ref_audio and os.path.isfile(self._ref_audio):
            logger.info(f"Pre-computing voice clone prompt from: {self._ref_audio}")
            self._voice_clone_prompt = await run_inference(
                self._build_voice_clone_prompt
            )
            logger.info("Voice clone prompt ready — voice will be consistent across calls")
        elif self._ref_audio:
            logger.warning(f"ref_audio not found: {self._ref_audio}, using default voice")

        # Warmup com texto dummy para capturar CUDA graphs
        logger.info("MacawStreamingTTS: executando warmup (CUDA graph capture)...")
        await run_inference(self._warmup)

        logger.info("MacawStreamingTTS carregado e pronto")

    def _build_voice_clone_prompt(self):
        """Pre-compute voice clone prompt from reference audio.

        Uses MacawTTS._model which is Qwen3TTSModel (the wrapper with
        create_voice_clone_prompt). NOT .model which is the inner nn.Module.
        The prompt is pre-computed ONCE and reused for all TTS calls,
        ensuring consistent voice across sentences in the pipeline.
        """
        # MacawTTS stores the Qwen3TTSModel wrapper as self._model
        wrapper = self._model._model  # Qwen3TTSModel
        prompt_items = wrapper.create_voice_clone_prompt(
            ref_audio=self._ref_audio,
            ref_text=self._ref_text,
        )
        return wrapper._prompt_items_to_voice_clone_prompt(prompt_items)

    def _warmup(self) -> None:
        """Executa warmup para capturar CUDA graphs."""
        if self._voice_clone_prompt is not None:
            for chunk, sr, meta in self._model.stream_voice_clone(
                "Olá, tudo bem?",
                language=self._language,
                voice_clone_prompt=self._voice_clone_prompt,
                ref_text=self._ref_text,
            ):
                pass
        else:
            for chunk, sr, meta in self._model.stream(
                "Olá, tudo bem?",
                language=self._language,
                speaker=self._speaker,
            ):
                pass
        logger.info("Warmup concluído")

    async def disconnect(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._voice_clone_prompt = None
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
            stream_fn = self._get_stream_fn(text)
            for chunk, sr, meta in stream_fn():
                audio_resampled = resample(chunk.astype(np.float32), sr, target_rate)
                all_audio.append(float32_to_pcm(audio_resampled))

        await run_inference(_generate)

        if not all_audio:
            return b""

        return b"".join(all_audio)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streaming real — yield PCM 8kHz chunks conforme gerados.

        🟢 GPU REQUIRED.

        Voice consistency: When voice_clone_prompt is pre-computed,
        every call produces the same voice — critical for sentence
        pipeline which calls TTS per sentence.
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
                stream_fn = self._get_stream_fn(text)
                for chunk, sr, meta in stream_fn():
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

    def _get_stream_fn(self, text: str):
        """Return the appropriate stream function (voice clone or regular).

        Returns a zero-arg callable that yields (chunk, sr, meta) tuples.
        Using voice_clone_prompt ensures consistent voice across all calls.
        """
        common_kwargs = dict(
            emit_every_phase1=self._emit_phase1,
            emit_every_phase2=self._emit_phase2,
            overlap_samples=self._overlap,
        )

        if self._voice_clone_prompt is not None:
            def _stream():
                return self._model.stream_voice_clone(
                    text,
                    language=self._language,
                    voice_clone_prompt=self._voice_clone_prompt,
                    ref_text=self._ref_text,
                    **common_kwargs,
                )
            return _stream
        else:
            def _stream():
                return self._model.stream(
                    text,
                    language=self._language,
                    speaker=self._speaker,
                    **common_kwargs,
                )
            return _stream

    @property
    def supports_streaming(self) -> bool:
        """MacawStreamingTTS suporta streaming real."""
        return True


# Auto-register quando o módulo é importado
register_tts_provider("macaw-streaming", MacawStreamingTTS)
