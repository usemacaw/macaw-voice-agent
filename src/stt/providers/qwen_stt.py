"""
Qwen3-ASR STT Provider - Transcricao local via modelos Qwen3-ASR.

Requer: pip install qwen-asr torch

Modelos disponiveis:
- Qwen/Qwen3-ASR-0.6B (leve, recomendado para CPU)
- Qwen/Qwen3-ASR-1.7B (maior precisao)

Configuracao via env vars:
    STT_PROVIDER=qwen
    QWEN_STT_MODEL=Qwen/Qwen3-ASR-0.6B
    QWEN_DEVICE=cpu              # ou cuda:0
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from common.config import STT_CONFIG, AUDIO_CONFIG
from common.audio_utils import pcm_to_float32, resample
from common.executor import run_inference
from stt.providers.base import STTProvider, register_stt_provider

if TYPE_CHECKING:
    from qwen_asr import Qwen3ASRModel

logger = logging.getLogger("ai-agent.stt.qwen")

_LANG_MAP = {
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "hu": "Hungarian",
    "ro": "Romanian",
    "ms": "Malay",
}

_INPUT_SAMPLE_RATE = 16000  # Qwen3-ASR espera 16kHz


class QwenSTT(STTProvider):
    """STT provider usando Qwen3-ASR para transcricao local.

    Carrega o modelo na GPU ou CPU conforme QWEN_DEVICE.
    Inferencia blocking e executada via asyncio.to_thread().
    """

    provider_name = "qwen"

    def __init__(self):
        self._model: Qwen3ASRModel | None = None
        self._model_name = os.getenv("QWEN_STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
        self._device = os.getenv("QWEN_DEVICE", "cpu")
        self._language = _LANG_MAP.get(
            STT_CONFIG.get("language", "pt"), "Portuguese"
        )

    async def connect(self) -> None:
        """Carrega modelo Qwen3-ASR."""
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = torch.float32 if self._device == "cpu" else torch.bfloat16

        logger.info(
            f"Carregando Qwen3-ASR: model={self._model_name}, "
            f"device={self._device}, dtype={dtype}"
        )

        self._model = await run_inference(
            Qwen3ASRModel.from_pretrained,
            self._model_name,
            device_map=self._device,
            dtype=dtype,
        )

        logger.info(f"Qwen3-ASR carregado: {self._model_name}")

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Qwen3-ASR descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcreve audio PCM 8kHz 16-bit para texto.

        1. PCM bytes -> numpy float32
        2. Resample 8kHz -> 16kHz
        3. Qwen3-ASR transcribe
        """
        if not audio_data:
            return ""

        if self._model is None:
            raise RuntimeError("QwenSTT nao conectado. Chame connect() primeiro.")

        # PCM -> float32
        float_audio = pcm_to_float32(audio_data)

        # Resample to 16kHz (Qwen3-ASR espera 16kHz)
        source_rate = AUDIO_CONFIG["sample_rate"]
        float_audio_16k = resample(float_audio, source_rate, _INPUT_SAMPLE_RATE)

        results = await run_inference(
            self._model.transcribe,
            audio=(float_audio_16k, _INPUT_SAMPLE_RATE),
            language=self._language,
        )

        text = results[0].text if results else ""
        if text:
            logger.info(f"STT (qwen): '{text}'")
        else:
            logger.debug("STT (qwen): nenhum texto detectado")

        return text


class QwenStreamingSTT(STTProvider):
    """STT provider usando Qwen3-ASR com streaming incremental via vLLM.

    Transcreve audio em tempo real enquanto o usuario fala, em vez de
    esperar speech_end e processar o bloco inteiro.

    Requer: pip install qwen-asr[vllm]

    Configuracao via env vars:
        STT_PROVIDER=qwen-streaming
        QWEN_STT_MODEL=Qwen/Qwen3-ASR-1.7B
        QWEN_DEVICE=cuda:0              # vLLM requer GPU
        QWEN_STT_CHUNK_SIZE_SEC=2.0     # Tamanho do chunk em segundos
        QWEN_STT_UNFIXED_CHUNK_NUM=2    # Chunks nao fixados para refinamento
        QWEN_STT_UNFIXED_TOKEN_NUM=5    # Tokens nao fixados para refinamento
    """

    provider_name = "qwen-streaming"

    def __init__(self):
        self._model = None
        self._model_name = os.getenv("QWEN_STT_MODEL", "Qwen/Qwen3-ASR-1.7B")
        self._device = os.getenv("QWEN_DEVICE", "cuda:0")
        self._language = _LANG_MAP.get(
            STT_CONFIG.get("language", "pt"), "Portuguese"
        )

        # Streaming config
        self._chunk_size_sec = float(os.getenv("QWEN_STT_CHUNK_SIZE_SEC", "2.0"))
        self._unfixed_chunk_num = int(os.getenv("QWEN_STT_UNFIXED_CHUNK_NUM", "2"))
        self._unfixed_token_num = int(os.getenv("QWEN_STT_UNFIXED_TOKEN_NUM", "5"))

        # Per-stream streaming states indexed by stream_id
        # Allows concurrent sessions on the same singleton provider
        self._streaming_states: dict[str, object] = {}

    async def connect(self) -> None:
        """Carrega modelo Qwen3-ASR com backend vLLM para streaming."""
        from qwen_asr import Qwen3ASRModel

        gpu_mem = float(os.getenv("QWEN_STT_GPU_MEM_UTIL", "0.25"))
        max_tokens = int(os.getenv("QWEN_STT_MAX_NEW_TOKENS", "32"))
        max_model_len = int(os.getenv("QWEN_STT_MAX_MODEL_LEN", "4096"))
        enforce_eager = os.getenv(
            "QWEN_STT_ENFORCE_EAGER", "true"
        ).lower() in ("true", "1", "yes")

        logger.info(
            f"Carregando Qwen3-ASR (vLLM streaming): model={self._model_name}, "
            f"device={self._device}, gpu_mem={gpu_mem}, "
            f"max_model_len={max_model_len}, enforce_eager={enforce_eager}"
        )

        self._model = await run_inference(
            Qwen3ASRModel.LLM,
            model=self._model_name,
            gpu_memory_utilization=gpu_mem,
            max_new_tokens=max_tokens,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )

        logger.info(f"Qwen3-ASR streaming carregado: {self._model_name}")

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            self._streaming_states.clear()
            logger.info("Qwen3-ASR streaming descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Fallback batch: processa bloco inteiro de audio.

        Usa streaming internamente: init → transcribe → finish.
        """
        if not audio_data:
            return ""

        if self._model is None:
            raise RuntimeError("QwenStreamingSTT nao conectado. Chame connect() primeiro.")

        float_audio = pcm_to_float32(audio_data)
        source_rate = AUDIO_CONFIG["sample_rate"]
        float_audio_16k = resample(float_audio, source_rate, _INPUT_SAMPLE_RATE)

        def _batch_transcribe():
            state = self._model.init_streaming_state(
                language=self._language,
                unfixed_chunk_num=self._unfixed_chunk_num,
                unfixed_token_num=self._unfixed_token_num,
                chunk_size_sec=self._chunk_size_sec,
            )
            self._model.streaming_transcribe(float_audio_16k, state)
            self._model.finish_streaming_transcribe(state)
            return state.text

        text = await run_inference(_batch_transcribe)

        if text:
            logger.info(f"STT (qwen-streaming batch): '{text}'")
        else:
            logger.debug("STT (qwen-streaming batch): nenhum texto detectado")

        return text

    # ==================== Streaming Interface ====================

    @property
    def supports_streaming(self) -> bool:
        return True

    async def start_streaming(self, stream_id: str = "") -> None:
        """Inicia/reinicia sessao de streaming STT.

        Args:
            stream_id: Identificador do stream (ex: session_id) para isolamento
                       de estado entre sessoes concorrentes.
        """
        if self._model is None:
            raise RuntimeError("QwenStreamingSTT nao conectado. Chame connect() primeiro.")

        state = await run_inference(
            self._model.init_streaming_state,
            language=self._language,
            unfixed_chunk_num=self._unfixed_chunk_num,
            unfixed_token_num=self._unfixed_token_num,
            chunk_size_sec=self._chunk_size_sec,
        )
        self._streaming_states[stream_id] = state
        logger.debug(f"STT streaming sessao iniciada (stream_id={stream_id[:8] or 'default'})")

    async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
        """Processa chunk de audio e retorna transcricao parcial acumulada.

        1. PCM bytes -> numpy float32
        2. Resample 8kHz -> 16kHz
        3. streaming_transcribe() atualiza state.text
        """
        state = self._streaming_states.get(stream_id)
        if state is None:
            raise RuntimeError("Streaming nao iniciado. Chame start_streaming() primeiro.")

        if not audio_chunk:
            return state.text or ""

        float_audio = pcm_to_float32(audio_chunk)
        source_rate = AUDIO_CONFIG["sample_rate"]
        float_audio_16k = resample(float_audio, source_rate, _INPUT_SAMPLE_RATE)

        await run_inference(
            self._model.streaming_transcribe,
            float_audio_16k,
            state,
        )

        text = state.text or ""
        if text:
            logger.debug(f"STT streaming parcial: '{text}'")
        return text

    async def finish_streaming(self, stream_id: str = "") -> str:
        """Finaliza sessao de streaming e retorna transcricao final."""
        state = self._streaming_states.pop(stream_id, None)
        if state is None:
            raise RuntimeError("Streaming nao iniciado. Chame start_streaming() primeiro.")

        await run_inference(
            self._model.finish_streaming_transcribe,
            state,
        )

        text = state.text or ""

        if text:
            logger.info(f"STT streaming final: '{text}'")
        else:
            logger.debug("STT streaming final: nenhum texto detectado")

        return text


# Auto-register quando o modulo e importado
register_stt_provider("qwen", QwenSTT)
register_stt_provider("qwen-streaming", QwenStreamingSTT)
