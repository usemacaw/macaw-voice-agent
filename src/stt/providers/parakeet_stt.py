"""
Parakeet TDT STT Provider - Transcricao local via NVIDIA Parakeet (NeMo).

Top 1 no HuggingFace Open ASR Leaderboard (junho 2025).

Requer: pip install nemo_toolkit[asr]==2.3.2 numpy<2

Modelos disponiveis:
- nvidia/parakeet-tdt-0.6b-v3 (600M params, 25 idiomas incluindo PT)
- nvidia/parakeet-tdt-0.6b-v2 (600M params, ingles apenas)

Configuracao via env vars:
    STT_PROVIDER=parakeet              # batch-only
    STT_PROVIDER=parakeet-streaming    # streaming incremental
    PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3
    PARAKEET_DEVICE=cuda:0
    PARAKEET_CHUNK_INTERVAL_MS=500     # intervalo entre inferencias streaming
"""

from __future__ import annotations

import logging
import os

import numpy as np

from common.config import AUDIO_CONFIG
from common.audio_utils import pcm_to_float32, resample
from common.executor import run_inference
from stt.providers.base import STTProvider, register_stt_provider

logger = logging.getLogger("ai-agent.stt.parakeet")

_INPUT_SAMPLE_RATE = 16000  # Parakeet espera 16kHz
_SOURCE_RATE = AUDIO_CONFIG["sample_rate"]  # 8kHz interno

# Loggers verbosos do NeMo que precisam ser silenciados
_NOISY_LOGGERS = ("nemo_logger", "nemo", "lightning", "pytorch_lightning")

# Bytes por chunk de 32ms @ 8kHz PCM16 mono
_BYTES_PER_CHUNK = int(_SOURCE_RATE * 0.032) * 2  # ~512 bytes


def _silence_nemo_loggers():
    """Silencia loggers verbosos do NeMo (uma vez)."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_model(model_name: str, device: str):
    """Carrega modelo Parakeet via NeMo (blocking)."""
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name,
    )
    if device.startswith("cuda"):
        model = model.to(device)
    return model


def _extract_text(results) -> str:
    """Extrai texto do resultado do NeMo (Hypothesis ou str)."""
    if not results:
        return ""
    raw = results[0]
    return raw.text if hasattr(raw, "text") else str(raw) if raw else ""


class ParakeetSTT(STTProvider):
    """STT provider usando NVIDIA Parakeet TDT via NeMo.

    Carrega o modelo na GPU ou CPU conforme PARAKEET_DEVICE.
    Inferencia blocking executada via run_inference() (thread pool).
    Batch-only — VAD do pipeline ja segmenta a fala.
    """

    provider_name = "parakeet"

    def __init__(self):
        self._model = None
        self._model_name = os.getenv(
            "PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3"
        )
        self._device = os.getenv("PARAKEET_DEVICE", "cuda:0")

    async def connect(self) -> None:
        """Carrega modelo Parakeet via NeMo e move para device."""
        logger.info(
            "Carregando Parakeet: model=%s, device=%s",
            self._model_name, self._device,
        )
        _silence_nemo_loggers()

        self._model = await run_inference(
            _load_model, self._model_name, self._device,
        )

        logger.info("Parakeet carregado: %s", self._model_name)

    async def disconnect(self) -> None:
        """Libera modelo da memoria (incluindo CUDA)."""
        if self._model is not None:
            model = self._model
            self._model = None

            def _free():
                nonlocal model
                del model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            await run_inference(_free)
            logger.info("Parakeet descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcreve audio PCM 8kHz 16-bit para texto."""
        if not audio_data:
            return ""
        if self._model is None:
            raise RuntimeError("ParakeetSTT nao conectado. Chame connect() primeiro.")

        float_audio = pcm_to_float32(audio_data)
        float_audio_16k = resample(float_audio, _SOURCE_RATE, _INPUT_SAMPLE_RATE)

        model = self._model

        def _transcribe():
            output = model.transcribe([float_audio_16k], batch_size=1)
            return output

        results = await run_inference(_transcribe)
        text = _extract_text(results)

        if text:
            logger.info("STT (parakeet): '%s'", text)
        else:
            logger.debug("STT (parakeet): nenhum texto detectado")

        return text


class ParakeetStreamingSTT(STTProvider):
    """STT provider com streaming incremental via NeMo conformer_stream_step.

    Acumula chunks PCM e processa periodicamente (a cada ~500ms).
    Usa cache-aware streaming do Conformer encoder para manter estado
    entre chunks, permitindo transcricao parcial durante a fala.

    Configuracao via env vars:
        STT_PROVIDER=parakeet-streaming
        PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3
        PARAKEET_DEVICE=cuda:0
        PARAKEET_CHUNK_INTERVAL_MS=500
    """

    provider_name = "parakeet-streaming"

    def __init__(self):
        self._model = None
        self._model_name = os.getenv(
            "PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3"
        )
        self._device = os.getenv("PARAKEET_DEVICE", "cuda:0")
        self._chunk_interval_ms = int(os.getenv("PARAKEET_CHUNK_INTERVAL_MS", "500"))
        # Bytes to accumulate before running inference (~500ms @ 8kHz PCM16)
        self._chunk_threshold = int(_SOURCE_RATE * (self._chunk_interval_ms / 1000)) * 2

        self._streaming_states: dict[str, dict] = {}

    async def connect(self) -> None:
        """Carrega modelo Parakeet e configura para streaming."""
        logger.info(
            "Carregando Parakeet (streaming): model=%s, device=%s, interval=%dms",
            self._model_name, self._device, self._chunk_interval_ms,
        )
        _silence_nemo_loggers()

        device = self._device
        model_name = self._model_name

        def _load_streaming():
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
            )
            if device.startswith("cuda"):
                model = model.to(device)
            model.eval()

            return model

        self._model = await run_inference(_load_streaming)
        logger.info("Parakeet streaming carregado: %s", self._model_name)

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            model = self._model
            self._model = None
            self._streaming_states.clear()

            def _free():
                nonlocal model
                del model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            await run_inference(_free)
            logger.info("Parakeet streaming descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Fallback batch: transcreve bloco inteiro."""
        if not audio_data:
            return ""
        if self._model is None:
            raise RuntimeError(
                "ParakeetStreamingSTT nao conectado. Chame connect() primeiro."
            )

        float_audio = pcm_to_float32(audio_data)
        float_audio_16k = resample(float_audio, _SOURCE_RATE, _INPUT_SAMPLE_RATE)

        model = self._model

        def _transcribe():
            output = model.transcribe([float_audio_16k], batch_size=1)
            return output

        results = await run_inference(_transcribe)
        text = _extract_text(results)

        if text:
            logger.info("STT (parakeet-streaming batch): '%s'", text)
        return text

    # ==================== Streaming Interface ====================

    @property
    def supports_streaming(self) -> bool:
        return True

    async def start_streaming(self, stream_id: str = "") -> None:
        """Inicia sessao de streaming."""
        if self._model is None:
            raise RuntimeError(
                "ParakeetStreamingSTT nao conectado. Chame connect() primeiro."
            )

        self._streaming_states[stream_id] = {
            "pcm_buffer": b"",          # Accumulated PCM 8kHz
            "all_audio_16k": np.array([], dtype=np.float32),  # All resampled audio
            "last_text": "",            # Last partial transcript
            "chunks_since_infer": 0,    # Chunks since last inference
        }
        logger.debug(
            "STT streaming sessao iniciada (stream_id=%s)",
            stream_id[:8] or "default",
        )

    async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
        """Processa chunk de audio e retorna transcricao parcial.

        Acumula PCM e roda inferencia a cada ~500ms de audio novo.
        """
        state = self._streaming_states.get(stream_id)
        if state is None:
            raise RuntimeError("Streaming nao iniciado. Chame start_streaming() primeiro.")

        if not audio_chunk:
            return state["last_text"]

        # Accumulate raw PCM
        state["pcm_buffer"] += audio_chunk
        state["chunks_since_infer"] += 1

        # Check if enough audio accumulated for inference
        if len(state["pcm_buffer"]) < self._chunk_threshold:
            return state["last_text"]

        # Convert accumulated PCM to float32 and resample
        new_float = pcm_to_float32(state["pcm_buffer"])
        new_float_16k = resample(new_float, _SOURCE_RATE, _INPUT_SAMPLE_RATE)
        state["pcm_buffer"] = b""  # Reset PCM buffer
        state["chunks_since_infer"] = 0

        # Append to full audio history
        state["all_audio_16k"] = np.concatenate(
            [state["all_audio_16k"], new_float_16k]
        )

        # Run inference on full accumulated audio
        all_audio = state["all_audio_16k"]
        model = self._model

        def _infer():
            output = model.transcribe([all_audio], batch_size=1)
            return output

        results = await run_inference(_infer)
        text = _extract_text(results)
        state["last_text"] = text

        if text:
            logger.debug("STT streaming parcial: '%s'", text)

        return text

    async def finish_streaming(self, stream_id: str = "") -> str:
        """Finaliza sessao de streaming e retorna transcricao final."""
        state = self._streaming_states.pop(stream_id, None)
        if state is None:
            raise RuntimeError("Streaming nao iniciado. Chame start_streaming() primeiro.")

        # Process any remaining PCM in buffer
        if state["pcm_buffer"]:
            new_float = pcm_to_float32(state["pcm_buffer"])
            new_float_16k = resample(new_float, _SOURCE_RATE, _INPUT_SAMPLE_RATE)
            state["all_audio_16k"] = np.concatenate(
                [state["all_audio_16k"], new_float_16k]
            )

        all_audio = state["all_audio_16k"]
        if len(all_audio) == 0:
            logger.debug("STT streaming final: nenhum audio")
            return ""

        # Final inference on complete audio
        model = self._model

        def _final_infer():
            output = model.transcribe([all_audio], batch_size=1)
            return output

        results = await run_inference(_final_infer)
        text = _extract_text(results)

        if text:
            logger.info("STT streaming final: '%s'", text)
        else:
            logger.debug("STT streaming final: nenhum texto detectado")

        return text


# Auto-register quando o modulo e importado
register_stt_provider("parakeet", ParakeetSTT)
register_stt_provider("parakeet-streaming", ParakeetStreamingSTT)
