"""
Parakeet TDT STT Provider - Transcricao local via NVIDIA Parakeet (NeMo).

Top 1 no HuggingFace Open ASR Leaderboard (junho 2025).

Requer: pip install nemo_toolkit[asr]==2.3.2 numpy<2

Modelo padrao:
- nvidia/parakeet-tdt-0.6b-v2 (600M params, ingles otimizado)

Configuracao via env vars:
    STT_PROVIDER=parakeet
    PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2
    PARAKEET_DEVICE=cuda:0       # ou cpu
"""

from __future__ import annotations

import logging
import os

from common.config import AUDIO_CONFIG
from common.audio_utils import pcm_to_float32, resample
from common.executor import run_inference
from stt.providers.base import STTProvider, register_stt_provider

logger = logging.getLogger("ai-agent.stt.parakeet")

_INPUT_SAMPLE_RATE = 16000  # Parakeet espera 16kHz
_SOURCE_RATE = AUDIO_CONFIG["sample_rate"]  # 8kHz interno

# Loggers verbosos do NeMo que precisam ser silenciados
_NOISY_LOGGERS = ("nemo_logger", "nemo", "lightning", "pytorch_lightning")


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
            "PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2"
        )
        self._device = os.getenv("PARAKEET_DEVICE", "cuda:0")

    async def connect(self) -> None:
        """Carrega modelo Parakeet via NeMo e move para device."""
        logger.info(
            f"Carregando Parakeet: model={self._model_name}, "
            f"device={self._device}"
        )

        # Silencia logs verbosos do NeMo (uma vez, antes do load)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)

        device = self._device
        model_name = self._model_name

        def _load():
            import nemo.collections.asr as nemo_asr

            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
            )
            if device.startswith("cuda"):
                model = model.to(device)
            return model

        self._model = await run_inference(_load)

        logger.info(f"Parakeet carregado: {self._model_name}")

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
        """Transcreve audio PCM 8kHz 16-bit para texto.

        1. PCM bytes -> numpy float32
        2. Resample 8kHz -> 16kHz
        3. Parakeet transcribe
        """
        if not audio_data:
            return ""

        if self._model is None:
            raise RuntimeError("ParakeetSTT nao conectado. Chame connect() primeiro.")

        # PCM -> float32
        float_audio = pcm_to_float32(audio_data)

        # Resample to 16kHz (Parakeet espera 16kHz)
        float_audio_16k = resample(float_audio, _SOURCE_RATE, _INPUT_SAMPLE_RATE)

        model = self._model

        def _transcribe():
            output = model.transcribe([float_audio_16k], batch_size=1)
            return output

        results = await run_inference(_transcribe)

        # NeMo pode retornar list[Hypothesis] ou list[str] dependendo da config
        raw = results[0] if results else ""
        text = raw.text if hasattr(raw, "text") else str(raw) if raw else ""

        if text:
            logger.info(f"STT (parakeet): '{text}'")
        else:
            logger.debug("STT (parakeet): nenhum texto detectado")

        return text


# Auto-register quando o modulo e importado
register_stt_provider("parakeet", ParakeetSTT)
