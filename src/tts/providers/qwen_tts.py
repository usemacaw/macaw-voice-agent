"""
Qwen3-TTS Provider - Sintese de voz local via modelos Qwen3-TTS.

Requer: pip install qwen-tts torch
Para GPU com Flash Attention 2: pip install flash-attn --no-build-isolation

Modelos disponiveis (CustomVoice):
- Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (leve, recomendado para CPU)
- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (maior qualidade, suporta instruct)

Configuracao via env vars:
    TTS_PROVIDER=qwen
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    QWEN_DEVICE=cpu              # ou cuda:0
    QWEN_TTS_SPEAKER=Ryan        # Voz pre-definida
    QWEN_TTS_LANGUAGE=Portuguese  # Override do idioma
    QWEN_TTS_INSTRUCT=           # Instrucao de estilo/emocao (1.7B only)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from common.config import TTS_CONFIG, AUDIO_CONFIG
from common.audio_utils import float32_to_pcm, resample, QWEN_LANGUAGE_MAP
from common.executor import run_inference
from tts.providers.base import TTSProvider, register_tts_provider

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("ai-agent.tts.qwen")

_OUTPUT_SAMPLE_RATE = 24000  # Qwen3-TTS gera audio a 24kHz


def _parse_generation_kwargs() -> dict:
    """Le generation kwargs de env vars.

    Retorna dict com apenas as chaves que foram explicitamente configuradas.
    Valores nao definidos ficam com o default do modelo.
    """
    kwargs = {}

    raw = os.getenv("QWEN_TTS_TEMPERATURE")
    if raw is not None:
        kwargs["temperature"] = float(raw)

    raw = os.getenv("QWEN_TTS_TOP_P")
    if raw is not None:
        kwargs["top_p"] = float(raw)

    raw = os.getenv("QWEN_TTS_TOP_K")
    if raw is not None:
        kwargs["top_k"] = int(raw)

    raw = os.getenv("QWEN_TTS_REPETITION_PENALTY")
    if raw is not None:
        kwargs["repetition_penalty"] = float(raw)

    return kwargs


class QwenTTS(TTSProvider):
    """TTS provider usando Qwen3-TTS para sintese de voz local.

    Carrega o modelo na GPU ou CPU conforme QWEN_DEVICE.
    Inferencia blocking e executada via run_inference() (ThreadPoolExecutor).

    Em GPU, usa Flash Attention 2 automaticamente quando flash-attn esta instalado.
    """

    provider_name = "qwen"

    def __init__(self):
        self._model: Qwen3TTSModel | None = None
        self._model_name = os.getenv(
            "QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
        self._device = os.getenv("QWEN_DEVICE", "cpu")
        self._speaker = os.getenv("QWEN_TTS_SPEAKER", "Ryan")

        # Instrucao de estilo/emocao (funciona apenas no modelo 1.7B;
        # o 0.6B ignora internamente)
        instruct_raw = os.getenv("QWEN_TTS_INSTRUCT", "")
        self._instruct = instruct_raw if instruct_raw else None

        # Generation kwargs opcionais (temperature, top_p, top_k, repetition_penalty)
        self._generation_kwargs = _parse_generation_kwargs()

        # Idioma: env var override > TTS_CONFIG > default
        lang_override = os.getenv("QWEN_TTS_LANGUAGE")
        if lang_override:
            self._language = lang_override.lower()
        else:
            self._language = QWEN_LANGUAGE_MAP.get(
                TTS_CONFIG.get("language", "pt"), "Portuguese"
            ).lower()

    async def connect(self) -> None:
        """Carrega modelo Qwen3-TTS."""
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = torch.float32 if self._device == "cpu" else torch.bfloat16

        # Flash Attention 2: usa em GPU quando flash-attn esta instalado
        from_pretrained_kwargs = {
            "device_map": self._device,
            "dtype": dtype,
        }
        if self._device != "cpu":
            try:
                import flash_attn  # noqa: F401
                from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 disponivel, sera usado")
            except ImportError:
                logger.info(
                    "flash-attn nao instalado, usando atencao manual do PyTorch. "
                    "Instale com: MAX_JOBS=4 pip install -U flash-attn --no-build-isolation"
                )

        logger.info(
            f"Carregando Qwen3-TTS: model={self._model_name}, "
            f"device={self._device}, dtype={dtype}, speaker={self._speaker}"
        )

        self._model = await run_inference(
            Qwen3TTSModel.from_pretrained,
            self._model_name,
            **from_pretrained_kwargs,
        )

        logger.info(f"Qwen3-TTS carregado: {self._model_name}")

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Qwen3-TTS descarregado")

    def _generate_and_convert(self, text: str, target_rate: int) -> bytes:
        """Generate + resample + convert em uma unica chamada sync.

        Roda inteiramente no executor (thread), evitando que resample e
        float32_to_pcm bloqueiem o event loop.

        Returns:
            Bytes PCM 16-bit LE no target_rate, ou b"" se modelo nao retornou audio.
        """
        wavs, sr = self._model.generate_custom_voice(
            text,
            self._speaker,
            language=self._language,
            instruct=self._instruct,
            **self._generation_kwargs,
        )

        if not wavs or len(wavs) == 0:
            return b""

        audio_float = wavs[0]
        audio_resampled = resample(audio_float, sr, target_rate)
        return float32_to_pcm(audio_resampled)

    async def synthesize(self, text: str) -> bytes:
        """Sintetiza texto em audio PCM 8kHz 16-bit.

        Todo o trabalho CPU-bound (generate + resample + convert) roda em
        uma unica chamada ao executor, liberando o event loop.
        """
        if not text.strip():
            return b""

        if self._model is None:
            raise RuntimeError("QwenTTS nao conectado. Chame connect() primeiro.")

        target_rate = AUDIO_CONFIG["sample_rate"]

        logger.info(
            f"TTS (qwen): sintetizando {len(text)} chars "
            f"(device={self._device}, speaker={self._speaker})..."
        )

        pcm_data = await run_inference(
            self._generate_and_convert, text, target_rate
        )

        if not pcm_data:
            logger.warning("TTS (qwen): modelo nao retornou audio")
            return b""

        logger.info(
            f"TTS (qwen): {len(pcm_data)} bytes, "
            f"{len(pcm_data) / (target_rate * 2):.1f}s"
        )
        return pcm_data


# Auto-register quando o modulo e importado
register_tts_provider("qwen", QwenTTS)
