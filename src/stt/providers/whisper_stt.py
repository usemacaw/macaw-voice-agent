"""
Faster-Whisper STT Provider — transcrição rápida via CTranslate2.

Suporta batch e streaming (pseudo-streaming via LocalAgreement).

Modo streaming: acumula audio durante a fala e re-transcreve incrementalmente.
Usa LocalAgreement-2: confirma texto quando 2 inferencias consecutivas concordam
no mesmo prefixo. Isso reduz latencia percebida — texto confirmado e retornado
antes do fim da fala.

Modelos recomendados:
- large-v3-turbo: melhor custo-beneficio (rapido + preciso)
- large-v3: maxima precisao
- medium: mais leve

Configuracao via env vars:
    STT_PROVIDER=whisper
    WHISPER_MODEL=large-v3-turbo
    WHISPER_DEVICE=cuda              # cuda | cpu | auto
    WHISPER_COMPUTE_TYPE=int8        # int8 | float16 | float32
    WHISPER_BEAM_SIZE=1              # 1 = greedy (mais rapido), 5 = beam search
    WHISPER_VAD_FILTER=true          # Filtra silencio antes de transcrever
    WHISPER_STREAM_CHUNK_MS=1000     # Intervalo minimo entre inferencias streaming
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from common.config import STT_CONFIG, AUDIO_CONFIG
from common.audio_utils import pcm_to_float32, resample
from common.executor import run_inference
from stt.providers.base import STTProvider, register_stt_provider

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger("ai-agent.stt.whisper")

_WHISPER_SAMPLE_RATE = 16000  # Whisper espera 16kHz


class WhisperSTT(STTProvider):
    """STT provider usando Faster-Whisper (CTranslate2).

    Otimizado para baixa latencia com int8 quantization em GPUs Turing+.
    Streaming via re-transcricao incremental com LocalAgreement-2.
    """

    provider_name = "whisper"

    def __init__(self):
        self._model: WhisperModel | None = None
        self._model_name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
        self._device = os.getenv("WHISPER_DEVICE", "auto")
        self._compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self._beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
        self._vad_filter = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
        self._language = STT_CONFIG.get("language", "pt")
        self._stream_chunk_ms = int(os.getenv("WHISPER_STREAM_CHUNK_MS", "1000"))

        # Per-stream states
        self._streaming_states: dict[str, dict] = {}

    async def connect(self) -> None:
        """Carrega modelo Faster-Whisper."""
        from faster_whisper import WhisperModel

        logger.info(
            f"Carregando Faster-Whisper: model={self._model_name}, "
            f"device={self._device}, compute_type={self._compute_type}, "
            f"beam_size={self._beam_size}, language={self._language}"
        )

        self._model = await run_inference(
            WhisperModel,
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

        logger.info(f"Faster-Whisper carregado: {self._model_name}")

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            self._streaming_states.clear()
            logger.info("Faster-Whisper descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcreve audio PCM 8kHz 16-bit para texto (modo batch)."""
        if not audio_data:
            return ""
        if self._model is None:
            raise RuntimeError("WhisperSTT nao conectado.")

        float_audio = pcm_to_float32(audio_data)
        source_rate = AUDIO_CONFIG["sample_rate"]
        float_audio_16k = resample(float_audio, source_rate, _WHISPER_SAMPLE_RATE)

        def _transcribe():
            return self._run_transcribe(float_audio_16k)

        text = await run_inference(_transcribe)

        if text:
            logger.info(f"STT (whisper): '{text}'")
        else:
            logger.debug("STT (whisper): nenhum texto detectado")

        return text

    def _run_transcribe(self, audio_16k: np.ndarray) -> str:
        """Executa transcricao Whisper (chamado em thread do executor)."""
        segments, _info = self._model.transcribe(
            audio_16k,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=200,
                speech_pad_ms=100,
            ),
        )
        texts = [s.text.strip() for s in segments]
        return " ".join(t for t in texts if t)

    # ==================== Streaming Interface ====================
    #
    # Pseudo-streaming via re-transcricao incremental com LocalAgreement-2.
    #
    # Fluxo:
    #   1. Audio chunks chegam durante a fala
    #   2. Acumula audio em buffer
    #   3. A cada ~1s, re-transcreve todo o audio acumulado
    #   4. Compara com transcricao anterior (LocalAgreement-2):
    #      - Palavras que aparecem no mesmo prefixo em 2+ inferencias → confirmadas
    #      - Retorna texto confirmado + texto nao-confirmado como parcial
    #   5. No finish_streaming, transcreve audio completo (resultado final)

    @property
    def supports_streaming(self) -> bool:
        return True

    async def start_streaming(self, stream_id: str = "") -> None:
        """Inicia sessao de streaming."""
        if self._model is None:
            raise RuntimeError("WhisperSTT nao conectado.")

        self._streaming_states[stream_id] = {
            "audio_buffer": bytearray(),  # PCM 8kHz acumulado
            "confirmed_words": [],  # Palavras confirmadas por LocalAgreement
            "prev_words": [],  # Palavras da inferencia anterior
            "last_inference_time": 0.0,  # Timestamp da ultima inferencia
        }

        logger.debug(f"Whisper streaming iniciado (stream_id={stream_id[:8] or 'default'})")

    async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
        """Processa chunk de audio e retorna transcricao parcial."""
        state = self._streaming_states.get(stream_id)
        if state is None:
            raise RuntimeError("Streaming nao iniciado.")

        if not audio_chunk:
            return self._get_current_text(state)

        state["audio_buffer"].extend(audio_chunk)

        # Verifica se ja passou tempo suficiente desde a ultima inferencia
        now = time.monotonic()
        elapsed_ms = (now - state["last_inference_time"]) * 1000
        if elapsed_ms < self._stream_chunk_ms:
            return self._get_current_text(state)

        # Minimo de audio para inferir (500ms)
        source_rate = AUDIO_CONFIG["sample_rate"]
        min_bytes = int(source_rate * 2 * 0.5)
        if len(state["audio_buffer"]) < min_bytes:
            return self._get_current_text(state)

        state["last_inference_time"] = now

        # Transcreve todo o audio acumulado
        audio_bytes = bytes(state["audio_buffer"])
        float_audio = pcm_to_float32(audio_bytes)
        float_audio_16k = resample(float_audio, source_rate, _WHISPER_SAMPLE_RATE)

        def _process():
            text = self._run_transcribe(float_audio_16k)
            current_words = text.split() if text else []
            self._update_local_agreement(state, current_words)
            return self._get_current_text(state)

        text = await run_inference(_process)
        return text

    async def finish_streaming(self, stream_id: str = "") -> str:
        """Finaliza streaming — transcreve audio completo para resultado final."""
        state = self._streaming_states.pop(stream_id, None)
        if state is None:
            raise RuntimeError("Streaming nao iniciado.")

        if not state["audio_buffer"]:
            text = self._get_current_text(state)
        else:
            audio_bytes = bytes(state["audio_buffer"])
            float_audio = pcm_to_float32(audio_bytes)
            source_rate = AUDIO_CONFIG["sample_rate"]
            float_audio_16k = resample(float_audio, source_rate, _WHISPER_SAMPLE_RATE)

            def _finalize():
                return self._run_transcribe(float_audio_16k)

            text = await run_inference(_finalize)

        if text:
            logger.info(f"STT streaming final: '{text}'")
        else:
            logger.debug("STT streaming final: nenhum texto detectado")

        return text

    @staticmethod
    def _update_local_agreement(state: dict, current_words: list[str]) -> None:
        """Atualiza palavras confirmadas via LocalAgreement-2.

        Compara palavras atuais com as da inferencia anterior.
        Palavras que coincidem no mesmo indice (prefixo) sao confirmadas.
        """
        prev_words = state["prev_words"]
        confirmed = state["confirmed_words"]
        confirmed_len = len(confirmed)

        # Compara a partir das palavras ja confirmadas
        prev_tail = prev_words[confirmed_len:]
        curr_tail = current_words[confirmed_len:]

        # Encontra prefixo comum entre prev_tail e curr_tail
        new_confirmed = 0
        for pw, cw in zip(prev_tail, curr_tail):
            if pw.lower() == cw.lower():
                new_confirmed += 1
            else:
                break

        if new_confirmed > 0:
            # Usa as palavras da inferencia atual (preserva casing)
            state["confirmed_words"] = (
                confirmed + current_words[confirmed_len : confirmed_len + new_confirmed]
            )

        state["prev_words"] = current_words

    @staticmethod
    def _get_current_text(state: dict) -> str:
        """Retorna texto atual: confirmado + nao-confirmado da ultima inferencia."""
        confirmed = state["confirmed_words"]
        prev = state["prev_words"]

        if not confirmed and not prev:
            return ""

        # Retorna tudo que temos (confirmado + tail nao-confirmado)
        if len(prev) > len(confirmed):
            return " ".join(prev)
        return " ".join(confirmed) if confirmed else ""


register_stt_provider("whisper", WhisperSTT)
