"""
NeMo Streaming STT Provider — streaming ASR via NVIDIA NeMo FastConformer.

Usa conformer_stream_step para streaming real chunk-by-chunk com cache de
atencao e convolucao, evitando reprocessar todo o audio a cada chunk.

Modelo recomendado para PT-BR:
    nvidia/stt_pt_fastconformer_hybrid_large_pc (~115M params, ~2200h PT-BR)

Configuracao via env vars:
    STT_PROVIDER=fastconformer
    FASTCONFORMER_MODEL=nvidia/stt_pt_fastconformer_hybrid_large_pc
    FASTCONFORMER_DEVICE=cuda:0
    FASTCONFORMER_CHUNK_MS=640
    FASTCONFORMER_LEFT_CHUNKS=4
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from common.config import STT_CONFIG, AUDIO_CONFIG
from common.audio_utils import pcm_to_float32, resample
from common.executor import run_inference
from stt.providers.base import STTProvider, register_stt_provider

logger = logging.getLogger("ai-agent.stt.fastconformer")

_INPUT_SAMPLE_RATE = 16000

# FastConformer usa subsampling 8x com features de 10ms → cada encoder frame = 80ms
_ENCODER_SUBSAMPLING = 8
_FEATURE_FRAME_MS = 10  # mel feature frame duration


class FastConformerSTT(STTProvider):
    """STT provider usando NVIDIA NeMo FastConformer PT-BR.

    Modo batch via model.transcribe() — robusto e confiavel.
    Modo streaming via conformer_stream_step — processa chunks com cache.
    """

    provider_name = "fastconformer"

    def __init__(self):
        self._model = None
        self._device = None
        self._model_name = os.getenv(
            "FASTCONFORMER_MODEL",
            "nvidia/stt_pt_fastconformer_hybrid_large_pc",
        )
        self._device_str = os.getenv("FASTCONFORMER_DEVICE", "cuda:0")
        self._language = STT_CONFIG.get("language", "pt")

        # Streaming config
        self._chunk_ms = int(os.getenv("FASTCONFORMER_CHUNK_MS", "640"))
        self._left_chunks = int(os.getenv("FASTCONFORMER_LEFT_CHUNKS", "4"))

        # Per-stream states (concurrent sessions)
        self._streaming_states: dict[str, dict] = {}

    async def connect(self) -> None:
        """Carrega modelo NeMo."""
        import nemo.collections.asr as nemo_asr

        logger.info(
            f"Carregando NeMo ASR: model={self._model_name}, "
            f"device={self._device_str}"
        )

        self._model = await run_inference(
            nemo_asr.models.ASRModel.from_pretrained,
            model_name=self._model_name,
        )
        self._model.eval()
        self._model.to(self._device_str)
        self._device = self._model.device

        # Configura streaming no encoder
        chunk_encoder_frames = max(
            1,
            int(self._chunk_ms / (_FEATURE_FRAME_MS * _ENCODER_SUBSAMPLING)),
        )
        self._chunk_encoder_frames = chunk_encoder_frames
        self._chunk_samples = int(_INPUT_SAMPLE_RATE * self._chunk_ms / 1000)

        self._model.encoder.setup_streaming_params(
            chunk_size=chunk_encoder_frames,
            shift_size=chunk_encoder_frames,
            left_chunks=self._left_chunks,
        )

        logger.info(
            f"NeMo ASR carregado: {self._model_name} "
            f"({sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M params), "
            f"streaming chunk={self._chunk_ms}ms ({chunk_encoder_frames} enc frames)"
        )

    async def disconnect(self) -> None:
        """Libera modelo da memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            self._streaming_states.clear()
            logger.info("NeMo ASR descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcreve audio PCM 8kHz 16-bit para texto (modo batch)."""
        if not audio_data:
            return ""
        if self._model is None:
            raise RuntimeError("FastConformerSTT nao conectado.")

        float_audio = pcm_to_float32(audio_data)
        source_rate = AUDIO_CONFIG["sample_rate"]
        float_audio_16k = resample(float_audio, source_rate, _INPUT_SAMPLE_RATE)

        def _transcribe():
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, float_audio_16k, _INPUT_SAMPLE_RATE)
                results = self._model.transcribe([f.name])

            if results and len(results) > 0:
                result = results[0]
                if isinstance(result, str):
                    return result
                if hasattr(result, "text"):
                    return result.text
                if isinstance(results, tuple) and results[0]:
                    r = results[0][0]
                    return r if isinstance(r, str) else getattr(r, "text", "")
            return ""

        text = await run_inference(_transcribe)

        if text:
            logger.info(f"STT (nemo): '{text}'")
        else:
            logger.debug("STT (nemo): nenhum texto detectado")

        return text

    # ==================== Streaming Interface ====================

    @property
    def supports_streaming(self) -> bool:
        return True

    async def start_streaming(self, stream_id: str = "") -> None:
        """Inicia sessao de streaming via conformer_stream_step."""
        if self._model is None:
            raise RuntimeError("FastConformerSTT nao conectado.")

        # Cache inicial do encoder (atencao + convolucao)
        cache_channel, cache_time, cache_channel_len = (
            self._model.encoder.get_initial_cache_state(
                batch_size=1,
                dtype=torch.float32,
                device=self._device_str,
            )
        )

        self._streaming_states[stream_id] = {
            "cache_last_channel": cache_channel,
            "cache_last_time": cache_time,
            "cache_last_channel_len": cache_channel_len,
            "previous_hypotheses": None,
            "accumulated_text": "",
            "audio_buffer": bytearray(),
        }

        logger.debug(
            f"NeMo streaming iniciado (stream_id={stream_id[:8] or 'default'})"
        )

    async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
        """Processa chunk de audio e retorna transcricao parcial."""
        state = self._streaming_states.get(stream_id)
        if state is None:
            raise RuntimeError("Streaming nao iniciado.")

        if not audio_chunk:
            return state["accumulated_text"]

        state["audio_buffer"].extend(audio_chunk)

        # Processa quando tiver pelo menos 1 chunk completo
        source_rate = AUDIO_CONFIG["sample_rate"]
        min_bytes = int(source_rate * 2 * self._chunk_ms / 1000)
        if len(state["audio_buffer"]) < min_bytes:
            return state["accumulated_text"]

        audio_bytes = bytes(state["audio_buffer"])
        state["audio_buffer"].clear()

        float_audio = pcm_to_float32(audio_bytes)
        float_audio_16k = resample(float_audio, source_rate, _INPUT_SAMPLE_RATE)

        def _process():
            return self._stream_step(state, float_audio_16k)

        text = await run_inference(_process)
        return text

    async def finish_streaming(self, stream_id: str = "") -> str:
        """Finaliza streaming e retorna transcricao final."""
        state = self._streaming_states.pop(stream_id, None)
        if state is None:
            raise RuntimeError("Streaming nao iniciado.")

        # Processa audio residual
        if state["audio_buffer"]:
            float_audio = pcm_to_float32(bytes(state["audio_buffer"]))
            source_rate = AUDIO_CONFIG["sample_rate"]
            float_audio_16k = resample(float_audio, source_rate, _INPUT_SAMPLE_RATE)

            def _finalize():
                return self._stream_step(state, float_audio_16k)

            text = await run_inference(_finalize)
        else:
            text = state["accumulated_text"]

        if text:
            logger.info(f"STT streaming final: '{text}'")
        else:
            logger.debug("STT streaming final: nenhum texto detectado")

        return text

    def _stream_step(self, state: dict, float_audio_16k: np.ndarray) -> str:
        """Processa audio float32 16kHz em chunks via conformer_stream_step.

        Divide o audio em chunks do tamanho configurado e processa cada um
        sequencialmente, mantendo cache de atencao entre chunks.
        """
        chunk_samples = self._chunk_samples
        model = self._model

        with torch.no_grad():
            for i in range(0, len(float_audio_16k), chunk_samples):
                chunk = float_audio_16k[i : i + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                audio_tensor = (
                    torch.tensor(chunk, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self._device_str)
                )
                audio_len = torch.tensor(
                    [len(chunk)], dtype=torch.long
                ).to(self._device_str)

                processed_signal, processed_signal_length = (
                    model.preprocessor(
                        input_signal=audio_tensor, length=audio_len
                    )
                )

                result = model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=state["cache_last_channel"],
                    cache_last_time=state["cache_last_time"],
                    cache_last_channel_len=state["cache_last_channel_len"],
                    keep_all_outputs=True,
                    previous_hypotheses=state["previous_hypotheses"],
                    return_transcription=True,
                )

                # Unpack: greedy_preds, all_hyp, cache_channel, cache_time,
                #         cache_channel_len, best_hyp
                (
                    _greedy_preds,
                    all_hyp,
                    state["cache_last_channel"],
                    state["cache_last_time"],
                    state["cache_last_channel_len"],
                    best_hyp,
                ) = result[:6]

                # Atualiza hypotheses para proximo chunk
                state["previous_hypotheses"] = (
                    all_hyp if isinstance(all_hyp, list) else None
                )

                # Extrai texto
                text = self._extract_text(best_hyp, all_hyp)
                if text:
                    state["accumulated_text"] = text

        return state["accumulated_text"]

    @staticmethod
    def _extract_text(best_hyp, all_hyp) -> str:
        """Extrai texto das hipoteses do decoder RNNT."""
        if best_hyp and len(best_hyp) > 0:
            hyp = best_hyp[0]
            if hasattr(hyp, "text") and hyp.text:
                return hyp.text
        if isinstance(all_hyp, list) and len(all_hyp) > 0:
            hyp = all_hyp[0]
            if hasattr(hyp, "text") and hyp.text:
                return hyp.text
            if isinstance(hyp, str) and hyp:
                return hyp
        return ""


# Auto-register
register_stt_provider("fastconformer", FastConformerSTT)
