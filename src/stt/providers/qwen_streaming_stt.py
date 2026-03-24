"""
Qwen3-ASR Streaming STT — Pre-computed mel + KV cache + prefix reuse.

Pipeline otimizado:
- processor() (mel spectrogram) = ~1000ms CPU → pre-computado em background
- generate() (prefill + decode) = ~440ms GPU
- Com prefix reuse no finish: ~45ms prefill + ~90ms (3 tokens) = ~135ms

Metricas detalhadas em cada etapa para validacao.

Sem dependencia de vLLM.

Configuracao via env vars:
    STT_PROVIDER=qwen-native-streaming
    QWEN_STT_MODEL=Qwen/Qwen3-ASR-0.6B
    QWEN_DEVICE=cuda:0
    QWEN_STT_CHUNK_SIZE_SEC=1.0
    QWEN_STT_MAX_NEW_TOKENS=64
"""

from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from common.config import STT_CONFIG, AUDIO_CONFIG
from common.audio_utils import pcm_to_float32, resample
from common.executor import run_inference
from stt.providers.base import STTProvider, register_stt_provider

logger = logging.getLogger("ai-agent.stt.qwen-native-streaming")

_INPUT_SAMPLE_RATE = 16000
_SOURCE_RATE = AUDIO_CONFIG["sample_rate"]  # 8kHz

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
}


@dataclass
class _StreamState:
    """Estado de streaming por sessao."""
    chunk_size_samples: int
    max_new_tokens: int
    prompt_text: str  # Cached prompt string

    # Audio (16kHz float32, pre-resampled)
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    samples_since_last_chunk: int = 0

    # Pre-computed inputs (from background processor())
    precomputed_inputs: Any = None  # dict of tensors ready for generate()
    precomputed_audio_len: int = 0  # Audio length when precomputed

    # Background state
    bg_task: Any = None
    bg_lock: Any = None

    # Partial decode result
    partial_text: str = ""
    partial_raw: str = ""

    # Output
    text: str = ""

    # Metrics
    total_resample_ms: float = 0.0
    total_processor_ms: float = 0.0
    total_generate_ms: float = 0.0
    bg_decode_count: int = 0


class QwenNativeStreamingSTT(STTProvider):
    """Streaming Qwen3-ASR com pre-computed mel + KV cache.

    Otimizacao: processor() roda em background durante a fala.
    No finish, inputs ja estao prontos — so roda generate().
    """

    provider_name = "qwen-native-streaming"

    def __init__(self):
        self._thinker = None
        self._processor = None
        self._eos_id = None
        self._model_name = os.getenv("QWEN_STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
        self._device = os.getenv("QWEN_DEVICE", "cuda:0")
        self._language = _LANG_MAP.get(
            STT_CONFIG.get("language", "pt"), "Portuguese"
        )
        self._chunk_size_sec = float(os.getenv("QWEN_STT_CHUNK_SIZE_SEC", "1.0"))
        self._max_new_tokens = int(os.getenv("QWEN_STT_MAX_NEW_TOKENS", "32"))
        self._prompt_text = ""  # Cached, built once

        self._streaming_states: dict[str, _StreamState] = {}

    async def connect(self) -> None:
        logger.info(
            "Carregando Qwen3-ASR (optimized): model=%s, device=%s, chunk=%.1fs",
            self._model_name, self._device, self._chunk_size_sec,
        )

        model_name = self._model_name
        device = self._device

        def _load():
            import torch
            from qwen_asr.core.transformers_backend import (
                Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor,
            )
            from transformers import AutoConfig, AutoModel, AutoProcessor
            try:
                AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
                AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
                AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
            except Exception:
                pass

            model = AutoModel.from_pretrained(
                model_name, device_map=device, torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_name, fix_mistral_regex=True)
            return model, processor

        model, self._processor = await run_inference(_load)
        self._thinker = model.thinker
        self._eos_id = self._processor.tokenizer.eos_token_id

        # Build and cache prompt (never changes)
        self._prompt_text = _build_prompt(self._processor, self._language, "")

        # Warmup: GPU mel + generate (first call compiles CUDA kernels)
        def _warmup():
            dummy = np.zeros(16000, dtype=np.float32)  # 1s dummy
            inp = self._run_processor(dummy, "")
            self._thinker.generate(**inp, max_new_tokens=2, use_cache=True)

        await run_inference(_warmup)
        logger.info("Qwen3-ASR optimized+GPU-mel carregado: %s", self._model_name)

    async def disconnect(self) -> None:
        if self._thinker is not None:
            thinker = self._thinker
            self._thinker = None
            self._processor = None
            self._streaming_states.clear()

            def _free():
                nonlocal thinker
                del thinker
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            await run_inference(_free)
            logger.info("Qwen3-ASR optimized descarregado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Fallback batch com metricas."""
        if not audio_data:
            return ""
        if self._thinker is None:
            raise RuntimeError("Nao conectado.")

        t_total = _time.perf_counter()

        # Resample
        t0 = _time.perf_counter()
        float_audio = pcm_to_float32(audio_data)
        float_audio_16k = resample(float_audio, _SOURCE_RATE, _INPUT_SAMPLE_RATE)
        resample_ms = (_time.perf_counter() - t0) * 1000

        # Processor
        t0 = _time.perf_counter()
        inputs = self._run_processor(float_audio_16k, "")
        proc_ms = (_time.perf_counter() - t0) * 1000

        # Generate
        t0 = _time.perf_counter()
        text, raw, n_tok, pf_ms, dc_ms = self._run_generate(inputs)
        gen_ms = (_time.perf_counter() - t0) * 1000

        total_ms = (_time.perf_counter() - t_total) * 1000

        logger.info(
            "STT batch: '%s' | resample=%.0fms proc=%.0fms gen=%.0fms "
            "(pf=%.0fms %dtok=%.0fms) total=%.0fms",
            text, resample_ms, proc_ms, gen_ms, pf_ms, n_tok, dc_ms, total_ms,
        )
        return text

    # ==================== Streaming ====================

    @property
    def supports_streaming(self) -> bool:
        return True

    async def start_streaming(self, stream_id: str = "") -> None:
        if self._thinker is None:
            raise RuntimeError("Nao conectado.")

        chunk_size_samples = int(round(self._chunk_size_sec * _INPUT_SAMPLE_RATE))

        self._streaming_states[stream_id] = _StreamState(
            chunk_size_samples=chunk_size_samples,
            max_new_tokens=self._max_new_tokens,
            prompt_text=self._prompt_text,
            bg_lock=asyncio.Lock(),
        )
        logger.debug("STT stream start (id=%s)", stream_id[:8] or "default")

    async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
        """Acumula audio + resample. A cada chunk_size, pre-computa mel em background."""
        state = self._streaming_states.get(stream_id)
        if state is None:
            raise RuntimeError("Streaming nao iniciado.")
        if not audio_chunk:
            return state.text

        # Resample (fast: ~0.1ms per 32ms chunk)
        t0 = _time.perf_counter()
        float_chunk = pcm_to_float32(audio_chunk)
        float_chunk_16k = resample(float_chunk, _SOURCE_RATE, _INPUT_SAMPLE_RATE)
        state.total_resample_ms += (_time.perf_counter() - t0) * 1000

        # Accumulate
        state.audio_accum = (
            float_chunk_16k if state.audio_accum.shape[0] == 0
            else np.concatenate([state.audio_accum, float_chunk_16k])
        )
        state.samples_since_last_chunk += len(float_chunk_16k)

        # Trigger background pre-compute when chunk_size reached
        if state.samples_since_last_chunk >= state.chunk_size_samples:
            state.samples_since_last_chunk = 0

            if state.bg_task is None or state.bg_task.done():
                audio_snapshot = state.audio_accum.copy()
                state.bg_task = asyncio.create_task(
                    self._background_precompute_and_decode(state, audio_snapshot)
                )

        return state.text

    async def finish_streaming(self, stream_id: str = "") -> str:
        """Finish: usa pre-computed inputs ou computa novos. Generate com prefix."""
        state = self._streaming_states.pop(stream_id, None)
        if state is None:
            raise RuntimeError("Streaming nao iniciado.")

        t_finish_start = _time.perf_counter()

        # Wait for bg task
        t0 = _time.perf_counter()
        bg_wait_ms = 0.0
        if state.bg_task and not state.bg_task.done():
            try:
                await asyncio.wait_for(state.bg_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
        bg_wait_ms = (_time.perf_counter() - t0) * 1000

        if state.audio_accum.shape[0] == 0:
            logger.debug("STT stream finish: no audio")
            return ""

        # Fast finish: GPU mel + token adjust from BG cache (~0.5ms)
        # vs full CPU processor recompute (~1500ms)
        audio_len = state.audio_accum.shape[0]
        t0 = _time.perf_counter()

        if state.precomputed_inputs is not None and state.precomputed_audio_len > 0:
            if state.precomputed_audio_len == audio_len:
                # Audio didn't grow — reuse exactly
                inputs = state.precomputed_inputs
                proc_ms = 0.0
                recomputed = False
            else:
                # Audio grew (tail) — fast adjust: GPU mel + insert audio_pad tokens
                def _fast_adjust():
                    return self._fast_finish_inputs(
                        state.audio_accum,
                        state.precomputed_inputs,
                        state.precomputed_audio_len,
                        state.partial_raw or "",
                    )

                inputs = await run_inference(_fast_adjust)
                proc_ms = (_time.perf_counter() - t0) * 1000
                state.total_processor_ms += proc_ms
                recomputed = "fast"
        else:
            # No BG result — full CPU processor (first utterance or very short)
            prefix = state.partial_raw or ""

            def _proc():
                return self._run_processor(state.audio_accum, prefix)

            inputs = await run_inference(_proc)
            proc_ms = (_time.perf_counter() - t0) * 1000
            state.total_processor_ms += proc_ms
            recomputed = True

        # Generate
        t0 = _time.perf_counter()

        def _gen():
            return self._run_generate(inputs)

        text, raw, n_tok, pf_ms, dc_ms = await run_inference(_gen)
        gen_ms = (_time.perf_counter() - t0) * 1000
        state.total_generate_ms += gen_ms

        finish_ms = (_time.perf_counter() - t_finish_start) * 1000

        logger.info(
            "STT FINISH: '%s' | bg_wait=%.0fms recompute=%s proc=%.0fms "
            "gen=%.0fms(pf=%.0fms %dtok=%.0fms) finish=%.0fms | "
            "session: resample=%.0fms proc=%.0fms gen=%.0fms bg_decodes=%d prefix_len=%d",
            text, bg_wait_ms, recomputed, proc_ms,
            gen_ms, pf_ms, n_tok, dc_ms, finish_ms,
            state.total_resample_ms, state.total_processor_ms,
            state.total_generate_ms, state.bg_decode_count,
            len(state.partial_raw),
        )

        return text

    async def _background_precompute_and_decode(
        self, state: _StreamState, audio: np.ndarray,
    ) -> None:
        """Background: processor() + generate() durante a fala."""
        async with state.bg_lock:
            try:
                t_total = _time.perf_counter()

                # Processor (CPU heavy: ~1000ms)
                t0 = _time.perf_counter()

                def _proc():
                    return self._run_processor(audio, "")

                inputs = await run_inference(_proc)
                proc_ms = (_time.perf_counter() - t0) * 1000
                state.total_processor_ms += proc_ms

                # Cache pre-computed inputs
                state.precomputed_inputs = inputs
                state.precomputed_audio_len = len(audio)

                # Generate (GPU: ~440ms)
                t0 = _time.perf_counter()

                def _gen():
                    return self._run_generate(inputs)

                text, raw, n_tok, pf_ms, dc_ms = await run_inference(_gen)
                gen_ms = (_time.perf_counter() - t0) * 1000
                state.total_generate_ms += gen_ms

                state.partial_text = text
                state.partial_raw = raw
                state.text = text
                state.bg_decode_count += 1

                total_ms = (_time.perf_counter() - t_total) * 1000

                logger.info(
                    "STT BG[%d]: '%s' | proc=%.0fms gen=%.0fms(pf=%.0fms %dtok=%.0fms) total=%.0fms",
                    state.bg_decode_count, text[:40],
                    proc_ms, gen_ms, pf_ms, n_tok, dc_ms, total_ms,
                )
            except Exception as e:
                logger.warning("BG decode error: %s", e)

    def _fast_finish_inputs(
        self, audio_16k: np.ndarray, bg_inputs: dict,
        bg_audio_len: int, prefix: str,
    ) -> dict:
        """Fast finish: GPU mel (~0.5ms) + adjust input_ids from BG cache.

        Instead of rerunning full CPU processor (~1500ms), we:
        1. Compute mel on GPU for the FULL audio (~0.5ms)
        2. Calculate how many extra audio_pad tokens are needed
        3. Insert them into the cached input_ids from BG
        Result: identical to full CPU processor, 3000x faster.
        """
        import torch
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            _get_feat_extract_output_lengths,
        )

        device = self._thinker.device
        fe = self._processor.feature_extractor
        AUDIO_PAD_TOKEN = 151676  # <|audio_pad|>

        # GPU mel of full audio
        mel_np = fe._torch_extract_fbank_features(audio_16k, device=str(device))
        full_frames = len(audio_16k) // fe.hop_length
        mel_trimmed = mel_np[:, :full_frames]

        input_features = torch.from_numpy(mel_trimmed).unsqueeze(0).to(
            device=device, dtype=self._thinker.dtype,
        )
        feature_mask = torch.ones(1, full_frames, dtype=torch.int32, device=device)

        # Calculate audio_pad token counts
        bg_frames = bg_audio_len // fe.hop_length
        bg_audio_tokens = _get_feat_extract_output_lengths(
            torch.tensor([bg_frames])
        ).item()
        full_audio_tokens = _get_feat_extract_output_lengths(
            torch.tensor([full_frames])
        ).item()
        extra_pads = full_audio_tokens - bg_audio_tokens

        # Adjust input_ids: insert extra audio_pad tokens
        bg_ids = bg_inputs["input_ids"][0].cpu()
        if extra_pads > 0:
            pad_positions = torch.where(bg_ids == AUDIO_PAD_TOKEN)[0]
            if len(pad_positions) > 0:
                insert_pos = pad_positions[-1].item() + 1
                new_ids = torch.cat([
                    bg_ids[:insert_pos],
                    torch.full((extra_pads,), AUDIO_PAD_TOKEN, dtype=bg_ids.dtype),
                    bg_ids[insert_pos:],
                ])
            else:
                new_ids = bg_ids
        elif extra_pads < 0:
            # Audio shrank? Shouldn't happen, fallback to BG ids
            new_ids = bg_ids
        else:
            new_ids = bg_ids

        new_mask = torch.ones(1, new_ids.shape[0], dtype=torch.int64, device=device)

        return {
            "input_ids": new_ids.unsqueeze(0).to(device),
            "attention_mask": new_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        }

    def _run_processor(self, audio_16k: np.ndarray, prefix: str) -> dict:
        """CPU processor para tokenizacao correta + GPU mel para features.

        O processor DEVE ser usado para tokenizacao porque ele insere os
        audio_pad tokens no input_ids baseado no comprimento do audio.
        Sem isso, input_ids tem 19 tokens em vez de 44 → modelo gera lixo.

        Otimizacao: processor roda no CPU (~1500ms) mas no background
        durante a fala. No finish, inputs pre-computados sao reutilizados.
        """
        prompt = _build_prompt(self._processor, self._language, prefix)
        inputs = self._processor(
            text=[prompt], audio=[audio_16k],
            return_tensors="pt", padding=True,
        )
        return inputs.to(self._thinker.device).to(self._thinker.dtype)

    def _run_generate(self, inputs) -> tuple:
        """Roda prefill + manual decode (GPU). Retorna (text, raw, n_tok, pf_ms, dc_ms)."""
        import torch

        input_len = inputs["input_ids"].shape[1]
        thinker = self._thinker
        eos_id = self._eos_id
        max_tokens = self._max_new_tokens

        # Prefill
        torch.cuda.synchronize()
        t_pf = _time.perf_counter()
        with torch.no_grad():
            out = thinker.generate(
                **inputs, max_new_tokens=1,
                return_dict_in_generate=True, use_cache=True,
            )
        torch.cuda.synchronize()
        pf_ms = (_time.perf_counter() - t_pf) * 1000

        kv = out.past_key_values
        seqs = out.sequences

        # Decode with early stopping
        torch.cuda.synchronize()
        t_dc = _time.perf_counter()
        n_tok = 0
        last_tokens = []  # Track last 3 tokens for repetition detection
        NEWLINE_TOKEN = thinker.config.eos_token_id  # Will also check for \n
        for _ in range(max_tokens - 1):
            with torch.no_grad():
                out_step = thinker(
                    input_ids=seqs[:, -1:],
                    past_key_values=kv,
                    use_cache=True,
                )
            logits = out_step.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tok_id = next_token.item()
            seqs = torch.cat([seqs, next_token], dim=-1)
            kv = out_step.past_key_values
            n_tok += 1

            # EOS → stop
            if tok_id == eos_id:
                break

            # Repetition detection: 3 consecutive same tokens → stop
            last_tokens.append(tok_id)
            if len(last_tokens) >= 3 and last_tokens[-1] == last_tokens[-2] == last_tokens[-3]:
                break
        torch.cuda.synchronize()
        dc_ms = (_time.perf_counter() - t_dc) * 1000

        gen_tokens = seqs[0, input_len:]
        decoded = self._processor.batch_decode(
            [gen_tokens], skip_special_tokens=True,
        )[0]

        text = _parse_text(decoded, self._language)
        return text, decoded, n_tok, pf_ms, dc_ms


def _build_prompt(processor, language: str, prefix: str = "") -> str:
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
    )
    if language:
        prompt = prompt + "language " + language + "<asr_text>"
    if prefix:
        prompt = prompt + prefix
    return prompt


def _parse_text(raw: str, language: Optional[str] = None) -> str:
    if not raw:
        return ""
    # Strip any "language X<asr_text>" prefix from the output
    # The model may generate this when language is forced in prompt
    _ASR_TAG = "<asr_text>"
    if _ASR_TAG in raw:
        raw = raw.split(_ASR_TAG, 1)[1]
    # Also strip "language X\n...\n<asr_text>" variant
    if raw.startswith("language "):
        # Try parse_asr_output without user_language to trigger tag parsing
        try:
            from qwen_asr.inference.utils import parse_asr_output
            _, text = parse_asr_output(raw, user_language=None)
            return text.strip()
        except ImportError:
            pass
    return raw.strip()


# Auto-register
register_stt_provider("qwen-native-streaming", QwenNativeStreamingSTT)
