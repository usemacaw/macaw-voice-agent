"""Core streaming generator with two-phase latency.

Combines:
- faster-qwen3-tts CUDA graph decode loop
- rekuenkdr two-phase latency (Phase 1: emit 1 frame, Phase 2: emit 4 frames)
- Hann crossfade between chunks
- Sliding window Code2Wav decode

Yields (audio_chunk, sample_rate, metadata) tuples as audio is generated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import torch

from macaw_tts.audio import SAMPLES_PER_FRAME, CODEC_SAMPLE_RATE
from macaw_tts.crossfade import HannCrossfader
from macaw_tts.decoder import StreamingDecoder
from macaw_tts.predictor_graph import PredictorCUDAGraph
from macaw_tts.sampling import CircularRepetitionPenalty, sample_logits
from macaw_tts.talker_graph import TalkerCUDAGraph

logger = logging.getLogger("macaw-tts.streaming")


@dataclass
class StreamingMetrics:
    """Timing metrics for a single streaming generation."""

    prefill_ms: float = 0.0
    ttfa_ms: float = 0.0          # Time to first audio
    total_ms: float = 0.0
    total_frames: int = 0
    total_chunks: int = 0
    rtf: float = 0.0              # Real-time factor (audio_duration / wall_time)
    phase1_chunks: int = 0
    phase2_chunks: int = 0
    step_times_ms: list[float] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """Metadata yielded with each audio chunk."""

    chunk_index: int
    num_frames: int               # Codec frames in this chunk
    phase: int                    # 1 = aggressive, 2 = stable
    is_final: bool
    decode_ms: float              # Decode time for this chunk
    total_frames_so_far: int
    ttfa_ms: float = 0.0         # Only set for first chunk


@torch.inference_mode()
def streaming_generate(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorCUDAGraph,
    talker_graph: TalkerCUDAGraph,
    decoder: StreamingDecoder,
    crossfader: HannCrossfader,
    *,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    rep_penalty_window: int = 256,
    emit_every_phase1: int = 1,
    emit_every_phase2: int = 4,
    phase1_frames: int = 1,
    ref_codes: torch.Tensor | None = None,
) -> Generator[tuple[np.ndarray, int, ChunkMetadata], None, None]:
    """Streaming generation with CUDA graphs and two-phase latency.

    🟢 GPU REQUIRED — runs Talker + Predictor CUDA graphs.

    Phase 1 (first frame): emit_every=1 → TTFA ~88ms
    Phase 2 (remaining):   emit_every=4 → stable throughput

    Yields:
        (audio_chunk, sample_rate, metadata) tuples.
        audio_chunk: float32 numpy array at 24kHz.
        sample_rate: 24000.
        metadata: ChunkMetadata with timing info.
    """
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    # Build suppress mask (last 1024 tokens minus EOS)
    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    # Model components
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor = talker.code_predictor
    predictor_codec_embeds = predictor.get_input_embeddings()
    num_code_groups = config.num_code_groups

    # Repetition penalty
    rep_penalty = CircularRepetitionPenalty(
        window=rep_penalty_window,
        penalty=repetition_penalty,
        vocab_size=vocab_size,
        device=device,
    )

    # === PREFILL (HF forward, variable-length) ===
    t_start = time.perf_counter()

    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )

    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step

    logits = out.logits[:, -1, :]
    suppress_eos = min_new_tokens > 0
    token = sample_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=suppress_mask,
        suppress_tokens=[eos_id] if suppress_eos else None,
    )

    # Copy HF KV cache → StaticCache for CUDA graphs
    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"Prefill: {prefill_ms:.0f}ms, seq_len={prefill_len}")

    # === DECODE LOOP with two-phase emission ===
    codes_buffer: list[torch.Tensor] = []  # GPU-resident
    frames_since_emit = 0
    total_frames_emitted = 0
    chunk_count = 0
    first_audio_time = None

    crossfader.reset()

    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break

        # --- Predictor CUDA graph: CB0 → CB1..CB15 ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        codebook_token_ids = predictor_graph.run(pred_input)

        # Store all 16 codebooks: [CB0, CB1, ..., CB15]
        all_cb = torch.cat([token.view(1), codebook_token_ids])
        codes_buffer.append(all_cb.detach())

        rep_penalty.update(token)

        # --- Build Talker input embedding ---
        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(
                predictor_codec_embeds[i](
                    codebook_token_ids[i].unsqueeze(0).unsqueeze(0)
                )
            )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        # --- Talker CUDA graph: single-token decode ---
        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            break

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)

        # Sample next CB0 token
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        logits = rep_penalty.apply(logits)

        suppress_eos = len(codes_buffer) < min_new_tokens
        token = sample_logits(
            logits.squeeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1

        # --- Two-phase emission ---
        frames_since_emit += 1
        total_frames = len(codes_buffer)

        # Determine current phase
        if total_frames <= phase1_frames:
            emit_every = emit_every_phase1
            phase = 1
        else:
            emit_every = emit_every_phase2
            phase = 2

        if frames_since_emit < emit_every:
            continue

        # === DECODE + YIELD ===
        t_decode = time.perf_counter()

        audio = decoder.decode_window(
            codes_buffer,
            num_new_frames=frames_since_emit,
            ref_codes=ref_codes,
        )

        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_decode) * 1000

        # Apply crossfade
        is_first = chunk_count == 0
        audio = crossfader.process(audio, is_first=is_first)

        # Record TTFA
        ttfa_ms = 0.0
        if first_audio_time is None:
            first_audio_time = time.perf_counter()
            ttfa_ms = (first_audio_time - t_start) * 1000
            logger.info(f"TTFA: {ttfa_ms:.0f}ms (phase {phase})")

        total_frames_emitted = total_frames
        frames_since_emit = 0

        meta = ChunkMetadata(
            chunk_index=chunk_count,
            num_frames=emit_every,
            phase=phase,
            is_final=False,
            decode_ms=decode_ms,
            total_frames_so_far=total_frames_emitted,
            ttfa_ms=ttfa_ms,
        )

        chunk_count += 1
        yield audio, CODEC_SAMPLE_RATE, meta

    # === FLUSH remaining frames ===
    remaining = len(codes_buffer) - total_frames_emitted
    if remaining > 0:
        t_decode = time.perf_counter()
        audio = decoder.decode_window(
            codes_buffer,
            num_new_frames=remaining,
            ref_codes=ref_codes,
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_decode) * 1000

        audio = crossfader.process(audio, is_last=True)

        if first_audio_time is None:
            first_audio_time = time.perf_counter()
            ttfa_ms = (first_audio_time - t_start) * 1000
        else:
            ttfa_ms = 0.0

        meta = ChunkMetadata(
            chunk_index=chunk_count,
            num_frames=remaining,
            phase=2,
            is_final=True,
            decode_ms=decode_ms,
            total_frames_so_far=len(codes_buffer),
            ttfa_ms=ttfa_ms,
        )
        yield audio, CODEC_SAMPLE_RATE, meta

    total_ms = (time.perf_counter() - t_start) * 1000
    total_frames = len(codes_buffer)
    audio_duration_s = total_frames * SAMPLES_PER_FRAME / CODEC_SAMPLE_RATE
    rtf = audio_duration_s / (total_ms / 1000) if total_ms > 0 else 0

    logger.info(
        f"Generation complete: {total_frames} frames, {chunk_count + 1} chunks, "
        f"{total_ms:.0f}ms wall, {audio_duration_s:.1f}s audio, RTF={rtf:.2f}x"
    )


@torch.inference_mode()
def streaming_generate_dynamic(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    decoder: StreamingDecoder,
    crossfader: HannCrossfader,
    *,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    rep_penalty_window: int = 256,
    emit_every_phase1: int = 1,
    emit_every_phase2: int = 4,
    phase1_frames: int = 1,
    ref_codes: torch.Tensor | None = None,
) -> Generator[tuple[np.ndarray, int, ChunkMetadata], None, None]:
    """Streaming with dynamic cache (for flash-attn compatibility).

    🟢 GPU REQUIRED.

    Uses the Talker's own forward() with dynamic KV cache instead of CUDA graphs.
    Compatible with flash_attention_2 which cannot be captured in CUDA graphs.
    """
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    rep_penalty = CircularRepetitionPenalty(
        window=rep_penalty_window, penalty=repetition_penalty,
        vocab_size=vocab_size, device=device,
    )

    # === PREFILL ===
    t_start = time.perf_counter()

    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )

    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step

    logits = out.logits[:, -1, :]
    suppress_eos = min_new_tokens > 0
    token = sample_logits(
        logits, temperature=temperature, top_k=top_k, top_p=top_p,
        do_sample=do_sample, suppress_mask=suppress_mask,
        suppress_tokens=[eos_id] if suppress_eos else None,
    )

    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"Prefill (dynamic): {prefill_ms:.0f}ms")

    # === DECODE LOOP ===
    codes_buffer: list[torch.Tensor] = []
    frames_since_emit = 0
    total_frames_emitted = 0
    chunk_count = 0
    first_audio_time = None
    crossfader.reset()

    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break

        # Extend attention mask for dynamic cache
        cache_position = None
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=1,
            )
            cache_position = torch.tensor(
                [attention_mask.shape[1] - 1], device=attention_mask.device
            )

        # Single-step forward with dynamic cache
        out = talker.forward(
            input_ids=token.view(1, 1),
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            generation_step=gen_step,
            past_hidden=past_hidden,
            past_key_values=talker_past_kv,
            subtalker_dosample=do_sample,
            subtalker_top_k=top_k,
            subtalker_top_p=top_p,
            subtalker_temperature=temperature,
            cache_position=cache_position,
        )

        codec_ids = out.hidden_states[1]
        if codec_ids is None:
            break

        codes_buffer.append(codec_ids.squeeze(0).detach())
        rep_penalty.update(token)

        # Sample next token
        logits = out.logits[:, -1, :]
        logits = rep_penalty.apply(logits.unsqueeze(0)).squeeze(0)

        suppress_eos = len(codes_buffer) < min_new_tokens
        token = sample_logits(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )

        talker_past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        # Two-phase emission
        frames_since_emit += 1
        total_frames = len(codes_buffer)

        if total_frames <= phase1_frames:
            emit_every = emit_every_phase1
            phase = 1
        else:
            emit_every = emit_every_phase2
            phase = 2

        if frames_since_emit < emit_every:
            continue

        # Decode + yield
        t_decode = time.perf_counter()
        audio = decoder.decode_window(
            codes_buffer, num_new_frames=frames_since_emit, ref_codes=ref_codes,
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_decode) * 1000

        is_first = chunk_count == 0
        audio = crossfader.process(audio, is_first=is_first)

        ttfa_ms = 0.0
        if first_audio_time is None:
            first_audio_time = time.perf_counter()
            ttfa_ms = (first_audio_time - t_start) * 1000
            logger.info(f"TTFA (dynamic): {ttfa_ms:.0f}ms (phase {phase})")

        total_frames_emitted = total_frames
        frames_since_emit = 0

        meta = ChunkMetadata(
            chunk_index=chunk_count, num_frames=emit_every, phase=phase,
            is_final=False, decode_ms=decode_ms,
            total_frames_so_far=total_frames_emitted, ttfa_ms=ttfa_ms,
        )
        chunk_count += 1
        yield audio, CODEC_SAMPLE_RATE, meta

    # Flush
    remaining = len(codes_buffer) - total_frames_emitted
    if remaining > 0:
        t_decode = time.perf_counter()
        audio = decoder.decode_window(
            codes_buffer, num_new_frames=remaining, ref_codes=ref_codes,
        )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t_decode) * 1000
        audio = crossfader.process(audio, is_last=True)

        ttfa_ms = 0.0
        if first_audio_time is None:
            first_audio_time = time.perf_counter()
            ttfa_ms = (first_audio_time - t_start) * 1000

        meta = ChunkMetadata(
            chunk_index=chunk_count, num_frames=remaining, phase=2,
            is_final=True, decode_ms=decode_ms,
            total_frames_so_far=len(codes_buffer), ttfa_ms=ttfa_ms,
        )
        yield audio, CODEC_SAMPLE_RATE, meta

    total_ms = (time.perf_counter() - t_start) * 1000
    total_frames = len(codes_buffer)
    audio_duration_s = total_frames * SAMPLES_PER_FRAME / CODEC_SAMPLE_RATE
    rtf = audio_duration_s / (total_ms / 1000) if total_ms > 0 else 0
    logger.info(
        f"Generation complete (dynamic): {total_frames} frames, {chunk_count + 1} chunks, "
        f"{total_ms:.0f}ms wall, {audio_duration_s:.1f}s audio, RTF={rtf:.2f}x"
    )


def compile_talker_for_streaming(talker, compile_mode: str = "reduce-overhead") -> None:
    """Apply torch.compile to talker and predictor for flash-attn compatibility.

    Uses torch.compile instead of manual CUDA graphs.
    The "reduce-overhead" mode internally manages CUDA graphs in a way
    compatible with flash-attn's dynamic operations.

    Based on dffdeeq approach:
    - Talker: mode="default" (KV-cache conflicts with reduce-overhead)
    - CodePredictor: mode="reduce-overhead"

    🟢 GPU REQUIRED.
    """
    # Compile codebook predictor
    predictor = talker.code_predictor
    if hasattr(predictor, 'forward') and not getattr(predictor, '_compiled', False):
        predictor.forward = torch.compile(
            predictor.forward,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        predictor._compiled = True
        logger.info(f"CodePredictor compiled (mode={compile_mode})")

    # Compile talker model — use "default" mode to avoid KV-cache CUDA graph conflicts
    if hasattr(talker, 'model') and not getattr(talker.model, '_compiled', False):
        talker.model.forward = torch.compile(
            talker.model.forward,
            mode="default",  # NOT reduce-overhead — KV cache is dynamic
            fullgraph=False,
            dynamic=False,
        )
        talker.model._compiled = True
        logger.info("Talker compiled (mode=default)")
