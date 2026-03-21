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
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import torch

from macaw_tts.audio import SAMPLES_PER_FRAME, CODEC_SAMPLE_RATE
from macaw_tts.crossfade import HannCrossfader
from macaw_tts.decoder import StreamingDecoder
from macaw_tts.predictor_graph import PredictorCUDAGraph
from macaw_tts.sampling import CircularRepetitionPenalty, sample_logits
from macaw_tts.talker_graph import TalkerCUDAGraph

logger = logging.getLogger("macaw-tts.streaming")

_cuda_available = torch.cuda.is_available()


def _sync_cuda() -> None:
    """Synchronize CUDA if available. No-op on CPU."""
    if _cuda_available:
        torch.cuda.synchronize()


def _log_vram(label: str) -> None:
    """Log GPU VRAM usage if CUDA is available. No-op on CPU."""
    if not _cuda_available:
        return
    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    logger.info(
        "vram_%s allocated_mb=%.0f reserved_mb=%.0f",
        label, allocated_mb, reserved_mb,
        extra={"vram_allocated_mb": allocated_mb, "vram_reserved_mb": reserved_mb},
    )


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


class _StreamingEmitter:
    """Shared two-phase emission, crossfade, and decode logic.

    Used by both CUDA graph and dynamic cache streaming paths to avoid
    duplicating the emission/flush/metrics logic.
    """

    def __init__(
        self,
        decoder: StreamingDecoder,
        crossfader: HannCrossfader,
        *,
        emit_every_phase1: int = 1,
        emit_every_phase2: int = 4,
        phase1_frames: int = 1,
        ref_codes: Optional[torch.Tensor] = None,
        t_start: float = 0.0,
    ):
        self._decoder = decoder
        self._crossfader = crossfader
        self._emit_phase1 = emit_every_phase1
        self._emit_phase2 = emit_every_phase2
        self._phase1_frames = phase1_frames
        self._ref_codes = ref_codes
        self._t_start = t_start

        self.codes_buffer: list[torch.Tensor] = []
        self._frames_since_emit = 0
        self._total_frames_emitted = 0
        self._chunk_count = 0
        self._first_audio_time: Optional[float] = None

        crossfader.reset()

    def add_frame(self, codes: torch.Tensor) -> None:
        """Add a codec frame to the buffer."""
        self.codes_buffer.append(codes.detach())
        self._frames_since_emit += 1

    def should_emit(self) -> bool:
        """Check if we have enough frames for the current phase."""
        total = len(self.codes_buffer)
        if total <= self._phase1_frames:
            return self._frames_since_emit >= self._emit_phase1
        return self._frames_since_emit >= self._emit_phase2

    def emit(self) -> tuple[np.ndarray, int, ChunkMetadata]:
        """Decode + crossfade accumulated frames and return chunk."""
        total = len(self.codes_buffer)
        if total <= self._phase1_frames:
            phase = 1
            emit_every = self._emit_phase1
        else:
            phase = 2
            emit_every = self._emit_phase2

        t_decode = time.perf_counter()
        audio = self._decoder.decode_window(
            self.codes_buffer,
            num_new_frames=self._frames_since_emit,
            ref_codes=self._ref_codes,
        )
        _sync_cuda()
        decode_ms = (time.perf_counter() - t_decode) * 1000

        is_first = self._chunk_count == 0
        audio = self._crossfader.process(audio, is_first=is_first)

        ttfa_ms = 0.0
        if self._first_audio_time is None:
            self._first_audio_time = time.perf_counter()
            ttfa_ms = (self._first_audio_time - self._t_start) * 1000
            logger.info(
                "ttfa_ms=%.0f phase=%d", ttfa_ms, phase,
                extra={"ttfa_ms": ttfa_ms, "phase": phase},
            )

        self._total_frames_emitted = total
        frames_in_chunk = self._frames_since_emit
        self._frames_since_emit = 0

        meta = ChunkMetadata(
            chunk_index=self._chunk_count,
            num_frames=frames_in_chunk,
            phase=phase,
            is_final=False,
            decode_ms=decode_ms,
            total_frames_so_far=self._total_frames_emitted,
            ttfa_ms=ttfa_ms,
        )
        self._chunk_count += 1
        return audio, CODEC_SAMPLE_RATE, meta

    def flush(self) -> Optional[tuple[np.ndarray, int, ChunkMetadata]]:
        """Flush remaining frames and crossfade tail. Returns None if nothing to flush."""
        remaining = len(self.codes_buffer) - self._total_frames_emitted

        if remaining > 0:
            t_decode = time.perf_counter()
            audio = self._decoder.decode_window(
                self.codes_buffer,
                num_new_frames=remaining,
                ref_codes=self._ref_codes,
            )
            _sync_cuda()
            decode_ms = (time.perf_counter() - t_decode) * 1000
            audio = self._crossfader.process(audio, is_last=True)
        elif self._crossfader.has_pending_tail:
            # No new frames to decode, but the crossfader withheld overlap
            # samples from the last emit(). Drain them with fade-out.
            audio = self._crossfader.drain()
            decode_ms = 0.0
        else:
            return None

        if len(audio) == 0:
            return None

        ttfa_ms = 0.0
        if self._first_audio_time is None:
            self._first_audio_time = time.perf_counter()
            ttfa_ms = (self._first_audio_time - self._t_start) * 1000

        meta = ChunkMetadata(
            chunk_index=self._chunk_count,
            num_frames=remaining,
            phase=2,
            is_final=True,
            decode_ms=decode_ms,
            total_frames_so_far=len(self.codes_buffer),
            ttfa_ms=ttfa_ms,
        )
        return audio, CODEC_SAMPLE_RATE, meta

    def log_summary(self) -> None:
        """Log final generation stats."""
        total_ms = (time.perf_counter() - self._t_start) * 1000
        total_frames = len(self.codes_buffer)
        audio_duration_s = total_frames * SAMPLES_PER_FRAME / CODEC_SAMPLE_RATE
        rtf = audio_duration_s / (total_ms / 1000) if total_ms > 0 else 0
        logger.info(
            "generation_complete frames=%d chunks=%d wall_ms=%.0f "
            "audio_s=%.1f rtf=%.2f",
            total_frames, self._chunk_count, total_ms, audio_duration_s, rtf,
            extra={
                "frames": total_frames,
                "chunks": self._chunk_count,
                "wall_ms": total_ms,
                "audio_s": audio_duration_s,
                "rtf": rtf,
            },
        )
        _log_vram("after_generation")


def _build_suppress_mask(
    vocab_size: int, eos_id: int, device: torch.device,
) -> torch.Tensor:
    """Build suppress mask for last 1024 tokens minus EOS."""
    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    suppress_mask[suppress_start:] = True
    if 0 <= eos_id < vocab_size:
        suppress_mask[eos_id] = False
    return suppress_mask


def _prefill(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    *,
    suppress_mask: torch.Tensor,
    eos_id: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
):
    """Run prefill forward and sample first token.

    Returns (out, token) where out contains past_key_values, past_hidden, etc.
    """
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
    return out, token


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
    max_wall_time_s: float = 60.0,
) -> Generator[tuple[np.ndarray, int, ChunkMetadata], None, None]:
    """Streaming generation with CUDA graphs and two-phase latency.

    GPU REQUIRED — runs Talker + Predictor CUDA graphs.

    Phase 1 (first frame): emit_every=1 → TTFA ~88ms
    Phase 2 (remaining):   emit_every=4 → stable throughput

    Args:
        max_wall_time_s: Maximum wall-clock time for generation in seconds.
            Prevents stuck generations if the model never produces EOS.
            Default 60s (~480 frames at RTF 3.5x, ~38s of audio).

    Yields:
        (audio_chunk, sample_rate, metadata) tuples.
        audio_chunk: float32 numpy array at 24kHz.
        sample_rate: 24000.
        metadata: ChunkMetadata with timing info.
    """
    eos_id = config.codec_eos_token_id
    eos_tensor = torch.tensor([eos_id], device=talker_input_embeds.device)
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    suppress_mask = _build_suppress_mask(vocab_size, eos_id, device)
    rep_penalty = CircularRepetitionPenalty(
        window=rep_penalty_window, penalty=repetition_penalty,
        vocab_size=vocab_size, device=device,
    )

    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor = talker.code_predictor
    predictor_codec_embeds = predictor.get_input_embeddings()
    num_code_groups = config.num_code_groups

    t_start = time.perf_counter()

    out, token = _prefill(
        talker, talker_input_embeds, attention_mask,
        trailing_text_hiddens, tts_pad_embed,
        suppress_mask=suppress_mask, eos_id=eos_id,
        min_new_tokens=min_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, do_sample=do_sample,
    )

    past_hidden = out.past_hidden
    gen_step = out.generation_step

    prefill_len = talker_graph.prefill_kv(out.past_key_values)
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "prefill_ms=%.0f seq_len=%d", prefill_ms, prefill_len,
        extra={"prefill_ms": prefill_ms, "seq_len": prefill_len},
    )
    _log_vram("after_prefill")

    emitter = _StreamingEmitter(
        decoder, crossfader,
        emit_every_phase1=emit_every_phase1,
        emit_every_phase2=emit_every_phase2,
        phase1_frames=phase1_frames,
        ref_codes=ref_codes,
        t_start=t_start,
    )

    # Pre-allocate past_hidden buffer to avoid .clone() allocation per step
    past_hidden_buf = torch.empty_like(past_hidden)
    _TIMEOUT_CHECK_INTERVAL = 100

    # EOS check strategy: we check EOS at the END of each iteration, after
    # submitting GPU work for the current step. This overlaps the CPU↔GPU sync
    # (.item()) with GPU compute that was already queued, reducing idle time.
    # The first token from prefill is checked before entering the loop.
    try:
        if token.eq(eos_tensor).item():
            flush_result = emitter.flush()
            if flush_result is not None:
                yield flush_result
            emitter.log_summary()
            return

        for step_idx in range(max_new_tokens):
            # Wall-clock timeout check (every N steps to avoid overhead).
            # Granularity: ~N * step_time (e.g., 100 * 8ms = 800ms).
            if step_idx > 0 and step_idx % _TIMEOUT_CHECK_INTERVAL == 0:
                elapsed = time.perf_counter() - t_start
                if elapsed > max_wall_time_s:
                    logger.warning(
                        "wall_timeout step=%d elapsed=%.1fs limit=%.1fs",
                        step_idx, elapsed, max_wall_time_s,
                    )
                    break

            # Predictor CUDA graph: CB0 → CB1..CB15
            last_id_hidden = talker_codec_embed(token.unsqueeze(1))
            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            codebook_token_ids = predictor_graph.run(pred_input)

            all_cb = torch.cat([token.view(1), codebook_token_ids])
            emitter.add_frame(all_cb)
            rep_penalty.update(token)

            # Build Talker input embedding
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

            # Talker CUDA graph: single-token decode
            current_pos = prefill_len + step_idx
            if current_pos >= talker_graph.max_seq_len - 1:
                logger.warning(
                    "max_seq_len_reached step=%d pos=%d max=%d",
                    step_idx, current_pos, talker_graph.max_seq_len,
                )
                break

            hidden_states = talker_graph.run(inputs_embeds, position=current_pos)

            # Sample next CB0 token — GPU kernels are queued asynchronously
            logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
            logits = rep_penalty.apply(logits)

            suppress_eos = len(emitter.codes_buffer) < min_new_tokens
            token = sample_logits(
                logits.squeeze(0),
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, suppress_mask=suppress_mask,
                suppress_tokens=[eos_id] if suppress_eos else None,
            )
            past_hidden_buf.copy_(hidden_states[:, -1:, :])
            past_hidden = past_hidden_buf
            gen_step += 1

            # Two-phase emission
            if emitter.should_emit():
                yield emitter.emit()

            # EOS check at end of step: the .item() sync overlaps with GPU work
            # already queued above (predictor/talker graph replays are async).
            if token.eq(eos_tensor).item():
                break

        # Flush remaining frames
        flush_result = emitter.flush()
        if flush_result is not None:
            yield flush_result

        emitter.log_summary()
    finally:
        # Clean up crossfader state on any exit path (normal, exception, or
        # GeneratorExit from barge-in). Without this, the crossfader retains
        # ~overlap_samples of audio that would leak into the next generation.
        crossfader.reset()
