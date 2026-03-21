"""CUDA graph capture for Talker single-token decode step.

Based on faster-qwen3-tts talker_graph.py.
Uses transformers StaticCache for fixed-size KV tensors compatible with CUDA graphs.

The Talker has 20-28 transformer layers. We capture the single-token decode
as a CUDA graph for ~8ms/step (vs ~50ms without graphs).

Strategy:
- Prefill via HF forward (variable-length) → copies KV into StaticCache
- Decode via CUDA graph replay (fixed-shape) → ~8ms/step
- LRU attention mask cache → O(1) per step, bounded VRAM usage
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

logger = logging.getLogger("macaw-tts.talker-graph")

# Maximum number of attention masks to cache. Each mask for a 2048 max_seq_len
# model is ~16MB in bfloat16. 16 entries = ~256MB worst case.
# Trade-off: lower = less VRAM, higher = fewer recomputations on long sequences.
# At 16, a 2048-frame generation recomputes ~99% of masks (sequential access
# pattern means each position is used exactly once, so LRU eviction is optimal).
_MASK_CACHE_MAX_SIZE = 16


class TalkerCUDAGraph:
    """CUDA graph for Talker single-token decode.

    ⚠️ REQUIRES GPU — capture() and run() need CUDA device.
    Code structure can be read/tested with mocks on CPU.

    Usage:
        graph = TalkerCUDAGraph(talker_model, config)
        graph.capture(prefill_len=seq_len)  # 🟢 GPU
        hidden = graph.run(input_embeds, position=pos)  # 🟢 GPU
    """

    def __init__(
        self,
        talker_model,
        talker_config,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
    ):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers

        # Reference to inner transformer model
        self.model = talker_model

        # Static I/O buffers (pre-allocated, reused every step)
        self.input_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)

        # Cache position buffer — updated before each graph replay
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)

        # RoPE deltas and position IDs.
        # num_position_types=3 is Qwen3-TTS specific: (text, codec, fused) positions.
        # Derived from upstream Qwen3-TTS model which uses [3, B, seq_len] position_ids.
        self._num_position_types = 3
        self.rope_deltas = torch.zeros(1, 1, dtype=torch.float32, device=device)
        self.position_ids = torch.zeros(self._num_position_types, 1, 1, dtype=torch.float32, device=device)

        # StaticCache and graph (initialized on capture)
        self.static_cache = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured = False
        self.attn_mask = None
        self._mask_key = None

    def __repr__(self) -> str:
        return (
            f"TalkerCUDAGraph(device={self.device!r}, captured={self.captured}, "
            f"max_seq_len={self.max_seq_len}, layers={self.num_layers})"
        )

    def _init_static_cache(self, config):
        """Create StaticCache and force lazy layer initialization."""
        self.static_cache = StaticCache(config=config, max_cache_len=self.max_seq_len)

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _build_attention_masks(self, attention_mask: torch.Tensor | None = None):
        """Prepare lazy mask generator. Masks are computed on-demand and cached.

        Replaces the old approach of pre-building masks for all max_seq_len
        positions (which consumed O(max_seq_len) GPU memory). Now uses a dict
        cache that only stores masks for positions actually used.
        """
        self._mask_dummy = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self._mask_pos_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self._mask_fn = (
            create_sliding_window_causal_mask
            if getattr(self.model.config, "sliding_window", None) is not None
            else create_causal_mask
        )
        self._mask_attention_mask = attention_mask
        self._mask_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        # Pre-compute position 0 to initialize attn_mask buffer
        mask_0 = self._compute_mask(0)
        if self.attn_mask is None:
            self.attn_mask = mask_0.clone()
        else:
            self.attn_mask.copy_(mask_0)

    def _compute_mask(self, position: int) -> torch.Tensor:
        """Compute and cache attention mask for a given position.

        Uses LRU eviction to bound VRAM usage at _MASK_CACHE_MAX_SIZE entries.
        Without eviction, generating 2048 frames would cache 2048 masks,
        consuming ~32GB VRAM and causing OOM.
        """
        if position in self._mask_cache:
            self._mask_cache.move_to_end(position)
            return self._mask_cache[position]

        # Reuse pre-allocated position buffer to avoid tensor allocation per call
        self._mask_pos_buf[0] = position
        mask = self._mask_fn(
            config=self.model.config,
            input_embeds=self._mask_dummy,
            attention_mask=self._mask_attention_mask,
            cache_position=self._mask_pos_buf,
            past_key_values=self.static_cache,
        )

        if len(self._mask_cache) >= _MASK_CACHE_MAX_SIZE:
            self._mask_cache.popitem(last=False)

        self._mask_cache[position] = mask
        return mask

    def _get_attn_mask(self, position: int) -> torch.Tensor:
        """Get attention mask for position. Computes lazily if not cached."""
        return self._compute_mask(position)

    def _decode_step(self):
        """Single-token decode through model forward."""
        out = self.model(
            inputs_embeds=self.input_buf,
            attention_mask=self.attn_mask,
            past_key_values=self.static_cache,
            cache_position=self.cache_position,
            position_ids=self.position_ids,
            use_cache=True,
        )
        self.output_buf.copy_(out.last_hidden_state)

    @torch.inference_mode()
    def capture(self, prefill_len: int = 100, num_warmup: int = 3) -> None:
        """Capture CUDA graph for single-token decode.

        🟢 GPU REQUIRED — This method runs on CUDA.

        Args:
            prefill_len: Simulated prefill length for warmup.
            num_warmup: Number of warmup iterations before capture.
        """
        logger.info("talker_warmup num_warmup=%d", num_warmup)

        self._init_static_cache(self.model.config)
        self._build_attention_masks()

        self.cache_position[0] = prefill_len
        self.attn_mask.copy_(self._get_attn_mask(prefill_len))

        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()

        logger.info("Capturing CUDA graph for talker decode...")

        try:
            device_index = torch.device(self.device).index or torch.cuda.current_device()
            with torch.cuda.device(device_index):
                self.graph = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self._decode_step()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(self.graph):
                        self._decode_step()

            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.captured = True
            logger.info("Talker CUDA graph captured!")
        except Exception:
            self.graph = None
            self.static_cache = None
            self._mask_cache = {}
            logger.error("Talker CUDA graph capture failed, state reset")
            raise

    def prefill_kv(self, past_key_values) -> int:
        """Copy HF DynamicCache from prefill into StaticCache.

        Args:
            past_key_values: DynamicCache from HF forward.

        Returns:
            Prefill sequence length.
        """
        self.static_cache.reset()
        seq_len = 0
        for li in range(self.num_layers):
            k, v = past_key_values[li]
            seq_len = k.shape[2]
            if seq_len > self.max_seq_len:
                raise RuntimeError(
                    f"Input too long: {seq_len} tokens > max_seq_len={self.max_seq_len}. "
                    "Use shorter text or reference audio."
                )
            cache_pos = torch.arange(seq_len, device=self.device)
            self.static_cache.update(k, v, li, {"cache_position": cache_pos})
        return seq_len

    def set_generation_state(
        self,
        attention_mask: torch.Tensor | None,
        rope_deltas: torch.Tensor | None,
    ) -> None:
        """Set padding-aware attention mask and rope deltas."""
        mask_key = None
        full_attention_mask = None
        if attention_mask is not None:
            pad_counts = (attention_mask == 0).sum(dim=-1)
            mask_key = tuple(pad_counts.tolist())
            # Build full mask tensorially: arange < pad_counts → 0, else → 1
            full_attention_mask = torch.ones(
                attention_mask.shape[0], self.max_seq_len,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            positions = torch.arange(
                self.max_seq_len, device=attention_mask.device,
            ).unsqueeze(0)
            full_attention_mask = (
                positions >= pad_counts.unsqueeze(1)
            ).to(attention_mask.dtype)

        if not hasattr(self, '_mask_cache') or mask_key != self._mask_key:
            self._build_attention_masks(full_attention_mask)
            self._mask_key = mask_key

        if rope_deltas is None:
            self.rope_deltas.zero_()
        else:
            if rope_deltas.dim() == 1:
                rope_deltas = rope_deltas.unsqueeze(1)
            self.rope_deltas.copy_(
                rope_deltas.to(self.rope_deltas.device, dtype=self.rope_deltas.dtype)
            )

    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        """Run one decode step via CUDA graph replay.

        🟢 GPU REQUIRED.

        Args:
            input_embeds: [1, 1, hidden_size] input tensor.
            position: Current sequence position.

        Returns:
            [1, 1, hidden_size] hidden states. Static buffer — use immediately or clone.
        """
        if not self.captured:
            raise RuntimeError(
                "CUDA graph not captured. Call capture() before run()."
            )

        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self.attn_mask.copy_(self._get_attn_mask(position))

        delta = self.rope_deltas + self.cache_position[0].to(self.rope_deltas.dtype)
        # position_ids has shape [num_position_types, 1, 1] — expanded for RoPE
        self.position_ids.copy_(delta.unsqueeze(0).expand(self._num_position_types, -1, -1))

        self.graph.replay()
        return self.output_buf

    def reset(self) -> None:
        """Reset cache for new sequence."""
        if self.static_cache is not None:
            self.static_cache.reset()
