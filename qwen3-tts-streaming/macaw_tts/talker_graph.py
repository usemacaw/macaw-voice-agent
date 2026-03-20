"""CUDA graph capture for Talker single-token decode step.

Based on faster-qwen3-tts talker_graph.py.
Uses transformers StaticCache for fixed-size KV tensors compatible with CUDA graphs.

The Talker has 20-28 transformer layers. We capture the single-token decode
as a CUDA graph for ~8ms/step (vs ~50ms without graphs).

Strategy:
- Prefill via HF forward (variable-length) → copies KV into StaticCache
- Decode via CUDA graph replay (fixed-shape) → ~8ms/step
- Pre-built attention mask table → O(1) per step
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger("macaw-tts.talker-graph")


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

        # RoPE deltas and position IDs
        self.rope_deltas = torch.zeros(1, 1, dtype=torch.float32, device=device)
        self.position_ids = torch.zeros(3, 1, 1, dtype=torch.float32, device=device)

        # StaticCache and graph (initialized on capture)
        self.static_cache = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured = False
        self.attn_mask = None
        self.attn_mask_table: Optional[list] = None
        self._mask_key = None

    def _init_static_cache(self, config):
        """Create StaticCache and force lazy layer initialization."""
        from transformers import StaticCache

        self.static_cache = StaticCache(config=config, max_cache_len=self.max_seq_len)

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _build_attention_masks(self, attention_mask: torch.Tensor | None = None):
        """Pre-build causal masks for all positions. O(1) lookup per step."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        dummy = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        mask_fn = (
            create_sliding_window_causal_mask
            if getattr(self.model.config, "sliding_window", None) is not None
            else create_causal_mask
        )

        self.attn_mask_table = [None] * self.max_seq_len
        for i in range(self.max_seq_len):
            pos = torch.tensor([i], device=self.device)
            self.attn_mask_table[i] = mask_fn(
                config=self.model.config,
                input_embeds=dummy,
                attention_mask=attention_mask,
                cache_position=pos,
                past_key_values=self.static_cache,
            )

        if self.attn_mask is None:
            self.attn_mask = self.attn_mask_table[0].clone()
        else:
            self.attn_mask.copy_(self.attn_mask_table[0])

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
        logger.info(f"Warming up talker graph ({num_warmup} runs)...")

        self._init_static_cache(self.model.config)
        self._build_attention_masks()

        self.cache_position[0] = prefill_len
        self.attn_mask.copy_(self.attn_mask_table[prefill_len])

        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()

        logger.info("Capturing CUDA graph for talker decode...")

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
            full_attention_mask = torch.ones(
                attention_mask.shape[0], self.max_seq_len,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            for b, pads in enumerate(pad_counts.tolist()):
                if pads > 0:
                    full_attention_mask[b, :pads] = 0

        if self.attn_mask_table is None or mask_key != self._mask_key:
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
        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self.attn_mask.copy_(self.attn_mask_table[position])

        delta = self.rope_deltas + self.cache_position[0].to(self.rope_deltas.dtype)
        self.position_ids.copy_(delta.unsqueeze(0).expand(3, -1, -1))

        self.graph.replay()
        return self.output_buf

    def reset(self) -> None:
        """Reset cache for new sequence."""
        if self.static_cache is not None:
            self.static_cache.reset()
