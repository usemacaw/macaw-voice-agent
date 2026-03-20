"""CUDA graph capture for CodePredictor 15-step decode loop.

Based on faster-qwen3-tts predictor_graph.py.
Generates 15 codebook tokens (CB1-CB15) autoregressively in a single CUDA graph.

Architecture:
- Step 0: Prefill with 2 tokens (past_hidden + CB0_embed) → sample CB1
- Steps 1-14: Decode 1 token at a time → sample CB2..CB15
- Entire loop captured as single CUDA graph → ~4ms total
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from macaw_tts.sampling import sample_logits

logger = logging.getLogger("macaw-tts.predictor-graph")


class PredictorCUDAGraph:
    """CUDA graph for CodePredictor 15-step loop.

    ⚠️ REQUIRES GPU — capture() and run() need CUDA device.

    Usage:
        graph = PredictorCUDAGraph(code_predictor, config, talker_hidden_size)
        graph.capture()  # 🟢 GPU
        tokens = graph.run(pred_input)  # 🟢 GPU → [15] long tensor
    """

    def __init__(
        self,
        code_predictor,
        pred_config,
        talker_hidden_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        self.device = device
        self.dtype = dtype
        self.num_layers = pred_config.num_hidden_layers
        self.hidden_size = pred_config.hidden_size
        self.num_code_groups = pred_config.num_code_groups
        self.num_codebooks = self.num_code_groups - 1  # 15
        self.max_seq = 2 + self.num_codebooks  # 17

        # Sampling parameters
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        # References to model components (not copies)
        cp = code_predictor
        self.small_to_mtp = cp.small_to_mtp_projection
        self.pred_model = cp.model  # Inner transformer (5 layers)
        self.lm_heads = cp.lm_head  # ModuleList[15]
        self.codec_embeds = cp.model.codec_embedding  # ModuleList[15]

        has_sliding = "sliding_attention" in getattr(
            self.pred_model.config, "layer_types", []
        )
        self._has_sliding = has_sliding

        # Pre-allocate cache position tensors (avoid CPU→GPU in graph)
        self.prefill_cache_pos = torch.arange(2, device=device)
        self.decode_cache_positions = [
            torch.tensor([2 + i], device=device) for i in range(self.num_codebooks - 1)
        ]

        # I/O buffers
        self.input_buf = torch.zeros(
            1, 2, talker_hidden_size, dtype=dtype, device=device
        )
        self.output_tokens = torch.zeros(
            self.num_codebooks, dtype=torch.long, device=device
        )

        # State (initialized on capture)
        self.static_cache = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.captured = False
        self.prefill_attn = None
        self.decode_attn: Optional[list] = None

    def _init_static_cache(self):
        """Create StaticCache and force lazy layer init."""
        from transformers import StaticCache

        self.static_cache = StaticCache(
            config=self.pred_model.config, max_cache_len=self.max_seq
        )

        config = self.pred_model.config
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dummy_k = torch.zeros(
            1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device
        )
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _make_attn_mask(self, input_embeds: torch.Tensor, cache_position: torch.Tensor):
        """Build attention mask dict for model forward."""
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        mask = create_causal_mask(
            config=self.pred_model.config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=self.static_cache,
        )
        if self._has_sliding:
            sliding = create_sliding_window_causal_mask(
                config=self.pred_model.config,
                input_embeds=input_embeds,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=self.static_cache,
            )
            return {"full_attention": mask, "sliding_attention": sliding}
        return {"full_attention": mask}

    def _build_attention_masks(self):
        """Pre-build masks for prefill (2 tokens) and each decode step (1 token)."""
        dummy_prefill = torch.zeros(
            1, 2, self.hidden_size, dtype=self.dtype, device=self.device
        )
        dummy_decode = torch.zeros(
            1, 1, self.hidden_size, dtype=self.dtype, device=self.device
        )
        self.prefill_attn = self._make_attn_mask(dummy_prefill, self.prefill_cache_pos)
        self.decode_attn = []
        for pos in self.decode_cache_positions:
            self.decode_attn.append(self._make_attn_mask(dummy_decode, pos))

    def _full_loop(self):
        """Full 15-step predictor loop on static buffers."""
        # Project from talker hidden → predictor hidden
        h = self.small_to_mtp(self.input_buf)

        # Prefill: 2 tokens through 5 layers
        out = self.pred_model(
            inputs_embeds=h,
            attention_mask=self.prefill_attn,
            past_key_values=self.static_cache,
            cache_position=self.prefill_cache_pos,
            use_cache=True,
        )
        h = out.last_hidden_state

        # First codebook: logits from last position
        logits = self.lm_heads[0](h[:, -1:, :])
        tok = sample_logits(
            logits[:, 0, :],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        self.output_tokens[0] = tok[0]

        # Remaining 14 codebooks
        for cb_idx in range(1, self.num_codebooks):
            emb = self.codec_embeds[cb_idx - 1](tok.unsqueeze(0))
            emb = self.small_to_mtp(emb)

            out = self.pred_model(
                inputs_embeds=emb,
                attention_mask=self.decode_attn[cb_idx - 1],
                past_key_values=self.static_cache,
                cache_position=self.decode_cache_positions[cb_idx - 1],
                use_cache=True,
            )
            h = out.last_hidden_state

            logits = self.lm_heads[cb_idx](h[:, -1:, :])
            tok = sample_logits(
                logits[:, 0, :],
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            self.output_tokens[cb_idx] = tok[0]

        return self.output_tokens

    @torch.inference_mode()
    def capture(self, num_warmup: int = 3) -> None:
        """Warmup and capture CUDA graph.

        🟢 GPU REQUIRED.
        """
        logger.info(f"Warming up predictor ({num_warmup} runs)...")

        self._init_static_cache()
        self._build_attention_masks()

        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()

        logger.info("Capturing CUDA graph for predictor...")

        device_index = torch.device(self.device).index or torch.cuda.current_device()
        with torch.cuda.device(device_index):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.graph = torch.cuda.CUDAGraph()
                self.static_cache.reset()
                self._full_loop()
                torch.cuda.synchronize()

                self.static_cache.reset()
                with torch.cuda.graph(self.graph):
                    self._full_loop()

        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        logger.info("Predictor CUDA graph captured!")

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """Run captured graph.

        🟢 GPU REQUIRED.

        Args:
            pred_input: [1, 2, talker_hidden_size] (past_hidden + CB0 embed).

        Returns:
            [15] long tensor of codebook token IDs.
        """
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self.graph.replay()
        return self.output_tokens.clone()
