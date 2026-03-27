"""Qwen3-ASR preprocessor — audio to model-ready tensors.

SRP: Only responsible for converting audio to input tensors.
Strategy Pattern: GPU mel (primary) with CPU processor (fallback).
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

import numpy as np

from macaw_asr.models.contracts import IPreprocessor
from macaw_asr.models.types import InputsWrapper

logger = logging.getLogger("macaw-asr.models.qwen.preprocessor")

AUDIO_PAD_TOKEN = 151676  # <|audio_pad|>


class QwenPreprocessor(IPreprocessor):
    """Converts audio to Qwen3-ASR input tensors.

    Primary path: GPU mel spectrogram (~2ms).
    Fallback: CPU processor (~15ms warm, ~1500ms cold).
    """

    def __init__(self, thinker, processor, prompt_builder) -> None:
        self._thinker = thinker
        self._processor = processor
        self._prompt_builder = prompt_builder

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        t0 = _time.perf_counter()
        try:
            result = self._prepare_gpu(audio, prefix)
        except Exception as e:
            logger.warning("GPU mel failed, falling back to CPU", exc_info=True)  # Fix #10: WARNING not DEBUG
            result = self._prepare_cpu(audio, prefix)
        result._prepare_ms = (_time.perf_counter() - t0) * 1000
        return result

    def fast_finish_inputs(
        self, audio: np.ndarray, cached_inputs: Any,
        cached_audio_len: int, prefix: str = "",
    ) -> Any | None:
        """GPU mel + token adjustment from background cache (~2ms vs ~15ms).

        Returns None if GPU mel fails — engine falls back to full prepare_inputs().
        """
        import torch
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths

        device = self._thinker.device
        fe = self._processor.feature_extractor

        try:
            mel_np = fe._torch_extract_fbank_features(audio, device=str(device))
        except Exception as e:
            logger.debug("fast_finish GPU mel failed: %s", e)
            return None  # GPU mel unavailable, fall back to full prepare_inputs
        full_frames = len(audio) // fe.hop_length
        mel_trimmed = mel_np[:, :full_frames]

        input_features = torch.from_numpy(mel_trimmed).unsqueeze(0).to(device=device, dtype=self._thinker.dtype)
        feature_mask = torch.ones(1, full_frames, dtype=torch.int32, device=device)

        bg_frames = cached_audio_len // fe.hop_length
        bg_tokens = _get_feat_extract_output_lengths(torch.tensor([bg_frames])).item()
        full_tokens = _get_feat_extract_output_lengths(torch.tensor([full_frames])).item()
        extra_pads = full_tokens - bg_tokens

        bg_ids = cached_inputs["input_ids"][0].cpu()
        if extra_pads > 0:
            pad_pos = torch.where(bg_ids == AUDIO_PAD_TOKEN)[0]
            if len(pad_pos) > 0:
                pos = pad_pos[-1].item() + 1
                new_ids = torch.cat([bg_ids[:pos], torch.full((extra_pads,), AUDIO_PAD_TOKEN, dtype=bg_ids.dtype), bg_ids[pos:]])
            else:
                new_ids = bg_ids
        else:
            new_ids = bg_ids

        return {
            "input_ids": new_ids.unsqueeze(0).to(device),
            "attention_mask": torch.ones(1, new_ids.shape[0], dtype=torch.int64, device=device),
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        }

    # ==================== Internal ====================

    def _prepare_gpu(self, audio: np.ndarray, prefix: str) -> InputsWrapper:
        """GPU mel + CPU tokenization."""
        import torch
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths

        device = self._thinker.device
        fe = self._processor.feature_extractor

        prompt = self._prompt_builder.build(prefix)
        tokenized = self._processor.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]

        mel_np = fe._torch_extract_fbank_features(audio, device=str(device))
        full_frames = len(audio) // fe.hop_length
        mel_trimmed = mel_np[:, :full_frames]

        input_features = torch.from_numpy(mel_trimmed).unsqueeze(0).to(device=device, dtype=self._thinker.dtype)
        feature_mask = torch.ones(1, full_frames, dtype=torch.int32, device=device)

        audio_tokens = _get_feat_extract_output_lengths(torch.tensor([full_frames])).item()

        pad_positions = torch.where(input_ids == AUDIO_PAD_TOKEN)[0]
        if len(pad_positions) > 0:
            existing = len(pad_positions)
            extra = audio_tokens - existing
            if extra > 0:
                pos = pad_positions[-1].item() + 1
                input_ids = torch.cat([input_ids[:pos], torch.full((extra,), AUDIO_PAD_TOKEN, dtype=input_ids.dtype), input_ids[pos:]])
            elif extra < 0:
                mask = torch.ones(len(input_ids), dtype=torch.bool)
                mask[pad_positions[audio_tokens:]] = False
                input_ids = input_ids[mask]
        else:
            raise RuntimeError("No audio_pad token found in tokenized prompt")

        return InputsWrapper({
            "input_ids": input_ids.unsqueeze(0).to(device),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.int64, device=device),
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        })

    def _prepare_cpu(self, audio: np.ndarray, prefix: str) -> Any:
        """CPU processor fallback."""
        prompt = self._prompt_builder.build(prefix)
        inputs = self._processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        return inputs.to(self._thinker.device).to(self._thinker.dtype)
