"""Streaming Code2Wav decoder with torch.compile and sliding window.

Combines dffdeeq fixed-shape padding with faster-qwen3-tts sliding window.
Decodes codec tokens → PCM float32 24kHz incrementally.

Key techniques:
- Fixed-shape left-padding for torch.compile (single compiled kernel)
- Sliding window of 25 frames context (O(window) per decode, not O(total))
- samples_per_frame = 1920 (12Hz codec @ 24kHz = 24000/12.5 = 1920 samples/frame)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from macaw_tts.audio import SAMPLES_PER_FRAME

logger = logging.getLogger("macaw-tts.decoder")


class StreamingDecoder:
    """Incremental Code2Wav decoder for streaming TTS.

    Maintains codec buffer and decodes sliding windows.
    Uses torch.compile with fixed-shape padding for consistent performance.

    Args:
        speech_tokenizer: Qwen3-TTS speech tokenizer (Code2Wav model).
        decode_window: Number of frames in decode context window (40 default).
        context_frames: Left context for sliding window quality (25 default).
    """

    def __init__(
        self,
        speech_tokenizer,
        decode_window: int = 40,
        context_frames: int = 25,
    ):
        self._tokenizer = speech_tokenizer
        self._decode_window = decode_window
        self._context_frames = context_frames
        self._compiled = False
        self._compiled_forward = None

    def compile(self, mode: str = "reduce-overhead") -> None:
        """Apply torch.compile to decoder forward.

        🟢 GPU REQUIRED — compilation happens on first call.

        Args:
            mode: torch.compile mode ("reduce-overhead" includes CUDA graphs).
        """
        if hasattr(self._tokenizer, "decoder"):
            decoder = self._tokenizer.decoder
            self._compiled_forward = torch.compile(
                decoder.forward,
                mode=mode,
                fullgraph=False,
                dynamic=False,
            )
            self._compiled = True
            logger.info(f"Decoder compiled with mode={mode}")
        else:
            logger.warning("Speech tokenizer has no decoder attribute, skipping compile")

    def decode_window(
        self,
        all_codes: list[torch.Tensor],
        num_new_frames: int,
        ref_codes: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Decode new frames using sliding window.

        🟢 GPU REQUIRED for actual decode.

        Args:
            all_codes: List of codec frame tensors (GPU-resident).
            num_new_frames: How many new frames to extract audio for.
            ref_codes: Optional reference codes for ICL voice cloning context.

        Returns:
            Float32 audio array for the new frames only.
        """
        total_frames = len(all_codes)
        if total_frames == 0 or num_new_frames == 0:
            return np.array([], dtype=np.float32)

        # Build sliding window
        window_start = max(0, total_frames - num_new_frames - self._context_frames)
        window_codes = torch.stack(all_codes[window_start:], dim=0)

        # Add ref_code context prefix if window is smaller than decode_window
        ref_prefix_frames = 0
        if ref_codes is not None and window_codes.shape[0] < self._decode_window:
            available = self._decode_window - window_codes.shape[0]
            ref_prefix_frames = min(available, ref_codes.shape[0])
            if ref_prefix_frames > 0:
                ref_prefix = ref_codes[-ref_prefix_frames:]
                window_codes = torch.cat([ref_prefix, window_codes], dim=0)

        # Decode (with optional fixed-shape padding for torch.compile)
        audio = self._decode_codes(window_codes)

        # Extract only new frames' audio
        context_frames_in_window = window_codes.shape[0] - num_new_frames
        if context_frames_in_window > 0:
            skip_samples = int(context_frames_in_window * SAMPLES_PER_FRAME)
            audio = audio[skip_samples:]

        return audio

    def _decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        """Decode codec frames → PCM float32.

        Uses fixed-shape padding if compiled, direct decode otherwise.

        Args:
            codes: [T, num_codebooks] tensor of codec IDs.

        Returns:
            Float32 audio array.
        """
        device = codes.device

        # Add batch dimension: [T, Q] → [1, T, Q]
        if codes.dim() == 2:
            codes_batch = codes.unsqueeze(0)
        else:
            codes_batch = codes

        if self._compiled and self._compiled_forward is not None:
            # Fixed-shape padding for torch.compile
            audio = self._decode_padded(codes_batch)
        else:
            # Direct decode
            audio = self._decode_direct(codes_batch)

        return audio

    def _decode_padded(self, codes: torch.Tensor) -> np.ndarray:
        """Decode with left-padding to fixed decode_window size.

        Fixed shape → single compiled kernel → maximum performance.
        Based on dffdeeq decode_padded implementation.
        """
        B, T, Q = codes.shape
        target = self._decode_window

        if T < target:
            # Left-pad with zeros (clamp to valid range before decode)
            pad = torch.zeros(B, target - T, Q, dtype=codes.dtype, device=codes.device)
            codes_padded = torch.cat([pad, codes], dim=1)
        elif T > target:
            # Trim to window (keep right side)
            codes_padded = codes[:, -target:, :]
        else:
            codes_padded = codes.contiguous()

        # Transpose for decoder: [B, T, Q] → [B, Q, T]
        codes_t = codes_padded.transpose(1, 2)

        wav_tensor = self._compiled_forward(codes_t)

        # Convert to numpy
        wav = wav_tensor.squeeze().float().detach().cpu().numpy()

        # Trim padding from output (left side)
        if T < target:
            pad_frames = target - T
            trim_samples = int(pad_frames * SAMPLES_PER_FRAME)
            wav = wav[trim_samples:]

        return wav

    def _decode_direct(self, codes: torch.Tensor) -> np.ndarray:
        """Direct decode without compilation (fallback path)."""
        # Speech tokenizer expects [T, Q] without batch dim in the dict
        if codes.dim() == 3:
            codes_for_decode = codes.squeeze(0)  # [1, T, Q] → [T, Q]
        else:
            codes_for_decode = codes

        wavs, sr = self._tokenizer.decode(
            [{"audio_codes": codes_for_decode}]
        )

        audio = wavs[0]
        if hasattr(audio, "cpu"):
            audio = audio.flatten().float().cpu().numpy()
        else:
            audio = np.asarray(audio, dtype=np.float32).flatten()

        return audio

    @property
    def decode_window_size(self) -> int:
        return self._decode_window

    @property
    def context_frames(self) -> int:
        return self._context_frames

    @property
    def is_compiled(self) -> bool:
        return self._compiled
