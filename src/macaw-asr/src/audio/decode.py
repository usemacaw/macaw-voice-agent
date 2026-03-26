"""Audio file decoder — converts any audio format to PCM16 float32.

Supports: wav, mp3, m4a, webm, ogg, flac, mpga via soundfile.
Falls back to raw PCM16 if soundfile fails.
"""

from __future__ import annotations

import io
import logging
import struct
import wave

import numpy as np

logger = logging.getLogger("macaw-asr.audio.decode")


def decode_audio(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Decode audio bytes to float32 array + sample_rate.

    Args:
        data: Raw audio file bytes.
        filename: Original filename (for format hint).

    Returns:
        (float32_array, sample_rate)

    Raises:
        ValueError: If audio cannot be decoded.
    """
    if not data:
        raise ValueError("Empty audio data")

    # Try soundfile first (handles wav, flac, ogg, mp3 via libsndfile)
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]  # mono
        logger.debug("Decoded via soundfile: %.1fs @ %dHz", len(audio) / sr, sr)
        return audio, sr
    except ImportError:
        pass
    except Exception as e:
        logger.debug("soundfile decode failed: %s", e)

    # Try wave module (stdlib, wav only)
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

            if sw == 2:
                samples = np.frombuffer(frames, dtype=np.int16)
            elif sw == 4:
                samples = np.frombuffer(frames, dtype=np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {sw}")

            if n_channels > 1:
                samples = samples[::n_channels]  # take first channel

            audio = samples.astype(np.float32) / (2 ** (sw * 8 - 1))
            logger.debug("Decoded via wave: %.1fs @ %dHz", len(audio) / sr, sr)
            return audio, sr
    except Exception as e:
        logger.debug("wave decode failed: %s", e)

    # Try librosa via temp file (handles mp3, webm, etc. via ffmpeg)
    # librosa/ffmpeg needs a real file for some formats like webm
    try:
        import librosa
        import tempfile
        import os

        ext = ""
        if filename:
            ext = os.path.splitext(filename)[1]
        if not ext:
            # Guess from magic bytes
            if data[:4] == b'\x1aE\xdf\xa3':
                ext = ".webm"
            elif data[:4] == b'ID3\x03' or data[:2] == b'\xff\xfb':
                ext = ".mp3"
            else:
                ext = ".bin"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
            logger.debug("Decoded via librosa: %.1fs @ %dHz", len(audio) / sr, sr)
            return audio, sr
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.debug("librosa decode failed: %s", e)

    # Last resort: treat as raw PCM16 at 16kHz
    if len(data) % 2 == 0:
        samples = np.frombuffer(data, dtype=np.int16)
        audio = samples.astype(np.float32) / 32768.0
        sr = 16000
        logger.warning("Treating as raw PCM16 @ %dHz: %d samples", sr, len(audio))
        return audio, sr

    raise ValueError(
        f"Cannot decode audio ({len(data)} bytes, filename={filename!r}). "
        "Supported formats: wav, mp3, m4a, webm, ogg, flac."
    )
