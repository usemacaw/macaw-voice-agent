"""
Utilitarios de conversao de audio para providers STT/TTS.

Funcoes puras para converter entre PCM 16-bit e numpy float32,
e para resample entre sample rates diferentes.

Usa scipy para resampling com anti-aliasing quando disponivel,
com fallback para interpolacao linear via numpy.
"""

import numpy as np

try:
    from scipy.signal import resample_poly as _scipy_resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def pcm_to_float32(pcm_data: bytes) -> np.ndarray:
    """PCM 16-bit signed little-endian -> numpy float32 normalizado [-1, 1].

    Args:
        pcm_data: Audio em bytes PCM 16-bit LE.

    Returns:
        Array float32 normalizado no range [-1.0, 1.0].

    Raises:
        ValueError: Se pcm_data tem tamanho impar (invalido para 16-bit).
    """
    if len(pcm_data) % 2 != 0:
        raise ValueError(
            f"PCM 16-bit requer numero par de bytes, recebeu {len(pcm_data)}"
        )
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm(samples: np.ndarray) -> bytes:
    """numpy float32 [-1, 1] -> PCM 16-bit signed little-endian bytes.

    Valores fora de [-1, 1] sao clipped para evitar overflow.

    Args:
        samples: Array float32 normalizado.

    Returns:
        Bytes de audio PCM 16-bit LE.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767.0).astype(np.int16)
    return int_samples.tobytes()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio com anti-aliasing quando possivel.

    Usa scipy.signal.resample_poly (filtro polifasico com anti-aliasing)
    quando disponivel, com fallback para interpolacao linear via numpy.

    Args:
        samples: Array de amostras (float32 ou int16).
        from_rate: Sample rate original (Hz).
        to_rate: Sample rate destino (Hz).

    Returns:
        Array resampled com o numero correto de amostras.
    """
    if from_rate == to_rate:
        return samples
    if len(samples) == 0:
        return np.array([], dtype=samples.dtype)

    if _HAS_SCIPY:
        from math import gcd
        g = gcd(from_rate, to_rate)
        up = to_rate // g
        down = from_rate // g
        return _scipy_resample_poly(samples, up, down).astype(samples.dtype)

    # Fallback: interpolacao linear (sem anti-aliasing)
    duration = len(samples) / from_rate
    target_length = int(duration * to_rate)

    if target_length == 0:
        return np.array([], dtype=samples.dtype)

    indices = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(indices, np.arange(len(samples)), samples).astype(samples.dtype)


# Mapeamento ISO-639-1 -> nome completo do idioma (titlecase).
# Compartilhado entre QwenTTS e VllmTTS.
# QwenTTS aplica .lower() (API espera lowercase); VllmTTS usa direto (API espera titlecase).
QWEN_LANGUAGE_MAP = {
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
