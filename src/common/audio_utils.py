"""
Utilitarios de conversao de audio para providers STT/TTS.

Funcoes puras para converter entre PCM 16-bit e numpy float32,
e para resample entre sample rates diferentes.

Usa apenas numpy (ja e dependencia do projeto).
"""

import numpy as np


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
    """Resample via interpolacao linear com numpy.

    Args:
        samples: Array de amostras (float32 ou int16).
        from_rate: Sample rate original (Hz).
        to_rate: Sample rate destino (Hz).

    Returns:
        Array resampled com o numero correto de amostras.
    """
    if from_rate == to_rate:
        return samples

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
