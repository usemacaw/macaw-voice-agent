"""macaw-asr: Self-contained ASR engine with pluggable models.

Usage (batch):
    import macaw_asr

    engine = await macaw_asr.create_engine(model="qwen")
    text = await engine.transcribe(pcm_bytes)
    await engine.stop()

Usage (streaming):
    engine = await macaw_asr.create_engine()
    await engine.create_session("s1")
    for chunk in audio_chunks:
        partial = await engine.push_audio("s1", chunk)
    final = await engine.finish_session("s1")
    await engine.stop()

CLI:
    macaw-asr pull Qwen/Qwen3-ASR-0.6B
    macaw-asr transcribe audio.wav
    macaw-asr serve
    macaw-asr list
"""

from macaw_asr.config import AudioConfig, EngineConfig, StreamingConfig
from macaw_asr.models.base import ModelOutput, list_models, register_model
from macaw_asr.runner.engine import ASREngine

__version__ = "0.1.0"

__all__ = [
    "ASREngine",
    "AudioConfig",
    "EngineConfig",
    "ModelOutput",
    "StreamingConfig",
    "create_engine",
    "list_models",
    "register_model",
]


async def create_engine(
    model: str = "qwen",
    model_id: str = "Qwen/Qwen3-ASR-0.6B",
    device: str = "cuda:0",
    language: str = "pt",
    **kwargs,
) -> ASREngine:
    """Create, configure, and start an ASR engine.

    Args:
        model: Model registry key (e.g. 'qwen', 'mock').
        model_id: HuggingFace model ID or local path.
        device: Device for inference.
        language: ISO-639-1 language code.
        **kwargs: Additional EngineConfig fields.

    Returns:
        Started ASREngine ready for transcription.
    """
    config = EngineConfig(
        model_name=model,
        model_id=model_id,
        device=device,
        language=language,
        **kwargs,
    )
    engine = ASREngine(config)
    await engine.start()
    return engine
