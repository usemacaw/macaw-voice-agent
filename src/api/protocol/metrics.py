"""
Typed response metrics — replaces untyped dict[str, object].

Every response emits a `macaw.metrics` event with these fields.
Using a dataclass ensures:
- Typos in field names are caught at write time (AttributeError)
- IDE autocomplete for all metric fields
- Default values document what's optional vs required
- to_dict() produces only populated fields (no None noise)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResponseMetrics:
    """Per-response observability metrics.

    Populated progressively during response execution.
    Fields left at default (0.0 / "" / empty) are omitted in to_dict().
    """

    # Identity
    response_id: str = ""
    turn: int = 0
    session_duration_s: float = 0.0
    barge_in_count: int = 0

    # ASR / input (populated by AudioInputHandler)
    asr_ms: float = 0.0
    speech_ms: float = 0.0
    asr_mode: str = ""
    input_chars: int = 0
    speech_rms: float = 0.0
    asr_partial_count: int = 0
    asr_last_partial: str = ""

    # VAD / turn detection
    vad_silence_wait_ms: float = 0.0
    smart_turn_inference_ms: float = 0.0
    smart_turn_waits: int = 0

    # LLM timing
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0

    # Pipeline
    pipeline_first_audio_ms: float = 0.0
    llm_first_sentence_ms: float = 0.0

    # TTS
    tts_synth_ms: float = 0.0
    tts_wait_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0

    # Encode + send (first audio chunk)
    encode_send_ms: float = 0.0

    # E2E
    e2e_ms: float = 0.0
    total_ms: float = 0.0

    # SLO
    slo_target_ms: float = 0.0
    slo_met: bool | None = None

    # Output
    output_chars: int = 0
    sentences: int = 0
    audio_chunks: int = 0

    # Tools
    tools_used: list[str] = field(default_factory=list)
    tool_rounds: int = 0
    tool_timings: list[dict] = field(default_factory=list)

    # Backpressure
    backpressure_level: int = 0
    events_dropped: int = 0

    # Early trigger (E2E streaming)
    early_trigger_words: int = 0
    early_trigger_partial: str = ""

    def to_dict(self) -> dict:
        """Convert to dict, omitting default/empty values for clean output."""
        result: dict = {}
        for key, val in self.__dict__.items():
            if val is None:
                continue
            if not isinstance(val, bool) and isinstance(val, (int, float)) and val == 0 and key not in (
                "turn", "barge_in_count", "tool_rounds", "backpressure_level",
                "events_dropped",
            ):
                continue
            if isinstance(val, str) and not val:
                continue
            if isinstance(val, list) and not val:
                continue
            result[key] = val
        return result

    def merge_prior(self, prior: dict[str, object]) -> None:
        """Merge ASR/VAD/early-trigger metrics from AudioInputHandler and session."""
        _MERGEABLE = (
            "asr_ms", "speech_ms", "asr_mode", "input_chars", "speech_rms",
            "vad_silence_wait_ms", "smart_turn_inference_ms", "smart_turn_waits",
            "asr_partial_count", "asr_last_partial",
            "early_trigger_words", "early_trigger_partial",
        )
        for key in _MERGEABLE:
            if key in prior:
                setattr(self, key, prior[key])
