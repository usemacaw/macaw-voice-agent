"""
TurnPipeline — explicit model for a voice conversation turn.

A "turn" is the atomic unit of voice conversation:
  User speaks → system understands → system responds → user hears

This module makes the turn a first-class concept instead of leaving
the flow implicit across session.py, audio_input.py, and response_runner.py.

Usage from RealtimeSession:
    turn = VoiceTurn(turn_id, store, emitter, ...)
    await turn.execute()
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from server.conversation_store import ConversationStore
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.turn")


class TurnStage(Enum):
    """Stages of a voice turn, used for observability."""
    CREATED = auto()
    INPUT_DETECTED = auto()
    TRANSCRIBED = auto()
    CONTEXT_BUILT = auto()
    LLM_STREAMING = auto()
    TOOL_EXECUTING = auto()
    TTS_SYNTHESIZING = auto()
    DELIVERING = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class TurnMetrics:
    """Per-stage timing for a single turn."""
    turn_id: str = ""
    turn_number: int = 0
    stage: TurnStage = TurnStage.CREATED
    created_at: float = 0.0

    # Input stage
    speech_ms: float = 0.0
    speech_rms: float = 0.0
    asr_ms: float = 0.0
    asr_mode: str = ""
    input_chars: int = 0

    # LLM stage
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    llm_first_sentence_ms: float = 0.0

    # Tool stage
    tool_rounds: int = 0
    tools_used: list[str] = field(default_factory=list)
    tool_timings: list[dict] = field(default_factory=list)

    # TTS/Pipeline stage
    tts_synth_ms: float = 0.0
    tts_wait_ms: float = 0.0
    pipeline_first_audio_ms: float = 0.0
    pipeline_total_ms: float = 0.0

    # Delivery stage
    e2e_ms: float = 0.0
    total_ms: float = 0.0

    # Output
    output_chars: int = 0
    sentences: int = 0
    audio_chunks: int = 0

    # Session context
    session_duration_s: float = 0.0
    barge_in_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for macaw.metrics emission."""
        d = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or key == "stage" or key == "created_at":
                continue
            if isinstance(value, list) and not value:
                continue
            if isinstance(value, (int, float)) and value == 0 and key not in ("turn_number", "barge_in_count"):
                continue
            d[key] = value
        return d


class VoiceTurn:
    """Represents a single voice conversation turn.

    Lifecycle:
        1. Created when speech is detected or response.create received
        2. Progresses through stages (input → transcribe → context → LLM → TTS → deliver)
        3. Completes or is cancelled (barge-in)
        4. Metrics emitted

    This class is the "narrative abstraction" — it makes the turn
    flow readable as a sequence of stages.
    """

    def __init__(
        self,
        turn_number: int,
        session_id: str,
        session_start: float,
        barge_in_count: int,
    ):
        self.turn_id = f"turn_{uuid.uuid4().hex[:16]}"
        self.metrics = TurnMetrics(
            turn_id=self.turn_id,
            turn_number=turn_number,
            created_at=time.perf_counter(),
            session_duration_s=round(time.perf_counter() - session_start, 1),
            barge_in_count=barge_in_count,
        )
        self._sid = session_id

    def advance(self, stage: TurnStage) -> None:
        """Move turn to next stage (for observability/debugging)."""
        self.metrics.stage = stage
        logger.debug(f"[{self._sid[:8]}] Turn {self.turn_id[:12]} → {stage.name}")

    def record_input(self, prior_metrics: dict) -> None:
        """Record input/ASR metrics from the speech phase."""
        for key in ("asr_ms", "speech_ms", "asr_mode", "input_chars", "speech_rms"):
            if key in prior_metrics:
                setattr(self.metrics, key, prior_metrics[key])
        self.advance(TurnStage.TRANSCRIBED)

    def record_e2e(self, speech_stopped_at: float | None) -> None:
        """Record E2E latency on first audio chunk sent."""
        if speech_stopped_at is not None:
            e2e_ms = (time.perf_counter() - speech_stopped_at) * 1000
            self.metrics.e2e_ms = round(e2e_ms, 1)
            logger.info(
                f"[{self._sid[:8]}] E2E LATENCY: {e2e_ms:.0f}ms "
                f"(speech_stopped → first_audio_sent)"
            )

    def finalize(self) -> dict:
        """Mark turn complete and return metrics dict."""
        self.metrics.total_ms = round(
            (time.perf_counter() - self.metrics.created_at) * 1000, 1
        )
        self.advance(TurnStage.COMPLETED)
        return self.metrics.to_dict()
