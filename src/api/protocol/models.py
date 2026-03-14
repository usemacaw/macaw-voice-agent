"""
Data models for OpenAI Realtime API protocol.

Matches the OpenAI Realtime API spec for session config, conversation items,
content parts, and turn detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SessionConfigValidationError(ValueError):
    """Raised when session config update contains invalid values."""


@dataclass
class TurnDetection:
    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 200
    create_response: bool = True
    interrupt_response: bool = True

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response,
            "interrupt_response": self.interrupt_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TurnDetection:
        return cls(
            type=data.get("type", "server_vad"),
            threshold=data.get("threshold", 0.5),
            prefix_padding_ms=data.get("prefix_padding_ms", 300),
            silence_duration_ms=data.get("silence_duration_ms", 200),
            create_response=data.get("create_response", True),
            interrupt_response=data.get("interrupt_response", True),
        )


@dataclass
class SessionConfig:
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    instructions: str = ""
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: dict | None = None
    turn_detection: TurnDetection | None = field(default_factory=TurnDetection)
    tools: list[dict] = field(default_factory=list)
    tool_choice: str | dict = "auto"
    temperature: float = 0.8
    max_response_output_tokens: int | str = "inf"

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "modalities": self.modalities,
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "input_audio_transcription": self.input_audio_transcription,
            "turn_detection": self.turn_detection.to_dict() if self.turn_detection else None,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        td = data.get("turn_detection")
        turn_detection = TurnDetection.from_dict(td) if isinstance(td, dict) else (None if td is None else TurnDetection())
        return cls(
            modalities=data.get("modalities", ["text", "audio"]),
            instructions=data.get("instructions", ""),
            voice=data.get("voice", "alloy"),
            input_audio_format=data.get("input_audio_format", "pcm16"),
            output_audio_format=data.get("output_audio_format", "pcm16"),
            input_audio_transcription=data.get("input_audio_transcription"),
            turn_detection=turn_detection,
            tools=data.get("tools", []),
            tool_choice=data.get("tool_choice", "auto"),
            temperature=data.get("temperature", 0.8),
            max_response_output_tokens=data.get("max_response_output_tokens", "inf"),
        )

    _VALID_AUDIO_FORMATS = {"pcm16", "g711_ulaw", "g711_alaw"}
    _VALID_MODALITIES = {"text", "audio"}
    _MAX_INSTRUCTIONS_CHARS = 50_000
    _MAX_TOOLS = 128
    _MAX_TOOL_SCHEMA_CHARS = 200_000  # total JSON chars across all tools

    def update(self, data: dict) -> None:
        """Update config in-place from partial dict (session.update semantics).

        Raises:
            SessionConfigValidationError: If any value is invalid.
        """
        if "modalities" in data:
            val = data["modalities"]
            if not isinstance(val, list) or not all(isinstance(m, str) for m in val):
                raise SessionConfigValidationError("modalities must be a list of strings")
            invalid = set(val) - self._VALID_MODALITIES
            if invalid:
                raise SessionConfigValidationError(
                    f"Invalid modalities: {invalid}. Valid: {self._VALID_MODALITIES}"
                )
            self.modalities = val
        if "instructions" in data:
            val = data["instructions"]
            if not isinstance(val, str):
                raise SessionConfigValidationError("instructions must be a string")
            if len(val) > self._MAX_INSTRUCTIONS_CHARS:
                raise SessionConfigValidationError(
                    f"instructions too long: {len(val)} chars (max {self._MAX_INSTRUCTIONS_CHARS})"
                )
            self.instructions = val
        if "voice" in data:
            val = data["voice"]
            if not isinstance(val, str):
                raise SessionConfigValidationError("voice must be a string")
            self.voice = val
        if "input_audio_format" in data:
            val = data["input_audio_format"]
            if val not in self._VALID_AUDIO_FORMATS:
                raise SessionConfigValidationError(
                    f"Invalid input_audio_format: {val}. Valid: {self._VALID_AUDIO_FORMATS}"
                )
            self.input_audio_format = val
        if "output_audio_format" in data:
            val = data["output_audio_format"]
            if val not in self._VALID_AUDIO_FORMATS:
                raise SessionConfigValidationError(
                    f"Invalid output_audio_format: {val}. Valid: {self._VALID_AUDIO_FORMATS}"
                )
            self.output_audio_format = val
        if "input_audio_transcription" in data:
            self.input_audio_transcription = data["input_audio_transcription"]
        if "turn_detection" in data:
            td = data["turn_detection"]
            self.turn_detection = TurnDetection.from_dict(td) if isinstance(td, dict) else None
        if "tools" in data:
            val = data["tools"]
            if not isinstance(val, list):
                raise SessionConfigValidationError("tools must be a list")
            if len(val) > self._MAX_TOOLS:
                raise SessionConfigValidationError(
                    f"Too many tools: {len(val)} (max {self._MAX_TOOLS})"
                )
            import json
            total_chars = len(json.dumps(val, separators=(",", ":")))
            if total_chars > self._MAX_TOOL_SCHEMA_CHARS:
                raise SessionConfigValidationError(
                    f"Tool schemas too large: {total_chars} chars (max {self._MAX_TOOL_SCHEMA_CHARS})"
                )
            self.tools = val
        if "tool_choice" in data:
            self.tool_choice = data["tool_choice"]
        if "temperature" in data:
            val = data["temperature"]
            if not isinstance(val, (int, float)):
                raise SessionConfigValidationError("temperature must be a number")
            if val < 0.0 or val > 2.0:
                raise SessionConfigValidationError(f"temperature must be between 0.0 and 2.0, got {val}")
            self.temperature = float(val)
        if "max_response_output_tokens" in data:
            val = data["max_response_output_tokens"]
            if val != "inf" and not isinstance(val, int):
                raise SessionConfigValidationError(
                    "max_response_output_tokens must be an integer or 'inf'"
                )
            if isinstance(val, int) and val < 1:
                raise SessionConfigValidationError(
                    f"max_response_output_tokens must be positive, got {val}"
                )
            self.max_response_output_tokens = val


@dataclass
class ContentPart:
    type: str
    text: str | None = None
    audio: str | None = None
    transcript: str | None = None

    def to_dict(self) -> dict:
        result: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            result["text"] = self.text
        if self.audio is not None:
            result["audio"] = self.audio
        if self.transcript is not None:
            result["transcript"] = self.transcript
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ContentPart:
        return cls(
            type=data["type"],
            text=data.get("text"),
            audio=data.get("audio"),
            transcript=data.get("transcript"),
        )


_VALID_ITEM_TYPES = {"message", "function_call", "function_call_output"}
_VALID_ITEM_ROLES = {"user", "assistant", "system", None}
_MAX_ITEM_CONTENT_PARTS = 50
_MAX_ITEM_TEXT_CHARS = 100_000


class ConversationItemValidationError(ValueError):
    """Raised when a conversation item contains invalid data."""


@dataclass
class ConversationItem:
    id: str
    type: str = "message"
    role: str | None = None
    content: list[ContentPart] = field(default_factory=list)
    status: str = "completed"
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "object": "realtime.item",
            "status": self.status,
        }
        if self.role is not None:
            result["role"] = self.role
        if self.content:
            result["content"] = [c.to_dict() for c in self.content]
        if self.call_id is not None:
            result["call_id"] = self.call_id
        if self.name is not None:
            result["name"] = self.name
        if self.arguments is not None:
            result["arguments"] = self.arguments
        if self.output is not None:
            result["output"] = self.output
        return result

    def validate(self) -> None:
        """Validate item data from client input.

        Raises:
            ConversationItemValidationError: If item contains invalid data.
        """
        if self.type not in _VALID_ITEM_TYPES:
            raise ConversationItemValidationError(
                f"Invalid item type: {self.type!r}. Valid: {_VALID_ITEM_TYPES}"
            )
        if self.role not in _VALID_ITEM_ROLES:
            raise ConversationItemValidationError(
                f"Invalid item role: {self.role!r}. Valid: {_VALID_ITEM_ROLES}"
            )
        if self.type == "message" and self.role == "system":
            raise ConversationItemValidationError(
                "Cannot inject system messages via conversation.item.create"
            )
        if len(self.content) > _MAX_ITEM_CONTENT_PARTS:
            raise ConversationItemValidationError(
                f"Too many content parts: {len(self.content)} (max {_MAX_ITEM_CONTENT_PARTS})"
            )
        for part in self.content:
            if part.text and len(part.text) > _MAX_ITEM_TEXT_CHARS:
                raise ConversationItemValidationError(
                    f"Content text too long: {len(part.text)} chars (max {_MAX_ITEM_TEXT_CHARS})"
                )

    @classmethod
    def from_dict(cls, data: dict) -> ConversationItem:
        content_data = data.get("content", [])
        content = [ContentPart.from_dict(c) for c in content_data] if content_data else []
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "message"),
            role=data.get("role"),
            content=content,
            status=data.get("status", "completed"),
            call_id=data.get("call_id"),
            name=data.get("name"),
            arguments=data.get("arguments"),
            output=data.get("output"),
        )
