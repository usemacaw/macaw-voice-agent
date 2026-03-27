"""Qwen3-ASR prompt builder."""

from __future__ import annotations


class QwenPromptBuilder:
    """Builds chat-template prompts for Qwen3-ASR."""

    def __init__(self, processor, language_name: str) -> None:
        self._processor = processor
        self._language_name = language_name

    def build(self, prefix: str = "") -> str:
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self._processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        if self._language_name:
            prompt = prompt + "language " + self._language_name + "<asr_text>"
        if prefix:
            prompt = prompt + prefix
        return prompt
