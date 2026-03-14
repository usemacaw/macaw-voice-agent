"""
Anthropic Claude LLM provider for OpenVoiceAPI.

Stateless: receives full messages each call. No conversation history management.

Config:
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-xxx
    LLM_MODEL=claude-sonnet-4-20250514
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncGenerator

from config import LLM_CONFIG
from providers.llm import LLMProvider, LLMStreamEvent, register_llm_provider

logger = logging.getLogger("open-voice-api.llm.anthropic")


class AnthropicLLM(LLMProvider):
    """Stateless Anthropic Claude provider with async streaming."""

    provider_name = "anthropic"

    def __init__(self):
        from anthropic import AsyncAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY env var is required")

        self._model = LLM_CONFIG["model"]
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=LLM_CONFIG["timeout"],
        )
        logger.info(f"Anthropic LLM initialized: model={self._model}")

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages to Anthropic format.

        Handles tool_calls → tool_use and tool responses → tool_result.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue  # system prompt handled separately

            if role == "assistant" and "tool_calls" in msg:
                # Convert OpenAI tool_calls to Anthropic content blocks
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                    })
                converted.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                # Convert OpenAI tool response to Anthropic tool_result
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })

            else:
                converted.append({"role": role, "content": msg.get("content", "")})

        return converted

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI-format tools to Anthropic format."""
        if not tools:
            return None
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        return anthropic_tools or None

    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        converted_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": converted_messages,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        t0 = time.perf_counter()
        first_token = False

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if not first_token:
                    first_token = True
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM first token in {ttft_ms:.0f}ms")
                yield text

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM stream complete: {total_ms:.0f}ms")

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream with full tool call support via Anthropic event stream."""
        converted_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": converted_messages,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        t0 = time.perf_counter()
        first_token = False
        current_tool_id = ""
        current_tool_name = ""
        in_tool_use = False

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if not first_token:
                    first_token = True
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM first event in {ttft_ms:.0f}ms")

                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        in_tool_use = True
                        current_tool_id = block.id
                        current_tool_name = block.name
                        yield LLMStreamEvent(
                            type="tool_call_start",
                            tool_call_id=block.id,
                            tool_name=block.name,
                        )
                    else:
                        in_tool_use = False

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield LLMStreamEvent(type="text_delta", text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        yield LLMStreamEvent(
                            type="tool_call_delta",
                            tool_call_id=current_tool_id,
                            tool_name=current_tool_name,
                            tool_arguments_delta=event.delta.partial_json,
                        )

                elif event.type == "content_block_stop":
                    if in_tool_use:
                        yield LLMStreamEvent(
                            type="tool_call_end",
                            tool_call_id=current_tool_id,
                            tool_name=current_tool_name,
                        )
                        in_tool_use = False

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM stream (with tools) complete: {total_ms:.0f}ms")

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        await self._client.close()


# DESIGN NOTE — Auto-discovery:
# This register call executes when the module is imported, but the module is only
# imported lazily by ProviderRegistry.create() when LLM_PROVIDER=anthropic.
# AnthropicLLM.__init__ validates ANTHROPIC_API_KEY at instantiation time, not
# at registration time — so importing this module is safe even without the key.
register_llm_provider("anthropic", AnthropicLLM)
