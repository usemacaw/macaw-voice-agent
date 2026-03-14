"""
OpenAI GPT LLM provider for OpenVoiceAPI.

Stateless: receives full messages each call.

Config:
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-xxx
    LLM_MODEL=gpt-4o
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncGenerator

from config import LLM_CONFIG
from providers.llm import LLMProvider, LLMStreamEvent, register_llm_provider

logger = logging.getLogger("open-voice-api.llm.openai")


class OpenAILLM(LLMProvider):
    """Stateless OpenAI GPT provider with async streaming."""

    provider_name = "openai"

    def __init__(self):
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY env var is required")

        self._model = LLM_CONFIG["model"]
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=LLM_CONFIG["timeout"],
        )
        logger.info(f"OpenAI LLM initialized: model={self._model}")

    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        # Prepend system message if provided
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        t0 = time.perf_counter()
        first_token = False

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if not first_token:
                    first_token = True
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM first token in {ttft_ms:.0f}ms")
                yield delta.content

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
        """Stream with full tool call support via OpenAI delta stream."""
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        t0 = time.perf_counter()
        first_token = False
        active_tool_calls: dict[int, dict] = {}

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if not first_token:
                first_token = True
                ttft_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"LLM first event in {ttft_ms:.0f}ms")

            if delta.content:
                yield LLMStreamEvent(type="text_delta", text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_tool_calls:
                        tc_id = tc.id or ""
                        tc_name = tc.function.name if tc.function and tc.function.name else ""
                        active_tool_calls[idx] = {"id": tc_id, "name": tc_name}
                        yield LLMStreamEvent(
                            type="tool_call_start",
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                        )
                    if tc.function and tc.function.arguments:
                        info = active_tool_calls[idx]
                        yield LLMStreamEvent(
                            type="tool_call_delta",
                            tool_call_id=info["id"],
                            tool_name=info["name"],
                            tool_arguments_delta=tc.function.arguments,
                        )

        # Emit end events for all tool calls
        for info in active_tool_calls.values():
            yield LLMStreamEvent(
                type="tool_call_end",
                tool_call_id=info["id"],
                tool_name=info["name"],
            )

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM stream (with tools) complete: {total_ms:.0f}ms")

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        await self._client.close()


register_llm_provider("openai", OpenAILLM)
