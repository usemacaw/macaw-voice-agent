"""
LLM Provider base — interface e factory para o LLM gRPC server.

Providers server-side implementam a inferencia real (vLLM, Anthropic, etc).
O gRPC server delega para o provider ativo.
"""

import importlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Type

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm-server.provider")


@dataclass
class LLMStreamEvent:
    """Event from LLM stream — text delta or tool call parts."""
    type: str  # "text_delta", "tool_call_start", "tool_call_delta", "tool_call_end"
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_arguments_delta: str = ""


class LLMProvider(ABC):
    """Interface base para provedores de LLM (server-side)."""

    provider_name: str = "base"

    async def connect(self) -> None:
        logger.info(f"{self.provider_name} connected")

    async def disconnect(self) -> None:
        logger.info(f"{self.provider_name} disconnected")

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream text tokens."""
        yield ""  # pragma: no cover

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream with tool call events. Default wraps generate_stream()."""
        async for chunk in self.generate_stream(
            messages, system=system, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        ):
            yield LLMStreamEvent(type="text_delta", text=chunk)


# ==================== Mock LLM Provider ====================

class MockLLM(LLMProvider):
    """LLM mock para testes."""

    provider_name = "mock"

    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        yield "Resposta mock do LLM."


# ==================== Factory ====================

_LLM_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "mock": MockLLM,
}

_KNOWN_LLM_MODULES = {
    "vllm": "llm.providers.vllm_provider",
}


def register_llm_provider(name: str, cls: Type[LLMProvider]) -> None:
    _LLM_PROVIDERS[name] = cls
    logger.info(f"LLM provider registrado: {name}")


async def create_llm_provider(provider_name: str | None = None) -> LLMProvider:
    name = provider_name or os.getenv("LLM_BACKEND_PROVIDER", "vllm")

    if name not in _LLM_PROVIDERS and name in _KNOWN_LLM_MODULES:
        try:
            importlib.import_module(_KNOWN_LLM_MODULES[name])
        except ImportError as e:
            raise ValueError(
                f"LLM provider '{name}' requer dependencias adicionais: {e}"
            ) from e

    if name not in _LLM_PROVIDERS:
        available = ", ".join(_LLM_PROVIDERS.keys())
        raise ValueError(
            f"LLM provider '{name}' nao registrado. Disponiveis: {available}"
        )

    provider = _LLM_PROVIDERS[name]()
    await provider.connect()
    return provider
