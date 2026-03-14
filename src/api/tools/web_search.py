"""
Web search + fetch tool via DuckDuckGo — zero API keys.

Provides a single `web_search` tool that searches the web and automatically
fetches the content of the top result, giving the LLM real information
instead of just short snippets.
"""

from __future__ import annotations

import html
import logging
import re
from typing import TYPE_CHECKING

import httpx
from duckduckgo_search import DDGS

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.tools.web")

_ddgs: DDGS | None = None
_http_client: httpx.AsyncClient | None = None


def _get_ddgs() -> DDGS:
    global _ddgs
    if _ddgs is None:
        _ddgs = DDGS()
    return _ddgs


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=5.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OpenVoiceAPI/1.0)"},
        )
    return _http_client


async def cleanup_web_search() -> None:
    """Close module-level singletons. Call on server shutdown."""
    global _ddgs, _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    _ddgs = None

# Tags whose content should be completely removed (not just the tags)
_STRIP_TAGS_RE = re.compile(
    r"<(script|style|nav|header|footer|aside|noscript|iframe)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# All remaining HTML tags
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Collapse whitespace
_WHITESPACE_RE = re.compile(r"\s+")


def _html_to_text(raw_html: str, max_chars: int = 800) -> str:
    """Extract readable text from HTML, stripping tags and boilerplate."""
    text = _STRIP_TAGS_RE.sub(" ", raw_html)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


async def _fetch_page(url: str) -> str:
    """Fetch a URL and extract text content. Returns empty string on failure."""
    try:
        resp = await _get_http_client().get(url)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        return _html_to_text(resp.text)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return ""


async def web_search(query: str, type: str = "general", max_results: int = 2) -> dict:
    """Busca informacoes na web via DuckDuckGo e extrai conteudo das paginas.

    Args:
        query: Search query.
        type: "general" for web search, "news" for recent news.
        max_results: Max results to return.
    """
    if not query:
        return {"error": "Informe o que deseja pesquisar."}

    search_type = type.lower().strip()
    is_news = search_type == "news"

    try:
        if is_news:
            results = _get_ddgs().news(query, max_results=max_results, region="br-pt")
        else:
            results = _get_ddgs().text(query, max_results=max_results, region="br-pt")
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return {"error": "Nao consegui realizar a busca no momento."}

    if not results:
        return {"query": query, "results": [], "message": "Nenhum resultado encontrado."}

    # Build simplified results with fetched content
    simplified = []
    for r in results:
        url = r.get("href") or r.get("url", "")
        entry: dict = {
            "title": r.get("title", "")[:80],
            "snippet": r.get("body", "")[:150],
        }
        if is_news:
            entry["source"] = r.get("source", "")

        # Fetch actual page content for the first result only (speed)
        if url and len(simplified) == 0 and not is_news:
            content = await _fetch_page(url)
            if content:
                entry["content"] = content[:400]

        simplified.append(entry)

    return {"query": query, "results": simplified}


_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Pesquisa informacoes na internet e extrai conteudo das paginas. "
            "Use para cotacoes, precos, clima, informacoes factuais, "
            "noticias e eventos recentes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "O que pesquisar (ex: 'cotacao dolar hoje', 'noticias guerra')",
                },
                "type": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": "Tipo: 'general' para informacoes, 'news' para noticias recentes",
                },
            },
            "required": ["query"],
        },
    },
}


def register_web_search_handlers(registry: ToolRegistry) -> None:
    """Register the unified web_search tool."""
    registry.register(
        "web_search",
        handler=web_search,
        schema=_TOOL_SCHEMA,
        filler_phrase="Um momento.",
    )
    logger.info("Registered web search tool (unified web_search with fetch)")
