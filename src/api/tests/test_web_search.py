"""Tests for unified web search + fetch tool (DuckDuckGo)."""

from __future__ import annotations

import json
import pytest

from tools.web_search import web_search, _html_to_text, register_web_search_handlers
from tools.registry import ToolRegistry


class TestHtmlToText:
    def test_strips_tags(self):
        assert _html_to_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_strips_script_and_style(self):
        html = "<script>var x=1;</script><style>.a{}</style><p>Content</p>"
        assert "Content" in _html_to_text(html)
        assert "var x" not in _html_to_text(html)

    def test_unescapes_entities(self):
        assert "R$5,20" in _html_to_text("<p>R$5,20</p>")
        assert "&amp;" not in _html_to_text("<p>A &amp; B</p>")

    def test_truncates_to_max_chars(self):
        result = _html_to_text("<p>" + "x" * 2000 + "</p>", max_chars=100)
        assert len(result) == 100

    def test_collapses_whitespace(self):
        result = _html_to_text("<p>Hello    \n\n   world</p>")
        assert result == "Hello world"


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_general_returns_results(self):
        result = await web_search(query="cotação dólar hoje", type="general")
        assert result["query"] == "cotação dólar hoje"
        assert len(result["results"]) > 0
        assert "title" in result["results"][0]
        assert "snippet" in result["results"][0]

    @pytest.mark.asyncio
    async def test_general_fetches_content(self):
        result = await web_search(query="python programming language")
        first = result["results"][0]
        # First result should have fetched content (if page is accessible)
        if "content" in first:
            assert len(first["content"]) > 50

    @pytest.mark.asyncio
    async def test_default_type_is_general(self):
        result = await web_search(query="python programming")
        assert "results" in result

    @pytest.mark.asyncio
    async def test_empty_query(self):
        result = await web_search(query="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_max_results(self):
        result = await web_search(query="python", max_results=2)
        assert len(result["results"]) <= 2

    @pytest.mark.asyncio
    async def test_news_returns_results(self):
        result = await web_search(query="brasil", type="news")
        assert result["query"] == "brasil"
        assert len(result["results"]) > 0
        assert "title" in result["results"][0]
        assert "source" in result["results"][0]

    @pytest.mark.asyncio
    async def test_news_no_content_fetch(self):
        """News results should not fetch page content (speed optimization)."""
        result = await web_search(query="brasil economia", type="news")
        for r in result["results"]:
            assert "content" not in r

    @pytest.mark.asyncio
    async def test_news_empty_query(self):
        result = await web_search(query="", type="news")
        assert "error" in result


class TestWebSearchRegistration:
    def test_register_handlers(self):
        registry = ToolRegistry()
        register_web_search_handlers(registry)

        assert registry.has_tool("web_search")
        assert len(registry.get_schemas()) == 1

    @pytest.mark.asyncio
    async def test_registered_executes(self):
        registry = ToolRegistry()
        register_web_search_handlers(registry)

        result_json = await registry.execute(
            "web_search", '{"query": "capital do brasil"}'
        )
        result = json.loads(result_json)
        assert "results" in result
        assert len(result["results"]) > 0
