"""Tests for unified web search + fetch tool (DuckDuckGo).

All tests that hit the network are mocked for determinism.
The _html_to_text tests are pure logic — no mocking needed.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from tools.web_search import web_search, _html_to_text, register_web_search_handlers
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# _html_to_text — pure logic, no network
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fake DuckDuckGo results
# ---------------------------------------------------------------------------

_FAKE_TEXT_RESULTS = [
    {
        "title": "Cotação do Dólar Hoje",
        "body": "Veja a cotação do dólar americano em tempo real.",
        "href": "https://example.com/dolar",
    },
    {
        "title": "Câmbio USD/BRL",
        "body": "Acompanhe o câmbio do dólar.",
        "href": "https://example.com/cambio",
    },
]

_FAKE_NEWS_RESULTS = [
    {
        "title": "Brasil anuncia nova política econômica",
        "body": "O governo anunciou medidas para conter a inflação.",
        "url": "https://example.com/news1",
        "source": "Reuters",
    },
]


def _mock_ddgs():
    """Create a mock DDGS instance with deterministic results."""
    mock = MagicMock()
    mock.text.return_value = _FAKE_TEXT_RESULTS
    mock.news.return_value = _FAKE_NEWS_RESULTS
    return mock


# ---------------------------------------------------------------------------
# web_search — mocked DuckDuckGo
# ---------------------------------------------------------------------------


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_general_returns_results(self):
        with patch("tools.web_search._get_ddgs", return_value=_mock_ddgs()), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value="Page content here"):
            result = await web_search(query="cotação dólar hoje", type="general")
            assert result["query"] == "cotação dólar hoje"
            assert len(result["results"]) == 2
            assert "title" in result["results"][0]
            assert "snippet" in result["results"][0]

    @pytest.mark.asyncio
    async def test_general_fetches_content_for_first_result(self):
        with patch("tools.web_search._get_ddgs", return_value=_mock_ddgs()), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value="Fetched page text"):
            result = await web_search(query="python programming")
            first = result["results"][0]
            assert "content" in first
            assert first["content"] == "Fetched page text"

    @pytest.mark.asyncio
    async def test_default_type_is_general(self):
        mock = _mock_ddgs()
        with patch("tools.web_search._get_ddgs", return_value=mock), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value=""):
            result = await web_search(query="python programming")
            assert "results" in result
            mock.text.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_query(self):
        result = await web_search(query="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_max_results(self):
        mock = _mock_ddgs()
        with patch("tools.web_search._get_ddgs", return_value=mock), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value=""):
            result = await web_search(query="python", max_results=2)
            assert len(result["results"]) <= 2
            mock.text.assert_called_once_with("python", max_results=2, region="br-pt")

    @pytest.mark.asyncio
    async def test_news_returns_results(self):
        with patch("tools.web_search._get_ddgs", return_value=_mock_ddgs()):
            result = await web_search(query="brasil", type="news")
            assert result["query"] == "brasil"
            assert len(result["results"]) > 0
            assert "title" in result["results"][0]
            assert "source" in result["results"][0]

    @pytest.mark.asyncio
    async def test_news_no_content_fetch(self):
        """News results should not fetch page content (speed optimization)."""
        with patch("tools.web_search._get_ddgs", return_value=_mock_ddgs()):
            result = await web_search(query="brasil economia", type="news")
            for r in result["results"]:
                assert "content" not in r

    @pytest.mark.asyncio
    async def test_news_empty_query(self):
        result = await web_search(query="", type="news")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ddgs_exception_returns_error(self):
        mock = _mock_ddgs()
        mock.text.side_effect = Exception("Rate limit")
        with patch("tools.web_search._get_ddgs", return_value=mock):
            result = await web_search(query="test")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_no_results_returns_message(self):
        mock = _mock_ddgs()
        mock.text.return_value = []
        with patch("tools.web_search._get_ddgs", return_value=mock):
            result = await web_search(query="xyznonexistent123")
            assert result["results"] == []
            assert "message" in result

    @pytest.mark.asyncio
    async def test_second_result_has_no_content(self):
        """Only first result gets page fetch."""
        with patch("tools.web_search._get_ddgs", return_value=_mock_ddgs()), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value="Content"):
            result = await web_search(query="test")
            assert "content" not in result["results"][1]

    @pytest.mark.asyncio
    async def test_title_truncated_to_80_chars(self):
        mock = _mock_ddgs()
        mock.text.return_value = [{"title": "x" * 200, "body": "b", "href": "http://x.com"}]
        with patch("tools.web_search._get_ddgs", return_value=mock), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value=""):
            result = await web_search(query="test")
            assert len(result["results"][0]["title"]) == 80

    @pytest.mark.asyncio
    async def test_snippet_truncated_to_150_chars(self):
        mock = _mock_ddgs()
        mock.text.return_value = [{"title": "t", "body": "b" * 300, "href": "http://x.com"}]
        with patch("tools.web_search._get_ddgs", return_value=mock), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value=""):
            result = await web_search(query="test")
            assert len(result["results"][0]["snippet"]) == 150


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

        with patch("tools.web_search._get_ddgs", return_value=_mock_ddgs()), \
             patch("tools.web_search._fetch_page", new_callable=AsyncMock, return_value=""):
            result_json = await registry.execute(
                "web_search", '{"query": "capital do brasil"}'
            )
            result = json.loads(result_json)
            assert "results" in result
            assert len(result["results"]) > 0
