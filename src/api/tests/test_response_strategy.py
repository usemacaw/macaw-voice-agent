"""Tests for intelligence/response_strategy.py — select_strategy() and ResponsePlan."""

from __future__ import annotations

import pytest

from config import SLO
from intelligence.response_strategy import ResponseMode, ResponsePlan, select_strategy
from protocol.models import SessionConfig
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


def _config(modalities: list[str] | None = None, tools: list[dict] | None = None) -> SessionConfig:
    """Build a SessionConfig with controlled modalities and tools."""
    cfg = SessionConfig()
    cfg.modalities = modalities if modalities is not None else ["text", "audio"]
    cfg.tools = tools if tools is not None else []
    return cfg


def _registry_with_tools(n: int = 1, max_rounds: int = 5) -> ToolRegistry:
    """Build a ToolRegistry pre-populated with n dummy tools."""
    registry = ToolRegistry(timeout=5.0, max_rounds=max_rounds)
    for i in range(n):
        name = f"tool_{i}"
        registry.register(
            name,
            handler=_dummy_handler,
            schema={"type": "function", "function": {"name": name, "description": f"Tool {i}"}},
        )
    return registry


async def _dummy_handler(**kwargs):
    return {"ok": True}


def _empty_registry() -> ToolRegistry:
    return ToolRegistry(timeout=5.0, max_rounds=5)


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


class TestModeSelection:
    def test_text_only_mode_when_no_audio_and_no_tools(self):
        """TEXT_ONLY is selected when modalities has no 'audio' and no tools exist."""
        # Arrange
        config = _config(modalities=["text"], tools=[])

        # Act
        plan = select_strategy(config, tool_registry=None)

        # Assert
        assert plan.mode is ResponseMode.TEXT_ONLY

    def test_audio_streaming_mode_when_audio_and_no_tools(self):
        """AUDIO_STREAMING is selected when modalities has 'audio' but no tools exist."""
        # Arrange
        config = _config(modalities=["text", "audio"], tools=[])

        # Act
        plan = select_strategy(config, tool_registry=None)

        # Assert
        assert plan.mode is ResponseMode.AUDIO_STREAMING

    def test_tool_calling_mode_when_tools_and_no_audio(self):
        """TOOL_CALLING is selected when tools exist but modalities has no 'audio'."""
        # Arrange
        config = _config(modalities=["text"], tools=[])
        registry = _registry_with_tools()

        # Act
        plan = select_strategy(config, tool_registry=registry)

        # Assert
        assert plan.mode is ResponseMode.TOOL_CALLING

    def test_tool_calling_audio_mode_when_tools_and_audio(self):
        """TOOL_CALLING_AUDIO is selected when tools exist AND modalities has 'audio'."""
        # Arrange
        config = _config(modalities=["text", "audio"], tools=[])
        registry = _registry_with_tools()

        # Act
        plan = select_strategy(config, tool_registry=registry)

        # Assert
        assert plan.mode is ResponseMode.TOOL_CALLING_AUDIO

    def test_tool_calling_audio_mode_when_config_tools_and_audio(self):
        """TOOL_CALLING_AUDIO is also triggered by tools in config (no registry)."""
        # Arrange
        client_tool = {"type": "function", "function": {"name": "client_fn", "description": "x"}}
        config = _config(modalities=["text", "audio"], tools=[client_tool])

        # Act
        plan = select_strategy(config, tool_registry=None)

        # Assert
        assert plan.mode is ResponseMode.TOOL_CALLING_AUDIO

    def test_tool_calling_mode_when_config_tools_and_no_audio(self):
        """TOOL_CALLING is selected when config.tools is non-empty and no audio modality."""
        # Arrange
        client_tool = {"type": "function", "function": {"name": "client_fn", "description": "x"}}
        config = _config(modalities=["text"], tools=[client_tool])

        # Act
        plan = select_strategy(config, tool_registry=None)

        # Assert
        assert plan.mode is ResponseMode.TOOL_CALLING


# ---------------------------------------------------------------------------
# ResponsePlan field: has_audio / has_tools
# ---------------------------------------------------------------------------


class TestPlanFlags:
    def test_has_audio_is_true_when_audio_in_modalities(self):
        config = _config(modalities=["text", "audio"])
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_audio is True

    def test_has_audio_is_false_when_audio_not_in_modalities(self):
        config = _config(modalities=["text"])
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_audio is False

    def test_has_tools_is_true_when_registry_has_server_tools(self):
        config = _config(modalities=["text"])
        registry = _registry_with_tools()
        plan = select_strategy(config, tool_registry=registry)
        assert plan.has_tools is True

    def test_has_tools_is_false_when_registry_is_empty(self):
        config = _config(modalities=["text", "audio"])
        registry = _empty_registry()
        plan = select_strategy(config, tool_registry=registry)
        assert plan.has_tools is False

    def test_has_tools_is_true_when_config_tools_non_empty(self):
        client_tool = {"type": "function", "function": {"name": "fn", "description": "x"}}
        config = _config(tools=[client_tool])
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_tools is True

    def test_has_tools_is_false_when_config_tools_empty_and_no_registry(self):
        config = _config(tools=[])
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_tools is False


# ---------------------------------------------------------------------------
# Server-side tools detection
# ---------------------------------------------------------------------------


class TestServerSideToolsDetection:
    def test_server_side_tools_true_when_registry_has_tools(self):
        """server_side_tools is True when tool_registry.has_server_tools is True."""
        config = _config(modalities=["text", "audio"])
        registry = _registry_with_tools()
        plan = select_strategy(config, tool_registry=registry)
        assert plan.server_side_tools is True

    def test_server_side_tools_false_when_registry_is_none(self):
        """server_side_tools is False when tool_registry is None (client-side tools only)."""
        client_tool = {"type": "function", "function": {"name": "fn", "description": "x"}}
        config = _config(tools=[client_tool])
        plan = select_strategy(config, tool_registry=None)
        assert plan.server_side_tools is False

    def test_server_side_tools_false_when_registry_is_empty(self):
        """server_side_tools is False when registry exists but no tools are registered."""
        config = _config(modalities=["text", "audio"])
        registry = _empty_registry()
        plan = select_strategy(config, tool_registry=registry)
        assert plan.server_side_tools is False

    def test_server_side_tools_false_when_only_config_tools_exist(self):
        """Tools from config.tools alone do not count as server-side tools."""
        client_tool = {"type": "function", "function": {"name": "client_fn", "description": "x"}}
        config = _config(tools=[client_tool])
        plan = select_strategy(config, tool_registry=_empty_registry())
        assert plan.server_side_tools is False


# ---------------------------------------------------------------------------
# Tool schema merging
# ---------------------------------------------------------------------------


class TestToolSchemaMerging:
    def _make_tool_schema(self, name: str) -> dict:
        return {"type": "function", "function": {"name": name, "description": f"desc of {name}"}}

    def test_registry_schemas_are_included_in_plan_tools(self):
        """Schemas from tool_registry appear in plan.tools."""
        config = _config(modalities=["text"], tools=[])
        registry = _registry_with_tools(n=2)
        plan = select_strategy(config, tool_registry=registry)
        names = {t.get("function", {}).get("name") for t in plan.tools}
        assert "tool_0" in names
        assert "tool_1" in names

    def test_config_tools_are_included_in_plan_tools(self):
        """Tools from config.tools appear in plan.tools."""
        client_tool = self._make_tool_schema("client_fn")
        config = _config(modalities=["text"], tools=[client_tool])
        plan = select_strategy(config, tool_registry=_empty_registry())
        names = {t.get("function", {}).get("name") for t in plan.tools}
        assert "client_fn" in names

    def test_no_duplicates_when_same_tool_in_config_and_registry(self):
        """A tool present in both config.tools and registry appears only once."""
        shared_tool = self._make_tool_schema("shared_fn")
        config = _config(modalities=["text"], tools=[shared_tool])
        registry = ToolRegistry(timeout=5.0, max_rounds=5)
        registry.register("shared_fn", handler=_dummy_handler, schema=shared_tool)

        plan = select_strategy(config, tool_registry=registry)

        names = [t.get("function", {}).get("name") for t in plan.tools]
        assert names.count("shared_fn") == 1

    def test_config_and_registry_tools_merged_without_duplicates(self):
        """When config has tool_a and registry has tool_b, plan has both exactly once."""
        tool_a = self._make_tool_schema("tool_a")
        tool_b = self._make_tool_schema("tool_b")
        config = _config(modalities=["text"], tools=[tool_a])
        registry = ToolRegistry(timeout=5.0, max_rounds=5)
        registry.register("tool_b", handler=_dummy_handler, schema=tool_b)

        plan = select_strategy(config, tool_registry=registry)

        names = [t.get("function", {}).get("name") for t in plan.tools]
        assert sorted(names) == ["tool_a", "tool_b"]

    def test_plan_tools_empty_when_no_tools_anywhere(self):
        """plan.tools is an empty list when neither config nor registry have tools."""
        config = _config(modalities=["text", "audio"], tools=[])
        plan = select_strategy(config, tool_registry=None)
        assert plan.tools == []

    def test_plan_tools_empty_when_registry_empty_and_config_tools_empty(self):
        """Empty registry + empty config.tools yields empty plan.tools."""
        config = _config(tools=[])
        plan = select_strategy(config, tool_registry=_empty_registry())
        assert plan.tools == []


# ---------------------------------------------------------------------------
# SLO target assignment
# ---------------------------------------------------------------------------


class TestSLOTargetAssignment:
    def test_non_tool_response_uses_standard_slo(self):
        """Responses without tools use SLO.first_audio_ms as the SLO target."""
        config = _config(modalities=["text", "audio"], tools=[])
        plan = select_strategy(config, tool_registry=None)
        assert plan.max_first_audio_ms == SLO.first_audio_ms

    def test_tool_response_uses_extended_slo(self):
        """Responses with tools use SLO.first_audio_tool_ms (larger budget)."""
        config = _config(modalities=["text", "audio"])
        registry = _registry_with_tools()
        plan = select_strategy(config, tool_registry=registry)
        assert plan.max_first_audio_ms == SLO.first_audio_tool_ms

    def test_tool_slo_is_larger_than_standard_slo(self):
        """The tool SLO target is always larger than the non-tool target."""
        assert SLO.first_audio_tool_ms > SLO.first_audio_ms

    def test_text_only_mode_uses_standard_slo(self):
        """TEXT_ONLY responses also use the standard SLO target."""
        config = _config(modalities=["text"], tools=[])
        plan = select_strategy(config, tool_registry=None)
        assert plan.max_first_audio_ms == SLO.first_audio_ms

    def test_config_tools_trigger_extended_slo(self):
        """Config-level tools (no registry) also trigger the extended SLO."""
        client_tool = {"type": "function", "function": {"name": "fn", "description": "x"}}
        config = _config(tools=[client_tool])
        plan = select_strategy(config, tool_registry=None)
        assert plan.max_first_audio_ms == SLO.first_audio_tool_ms


# ---------------------------------------------------------------------------
# max_rounds
# ---------------------------------------------------------------------------


class TestMaxRounds:
    def test_max_rounds_comes_from_registry(self):
        """plan.max_rounds reflects the registry's max_rounds setting."""
        config = _config(modalities=["text", "audio"])
        registry = _registry_with_tools(max_rounds=7)
        plan = select_strategy(config, tool_registry=registry)
        assert plan.max_rounds == 7

    def test_max_rounds_defaults_to_5_when_no_registry(self):
        """plan.max_rounds defaults to 5 when tool_registry is None."""
        config = _config(modalities=["text", "audio"])
        plan = select_strategy(config, tool_registry=None)
        assert plan.max_rounds == 5

    def test_max_rounds_from_empty_registry(self):
        """An empty registry still provides its configured max_rounds."""
        config = _config(modalities=["text", "audio"])
        registry = ToolRegistry(timeout=5.0, max_rounds=3)
        plan = select_strategy(config, tool_registry=registry)
        assert plan.max_rounds == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_tool_registry_does_not_raise(self):
        """select_strategy does not raise when tool_registry is None."""
        config = _config(modalities=["text", "audio"], tools=[])
        plan = select_strategy(config, tool_registry=None)
        assert plan is not None

    def test_empty_tool_registry_does_not_activate_tool_mode(self):
        """An empty ToolRegistry (no registered tools) behaves like no registry."""
        config = _config(modalities=["text", "audio"], tools=[])
        registry = _empty_registry()
        plan = select_strategy(config, tool_registry=registry)
        assert plan.mode is ResponseMode.AUDIO_STREAMING
        assert plan.has_tools is False
        assert plan.server_side_tools is False

    def test_config_tools_empty_list_means_no_client_tools(self):
        """config.tools = [] is treated as no client-side tools."""
        config = _config(tools=[])
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_tools is False
        assert plan.tools == []

    def test_config_tools_none_treated_same_as_empty_list(self):
        """config.tools = None edge: SessionConfig defaults to [] so no tools."""
        config = SessionConfig()
        config.modalities = ["text", "audio"]
        config.tools = []  # default is []
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_tools is False

    def test_response_plan_is_immutable(self):
        """ResponsePlan is a frozen dataclass — mutations must raise."""
        config = _config(modalities=["text", "audio"])
        plan = select_strategy(config, tool_registry=None)
        with pytest.raises((AttributeError, TypeError)):
            plan.mode = ResponseMode.TEXT_ONLY  # type: ignore[misc]

    def test_only_audio_modality_still_detected(self):
        """Modalities=['audio'] (no 'text') still sets has_audio=True."""
        config = _config(modalities=["audio"])
        plan = select_strategy(config, tool_registry=None)
        assert plan.has_audio is True
        assert plan.mode is ResponseMode.AUDIO_STREAMING

    def test_registry_with_multiple_tools_all_schemas_included(self):
        """All schemas from a multi-tool registry appear in plan.tools."""
        config = _config(modalities=["text"])
        registry = _registry_with_tools(n=5)
        plan = select_strategy(config, tool_registry=registry)
        assert len(plan.tools) == 5

    def test_select_strategy_returns_response_plan_instance(self):
        """select_strategy always returns a ResponsePlan."""
        config = _config()
        plan = select_strategy(config, tool_registry=None)
        assert isinstance(plan, ResponsePlan)
