"""Tests for tool-use scenario."""

from __future__ import annotations

from promptgauntlet.adapters.mock import MockModelClient
from promptgauntlet.engine.prompter import ScriptedPrompter
from promptgauntlet.engine.runner import run_single_scenario
from promptgauntlet.scenarios.tool_use.scenario import ToolUseScenario
from promptgauntlet.types import TaskFamily


class TestToolUseScenario:
    """Tests for the tool-use scenario."""

    def test_scenario_config(self) -> None:
        s = ToolUseScenario()
        assert s.config.id == "tool_use/research_calculate"
        assert s.config.family == TaskFamily.TOOL_USE

    def test_tools_available(self) -> None:
        """Scenario provides tool schemas."""
        s = ToolUseScenario()
        s.setup(seed=0)
        tools = s.get_tools()
        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert "search" in tool_names
        assert "calculator" in tool_names
        assert "file_store" in tool_names

    def test_mock_run_uses_tools(self) -> None:
        """Mock model calls tools during the scenario."""
        s = ToolUseScenario()
        client = MockModelClient()
        prompter = ScriptedPrompter()

        result = run_single_scenario(
            scenario=s,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=8000,
            budget_turns=15,
        )
        assert result.metrics["tool_call_correctness"] > 0.0
        assert result.metrics["tools_used"] >= 2.0
