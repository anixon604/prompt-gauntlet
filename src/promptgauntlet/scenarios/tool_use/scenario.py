"""Tool-using multi-turn scenario: search + calculate + store."""

from __future__ import annotations

import re
from typing import Any

from promptgauntlet.scenarios.base import Scenario, ScenarioResult, ScriptedPolicy
from promptgauntlet.scenarios.registry import register_scenario
from promptgauntlet.tools.base import ToolRegistry
from promptgauntlet.tools.calculator import CalculatorTool
from promptgauntlet.tools.filestore import FileStoreTool
from promptgauntlet.tools.search import SearchTool
from promptgauntlet.types import (
    Message,
    Role,
    ScenarioConfig,
    TaskFamily,
    ToolCallRequest,
    ToolCallResult,
    ToolSchema,
)


class ToolUsePolicy(ScriptedPolicy):
    """Scripted baseline: step-by-step instruction strategy."""

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str:
        if turn == 0:
            return (
                "Find the population of Springfield, IL using the search tool. "
                "Then calculate the GDP per capita given that the GDP is 7.6 billion dollars. "
                "Finally, store the result using the file_store tool with key 'gdp_per_capita'. "
                "Show your work step by step."
            )

        # Check if we got a final answer
        for msg in reversed(messages):
            if (
                msg.role == Role.ASSISTANT
                and not msg.tool_calls
                and any(
                    kw in msg.content.lower()
                    for kw in ["per capita", "result", "approximately", "answer"]
                )
            ):
                return ""  # Done
        return ""  # Let the model continue if it has tool calls pending


@register_scenario
class ToolUseScenario(Scenario):
    """Multi-turn tool-using scenario.

    Requires: search corpus, calculator, file store.
    Goal: find information, compute derived values, store results.
    """

    def __init__(self) -> None:
        self._tool_registry = ToolRegistry()
        self._search = SearchTool()
        self._calculator = CalculatorTool()
        self._filestore = FileStoreTool()
        self._tool_registry.register(self._search)
        self._tool_registry.register(self._calculator)
        self._tool_registry.register(self._filestore)
        self._tool_calls_made: list[dict[str, Any]] = []
        self._seed = 0

        # Ground truth
        self._expected_population = 116250
        self._expected_gdp = 7.6e9
        self._expected_per_capita = self._expected_gdp / self._expected_population

    @property
    def config(self) -> ScenarioConfig:
        return ScenarioConfig(
            id="tool_use/research_calculate",
            family=TaskFamily.TOOL_USE,
            name="Research and Calculate",
            description=(
                "Use search, calculator, and file store tools to research a topic, "
                "compute derived values, and store results. Validates final answer "
                "against ground truth."
            ),
            budget_tokens=8000,
            budget_turns=15,
        )

    def setup(self, seed: int) -> list[Message]:
        self._seed = seed
        self._tool_calls_made = []
        self._filestore.reset()

        # Pre-load search corpus
        self._search.load_corpus()

        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a research assistant with access to tools. "
                    "Use the search tool to find information, the calculator "
                    "for computations, and file_store to save results. "
                    "Always show your reasoning."
                ),
            )
        ]

    def get_tools(self) -> list[ToolSchema]:
        return self._tool_registry.get_schemas()

    def handle_tool_call(self, call: ToolCallRequest) -> ToolCallResult:
        self._tool_calls_made.append({
            "name": call.name,
            "arguments": call.arguments,
        })
        return self._tool_registry.handle_call(call)

    def check_termination(
        self, messages: list[Message], turn: int, tokens: int
    ) -> bool:
        # Terminate once we have a final text response after tool calls
        if len(self._tool_calls_made) >= 2:
            for msg in reversed(messages):
                if msg.role == Role.ASSISTANT and not msg.tool_calls:
                    return True
        return False

    def grade(self, result: ScenarioResult) -> dict[str, float]:
        """Grade tool use scenario."""
        # Check which tools were called
        tools_used = set(tc["name"] for tc in self._tool_calls_made)
        used_search = "search" in tools_used
        used_calc = "calculator" in tools_used
        used_store = "file_store" in tools_used

        # Tool call correctness
        tool_correct = sum([used_search, used_calc, used_store]) / 3.0

        # Check final answer accuracy
        final_answer = ""
        for msg in reversed(result.messages):
            if msg.role == Role.ASSISTANT and not msg.tool_calls:
                final_answer = msg.content
                break

        # Extract numerical answer
        answer_correct = 0.0
        numbers = re.findall(r"[\d,]+\.?\d*", final_answer.replace(",", ""))
        for num_str in numbers:
            try:
                num = float(num_str)
                # Check if close to expected per-capita
                if abs(num - self._expected_per_capita) / self._expected_per_capita < 0.1:
                    answer_correct = 1.0
                    break
            except ValueError:
                continue

        # Recovery: did model recover from tool errors?
        error_count = sum(
            1 for msg in result.messages
            if msg.role == Role.TOOL and "Error" in msg.content
        )
        total_tool_calls = len(self._tool_calls_made)
        recovery = 1.0 if (error_count > 0 and answer_correct > 0) else 0.0

        # Efficiency
        efficiency = max(0.0, 1.0 - result.total_tokens / 3000.0)

        # Task success: need correct answer + used tools properly
        task_success = float(answer_correct > 0 and tool_correct >= 0.66)

        return {
            "task_success": task_success,
            "answer_accuracy": answer_correct,
            "tool_call_correctness": tool_correct,
            "tools_used": float(len(tools_used)),
            "recovery_rate": recovery,
            "efficiency": efficiency,
            "total_tool_calls": float(total_tool_calls),
        }

    def get_human_brief(self) -> str | None:
        return (
            "OBJECTIVE: Get the model to (1) search for Springfield IL population, "
            "(2) calculate GDP per capita (Springfield GDP is 7.6 billion dollars), "
            "(3) store the result in the file_store with key 'gdp_per_capita'.\n\n"
            "TOOLS: The model has access to: search (query a knowledge corpus), "
            "calculator (evaluate math expressions), file_store (read/write/list). "
            "You instruct the model; it will call tools and reply with results.\n\n"
            "SUCCESS: The run succeeds when the model gives a final answer with the "
            "per-capita value (approximately 1788) and has used the tools correctly. "
            "Guide it step by step if needed (e.g. 'Search for Springfield IL population', "
            "then 'Calculate GDP per capita using 7.6 billion and that population')."
        )

    def get_scripted_prompter_policy(self) -> ScriptedPolicy:
        return ToolUsePolicy()
