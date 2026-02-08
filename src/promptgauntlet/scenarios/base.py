"""Base scenario protocol and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from promptgauntlet.types import (
    Message,
    ScenarioConfig,
    TaskFamily,
    ToolCallRequest,
    ToolCallResult,
    ToolSchema,
)


class ScenarioResult(BaseModel):
    """Result of a single scenario run (single seed)."""

    scenario_id: str
    seed: int
    success: bool = False
    messages: list[Message] = Field(default_factory=list)
    total_tokens: int = 0
    total_turns: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scenario(ABC):
    """Abstract base class for all scenarios.

    A scenario defines:
    - A system prompt / initial setup
    - Available tools (if any)
    - A termination condition
    - A grading function
    """

    @property
    @abstractmethod
    def config(self) -> ScenarioConfig:
        """Return the scenario configuration."""

    @abstractmethod
    def setup(self, seed: int) -> list[Message]:
        """Initialize the scenario for a given seed.

        Returns initial messages (typically a system message).
        """

    @abstractmethod
    def get_tools(self) -> list[ToolSchema]:
        """Return tool schemas available in this scenario."""

    @abstractmethod
    def handle_tool_call(self, call: ToolCallRequest) -> ToolCallResult:
        """Execute a tool call and return the result."""

    @abstractmethod
    def check_termination(
        self, messages: list[Message], turn: int, tokens: int
    ) -> bool:
        """Return True if the scenario should terminate."""

    @abstractmethod
    def grade(self, result: ScenarioResult) -> dict[str, float]:
        """Compute metrics from a scenario result.

        Returns a dict of metric_name -> value.
        """

    def get_scripted_prompter_policy(self) -> ScriptedPolicy | None:
        """Return a scripted prompter policy, or None for default."""
        return None

    def get_human_brief(self) -> str | None:
        """Return full task instructions for human-in-the-loop mode.

        Shown at the start of human mode so the human knows exactly what
        to do (objective, schema, rubric, success criteria). Return None
        to rely only on config.description.
        """
        return None


class ScriptedPolicy:
    """A scripted prompter policy that generates user messages."""

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str:
        """Generate the next user message given conversation history."""
        raise NotImplementedError


class ScenarioInfo(BaseModel):
    """Lightweight info about a registered scenario (for listing)."""

    id: str
    family: TaskFamily
    name: str
    description: str = ""
