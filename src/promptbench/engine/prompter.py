"""Prompter implementations: scripted, human, replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from promptbench.scenarios.base import Scenario
    from promptbench.types import Message


class Prompter(ABC):
    """Base class for prompter policies."""

    @abstractmethod
    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str | None:
        """Return the next user message, or None to stop."""


class ScriptedPrompter(Prompter):
    """Executes a scenario's scripted prompter policy."""

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str | None:
        policy = scenario.get_scripted_prompter_policy()
        if policy is None:
            return self._default_policy(messages, turn, scenario)
        return policy.next_message(messages, turn, scenario)

    def _default_policy(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str | None:
        """Default scripted policy: send the initial prompt then stop."""
        if turn == 0:
            return "Please begin the task as described in the system prompt."
        return None


class HumanPrompter(Prompter):
    """Interactive human-in-the-loop prompter using rich terminal UI."""

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str | None:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        # Display recent messages
        if messages:
            last = messages[-1]
            role_colors = {
                "system": "yellow",
                "assistant": "green",
                "tool": "cyan",
                "user": "blue",
            }
            color = role_colors.get(last.role.value, "white")
            console.print(
                Panel(
                    last.content[:2000],
                    title=f"[{color}]{last.role.value.upper()}[/{color}]",
                    border_style=color,
                )
            )

        # Prompt for input
        console.print(f"\n[dim]Turn {turn + 1} | Type 'quit' to exit[/dim]")
        try:
            user_input = console.input("[bold blue]You>[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            return None

        if user_input.strip().lower() in ("quit", "exit", "q"):
            return None
        return user_input


class ReplayPrompter(Prompter):
    """Replays user messages from a trace."""

    def __init__(self, user_messages: list[str]) -> None:
        self._messages = user_messages
        self._index = 0

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str | None:
        if self._index >= len(self._messages):
            return None
        msg = self._messages[self._index]
        self._index += 1
        return msg
