"""Base tool protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from promptbench.types import ToolCallRequest, ToolCallResult, ToolSchema


class Tool(ABC):
    """Abstract base class for simulated tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must be unique)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's parameters."""

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with given arguments.

        Args:
            arguments: Tool arguments matching the parameters schema.

        Returns:
            String result of the tool execution.

        Raises:
            ValueError: If arguments are invalid.
        """

    def to_schema(self) -> ToolSchema:
        """Convert to a ToolSchema for the model."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    def handle_call(self, call: ToolCallRequest) -> ToolCallResult:
        """Handle a tool call request and return a result."""
        try:
            result = self.execute(call.arguments)
            return ToolCallResult(
                call_id=call.id,
                name=self.name,
                result=result,
                is_error=False,
            )
        except Exception as e:
            return ToolCallResult(
                call_id=call.id,
                name=self.name,
                result=f"Error: {type(e).__name__}: {e}",
                is_error=True,
            )


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def get_schemas(self) -> list[ToolSchema]:
        """Return schemas for all registered tools."""
        return [t.to_schema() for t in self._tools.values()]

    def handle_call(self, call: ToolCallRequest) -> ToolCallResult:
        """Route a tool call to the appropriate tool."""
        try:
            tool = self.get(call.name)
            return tool.handle_call(call)
        except KeyError:
            return ToolCallResult(
                call_id=call.id,
                name=call.name,
                result=f"Error: Unknown tool '{call.name}'",
                is_error=True,
            )
