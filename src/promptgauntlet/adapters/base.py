"""Model adapter base protocol."""

from __future__ import annotations

from typing import Protocol

from promptgauntlet.types import Message, Response, ToolSchema


class ModelClient(Protocol):
    """Protocol for model adapters.

    All adapters must implement this interface to be usable
    with the PromptGauntlet engine.
    """

    @property
    def name(self) -> str:
        """Return the adapter/model name."""
        ...

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Send messages to the model and return a response.

        Args:
            messages: Conversation history.
            tools: Optional tool schemas the model can call.
            seed: Optional seed for deterministic output.
            temperature: Optional temperature override.

        Returns:
            A Response with content, tool_calls, and usage.
        """
        ...


def get_adapter(model_name: str, **kwargs: object) -> ModelClient:
    """Factory function to get a model adapter by name.

    Args:
        model_name: One of 'mock', 'openai', 'local'.
        **kwargs: Extra configuration passed to the adapter.

    Returns:
        A ModelClient instance.
    """
    if model_name == "mock":
        from promptgauntlet.adapters.mock import MockModelClient

        return MockModelClient()
    elif model_name in ("openai", "openai-compat"):
        from promptgauntlet.adapters.openai_compat import OpenAIClient

        return OpenAIClient(**kwargs)  # type: ignore[arg-type]
    elif model_name == "local":
        from promptgauntlet.adapters.local import LocalClient

        return LocalClient()
    else:
        # Treat as OpenAI-compatible with model name
        from promptgauntlet.adapters.openai_compat import OpenAIClient

        return OpenAIClient(model=model_name, **kwargs)  # type: ignore[arg-type]
