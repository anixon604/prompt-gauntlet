"""Local model adapter stub."""

from __future__ import annotations

from promptgauntlet.types import Message, Response, ToolSchema


class LocalClient:
    """Stub adapter for local model inference.

    This is a placeholder for future local model support
    (e.g., llama.cpp, vLLM, Ollama).
    """

    @property
    def name(self) -> str:
        return "local"

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Not yet implemented.

        To add local model support, implement this method with
        your preferred inference backend (llama.cpp, vLLM, Ollama, etc.).
        """
        raise NotImplementedError(
            "Local model adapter not yet implemented. "
            "To use a local model, implement the complete() method in "
            "src/promptgauntlet/adapters/local.py or use an OpenAI-compatible "
            "local server with --model openai and OPENAI_BASE_URL env var."
        )
