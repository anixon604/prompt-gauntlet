"""Deterministic mock model adapter for offline testing."""

from __future__ import annotations

import hashlib
import json

from promptgauntlet.types import Message, Response, ToolCallRequest, ToolSchema, Usage


class MockModelClient:
    """A deterministic mock model that produces predictable responses.

    Responses are determined by hashing the scenario context + turn + seed,
    ensuring reproducibility. Supports tool-call simulation.
    """

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Generate a deterministic mock response."""
        self._call_count += 1
        context_hash = self._hash_context(messages, seed)

        # Detect scenario type from system message
        system_msg = next((m for m in messages if m.role.value == "system"), None)
        system_text = system_msg.content if system_msg else ""
        last_user = next(
            (m for m in reversed(messages) if m.role.value == "user"), None
        )
        last_user_text = last_user.content if last_user else ""

        # Route to scenario-specific mock response
        if "classification" in system_text.lower() or "classify" in system_text.lower():
            return self._classification_response(messages, context_hash, tools)
        elif "json" in system_text.lower() or "schema" in system_text.lower():
            return self._constraint_response(messages, context_hash, tools)
        elif tools and any("calculator" in t.name or "search" in t.name for t in tools):
            return self._tool_use_response(messages, context_hash, tools)
        elif "rubric" in system_text.lower() or "converge" in system_text.lower():
            return self._convergence_response(messages, context_hash, tools)
        elif "score" in last_user_text.lower() and "rubric" in last_user_text.lower():
            # Judge rubric scoring request
            return self._judge_response(messages, context_hash)
        else:
            return self._default_response(messages, context_hash)

    def _hash_context(self, messages: list[Message], seed: int | None) -> int:
        """Create a deterministic hash from messages and seed."""
        h = hashlib.sha256()
        h.update(str(seed or 0).encode())
        h.update(str(len(messages)).encode())
        for m in messages[-3:]:  # Last 3 messages for variety
            h.update(m.content[:200].encode())
        return int(h.hexdigest()[:8], 16)

    def _classification_response(
        self,
        messages: list[Message],
        ctx: int,
        tools: list[ToolSchema] | None,
    ) -> Response:
        """Mock response for classification scenarios."""
        # Count how many user messages there have been (turns)
        user_msgs = [m for m in messages if m.role.value == "user"]
        turn = len(user_msgs)

        # Extract text to classify from the last user message
        last_user = user_msgs[-1].content if user_msgs else ""

        # Deterministic label based on hash of the text
        text_hash = hashlib.sha256(last_user.encode()).hexdigest()
        labels = ["positive", "negative", "neutral"]
        label_idx = int(text_hash[:4], 16) % 3
        chosen_label = labels[label_idx]

        # For the first turn, acknowledge the task
        if turn <= 1:
            content = (
                "I'll classify the text you provide. "
                "I'll label each as positive, negative, or neutral based on sentiment."
            )
        else:
            content = chosen_label

        tokens = len(content.split()) * 2
        return Response(
            content=content,
            usage=Usage(prompt_tokens=tokens, completion_tokens=tokens // 2),
            model="mock",
        )

    def _constraint_response(
        self,
        messages: list[Message],
        ctx: int,
        tools: list[ToolSchema] | None,
    ) -> Response:
        """Mock response for constraint satisfaction scenarios."""
        user_msgs = [m for m in messages if m.role.value == "user"]
        turn = len(user_msgs)
        last_user = user_msgs[-1].content if user_msgs else ""

        # Try to extract the expected schema from the conversation
        if turn <= 1:
            # First response: attempt to produce valid JSON
            content = json.dumps(
                {
                    "name": "Alice Smith",
                    "age": 30,
                    "email": "alice@example.com",
                    "address": {
                        "street": "123 Main St",
                        "city": "Springfield",
                        "state": "IL",
                        "zip": "62701",
                    },
                },
                indent=2,
            )
        elif "error" in last_user.lower() or "fix" in last_user.lower():
            # Recovery attempt: produce corrected JSON
            content = json.dumps(
                {
                    "name": "Alice Smith",
                    "age": 30,
                    "email": "alice@example.com",
                    "address": {
                        "street": "123 Main St",
                        "city": "Springfield",
                        "state": "IL",
                        "zip": "62701",
                    },
                    "phone": "+1-555-0123",
                },
                indent=2,
            )
        else:
            content = json.dumps({"result": "acknowledged", "turn": turn})

        tokens = len(content.split()) * 2
        return Response(
            content=content,
            usage=Usage(prompt_tokens=tokens, completion_tokens=tokens // 2),
            model="mock",
        )

    def _tool_use_response(
        self,
        messages: list[Message],
        ctx: int,
        tools: list[ToolSchema] | None,
    ) -> Response:
        """Mock response for tool-use scenarios."""
        user_msgs = [m for m in messages if m.role.value == "user"]
        tool_results = [m for m in messages if m.role.value == "tool"]
        turn = len(user_msgs)

        # Simulate a multi-step tool-using workflow
        if not tool_results:
            # First step: search for information
            tool_call = ToolCallRequest(
                id=f"call_{ctx}_{turn}",
                name="search",
                arguments={"query": "population Springfield"},
            )
            return Response(
                content="Let me search for that information.",
                tool_calls=[tool_call],
                usage=Usage(prompt_tokens=50, completion_tokens=30),
                model="mock",
            )
        elif len(tool_results) == 1:
            # Second step: calculate
            tool_call = ToolCallRequest(
                id=f"call_{ctx}_{turn}_calc",
                name="calculator",
                arguments={"expression": "116250 / 65.0"},
            )
            return Response(
                content="Now let me calculate the per-capita value.",
                tool_calls=[tool_call],
                usage=Usage(prompt_tokens=60, completion_tokens=40),
                model="mock",
            )
        elif len(tool_results) == 2:
            # Third step: store result
            tool_call = ToolCallRequest(
                id=f"call_{ctx}_{turn}_store",
                name="file_store",
                arguments={
                    "action": "write",
                    "key": "result",
                    "value": "The per-capita value is approximately 1788.46",
                },
            )
            return Response(
                content="Let me store the result.",
                tool_calls=[tool_call],
                usage=Usage(prompt_tokens=70, completion_tokens=40),
                model="mock",
            )
        else:
            # Final answer
            return Response(
                content="Based on my research and calculations, the per-capita value for Springfield is approximately 1788.46.",
                usage=Usage(prompt_tokens=80, completion_tokens=30),
                model="mock",
            )

    def _convergence_response(
        self,
        messages: list[Message],
        ctx: int,
        tools: list[ToolSchema] | None,
    ) -> Response:
        """Mock response for convergence scenarios."""
        user_msgs = [m for m in messages if m.role.value == "user"]
        turn = len(user_msgs)

        # Progressively improve toward target
        if turn <= 1:
            content = (
                "A robust error handling system should include: "
                "input validation, exception catching, logging, "
                "and graceful degradation."
            )
        elif turn == 2:
            content = (
                "A robust error handling system must include: "
                "1) Input validation at boundaries, "
                "2) Structured exception hierarchy, "
                "3) Centralized logging with severity levels, "
                "4) Graceful degradation with fallback behaviors, "
                "5) User-friendly error messages."
            )
        else:
            content = (
                "A comprehensive error handling system requires: "
                "1) Input validation at all entry points using schema validation, "
                "2) A typed exception hierarchy (ValueError, IOError, AuthError), "
                "3) Centralized structured logging with correlation IDs, "
                "4) Circuit breakers for external service calls, "
                "5) Graceful degradation with cached fallbacks, "
                "6) User-facing error messages that hide internal details, "
                "7) Retry logic with exponential backoff for transient failures."
            )

        tokens = len(content.split()) * 2
        return Response(
            content=content,
            usage=Usage(prompt_tokens=tokens, completion_tokens=tokens // 2),
            model="mock",
        )

    def _judge_response(
        self,
        messages: list[Message],
        ctx: int,
    ) -> Response:
        """Mock response for LLM-as-judge scoring requests."""
        # Return a deterministic score
        score = 0.75  # Reasonable mock score
        content = json.dumps({"score": score, "rationale": "Mock judge: reasonable quality output."})
        return Response(
            content=content,
            usage=Usage(prompt_tokens=100, completion_tokens=30),
            model="mock",
        )

    def _default_response(
        self,
        messages: list[Message],
        ctx: int,
    ) -> Response:
        """Default mock response."""
        content = f"Mock response (turn {len(messages)}, hash {ctx})"
        return Response(
            content=content,
            usage=Usage(prompt_tokens=20, completion_tokens=10),
            model="mock",
        )
