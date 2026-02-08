"""OpenAI-compatible model adapter."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from promptgauntlet.types import Message, Response, ToolCallRequest, ToolSchema, Usage


class OpenAIClient:
    """Adapter for OpenAI-compatible APIs.

    Uses the openai Python package if available, otherwise falls back
    to httpx for raw API calls.

    Env vars:
        OPENAI_API_KEY: API key.
        OPENAI_BASE_URL: Optional custom base URL.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        # Try to use openai package
        try:
            import openai

            self._client = openai.OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
            self._use_openai_pkg = True
        except ImportError:
            import httpx

            self._http = httpx.Client(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=120.0,
            )
            self._use_openai_pkg = False

    @property
    def name(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Send completion request to OpenAI-compatible API."""
        if self._use_openai_pkg:
            return self._complete_openai(messages, tools, seed, temperature)
        return self._complete_httpx(messages, tools, seed, temperature)

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        result: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role.value, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            result.append(msg)
        return result

    def _format_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """Convert tool schemas to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _complete_openai(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        seed: int | None,
        temperature: float | None,
    ) -> Response:
        """Complete using the openai Python package."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._format_messages(messages),
        }
        if tools:
            kwargs["tools"] = self._format_tools(tools)
        if seed is not None:
            kwargs["seed"] = seed
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls: list[ToolCallRequest] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = Usage()
        if resp.usage:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )

        return Response(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            model=resp.model,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    def _complete_httpx(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        seed: int | None,
        temperature: float | None,
    ) -> Response:
        """Complete using raw httpx calls."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._format_messages(messages),
        }
        if tools:
            payload["tools"] = self._format_tools(tools)
        if seed is not None:
            payload["seed"] = seed
        if temperature is not None:
            payload["temperature"] = temperature

        resp = self._http.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls: list[ToolCallRequest] = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {"raw": fn.get("arguments", "")}
            tool_calls.append(
                ToolCallRequest(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        # Parse tool calls from text if none found natively
        if not tool_calls and tools:
            tool_calls = self._parse_tool_calls_from_text(msg.get("content", ""))

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return Response(
            content=msg.get("content", ""),
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model", self._model),
            raw=data,
        )

    @staticmethod
    def _parse_tool_calls_from_text(text: str) -> list[ToolCallRequest]:
        """Fallback: parse tool calls from XML-style tags in text.

        Expected format:
            <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        """
        calls: list[ToolCallRequest] = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                calls.append(
                    ToolCallRequest(
                        name=data.get("name", ""),
                        arguments=data.get("arguments", {}),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return calls
