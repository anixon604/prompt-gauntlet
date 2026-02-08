"""Tests for model adapters."""

from __future__ import annotations

import pytest

from promptgauntlet.adapters.base import get_adapter
from promptgauntlet.adapters.local import LocalClient
from promptgauntlet.adapters.mock import MockModelClient
from promptgauntlet.types import Message, Role


class TestMockAdapter:
    """Tests for the deterministic mock model."""

    def test_mock_returns_response(self, mock_client: MockModelClient) -> None:
        """Mock adapter produces a response with content."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello"),
        ]
        response = mock_client.complete(messages)
        assert response.content
        assert response.model == "mock"

    def test_mock_deterministic_same_seed(self, mock_client: MockModelClient) -> None:
        """Same messages + seed produce identical responses."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a classification assistant."),
            Message(role=Role.USER, content="Classify: great product"),
        ]
        r1 = mock_client.complete(messages, seed=42)
        # Fresh client for clean state
        client2 = MockModelClient()
        r2 = client2.complete(messages, seed=42)
        assert r1.content == r2.content

    def test_mock_classification_response(self, mock_client: MockModelClient) -> None:
        """Mock produces classification-relevant responses."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a classification assistant. Classify text."),
            Message(role=Role.USER, content="Classify this text"),
        ]
        response = mock_client.complete(messages, seed=0)
        assert response.content  # Should produce some content

    def test_mock_tool_use_response(self, mock_client: MockModelClient) -> None:
        """Mock produces tool calls when tools are available."""
        from promptgauntlet.types import ToolSchema

        messages = [
            Message(role=Role.SYSTEM, content="You are a research assistant with tools."),
            Message(role=Role.USER, content="Find population of Springfield"),
        ]
        tools = [
            ToolSchema(name="search", description="Search corpus", parameters={}),
            ToolSchema(name="calculator", description="Calculate", parameters={}),
        ]
        response = mock_client.complete(messages, tools=tools, seed=0)
        assert response.tool_calls  # Should produce tool calls

    def test_mock_usage_tracking(self, mock_client: MockModelClient) -> None:
        """Mock adapter reports token usage."""
        messages = [
            Message(role=Role.SYSTEM, content="Test."),
            Message(role=Role.USER, content="Hello"),
        ]
        response = mock_client.complete(messages)
        assert response.usage.prompt_tokens >= 0
        assert response.usage.completion_tokens >= 0


class TestAdapterFactory:
    """Tests for the adapter factory function."""

    def test_get_mock_adapter(self) -> None:
        """Factory returns mock adapter."""
        client = get_adapter("mock")
        assert isinstance(client, MockModelClient)
        assert client.name == "mock"

    def test_get_local_adapter(self) -> None:
        """Factory returns local adapter (stub)."""
        client = get_adapter("local")
        assert isinstance(client, LocalClient)

    def test_local_adapter_raises(self) -> None:
        """Local adapter raises NotImplementedError."""
        client = LocalClient()
        with pytest.raises(NotImplementedError):
            client.complete([Message(role=Role.USER, content="test")])
