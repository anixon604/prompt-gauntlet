"""Shared test fixtures for PromptBench."""

from __future__ import annotations

import pytest

from promptbench.adapters.mock import MockModelClient
from promptbench.types import Message, Role


@pytest.fixture
def mock_client() -> MockModelClient:
    """Provide a fresh mock model client."""
    return MockModelClient()


@pytest.fixture
def sample_messages() -> list[Message]:
    """Provide a simple conversation for testing."""
    return [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="Hello, how are you?"),
    ]
