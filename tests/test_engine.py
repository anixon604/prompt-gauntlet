"""Tests for the execution engine."""

from __future__ import annotations

from pathlib import Path

from promptgauntlet.adapters.mock import MockModelClient
from promptgauntlet.engine.prompter import ReplayPrompter, ScriptedPrompter
from promptgauntlet.engine.runner import run_single_scenario
from promptgauntlet.engine.trace import TraceReader, TraceWriter
from promptgauntlet.scenarios.registry import get_registry
from promptgauntlet.types import Message, Role, TraceEvent, TraceEventType, Usage


class TestTraceWriterReader:
    """Tests for trace recording and reading."""

    def test_write_and_read_events(self, tmp_path: Path) -> None:
        """Write events and read them back identically."""
        trace_path = tmp_path / "test_trace.jsonl"
        writer = TraceWriter(trace_path)

        event = TraceEvent(
            type=TraceEventType.USER_MESSAGE,
            data={"role": "user", "content": "hello"},
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )
        writer.write_event(event)
        writer.close()

        reader = TraceReader(trace_path)
        events = reader.read_events()
        assert len(events) == 1
        assert events[0].type == TraceEventType.USER_MESSAGE
        assert events[0].data["content"] == "hello"
        assert events[0].usage is not None
        assert events[0].usage.prompt_tokens == 10

    def test_write_message(self, tmp_path: Path) -> None:
        """Write a Message and extract it back."""
        trace_path = tmp_path / "msg_trace.jsonl"
        writer = TraceWriter(trace_path)
        msg = Message(role=Role.USER, content="test message")
        writer.write_message(msg)
        writer.close()

        reader = TraceReader(trace_path)
        messages = reader.extract_messages()
        assert len(messages) == 1
        assert messages[0].content == "test message"

    def test_metadata_and_scores(self, tmp_path: Path) -> None:
        """Write and extract metadata and scores."""
        trace_path = tmp_path / "meta_trace.jsonl"
        writer = TraceWriter(trace_path)
        writer.write_metadata({"scenario_id": "test/scenario", "seed": 42})
        writer.write_score({"accuracy": 0.95, "efficiency": 0.8})
        writer.close()

        reader = TraceReader(trace_path)
        meta = reader.extract_metadata()
        assert meta["scenario_id"] == "test/scenario"
        assert meta["seed"] == 42

        scores = reader.extract_scores()
        assert scores["accuracy"] == 0.95


class TestEngineRunner:
    """Tests for scenario execution."""

    def test_budget_turns_enforced(self) -> None:
        """Engine respects the turn budget."""
        registry = get_registry()
        scenario = registry.get_scenario("constraint/json_schema")
        client = MockModelClient()
        prompter = ScriptedPrompter()

        result = run_single_scenario(
            scenario=scenario,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=100000,
            budget_turns=2,
        )
        assert result.total_turns <= 2

    def test_budget_tokens_enforced(self) -> None:
        """Engine respects the token budget."""
        registry = get_registry()
        scenario = registry.get_scenario("classification/sentiment")
        client = MockModelClient()
        prompter = ScriptedPrompter()

        result = run_single_scenario(
            scenario=scenario,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=100,  # Very low budget
            budget_turns=100,
        )
        # Should stop early due to token budget
        assert result.total_tokens <= 200  # Allow small overshoot on last turn

    def test_replay_prompter(self) -> None:
        """ReplayPrompter replays messages in order."""
        prompter = ReplayPrompter(["msg1", "msg2", "msg3"])
        from promptgauntlet.scenarios.registry import get_registry

        scenario = get_registry().get_scenario("constraint/json_schema")
        msgs: list[Message] = []

        r1 = prompter.next_message(msgs, 0, scenario)
        assert r1 == "msg1"
        r2 = prompter.next_message(msgs, 1, scenario)
        assert r2 == "msg2"
        r3 = prompter.next_message(msgs, 2, scenario)
        assert r3 == "msg3"
        r4 = prompter.next_message(msgs, 3, scenario)
        assert r4 is None
