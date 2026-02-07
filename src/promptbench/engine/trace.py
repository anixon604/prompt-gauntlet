"""Trace recording and reading for reproducibility."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from promptbench.types import Message, TraceEvent, TraceEventType, Usage


class TraceWriter:
    """Appends trace events to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a")  # noqa: SIM115

    def write_event(self, event: TraceEvent) -> None:
        """Write a single event to the trace file."""
        self._file.write(event.model_dump_json() + "\n")
        self._file.flush()

    def write_message(self, message: Message, usage: Usage | None = None) -> None:
        """Convenience: write a message as a trace event."""
        event_type_map = {
            "system": TraceEventType.SYSTEM_SETUP,
            "user": TraceEventType.USER_MESSAGE,
            "assistant": TraceEventType.ASSISTANT_MESSAGE,
            "tool": TraceEventType.TOOL_RESULT,
        }
        event_type = event_type_map.get(
            message.role.value, TraceEventType.METADATA
        )
        self.write_event(
            TraceEvent(
                type=event_type,
                timestamp=time.time(),
                data=message.model_dump(),
                usage=usage,
            )
        )

    def write_metadata(self, data: dict[str, Any]) -> None:
        """Write metadata to the trace."""
        self.write_event(
            TraceEvent(
                type=TraceEventType.METADATA,
                timestamp=time.time(),
                data=data,
            )
        )

    def write_score(self, metrics: dict[str, float]) -> None:
        """Write final scores to the trace."""
        self.write_event(
            TraceEvent(
                type=TraceEventType.SCORE,
                timestamp=time.time(),
                data={"metrics": metrics},
            )
        )

    def close(self) -> None:
        """Close the trace file."""
        self._file.close()


class TraceReader:
    """Reads trace events from a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def read_events(self) -> list[TraceEvent]:
        """Read all events from the trace file."""
        events: list[TraceEvent] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(TraceEvent.model_validate_json(line))
        return events

    def iter_events(self) -> Iterator[TraceEvent]:
        """Iterate over events lazily."""
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield TraceEvent.model_validate_json(line)

    def extract_messages(self) -> list[Message]:
        """Extract conversation messages from the trace."""
        messages: list[Message] = []
        for event in self.read_events():
            if event.type in (
                TraceEventType.SYSTEM_SETUP,
                TraceEventType.USER_MESSAGE,
                TraceEventType.ASSISTANT_MESSAGE,
                TraceEventType.TOOL_RESULT,
            ):
                messages.append(Message.model_validate(event.data))
        return messages

    def extract_metadata(self) -> dict[str, Any]:
        """Extract metadata from the trace."""
        meta: dict[str, Any] = {}
        for event in self.read_events():
            if event.type == TraceEventType.METADATA:
                meta.update(event.data)
        return meta

    def extract_scores(self) -> dict[str, float]:
        """Extract the last score entry from the trace."""
        scores: dict[str, float] = {}
        for event in self.read_events():
            if event.type == TraceEventType.SCORE:
                scores = event.data.get("metrics", {})
        return scores

    def total_tokens(self) -> int:
        """Sum all token usage in the trace."""
        total = 0
        for event in self.read_events():
            if event.usage:
                total += event.usage.total_tokens
        return total
