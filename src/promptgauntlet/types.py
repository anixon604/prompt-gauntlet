"""Shared type definitions for PromptGauntlet."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class Role(StrEnum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallRequest(BaseModel):
    """A request from the model to call a tool."""

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Result of executing a tool call."""

    call_id: str = ""
    name: str
    result: str
    is_error: bool = False


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class Usage(BaseModel):
    """Token usage information from a model response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Response(BaseModel):
    """Response from a model adapter."""

    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------

class TraceEventType(StrEnum):
    """Types of events recorded in a trace."""

    SYSTEM_SETUP = "system_setup"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    METADATA = "metadata"
    SCORE = "score"


class TraceEvent(BaseModel):
    """A single event in a scenario trace."""

    type: TraceEventType
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Scenario types
# ---------------------------------------------------------------------------

class TaskFamily(StrEnum):
    """Task family identifiers."""

    CLASSIFICATION = "classification"
    CONSTRAINT = "constraint"
    TOOL_USE = "tool_use"
    CONVERGENCE = "convergence"


class ScenarioConfig(BaseModel):
    """Configuration for a single scenario."""

    id: str
    family: TaskFamily
    name: str
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    budget_tokens: int = 10000
    budget_turns: int = 20


class ToolSchema(BaseModel):
    """Schema for a tool exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run / scoring types
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """Configuration for a batch run."""

    model: str = "mock"
    scenarios: list[str] = Field(default_factory=lambda: ["all"])
    seeds: int = 3
    budget_tokens: int = 10000
    budget_turns: int = 20
    temperature: float = 0.0
    config_path: str | None = None


class MetricValue(BaseModel):
    """A single metric with robust statistics."""

    name: str
    values: list[float] = Field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p10: float = 0.0
    p90: float = 0.0
    failure_rate: float = 0.0


class ScorecardEntry(BaseModel):
    """Scorecard for a single scenario across seeds."""

    scenario_id: str
    family: TaskFamily
    scenario_name: str
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    seeds_run: int = 0


class Scorecard(BaseModel):
    """Full scorecard for a run."""

    schema_version: str = "1.0"
    run_id: str = ""
    model: str = ""
    entries: list[ScorecardEntry] = Field(default_factory=list)
    weighted_score: float | None = None
