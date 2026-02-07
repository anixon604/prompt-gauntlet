"""Constraint satisfaction scenario: JSON schema + formatting tasks."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from promptbench.scenarios.base import Scenario, ScenarioResult, ScriptedPolicy
from promptbench.scenarios.registry import register_scenario
from promptbench.types import (
    Message,
    Role,
    ScenarioConfig,
    TaskFamily,
    ToolCallRequest,
    ToolCallResult,
    ToolSchema,
)

# Define the target JSON schema
_PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "format": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string", "minLength": 1},
                "city": {"type": "string", "minLength": 1},
                "state": {"type": "string", "minLength": 2, "maxLength": 2},
                "zip": {"type": "string", "pattern": r"^\d{5}$"},
            },
            "required": ["street", "city", "state", "zip"],
        },
    },
    "required": ["name", "age", "email", "address"],
}


def validate_json_against_schema(
    text: str, schema: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate JSON text against a schema.

    Returns:
        (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Try to parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Validate against schema
    validator = jsonschema.Draft7Validator(schema)
    for error in validator.iter_errors(data):
        errors.append(f"{error.json_path}: {error.message}")

    return len(errors) == 0, errors


class ConstraintPolicy(ScriptedPolicy):
    """Scripted baseline: direct instruction, then fix-up on failure."""

    def __init__(self) -> None:
        self._attempt = 0
        self._last_errors: list[str] = []

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str:
        if turn == 0:
            schema_str = json.dumps(_PERSON_SCHEMA, indent=2)
            return (
                f"Generate a valid JSON object that matches this schema exactly:\n"
                f"```json\n{schema_str}\n```\n"
                f"Output ONLY the JSON, no other text."
            )

        # Check last assistant response for validity
        last_assistant = None
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT:
                last_assistant = msg.content
                break

        if last_assistant:
            is_valid, errors = validate_json_against_schema(
                last_assistant, _PERSON_SCHEMA
            )
            if is_valid:
                return ""  # Done
            self._last_errors = errors
            self._attempt += 1
            if self._attempt >= 3:
                return ""  # Give up
            return (
                "The JSON has validation errors:\n"
                + "\n".join(f"- {e}" for e in errors[:5])
                + "\nPlease fix and output ONLY the corrected JSON."
            )

        return ""


@register_scenario
class ConstraintScenario(Scenario):
    """Constraint satisfaction scenario.

    Tasks: produce JSON matching a schema, extract fields, follow formatting.
    Deterministic validators using jsonschema + custom checks.
    """

    def __init__(self) -> None:
        self._schema = _PERSON_SCHEMA
        self._seed = 0

    @property
    def config(self) -> ScenarioConfig:
        return ScenarioConfig(
            id="constraint/json_schema",
            family=TaskFamily.CONSTRAINT,
            name="JSON Schema Conformance",
            description=(
                "Produce a valid JSON object matching a given schema. "
                "Scored by pass rate, retries, and token cost."
            ),
            budget_tokens=5000,
            budget_turns=10,
        )

    def setup(self, seed: int) -> list[Message]:
        self._seed = seed
        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a data generation assistant. When asked to produce "
                    "JSON matching a schema, output ONLY valid JSON with no "
                    "surrounding text, markdown, or explanation."
                ),
            )
        ]

    def get_tools(self) -> list[ToolSchema]:
        return []

    def handle_tool_call(self, call: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            call_id=call.id,
            name=call.name,
            result="Error: No tools available.",
            is_error=True,
        )

    def check_termination(
        self, messages: list[Message], turn: int, tokens: int
    ) -> bool:
        # Check if last assistant message is valid JSON
        for msg in reversed(messages):
            if msg.role == Role.ASSISTANT:
                is_valid, _ = validate_json_against_schema(
                    msg.content, self._schema
                )
                return is_valid
        return False

    def grade(self, result: ScenarioResult) -> dict[str, float]:
        """Grade constraint satisfaction."""
        attempts = 0
        passed = False
        errors_per_attempt: list[int] = []

        for msg in result.messages:
            if msg.role == Role.ASSISTANT:
                attempts += 1
                is_valid, errors = validate_json_against_schema(
                    msg.content, self._schema
                )
                errors_per_attempt.append(len(errors))
                if is_valid:
                    passed = True

        # First-attempt pass
        first_pass = errors_per_attempt[0] == 0 if errors_per_attempt else False

        # Recovery: did it eventually pass?
        recovery = 1.0 if (passed and not first_pass and attempts > 1) else 0.0

        # Efficiency
        efficiency = max(0.0, 1.0 - result.total_tokens / 2000.0)

        return {
            "task_success": 1.0 if passed else 0.0,
            "pass_rate": 1.0 if passed else 0.0,
            "first_attempt_pass": 1.0 if first_pass else 0.0,
            "recovery_rate": recovery,
            "attempts": float(attempts),
            "efficiency": efficiency,
        }

    def get_scripted_prompter_policy(self) -> ScriptedPolicy:
        return ConstraintPolicy()
