"""Tests for constraint satisfaction scenario."""

from __future__ import annotations

import json

from promptgauntlet.adapters.mock import MockModelClient
from promptgauntlet.engine.prompter import ScriptedPrompter
from promptgauntlet.engine.runner import run_single_scenario
from promptgauntlet.scenarios.constraint.scenario import (
    _PERSON_SCHEMA,
    ConstraintScenario,
    validate_json_against_schema,
)
from promptgauntlet.types import TaskFamily


class TestConstraintValidator:
    """Tests for the JSON schema validator."""

    def test_valid_json_passes(self) -> None:
        """Valid JSON matching schema passes validation."""
        valid_json = json.dumps({
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com",
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "state": "IL",
                "zip": "62701",
            },
        })
        is_valid, errors = validate_json_against_schema(valid_json, _PERSON_SCHEMA)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_json_fails(self) -> None:
        """Invalid JSON fails parsing."""
        is_valid, errors = validate_json_against_schema("not json", _PERSON_SCHEMA)
        assert not is_valid
        assert any("Invalid JSON" in e for e in errors)

    def test_missing_required_field_fails(self) -> None:
        """JSON missing required fields fails validation."""
        incomplete = json.dumps({"name": "Alice"})
        is_valid, errors = validate_json_against_schema(incomplete, _PERSON_SCHEMA)
        assert not is_valid
        assert len(errors) > 0

    def test_invalid_email_fails(self) -> None:
        """Invalid email format fails validation."""
        bad_email = json.dumps({
            "name": "Alice",
            "age": 30,
            "email": "not-an-email",
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "state": "IL",
                "zip": "62701",
            },
        })
        is_valid, errors = validate_json_against_schema(bad_email, _PERSON_SCHEMA)
        assert not is_valid

    def test_invalid_zip_fails(self) -> None:
        """Invalid zip code fails validation."""
        bad_zip = json.dumps({
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com",
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "state": "IL",
                "zip": "invalid",
            },
        })
        is_valid, errors = validate_json_against_schema(bad_zip, _PERSON_SCHEMA)
        assert not is_valid


class TestConstraintScenario:
    """Tests for the constraint scenario end-to-end."""

    def test_scenario_config(self) -> None:
        s = ConstraintScenario()
        assert s.config.id == "constraint/json_schema"
        assert s.config.family == TaskFamily.CONSTRAINT

    def test_mock_run_passes(self) -> None:
        """Mock model produces valid JSON that passes the constraint."""
        s = ConstraintScenario()
        client = MockModelClient()
        prompter = ScriptedPrompter()

        result = run_single_scenario(
            scenario=s,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=5000,
            budget_turns=10,
        )
        assert result.metrics["task_success"] == 1.0
        assert result.metrics["pass_rate"] == 1.0
