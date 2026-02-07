"""Tests for convergence scenario."""

from __future__ import annotations

from promptbench.adapters.mock import MockModelClient
from promptbench.engine.prompter import ScriptedPrompter
from promptbench.engine.runner import run_single_scenario
from promptbench.scenarios.convergence.scenario import (
    _DEFAULT_RUBRIC,
    ConvergenceScenario,
    check_invariants,
    check_keywords,
)
from promptbench.types import TaskFamily


class TestConvergenceChecks:
    """Tests for convergence validation functions."""

    def test_invariant_checking(self) -> None:
        """Invariant checker detects required concepts."""
        text = (
            "The system includes input validation, exception hierarchy, "
            "comprehensive logging, graceful degradation, and "
            "user-friendly error messages."
        )
        matched, total, names = check_invariants(text, _DEFAULT_RUBRIC)
        assert total == 5
        assert matched >= 4

    def test_keyword_checking(self) -> None:
        """Keyword checker detects target keywords."""
        text = (
            "Validation and exception handling with logging and "
            "degradation patterns plus error messages."
        )
        coverage = check_keywords(text, _DEFAULT_RUBRIC)
        assert coverage > 0.3

    def test_empty_text_scores_zero(self) -> None:
        """Empty text scores zero invariants."""
        matched, total, names = check_invariants("", _DEFAULT_RUBRIC)
        assert matched == 0


class TestConvergenceScenario:
    """Tests for the convergence scenario end-to-end."""

    def test_scenario_config(self) -> None:
        s = ConvergenceScenario()
        assert s.config.id == "convergence/error_handling"
        assert s.config.family == TaskFamily.CONVERGENCE

    def test_mock_run_produces_metrics(self) -> None:
        """Running with mock model produces expected metrics."""
        s = ConvergenceScenario()
        client = MockModelClient()
        prompter = ScriptedPrompter()

        result = run_single_scenario(
            scenario=s,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=8000,
            budget_turns=10,
        )
        assert "invariant_coverage" in result.metrics
        assert "keyword_coverage" in result.metrics
        assert "task_success" in result.metrics
        assert result.metrics["invariant_coverage"] > 0.0
