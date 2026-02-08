"""Tests for classification scenario."""

from __future__ import annotations

from promptgauntlet.adapters.mock import MockModelClient
from promptgauntlet.engine.prompter import ScriptedPrompter
from promptgauntlet.engine.runner import run_single_scenario
from promptgauntlet.scenarios.classification.scenario import ClassificationScenario
from promptgauntlet.types import TaskFamily


class TestClassificationScenario:
    """Tests for sentiment classification scenario."""

    def test_scenario_config(self) -> None:
        """Scenario has correct configuration."""
        s = ClassificationScenario()
        assert s.config.id == "classification/sentiment"
        assert s.config.family == TaskFamily.CLASSIFICATION

    def test_setup_creates_train_test_split(self) -> None:
        """Setup creates a train/test split from the dataset."""
        s = ClassificationScenario()
        messages = s.setup(seed=42)
        assert len(messages) == 1
        assert messages[0].role.value == "system"
        assert len(s._train) > 0
        assert len(s._test) > 0
        assert len(s._train) + len(s._test) <= 200

    def test_deterministic_split(self) -> None:
        """Same seed produces same split."""
        s1 = ClassificationScenario()
        s1.setup(seed=42)

        s2 = ClassificationScenario()
        s2.setup(seed=42)

        assert [x["id"] for x in s1._test] == [x["id"] for x in s2._test]

    def test_run_produces_metrics(self) -> None:
        """Running the scenario produces expected metrics."""
        s = ClassificationScenario()
        client = MockModelClient()
        prompter = ScriptedPrompter()

        result = run_single_scenario(
            scenario=s,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=10000,
            budget_turns=20,
        )
        assert "accuracy" in result.metrics
        assert "task_success" in result.metrics
        assert "efficiency" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
