"""Tests for grading, statistics, and scoring."""

from __future__ import annotations

from promptgauntlet.grading.pareto import pareto_rank, weighted_score
from promptgauntlet.grading.scorer import compute_scorecard
from promptgauntlet.grading.stats import bootstrap_ci, robust_stats
from promptgauntlet.scenarios.base import ScenarioResult
from promptgauntlet.types import MetricValue, Scorecard, ScorecardEntry, TaskFamily


class TestRobustStats:
    """Tests for statistical computations."""

    def test_basic_stats(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = robust_stats(values)
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_empty_values(self) -> None:
        stats = robust_stats([])
        assert stats["mean"] == 0.0
        assert stats["failure_rate"] == 1.0

    def test_single_value(self) -> None:
        stats = robust_stats([0.75])
        assert stats["mean"] == 0.75
        assert stats["median"] == 0.75

    def test_failure_rate(self) -> None:
        values = [0.0, 0.0, 1.0, 1.0]
        stats = robust_stats(values)
        assert stats["failure_rate"] == 0.5

    def test_bootstrap_ci(self) -> None:
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        lower, upper = bootstrap_ci(values, seed=42)
        assert lower < upper
        assert lower >= 0.4
        assert upper <= 1.0


class TestScorecard:
    """Tests for scorecard computation."""

    def test_compute_scorecard(self) -> None:
        results = [
            ScenarioResult(
                scenario_id="test/scenario",
                seed=0,
                metrics={"accuracy": 0.8, "efficiency": 0.9},
            ),
            ScenarioResult(
                scenario_id="test/scenario",
                seed=1,
                metrics={"accuracy": 0.7, "efficiency": 0.85},
            ),
        ]
        scorecard = compute_scorecard(results, run_id="test_run", model="mock")
        assert len(scorecard.entries) == 1
        entry = scorecard.entries[0]
        assert entry.seeds_run == 2
        assert "accuracy" in entry.metrics
        assert entry.metrics["accuracy"].median == 0.75

    def test_multiple_scenarios(self) -> None:
        results = [
            ScenarioResult(scenario_id="classification/a", seed=0, metrics={"x": 1.0}),
            ScenarioResult(scenario_id="constraint/b", seed=0, metrics={"x": 0.5}),
        ]
        scorecard = compute_scorecard(results)
        assert len(scorecard.entries) == 2


class TestParetoRanking:
    """Tests for Pareto ranking."""

    def test_pareto_ranking(self) -> None:
        scorecard = Scorecard(
            entries=[
                ScorecardEntry(
                    scenario_id="a",
                    family=TaskFamily.CLASSIFICATION,
                    scenario_name="A",
                    metrics={
                        "task_success": MetricValue(name="task_success", median=1.0),
                        "efficiency": MetricValue(name="efficiency", median=0.8),
                    },
                ),
                ScorecardEntry(
                    scenario_id="b",
                    family=TaskFamily.CONSTRAINT,
                    scenario_name="B",
                    metrics={
                        "task_success": MetricValue(name="task_success", median=0.5),
                        "efficiency": MetricValue(name="efficiency", median=0.5),
                    },
                ),
            ]
        )
        ranking = pareto_rank(scorecard)
        assert len(ranking) == 2
        # 'a' dominates 'b'
        a_entry = next(r for r in ranking if r.scenario_id == "a")
        b_entry = next(r for r in ranking if r.scenario_id == "b")
        assert a_entry.is_pareto_optimal
        assert a_entry.rank < b_entry.rank

    def test_weighted_score(self) -> None:
        entry = ScorecardEntry(
            scenario_id="test",
            family=TaskFamily.CLASSIFICATION,
            scenario_name="Test",
            metrics={
                "task_success": MetricValue(name="task_success", median=1.0),
                "efficiency": MetricValue(name="efficiency", median=0.5),
            },
        )
        ws = weighted_score(entry, {"task_success": 0.7, "efficiency": 0.3})
        assert 0.0 <= ws <= 1.0
        # Should be closer to 1.0 since task_success is weighted higher
        assert ws > 0.7
