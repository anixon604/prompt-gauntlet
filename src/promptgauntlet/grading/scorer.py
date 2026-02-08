"""Multi-objective metric computation and scorecard generation."""

from __future__ import annotations

from collections import defaultdict

from promptgauntlet.grading.stats import robust_stats
from promptgauntlet.scenarios.base import ScenarioResult
from promptgauntlet.types import MetricValue, Scorecard, ScorecardEntry, TaskFamily


def compute_scorecard(
    results: list[ScenarioResult],
    run_id: str = "",
    model: str = "",
) -> Scorecard:
    """Compute a scorecard from a list of scenario results.

    Groups results by scenario_id, computes robust statistics
    per metric, and produces a Scorecard.

    Args:
        results: List of ScenarioResult from running scenarios.
        run_id: The run identifier.
        model: Model name.

    Returns:
        A Scorecard with entries per scenario.
    """
    # Group by scenario_id
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        grouped[r.scenario_id].append(r)

    entries: list[ScorecardEntry] = []
    for scenario_id, scenario_results in sorted(grouped.items()):
        # Determine family from scenario_id
        family = _detect_family(scenario_id)

        # Collect all metric names
        all_metric_names: set[str] = set()
        for r in scenario_results:
            all_metric_names.update(r.metrics.keys())

        # Compute stats per metric
        metrics: dict[str, MetricValue] = {}
        for metric_name in sorted(all_metric_names):
            values = [r.metrics.get(metric_name, 0.0) for r in scenario_results]
            stats = robust_stats(values)
            metrics[metric_name] = MetricValue(
                name=metric_name,
                values=values,
                mean=stats["mean"],
                median=stats["median"],
                std=stats["std"],
                p10=stats["p10"],
                p90=stats["p90"],
                failure_rate=stats["failure_rate"],
            )

        # Infer scenario name from ID
        name = scenario_id.replace("/", " - ").replace("_", " ").title()

        entries.append(
            ScorecardEntry(
                scenario_id=scenario_id,
                family=family,
                scenario_name=name,
                metrics=metrics,
                seeds_run=len(scenario_results),
            )
        )

    return Scorecard(
        run_id=run_id,
        model=model,
        entries=entries,
    )


def _detect_family(scenario_id: str) -> TaskFamily:
    """Detect task family from scenario ID prefix."""
    if scenario_id.startswith("classification"):
        return TaskFamily.CLASSIFICATION
    elif scenario_id.startswith("constraint"):
        return TaskFamily.CONSTRAINT
    elif scenario_id.startswith("tool_use"):
        return TaskFamily.TOOL_USE
    elif scenario_id.startswith("convergence"):
        return TaskFamily.CONVERGENCE
    else:
        return TaskFamily.CLASSIFICATION  # Default fallback
