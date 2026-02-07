"""Pareto ranking and weighted scoring for multi-objective evaluation."""

from __future__ import annotations

from typing import Any

from promptbench.types import Scorecard, ScorecardEntry


class ParetoEntry:
    """An entry in a Pareto ranking."""

    def __init__(
        self,
        scenario_id: str,
        rank: int,
        is_pareto_optimal: bool,
        metrics: dict[str, float],
    ) -> None:
        self.scenario_id = scenario_id
        self.rank = rank
        self.is_pareto_optimal = is_pareto_optimal
        self.metrics = metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "rank": self.rank,
            "is_pareto_optimal": self.is_pareto_optimal,
            "metrics": self.metrics,
        }


def _dominates(a: list[float], b: list[float]) -> bool:
    """Return True if a Pareto-dominates b (all >= and at least one >)."""
    at_least_one_better = False
    for ai, bi in zip(a, b, strict=True):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better


def pareto_rank(
    scorecard: Scorecard,
    objectives: list[str] | None = None,
) -> list[ParetoEntry]:
    """Compute Pareto ranking over scorecard entries.

    Args:
        scorecard: The scorecard to rank.
        objectives: Metric names to use as objectives (default: task_success, efficiency).

    Returns:
        List of ParetoEntry sorted by rank.
    """
    if objectives is None:
        objectives = ["task_success", "efficiency"]

    # Extract objective vectors
    entries: list[tuple[str, list[float]]] = []
    for entry in scorecard.entries:
        obj_vals: list[float] = []
        for obj in objectives:
            mv = entry.metrics.get(obj)
            obj_vals.append(mv.median if mv else 0.0)
        entries.append((entry.scenario_id, obj_vals))

    # Compute Pareto fronts
    remaining = list(range(len(entries)))
    ranks: dict[int, int] = {}
    current_rank = 1

    while remaining:
        # Find non-dominated points in remaining
        front: list[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and _dominates(entries[j][1], entries[i][1]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for idx in front:
            ranks[idx] = current_rank
            remaining.remove(idx)
        current_rank += 1

    # Build result
    result: list[ParetoEntry] = []
    for i, (sid, vals) in enumerate(entries):
        result.append(
            ParetoEntry(
                scenario_id=sid,
                rank=ranks.get(i, current_rank),
                is_pareto_optimal=ranks.get(i, current_rank) == 1,
                metrics=dict(zip(objectives, vals, strict=True)),
            )
        )

    return sorted(result, key=lambda e: e.rank)


def weighted_score(
    entry: ScorecardEntry,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted scalar score from a scorecard entry.

    Args:
        entry: The scorecard entry.
        weights: Metric name -> weight mapping.

    Returns:
        Weighted score (0.0-1.0).
    """
    if weights is None:
        weights = {
            "task_success": 0.5,
            "efficiency": 0.2,
            "recovery_rate": 0.15,
        }

    total_weight = 0.0
    total_score = 0.0
    for metric_name, weight in weights.items():
        mv = entry.metrics.get(metric_name)
        if mv:
            total_score += weight * mv.median
            total_weight += weight

    if total_weight == 0:
        return 0.0
    return total_score / total_weight
