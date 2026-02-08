"""Ensemble judge: aggregates multiple judges with disagreement penalty."""

from __future__ import annotations

from typing import Any

import numpy as np

from promptgauntlet.judges.base import Judge, JudgeScore


class EnsembleJudge(Judge):
    """Aggregates scores from multiple judges.

    Features:
    - Configurable weights per judge
    - Disagreement penalty (based on score variance)
    - Per-judge breakdown in metadata
    """

    def __init__(
        self,
        judges: list[Judge],
        weights: dict[str, float] | None = None,
        disagreement_penalty: float = 0.1,
    ) -> None:
        self._judges = judges
        self._weights = weights or {}
        self._disagreement_penalty = disagreement_penalty

    @property
    def name(self) -> str:
        return "ensemble"

    def score(
        self,
        output: str,
        rubric: dict[str, Any],
        **kwargs: Any,
    ) -> JudgeScore:
        """Compute ensemble score from all judges."""
        judge_scores: list[JudgeScore] = []
        for judge in self._judges:
            try:
                js = judge.score(output, rubric, **kwargs)
                judge_scores.append(js)
            except Exception as e:
                # Record failure but don't crash
                judge_scores.append(
                    JudgeScore(
                        judge_name=judge.name,
                        score=0.0,
                        rationale=f"Judge error: {e}",
                        metadata={"error": True},
                    )
                )

        if not judge_scores:
            return JudgeScore(
                judge_name=self.name,
                score=0.0,
                rationale="No judges available",
            )

        # Compute weighted average
        scores: list[float] = []
        weights: list[float] = []
        for js in judge_scores:
            w = self._weights.get(js.judge_name, 1.0 / len(judge_scores))
            scores.append(js.score)
            weights.append(w)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        weighted_avg = float(np.average(scores, weights=weights))

        # Compute disagreement penalty
        if len(scores) > 1:
            variance = float(np.var(scores))
            penalty = self._disagreement_penalty * variance
        else:
            variance = 0.0
            penalty = 0.0

        final_score = max(0.0, min(1.0, weighted_avg - penalty))

        # Build rationale
        parts: list[str] = []
        for js in judge_scores:
            parts.append(f"{js.judge_name}: {js.score:.3f}")
        rationale = (
            f"Ensemble ({', '.join(parts)}), "
            f"weighted avg: {weighted_avg:.3f}, "
            f"variance: {variance:.4f}, "
            f"penalty: {penalty:.4f}"
        )

        return JudgeScore(
            judge_name=self.name,
            score=final_score,
            rationale=rationale,
            metadata={
                "per_judge": {js.judge_name: js.score for js in judge_scores},
                "weighted_average": weighted_avg,
                "variance": variance,
                "penalty": penalty,
                "judge_details": [js.model_dump() for js in judge_scores],
            },
        )
