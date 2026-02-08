"""Robust statistics for evaluation metrics."""

from __future__ import annotations

import numpy as np


def robust_stats(values: list[float]) -> dict[str, float]:
    """Compute robust statistics from a list of values.

    Returns:
        Dict with: mean, median, std, p10, p90, failure_rate, min, max.
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "failure_rate": 1.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "failure_rate": float(np.mean(arr <= 0.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: Sample values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if len(values) < 2:
        v = values[0] if values else 0.0
        return (v, v)

    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    means: list[float] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)
