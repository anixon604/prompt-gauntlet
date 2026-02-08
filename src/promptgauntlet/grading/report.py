"""Report generation: scorecard.json, scorecard.csv, report.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from promptgauntlet.grading.pareto import pareto_rank, weighted_score
from promptgauntlet.types import Scorecard


def _runs_dir() -> Path:
    return Path("runs")


def _load_scorecard(run_id: str) -> Scorecard | None:
    """Load scorecard from a run directory."""
    path = _runs_dir() / run_id / "scorecard.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return Scorecard.model_validate(data)


def generate_report(run_id: str, formats: list[str]) -> None:
    """Generate report artifacts for a run.

    Args:
        run_id: The run ID (directory under runs/).
        formats: List of formats to generate (md, csv, json).
    """
    scorecard = _load_scorecard(run_id)
    if scorecard is None:
        return

    run_path = _runs_dir() / run_id

    if "json" in formats:
        _write_json(scorecard, run_path / "scorecard.json")
    if "csv" in formats:
        _write_csv(scorecard, run_path / "scorecard.csv")
    if "md" in formats:
        _write_markdown(scorecard, run_path / "report.md")


def _write_json(scorecard: Scorecard, path: Path) -> None:
    """Write scorecard as formatted JSON."""
    with open(path, "w") as f:
        f.write(scorecard.model_dump_json(indent=2))


def _write_csv(scorecard: Scorecard, path: Path) -> None:
    """Write scorecard as a flat CSV table."""
    if not scorecard.entries:
        return

    # Collect all metric names
    all_metrics: set[str] = set()
    for entry in scorecard.entries:
        all_metrics.update(entry.metrics.keys())
    metric_names = sorted(all_metrics)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["scenario_id", "family", "seeds"]
        for mn in metric_names:
            header.extend([f"{mn}_median", f"{mn}_mean", f"{mn}_std", f"{mn}_p10"])
        writer.writerow(header)

        # Rows
        for entry in scorecard.entries:
            row: list[Any] = [entry.scenario_id, entry.family.value, entry.seeds_run]
            for mn in metric_names:
                mv = entry.metrics.get(mn)
                if mv:
                    row.extend([
                        f"{mv.median:.4f}",
                        f"{mv.mean:.4f}",
                        f"{mv.std:.4f}",
                        f"{mv.p10:.4f}",
                    ])
                else:
                    row.extend(["", "", "", ""])
            writer.writerow(row)


def _write_markdown(scorecard: Scorecard, path: Path) -> None:
    """Write a human-readable markdown report."""
    lines: list[str] = []
    lines.append("# PromptGauntlet Report")
    lines.append("")
    lines.append(f"**Run ID:** {scorecard.run_id}")
    lines.append(f"**Model:** {scorecard.model}")
    lines.append(f"**Schema Version:** {scorecard.schema_version}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Scenario | Family | Seeds | Task Success (median) | Efficiency (median) |")
    lines.append("|----------|--------|-------|-----------------------|---------------------|")

    for entry in scorecard.entries:
        ts = entry.metrics.get("task_success")
        eff = entry.metrics.get("efficiency")
        ts_val = f"{ts.median:.3f}" if ts else "N/A"
        eff_val = f"{eff.median:.3f}" if eff else "N/A"
        lines.append(
            f"| {entry.scenario_id} | {entry.family.value} | "
            f"{entry.seeds_run} | {ts_val} | {eff_val} |"
        )

    lines.append("")

    # Per-family breakdown
    families: dict[str, list[Any]] = {}
    for entry in scorecard.entries:
        families.setdefault(entry.family.value, []).append(entry)

    for family_name, entries in sorted(families.items()):
        lines.append(f"## Family: {family_name.upper()}")
        lines.append("")

        for entry in entries:
            lines.append(f"### {entry.scenario_name}")
            lines.append(f"*{entry.scenario_id}* | Seeds: {entry.seeds_run}")
            lines.append("")
            lines.append("| Metric | Median | Mean | Std | P10 | P90 |")
            lines.append("|--------|--------|------|-----|-----|-----|")

            for mn, mv in sorted(entry.metrics.items()):
                lines.append(
                    f"| {mn} | {mv.median:.4f} | {mv.mean:.4f} | "
                    f"{mv.std:.4f} | {mv.p10:.4f} | {mv.p90:.4f} |"
                )
            lines.append("")

    # Pareto ranking
    lines.append("## Pareto Ranking")
    lines.append("")
    try:
        ranking = pareto_rank(scorecard)
        lines.append("| Rank | Scenario | Pareto Optimal | Objectives |")
        lines.append("|------|----------|----------------|------------|")
        for pe in ranking:
            obj_str = ", ".join(f"{k}: {v:.3f}" for k, v in pe.metrics.items())
            opt = "Yes" if pe.is_pareto_optimal else "No"
            lines.append(f"| {pe.rank} | {pe.scenario_id} | {opt} | {obj_str} |")
    except Exception:
        lines.append("*Pareto ranking not available.*")
    lines.append("")

    # Weighted scores
    lines.append("## Weighted Scores")
    lines.append("")
    lines.append("| Scenario | Weighted Score |")
    lines.append("|----------|----------------|")
    for entry in scorecard.entries:
        ws = weighted_score(entry)
        lines.append(f"| {entry.scenario_id} | {ws:.4f} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
