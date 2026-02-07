"""Replay and re-grade from stored traces."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from promptbench.engine.trace import TraceReader
from promptbench.grading.report import generate_report
from promptbench.grading.scorer import compute_scorecard
from promptbench.scenarios.base import ScenarioResult
from promptbench.scenarios.registry import get_registry

console = Console()


def _runs_dir() -> Path:
    return Path("runs")


def replay_single_trace(trace_path: Path) -> ScenarioResult:
    """Replay a single trace file and re-grade.

    Args:
        trace_path: Path to a .jsonl trace file.

    Returns:
        ScenarioResult with re-computed metrics.
    """
    reader = TraceReader(trace_path)
    metadata = reader.extract_metadata()
    messages = reader.extract_messages()
    total_tokens = reader.total_tokens()

    scenario_id = metadata.get("scenario_id", "unknown")
    seed = metadata.get("seed", 0)

    # Re-instantiate scenario and re-grade
    registry = get_registry()
    try:
        scenario = registry.get_scenario(scenario_id)
    except KeyError:
        console.print(f"[yellow]Warning: scenario {scenario_id} not found, skipping[/yellow]")
        return ScenarioResult(
            scenario_id=scenario_id,
            seed=seed,
            messages=messages,
            total_tokens=total_tokens,
            total_turns=len([m for m in messages if m.role.value == "user"]),
        )

    # Rebuild scenario state from trace
    scenario.setup(seed)

    result = ScenarioResult(
        scenario_id=scenario_id,
        seed=seed,
        messages=messages,
        total_tokens=total_tokens,
        total_turns=len([m for m in messages if m.role.value == "user"]),
    )

    metrics = scenario.grade(result)
    result.metrics = metrics
    result.success = metrics.get("task_success", 0.0) > 0.5

    return result


def replay_and_grade(run_id: str) -> None:
    """Re-grade all traces in a run directory.

    Args:
        run_id: The run ID (directory name under runs/).
    """
    run_path = _runs_dir() / run_id
    if not run_path.exists():
        console.print(f"[red]Run not found: {run_id}[/red]")
        return

    trace_files = sorted(run_path.rglob("*.jsonl"))
    if not trace_files:
        console.print(f"[red]No trace files found in {run_path}[/red]")
        return

    console.print(f"\n[bold]Replaying run: {run_id}[/bold]")
    console.print(f"Found {len(trace_files)} trace files")

    results: list[ScenarioResult] = []
    for tf in trace_files:
        console.print(f"  Replaying {tf.name}...")
        result = replay_single_trace(tf)
        results.append(result)

    # Detect model from first trace metadata
    reader = TraceReader(trace_files[0])
    meta = reader.extract_metadata()
    model = meta.get("model", "unknown")

    # Compute scorecard
    scorecard = compute_scorecard(results, run_id=run_id, model=model)

    # Write scorecard
    scorecard_path = run_path / "scorecard.json"
    with open(scorecard_path, "w") as f:
        f.write(scorecard.model_dump_json(indent=2))

    console.print(f"\n[green]Re-graded scorecard written: {scorecard_path}[/green]")

    # Generate report
    generate_report(run_id, ["md", "csv", "json"])
