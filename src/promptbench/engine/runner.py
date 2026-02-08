"""Core scenario execution engine."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from promptbench.adapters.base import get_adapter
from promptbench.config import BenchConfig
from promptbench.engine.prompter import HumanPrompter, Prompter, ScriptedPrompter
from promptbench.engine.trace import TraceWriter
from promptbench.grading.report import generate_report
from promptbench.grading.scorer import compute_scorecard
from promptbench.scenarios.base import Scenario, ScenarioResult
from promptbench.scenarios.registry import get_registry
from promptbench.types import Message, Response, Role, ToolCallResult

if TYPE_CHECKING:
    from promptbench.adapters.base import ModelClient

console = Console()


def _run_dir() -> Path:
    """Return the base runs directory."""
    return Path("runs")


def run_single_scenario(
    scenario: Scenario,
    client: ModelClient,
    prompter: Prompter,
    seed: int,
    budget_tokens: int,
    budget_turns: int,
    trace_writer: TraceWriter | None = None,
) -> ScenarioResult:
    """Execute a single scenario run with one seed.

    Args:
        scenario: The scenario to run.
        client: Model adapter.
        prompter: Prompter policy (scripted, human, or replay).
        seed: Random seed for this run.
        budget_tokens: Maximum tokens allowed.
        budget_turns: Maximum turns allowed.
        trace_writer: Optional trace writer for recording.

    Returns:
        ScenarioResult with metrics.
    """
    # Initialize scenario
    messages = scenario.setup(seed)
    tools = scenario.get_tools()
    total_tokens = 0
    turn = 0

    # Write initial messages to trace
    if trace_writer:
        trace_writer.write_metadata({
            "scenario_id": scenario.config.id,
            "seed": seed,
            "budget_tokens": budget_tokens,
            "budget_turns": budget_turns,
            "model": client.name,
            "timestamp_start": time.time(),
        })
        for msg in messages:
            trace_writer.write_message(msg)

    while turn < budget_turns and total_tokens < budget_tokens:
        # Get next user message from prompter
        user_text = prompter.next_message(messages, turn, scenario)
        if user_text is None:
            break

        user_msg = Message(role=Role.USER, content=user_text)
        messages.append(user_msg)
        if trace_writer:
            trace_writer.write_message(user_msg)

        # Get model response
        response: Response = client.complete(
            messages=messages,
            tools=tools if tools else None,
            seed=seed,
            temperature=0.0,
        )
        total_tokens += response.usage.total_tokens

        # Process response
        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls if response.tool_calls else None,
        )
        messages.append(assistant_msg)
        if trace_writer:
            trace_writer.write_message(assistant_msg, response.usage)

        # Handle tool calls
        if response.tool_calls:
            for tc in response.tool_calls:
                try:
                    result = scenario.handle_tool_call(tc)
                except Exception as e:
                    result = ToolCallResult(
                        call_id=tc.id,
                        name=tc.name,
                        result=f"Error: {e}",
                        is_error=True,
                    )
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.result,
                    tool_call_id=result.call_id,
                    name=result.name,
                )
                messages.append(tool_msg)
                if trace_writer:
                    trace_writer.write_message(tool_msg)

        turn += 1

        # Check termination
        if scenario.check_termination(messages, turn, total_tokens):
            break

    # Build result
    result = ScenarioResult(
        scenario_id=scenario.config.id,
        seed=seed,
        messages=messages,
        total_tokens=total_tokens,
        total_turns=turn,
    )

    # Grade
    metrics = scenario.grade(result)
    result.metrics = metrics
    result.success = metrics.get("task_success", 0.0) > 0.5

    if trace_writer:
        trace_writer.write_metadata({"timestamp_end": time.time()})
        trace_writer.write_score(metrics)

    return result


def run_batch(config: BenchConfig) -> str:
    """Run a batch of scenarios according to config.

    Returns:
        The run ID.
    """
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_path = _run_dir() / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    registry = get_registry()
    scenario_ids = registry.resolve_scenario_ids(config.scenarios)

    if not scenario_ids:
        console.print("[red]No scenarios matched the given patterns.[/red]")
        return run_id

    client = get_adapter(config.model.name)
    all_results: list[ScenarioResult] = []

    console.print(f"\n[bold]PromptBench Run: {run_id}[/bold]")
    console.print(f"Model: {config.model.name} | Seeds: {config.seeds} | Scenarios: {len(scenario_ids)}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for sid in scenario_ids:
            scenario = registry.get_scenario(sid)
            task = progress.add_task(f"Running {sid}...", total=config.seeds)

            for seed in range(config.seeds):
                trace_path = run_path / f"{sid}_seed{seed}.jsonl"
                trace_writer = TraceWriter(trace_path)

                try:
                    prompter = ScriptedPrompter()
                    result = run_single_scenario(
                        scenario=scenario,
                        client=client,
                        prompter=prompter,
                        seed=seed,
                        budget_tokens=config.budget.tokens,
                        budget_turns=config.budget.turns,
                        trace_writer=trace_writer,
                    )
                    all_results.append(result)
                finally:
                    trace_writer.close()

                progress.advance(task)

    # Compute scorecard
    scorecard = compute_scorecard(all_results, run_id=run_id, model=config.model.name)

    # Write scorecard
    scorecard_path = run_path / "scorecard.json"
    with open(scorecard_path, "w") as f:
        f.write(scorecard.model_dump_json(indent=2))

    # Generate report
    generate_report(run_id, ["md", "csv", "json"])

    console.print(f"\n[green]Run complete: {run_id}[/green]")
    console.print(f"Results in: {run_path}")
    return run_id


def run_human(config: BenchConfig, scenario_id: str) -> None:
    """Run a single scenario in human-in-the-loop mode."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)
    client = get_adapter(config.model.name)

    run_id = f"human_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_path = _run_dir() / run_id
    trace_path = run_path / f"{scenario_id}_human.jsonl"
    trace_writer = TraceWriter(trace_path)

    console.print(f"\n[bold]Interactive Mode: {scenario.config.name}[/bold]")
    console.print(f"[dim]{scenario.config.description}[/dim]")
    console.print(f"[dim]Budget: {config.budget.tokens} tokens, {config.budget.turns} turns[/dim]\n")

    brief = scenario.get_human_brief()
    if brief:
        from rich.panel import Panel

        console.print(Panel(brief, title="[bold]TASK — read before you start[/bold]", border_style="cyan"))
        console.print()

    try:
        prompter = HumanPrompter()
        result = run_single_scenario(
            scenario=scenario,
            client=client,
            prompter=prompter,
            seed=0,
            budget_tokens=config.budget.tokens,
            budget_turns=config.budget.turns,
            trace_writer=trace_writer,
        )

        console.print("\n[bold]Results:[/bold]")
        for k, v in result.metrics.items():
            console.print(f"  {k}: {v:.4f}")

    finally:
        trace_writer.close()
    console.print(f"\nTrace saved: {trace_path}")
