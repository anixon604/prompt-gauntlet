"""CLI entrypoint for PromptBench."""

from __future__ import annotations

import click

from promptbench import __version__


@click.group()
@click.version_option(version=__version__, prog_name="promptbench")
def main() -> None:
    """PromptBench: A reproducible, model-agnostic Prompt Aptitude Test Suite."""


@main.command("list")
@click.option("--family", "-f", default=None, help="Filter by task family.")
def list_scenarios(family: str | None) -> None:
    """List available scenarios and task families."""
    from promptbench.scenarios.registry import get_registry

    registry = get_registry()
    scenarios = registry.list_scenarios(family=family)
    if not scenarios:
        click.echo("No scenarios found.")
        return
    current_family = None
    for s in scenarios:
        if s.family.value != current_family:
            current_family = s.family.value
            click.echo(f"\n[{current_family.upper()}]")
        click.echo(f"  {s.id:<40} {s.name}")
        if s.description:
            click.echo(f"    {s.description}")


@main.command("run")
@click.option("--model", "-m", default="mock", help="Model adapter name.")
@click.option(
    "--scenarios", "-s", default="all", help="Scenario glob or 'all'."
)
@click.option("--seeds", default=3, type=int, help="Number of random seeds.")
@click.option("--budget-tokens", default=None, type=int, help="Token budget per scenario.")
@click.option("--budget-turns", default=None, type=int, help="Turn budget per scenario.")
@click.option("--config", "config_path", default=None, type=click.Path(), help="YAML config path.")
@click.option("--temperature", default=None, type=float, help="Model temperature.")
def run(
    model: str,
    scenarios: str,
    seeds: int,
    budget_tokens: int | None,
    budget_turns: int | None,
    config_path: str | None,
    temperature: float | None,
) -> None:
    """Run scenarios against a model and produce a scorecard."""
    from promptbench.config import load_config, merge_cli_overrides
    from promptbench.engine.runner import run_batch

    cfg = load_config(config_path)
    cfg = merge_cli_overrides(
        cfg,
        model=model,
        seeds=seeds,
        budget_tokens=budget_tokens,
        budget_turns=budget_turns,
        scenarios=[scenarios] if scenarios != "all" else ["all"],
        temperature=temperature,
    )
    run_batch(cfg)


@main.command("human")
@click.option("--scenario", "-s", required=True, help="Scenario ID.")
@click.option("--model", "-m", default="mock", help="Model adapter name.")
@click.option("--config", "config_path", default=None, type=click.Path(), help="YAML config path.")
def human(scenario: str, model: str, config_path: str | None) -> None:
    """Run a scenario in human-in-the-loop mode."""
    from promptbench.config import load_config, merge_cli_overrides
    from promptbench.engine.runner import run_human

    cfg = load_config(config_path)
    cfg = merge_cli_overrides(cfg, model=model)
    run_human(cfg, scenario)


@main.command("grade")
@click.option("--run", "run_id", required=True, help="Run ID to re-grade.")
def grade(run_id: str) -> None:
    """Re-grade an existing run from traces."""
    from promptbench.engine.replay import replay_and_grade

    replay_and_grade(run_id)


@main.command("report")
@click.option("--run", "run_id", required=True, help="Run ID to report on.")
@click.option(
    "--format",
    "formats",
    default="md,csv,json",
    help="Output formats (comma-separated: md,csv,json).",
)
def report(run_id: str, formats: str) -> None:
    """Generate report artifacts for a run."""
    from promptbench.grading.report import generate_report

    fmt_list = [f.strip() for f in formats.split(",")]
    generate_report(run_id, fmt_list)
