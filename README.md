# PromptBench

A reproducible, model-agnostic **Prompt Aptitude Test Suite** that measures multi-turn prompting skill across multiple task families. Runs the same scenarios against different LLM providers/models and produces a comparable multi-objective scorecard.

## Quickstart

```bash
# Install with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run all scenarios with the mock model (fully offline)
promptbench run --model mock --scenarios all --seeds 3

# List available scenarios
promptbench list

# Re-grade a previous run
promptbench grade --run <run_id>

# Generate report
promptbench report --run <run_id>
```

## Architecture

```
src/promptbench/
  cli.py          # CLI entrypoint (click)
  types.py        # Shared Pydantic models
  config.py       # YAML config loader
  adapters/       # Model adapters (mock, OpenAI, local)
  engine/         # Execution engine (runner, trace, replay, prompter)
  scenarios/      # Task families (classification, constraint, tool_use, convergence)
  tools/          # Simulated tools (calculator, search, file_store)
  judges/         # Evaluation judges (embedding, rubric, constraint, ensemble)
  grading/        # Scoring, statistics, Pareto ranking, report generation
```

## Task Families

### A) Classification (Prompting-in-the-Dark)
Classify text snippets by sentiment without seeing ground-truth labels. The prompter must elicit a consistent labeling policy from the model. Scored by accuracy on a held-out set within a token/turn budget.

### B) Constraint Satisfaction (JSON Schema)
Produce JSON matching a given schema, extract fields, follow exact formatting constraints. Validated deterministically with jsonschema + custom checks.

### C) Tool Use (Multi-Turn)
Multi-step problems requiring search, calculator, and file store tools. Validates final answer against ground truth and tracks tool-call correctness.

### D) Convergence (Hidden-Target Manifold)
Converge toward a hidden target defined by a rubric with required invariants. Evaluated by multiple judges (embedding similarity, rubric LLM-judge, constraint checks) aggregated via ensemble.

## CLI Reference

```
promptbench list [--family <name>]
    List available scenarios, optionally filtered by task family.

promptbench run --model <name> --scenarios <glob|all> --seeds <N>
               [--budget-tokens <N>] [--budget-turns <N>]
               [--config <path>] [--temperature <float>]
    Run scenarios and produce scorecard + report in runs/<run_id>/.

promptbench human --scenario <id> --model <name> [--config <path>]
    Interactive human-in-the-loop mode.

promptbench grade --run <run_id>
    Re-grade from stored traces (deterministic replay).

promptbench report --run <run_id> [--format md,csv,json]
    Generate report artifacts.
```

## Configuration

Default config in `configs/default.yaml`. Override with `--config <path>` or CLI flags.

```yaml
model:
  name: mock
  temperature: 0.0
budget:
  tokens: 10000
  turns: 20
seeds: 3
scoring:
  family_weights:
    classification: 0.25
    constraint: 0.25
    tool_use: 0.25
    convergence: 0.25
```

## How to Add a Scenario

1. Create a new directory under `src/promptbench/scenarios/<family>/`
2. Implement a class extending `Scenario` (see `scenarios/base.py`):
   ```python
   from promptbench.scenarios.base import Scenario, ScenarioResult
   from promptbench.scenarios.registry import register_scenario

   @register_scenario
   class MyScenario(Scenario):
       @property
       def config(self) -> ScenarioConfig:
           return ScenarioConfig(
               id="family/my_scenario",
               family=TaskFamily.CONSTRAINT,
               name="My Scenario",
               description="...",
           )

       def setup(self, seed: int) -> list[Message]: ...
       def get_tools(self) -> list[ToolSchema]: ...
       def handle_tool_call(self, call) -> ToolCallResult: ...
       def check_termination(self, messages, turn, tokens) -> bool: ...
       def grade(self, result: ScenarioResult) -> dict[str, float]: ...
   ```
3. Add a unit test in `tests/`
4. Register import in `scenarios/registry.py` `_ensure_loaded()`

## Scoring

Multi-objective metrics per scenario:
- **task_success**: Primary success indicator (0 or 1)
- **accuracy / pass_rate**: Domain-specific quality metric
- **efficiency**: Token cost normalized
- **recovery_rate**: Ability to recover from errors
- **robustness**: Consistency across seeds (via median + p10)

Output formats: `scorecard.json`, `scorecard.csv`, `report.md`

Pareto ranking and configurable weighted scoring available.

## Judge Variance

The rubric LLM-judge introduces non-determinism. To mitigate:
- Ensemble with deterministic judges (constraint, embedding)
- Disagreement penalty reduces score when judges disagree
- Calibration mode available to baseline judge behavior
- Mock model provides fully deterministic judge responses

## Development

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## License

MIT
