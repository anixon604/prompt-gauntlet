"""Configuration loading and validation for PromptGauntlet."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model adapter configuration."""

    name: str = "mock"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096


class BudgetConfig(BaseModel):
    """Token and turn budget configuration."""

    tokens: int = 10000
    turns: int = 20


class JudgeConfig(BaseModel):
    """Judge configuration."""

    rubric_model: str = "mock"
    rubric_prompt_template: str = (
        "Score the following output against the rubric.\n"
        "Rubric: {rubric}\n"
        "Output: {output}\n"
        "Score (0.0-1.0):"
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "embedding": 0.3,
            "rubric": 0.4,
            "constraint": 0.3,
        }
    )
    disagreement_penalty: float = 0.1


class ScoringConfig(BaseModel):
    """Scoring weights configuration."""

    family_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "classification": 0.25,
            "constraint": 0.25,
            "tool_use": 0.25,
            "convergence": 0.25,
        }
    )
    metric_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "task_success": 0.5,
            "efficiency": 0.2,
            "robustness": 0.15,
            "recovery": 0.15,
        }
    )


class BenchConfig(BaseModel):
    """Top-level PromptGauntlet configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    judges: JudgeConfig = Field(default_factory=JudgeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    scenarios: list[str] = Field(default_factory=lambda: ["all"])
    seeds: int = 3


def _default_config_path() -> Path:
    """Return path to the bundled default config."""
    # Try package data first, fall back to relative path
    try:
        pkg = importlib.resources.files("promptgauntlet")
        cfg = pkg.joinpath("..", "..", "configs", "default.yaml")
        p = Path(str(cfg))
        if p.exists():
            return p
    except Exception:
        pass
    # Fall back to workspace-relative
    return Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> BenchConfig:
    """Load configuration from YAML, falling back to defaults.

    Args:
        path: Optional path to a YAML config file.

    Returns:
        Validated BenchConfig instance.
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
    else:
        default = _default_config_path()
        if default.exists():
            with open(default) as f:
                data = yaml.safe_load(f) or {}
    return BenchConfig(**data)


def merge_cli_overrides(config: BenchConfig, **overrides: Any) -> BenchConfig:
    """Merge CLI flag overrides into an existing config.

    Only non-None overrides are applied.
    """
    updates: dict[str, Any] = {}
    if overrides.get("model"):
        updates["model"] = ModelConfig(name=overrides["model"])
    if overrides.get("seeds") is not None:
        updates["seeds"] = overrides["seeds"]
    if overrides.get("budget_tokens") is not None:
        budget = config.budget.model_copy()
        budget.tokens = overrides["budget_tokens"]
        updates["budget"] = budget
    if overrides.get("budget_turns") is not None:
        budget = updates.get("budget", config.budget.model_copy())
        if isinstance(budget, dict):
            budget = BudgetConfig(**budget)
        budget.turns = overrides["budget_turns"]
        updates["budget"] = budget
    if overrides.get("scenarios"):
        updates["scenarios"] = overrides["scenarios"]
    if overrides.get("temperature") is not None:
        model_cfg = updates.get("model", config.model.model_copy())
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        model_cfg.temperature = overrides["temperature"]
        updates["model"] = model_cfg
    if updates:
        return config.model_copy(update=updates)
    return config
