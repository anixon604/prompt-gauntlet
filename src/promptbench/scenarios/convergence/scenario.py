"""Hidden-target / latent-manifold convergence scenario."""

from __future__ import annotations

from typing import Any

from promptbench.scenarios.base import Scenario, ScenarioResult, ScriptedPolicy
from promptbench.scenarios.registry import register_scenario
from promptbench.types import (
    Message,
    Role,
    ScenarioConfig,
    TaskFamily,
    ToolCallRequest,
    ToolCallResult,
    ToolSchema,
)

# Default rubric for the error handling convergence target
_DEFAULT_RUBRIC = {
    "description": "Design a comprehensive error handling system",
    "required_invariants": [
        "input validation",
        "exception hierarchy",
        "logging",
        "graceful degradation",
        "user-friendly error messages",
    ],
    "bonus_concepts": [
        "circuit breaker",
        "retry logic",
        "exponential backoff",
        "correlation id",
        "structured logging",
        "fallback",
    ],
    "min_invariants_for_pass": 4,
    "target_keywords": [
        "validation",
        "exception",
        "log",
        "degrad",
        "error message",
        "retry",
        "circuit",
        "fallback",
    ],
}


def check_invariants(text: str, rubric: dict[str, Any]) -> tuple[int, int, list[str]]:
    """Check how many required invariants are present in the text.

    Returns:
        (matched_count, total_required, list of matched invariant names)
    """
    text_lower = text.lower()
    required = rubric.get("required_invariants", [])
    matched: list[str] = []
    for inv in required:
        # Check if the invariant concept is present
        inv_lower = inv.lower()
        # Split into words and check if all key words appear
        key_words = inv_lower.split()
        if all(w in text_lower for w in key_words):
            matched.append(inv)
        elif any(w in text_lower for w in key_words if len(w) > 3):
            # Partial match for longer words
            matched.append(inv)
    return len(matched), len(required), matched


def check_keywords(text: str, rubric: dict[str, Any]) -> float:
    """Check keyword coverage in the text.

    Returns:
        Fraction of target keywords found.
    """
    text_lower = text.lower()
    keywords = rubric.get("target_keywords", [])
    if not keywords:
        return 1.0
    found = sum(1 for kw in keywords if kw in text_lower)
    return found / len(keywords)


def check_bonus_concepts(text: str, rubric: dict[str, Any]) -> float:
    """Check bonus concept coverage.

    Returns:
        Fraction of bonus concepts found.
    """
    text_lower = text.lower()
    bonus = rubric.get("bonus_concepts", [])
    if not bonus:
        return 0.0
    found = sum(1 for b in bonus if b.lower() in text_lower)
    return found / len(bonus)


class ConvergencePolicy(ScriptedPolicy):
    """Scripted baseline: iterative refinement prompting."""

    def __init__(self) -> None:
        self._phase = 0

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str:
        if turn == 0:  # noqa: SIM116
            return (
                "Describe a comprehensive error handling system for a "
                "production web application. Include all major components "
                "and best practices."
            )
        elif turn == 1:
            return (
                "Good start. Now expand on each point: specifically address "
                "input validation strategies, exception hierarchy design, "
                "and logging architecture with structured formats."
            )
        elif turn == 2:
            return (
                "Almost there. Add details about: graceful degradation "
                "patterns, circuit breakers for external services, "
                "retry logic with exponential backoff, and how to "
                "generate user-friendly error messages while hiding "
                "internal details. Also mention correlation IDs for "
                "request tracing."
            )
        else:
            return ""  # Done after 3 refinement rounds


@register_scenario
class ConvergenceScenario(Scenario):
    """Hidden-target convergence scenario.

    Target is defined as a rubric with required invariants.
    Multiple acceptable endpoints. Evaluated by:
    - Invariant coverage (deterministic)
    - Keyword coverage (deterministic)
    - Bonus concept coverage (deterministic)
    - Judge ensemble (when judges available)
    """

    def __init__(self) -> None:
        self._rubric = _DEFAULT_RUBRIC
        self._seed = 0

    @property
    def config(self) -> ScenarioConfig:
        return ScenarioConfig(
            id="convergence/error_handling",
            family=TaskFamily.CONVERGENCE,
            name="Error Handling System Design",
            description=(
                "Converge on a comprehensive error handling system design "
                "through iterative refinement. Evaluated against a rubric "
                "with required invariants and bonus concepts."
            ),
            budget_tokens=8000,
            budget_turns=10,
        )

    def setup(self, seed: int) -> list[Message]:
        self._seed = seed
        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a software architecture expert. When asked about "
                    "system design topics, provide comprehensive, detailed "
                    "responses covering all important aspects. Build on "
                    "feedback to converge toward a complete solution."
                ),
            )
        ]

    def get_tools(self) -> list[ToolSchema]:
        return []

    def handle_tool_call(self, call: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            call_id=call.id,
            name=call.name,
            result="Error: No tools available.",
            is_error=True,
        )

    def check_termination(
        self, messages: list[Message], turn: int, tokens: int
    ) -> bool:
        # Let the scripted prompter decide
        user_msgs = [m for m in messages if m.role == Role.USER]
        return len(user_msgs) >= 4

    def grade(self, result: ScenarioResult) -> dict[str, float]:
        """Grade convergence quality using deterministic checks + optional judges."""
        # Get the final (best) assistant message
        best_text = ""
        best_score = -1.0
        for msg in result.messages:
            if msg.role == Role.ASSISTANT:
                matched, total, _ = check_invariants(msg.content, self._rubric)
                score = matched / max(total, 1)
                if score >= best_score:
                    best_score = score
                    best_text = msg.content

        if not best_text:
            return {
                "task_success": 0.0,
                "invariant_coverage": 0.0,
                "keyword_coverage": 0.0,
                "bonus_coverage": 0.0,
                "convergence_rate": 0.0,
                "efficiency": 0.0,
            }

        # Invariant coverage
        matched, total, matched_names = check_invariants(best_text, self._rubric)
        invariant_coverage = matched / max(total, 1)

        # Keyword coverage
        keyword_cov = check_keywords(best_text, self._rubric)

        # Bonus coverage
        bonus_cov = check_bonus_concepts(best_text, self._rubric)

        # Convergence rate: how much did the output improve over turns?
        assistant_msgs = [
            m for m in result.messages if m.role == Role.ASSISTANT
        ]
        if len(assistant_msgs) >= 2:
            first_matched, _, _ = check_invariants(
                assistant_msgs[0].content, self._rubric
            )
            first_score = first_matched / max(total, 1)
            convergence_rate = max(0.0, best_score - first_score)
        else:
            convergence_rate = 0.0

        # Task success: need minimum invariants
        min_req = self._rubric.get("min_invariants_for_pass", 4)
        task_success = 1.0 if matched >= min_req else 0.0

        # Efficiency
        efficiency = max(0.0, 1.0 - result.total_tokens / 4000.0)

        return {
            "task_success": task_success,
            "invariant_coverage": invariant_coverage,
            "keyword_coverage": keyword_cov,
            "bonus_coverage": bonus_cov,
            "convergence_rate": convergence_rate,
            "efficiency": efficiency,
            "invariants_matched": float(matched),
            "invariants_total": float(total),
        }

    def get_human_brief(self) -> str | None:
        required = ", ".join(_DEFAULT_RUBRIC["required_invariants"])
        bonus = ", ".join(_DEFAULT_RUBRIC["bonus_concepts"])
        return (
            "OBJECTIVE: Get the model to produce a short design document for a "
            "comprehensive error-handling system (e.g. for a web app).\n\n"
            "REQUIRED (must cover at least 4 of these): " + required + ".\n\n"
            "BONUS (improves score): " + bonus + ".\n\n"
            "HOW: Describe the task in your first message (e.g. 'Describe a comprehensive "
            "error handling system...'). If the first response is thin, follow up with "
            "'Expand on X' or 'Add details about Y' until the output covers enough of the "
            "required concepts. Success = at least 4 required invariants clearly present "
            "in the model's final output."
        )

    def get_scripted_prompter_policy(self) -> ScriptedPolicy:
        return ConvergencePolicy()
