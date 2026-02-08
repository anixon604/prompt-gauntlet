"""LLM-as-judge with rubric scoring and calibration support."""

from __future__ import annotations

import json
from typing import Any

from promptgauntlet.judges.base import Judge, JudgeScore


class RubricJudge(Judge):
    """LLM-as-judge that scores output against a rubric.

    Uses a model adapter to evaluate quality. Includes:
    - Configurable prompt template
    - Calibration mode (score known examples first)
    - Structured output parsing

    Rubric format:
        description: str - what to evaluate
        criteria: list[str] - specific criteria to check
        rubric_text: str - full rubric text (optional, overrides criteria)
    """

    def __init__(
        self,
        model_client: Any = None,
        prompt_template: str | None = None,
    ) -> None:
        self._client = model_client
        self._template = prompt_template or (
            "You are an expert evaluator. Score the following output against the rubric.\n\n"
            "## Rubric\n{rubric}\n\n"
            "## Output to Evaluate\n{output}\n\n"
            "Respond with a JSON object containing:\n"
            '- "score": a float from 0.0 (worst) to 1.0 (best)\n'
            '- "rationale": a brief explanation of the score\n\n'
            "JSON response:"
        )
        self._calibration_scores: list[tuple[float, float]] = []

    @property
    def name(self) -> str:
        return "rubric"

    def calibrate(
        self,
        examples: list[tuple[str, float]],
        rubric: dict[str, Any],
    ) -> None:
        """Run calibration by scoring known examples.

        Args:
            examples: List of (text, expected_score) tuples.
            rubric: The rubric to calibrate against.
        """
        self._calibration_scores = []
        for text, expected in examples:
            result = self.score(text, rubric)
            self._calibration_scores.append((expected, result.score))

    def score(
        self,
        output: str,
        rubric: dict[str, Any],
        **kwargs: Any,
    ) -> JudgeScore:
        """Score output using LLM judge."""
        if self._client is None:
            # No model client: use deterministic fallback
            return self._deterministic_score(output, rubric)

        # Build rubric text
        rubric_text = rubric.get("rubric_text", "")
        if not rubric_text:
            desc = rubric.get("description", "Evaluate quality")
            criteria = rubric.get("criteria", rubric.get("required_invariants", []))
            rubric_text = f"{desc}\n\nCriteria:\n" + "\n".join(
                f"- {c}" for c in criteria
            )

        # Build prompt
        prompt = self._template.format(rubric=rubric_text, output=output[:3000])

        from promptgauntlet.types import Message, Role

        messages = [Message(role=Role.USER, content=prompt)]
        response = self._client.complete(messages)

        # Parse structured response
        try:
            # Try to parse JSON from response
            content = response.content.strip()
            # Handle markdown code blocks
            if "```" in content:
                import re
                json_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
                if json_match:
                    content = json_match.group(1).strip()

            data = json.loads(content)
            raw_score = float(data.get("score", 0.5))
            rationale = str(data.get("rationale", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback: try to extract a number
            import re
            numbers = re.findall(r"0?\.\d+|\d+\.?\d*", response.content)
            raw_score = min(1.0, max(0.0, float(numbers[0]))) if numbers else 0.5
            rationale = response.content[:200]

        score = max(0.0, min(1.0, raw_score))
        return JudgeScore(
            judge_name=self.name,
            score=score,
            rationale=rationale,
            metadata={"raw_score": raw_score, "model": getattr(self._client, "name", "unknown")},
        )

    def _deterministic_score(
        self, output: str, rubric: dict[str, Any]
    ) -> JudgeScore:
        """Deterministic fallback when no model is available.

        Scores based on criteria coverage in the output text.
        """
        criteria = rubric.get("criteria", rubric.get("required_invariants", []))
        if not criteria:
            return JudgeScore(
                judge_name=self.name,
                score=0.5,
                rationale="No criteria in rubric; default score.",
                metadata={"deterministic": True},
            )

        text_lower = output.lower()
        matched = 0
        for criterion in criteria:
            words = criterion.lower().split()
            if any(w in text_lower for w in words if len(w) > 3):
                matched += 1

        score = matched / len(criteria)
        return JudgeScore(
            judge_name=self.name,
            score=score,
            rationale=f"Deterministic: {matched}/{len(criteria)} criteria matched",
            metadata={"deterministic": True, "matched": matched, "total": len(criteria)},
        )
