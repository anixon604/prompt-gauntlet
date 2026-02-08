"""Constraint judge: regex/keyword/structured checks."""

from __future__ import annotations

import re
from typing import Any

from promptgauntlet.judges.base import Judge, JudgeScore


class ConstraintJudge(Judge):
    """Deterministic judge using keyword, regex, and structural checks.

    Rubric format:
        required_invariants: list[str] - required concept mentions
        target_keywords: list[str] - keywords to check
        regex_patterns: list[str] - optional regex patterns to match
        min_length: int - optional minimum text length
    """

    @property
    def name(self) -> str:
        return "constraint"

    def score(
        self,
        output: str,
        rubric: dict[str, Any],
        **kwargs: Any,
    ) -> JudgeScore:
        """Score based on deterministic constraint checks."""
        text_lower = output.lower()
        checks_passed = 0
        checks_total = 0
        details: list[str] = []

        # Check required invariants
        invariants = rubric.get("required_invariants", [])
        if invariants:
            for inv in invariants:
                checks_total += 1
                inv_words = inv.lower().split()
                if any(w in text_lower for w in inv_words if len(w) > 3):
                    checks_passed += 1
                    details.append(f"PASS: invariant '{inv}'")
                else:
                    details.append(f"FAIL: invariant '{inv}'")

        # Check keywords
        keywords = rubric.get("target_keywords", [])
        if keywords:
            for kw in keywords:
                checks_total += 1
                if kw.lower() in text_lower:
                    checks_passed += 1
                    details.append(f"PASS: keyword '{kw}'")
                else:
                    details.append(f"FAIL: keyword '{kw}'")

        # Check regex patterns
        patterns = rubric.get("regex_patterns", [])
        if patterns:
            for pattern in patterns:
                checks_total += 1
                if re.search(pattern, output, re.IGNORECASE):
                    checks_passed += 1
                    details.append(f"PASS: regex '{pattern}'")
                else:
                    details.append(f"FAIL: regex '{pattern}'")

        # Check minimum length
        min_length = rubric.get("min_length", 0)
        if min_length > 0:
            checks_total += 1
            if len(output) >= min_length:
                checks_passed += 1
                details.append(f"PASS: length >= {min_length}")
            else:
                details.append(f"FAIL: length {len(output)} < {min_length}")

        score = checks_passed / max(checks_total, 1)
        return JudgeScore(
            judge_name=self.name,
            score=score,
            rationale="; ".join(details[:10]),  # Limit rationale length
            metadata={
                "checks_passed": checks_passed,
                "checks_total": checks_total,
            },
        )
