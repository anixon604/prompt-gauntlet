"""Base judge protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """Score from a single judge."""

    judge_name: str
    score: float  # 0.0 to 1.0
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Judge(ABC):
    """Abstract base class for evaluation judges."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Judge name."""

    @abstractmethod
    def score(
        self,
        output: str,
        rubric: dict[str, Any],
        **kwargs: Any,
    ) -> JudgeScore:
        """Score an output against a rubric.

        Args:
            output: The text to evaluate.
            rubric: Rubric definition with criteria.
            **kwargs: Extra context (e.g., reference text).

        Returns:
            JudgeScore with score, rationale, and metadata.
        """
