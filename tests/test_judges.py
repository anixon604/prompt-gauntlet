"""Tests for judge implementations."""

from __future__ import annotations

from promptbench.judges.constraint import ConstraintJudge
from promptbench.judges.embedding import EmbeddingJudge
from promptbench.judges.ensemble import EnsembleJudge
from promptbench.judges.rubric import RubricJudge


class TestConstraintJudge:
    """Tests for the deterministic constraint judge."""

    def test_keyword_matching(self) -> None:
        rubric = {"target_keywords": ["python", "programming", "code"]}
        judge = ConstraintJudge()
        result = judge.score("Python is a programming language for writing code.", rubric)
        assert result.score == 1.0

    def test_partial_match(self) -> None:
        rubric = {"target_keywords": ["python", "java", "rust"]}
        judge = ConstraintJudge()
        result = judge.score("Python is great.", rubric)
        assert 0.0 < result.score < 1.0

    def test_no_match(self) -> None:
        rubric = {"target_keywords": ["quantum", "physics"]}
        judge = ConstraintJudge()
        result = judge.score("Hello world.", rubric)
        assert result.score == 0.0

    def test_invariant_checking(self) -> None:
        rubric = {"required_invariants": ["input validation", "error handling"]}
        judge = ConstraintJudge()
        result = judge.score(
            "The system uses input validation and error handling.", rubric
        )
        assert result.score > 0.5


class TestRubricJudge:
    """Tests for the rubric (LLM-as-judge) implementation."""

    def test_deterministic_fallback(self) -> None:
        """Without a model, rubric judge uses deterministic scoring."""
        judge = RubricJudge(model_client=None)
        rubric = {"required_invariants": ["logging", "validation"]}
        result = judge.score("We use logging and input validation.", rubric)
        assert result.score > 0.0
        assert result.metadata.get("deterministic") is True

    def test_no_criteria_returns_default(self) -> None:
        """Empty rubric returns a default score."""
        judge = RubricJudge(model_client=None)
        result = judge.score("Some text.", {})
        assert result.score == 0.5


class TestEnsembleJudge:
    """Tests for ensemble judge aggregation."""

    def test_ensemble_aggregation(self) -> None:
        """Ensemble combines multiple judge scores."""
        constraint = ConstraintJudge()
        rubric_judge = RubricJudge(model_client=None)

        ensemble = EnsembleJudge(
            judges=[constraint, rubric_judge],
            weights={"constraint": 0.5, "rubric": 0.5},
            disagreement_penalty=0.1,
        )

        rubric = {
            "target_keywords": ["python", "programming"],
            "required_invariants": ["coding", "development"],
        }
        result = ensemble.score("Python is a programming language for coding.", rubric)
        assert 0.0 <= result.score <= 1.0
        assert "per_judge" in result.metadata

    def test_disagreement_penalty_applied(self) -> None:
        """Disagreement penalty reduces score when judges disagree."""
        constraint = ConstraintJudge()
        rubric_judge = RubricJudge(model_client=None)

        # High penalty
        ensemble = EnsembleJudge(
            judges=[constraint, rubric_judge],
            weights={"constraint": 0.5, "rubric": 0.5},
            disagreement_penalty=1.0,
        )

        rubric = {
            "target_keywords": ["python"],
            "required_invariants": ["nonexistent_concept"],
        }
        result = ensemble.score("Python is great.", rubric)
        assert result.metadata["penalty"] > 0

    def test_embedding_fallback(self) -> None:
        """Embedding judge works in fallback mode without sentence-transformers."""
        judge = EmbeddingJudge()
        rubric = {"target_keywords": ["error", "handling"]}
        result = judge.score("Error handling is important.", rubric)
        assert result.score > 0.0
