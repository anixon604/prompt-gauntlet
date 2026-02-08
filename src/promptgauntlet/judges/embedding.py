"""Embedding similarity judge using sentence-transformers."""

from __future__ import annotations

from typing import Any

from promptgauntlet.judges.base import Judge, JudgeScore


class EmbeddingJudge(Judge):
    """Judge using embedding cosine similarity.

    Uses sentence-transformers for encoding. Lazy imports to avoid
    requiring the dependency when not used.

    Rubric format:
        reference_text: str - text to compare against
        threshold: float - minimum similarity for full score (default 0.8)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingJudge. "
                    "Install with: pip install sentence-transformers"
                ) from exc
        return self._model

    @property
    def name(self) -> str:
        return "embedding"

    def score(
        self,
        output: str,
        rubric: dict[str, Any],
        **kwargs: Any,
    ) -> JudgeScore:
        """Score based on embedding similarity to reference text."""
        reference = rubric.get("reference_text", "")
        if not reference:
            # Fall back to a keyword-based similarity estimate
            return self._fallback_score(output, rubric)

        try:
            model = self._load_model()
            import numpy as np

            embeddings = model.encode([output, reference])
            # Cosine similarity
            sim = float(
                np.dot(embeddings[0], embeddings[1])
                / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8)
            )
            sim = max(0.0, min(1.0, sim))

            threshold = rubric.get("threshold", 0.8)
            # Scale: below threshold maps to 0-0.5, above maps to 0.5-1.0
            if sim >= threshold:
                scaled = 0.5 + 0.5 * (sim - threshold) / (1.0 - threshold + 1e-8)
            else:
                scaled = 0.5 * sim / (threshold + 1e-8)

            return JudgeScore(
                judge_name=self.name,
                score=scaled,
                rationale=f"Cosine similarity: {sim:.4f} (threshold: {threshold})",
                metadata={"raw_similarity": sim, "threshold": threshold},
            )
        except ImportError:
            return self._fallback_score(output, rubric)

    def _fallback_score(self, output: str, rubric: dict[str, Any]) -> JudgeScore:
        """Fallback when sentence-transformers is not available.

        Uses simple word overlap as a rough proxy.
        """
        reference = rubric.get("reference_text", "")
        if not reference:
            # Use keywords from rubric
            keywords = rubric.get("target_keywords", [])
            if keywords:
                text_lower = output.lower()
                found = sum(1 for kw in keywords if kw.lower() in text_lower)
                score = found / len(keywords)
            else:
                score = 0.5  # No reference available
            return JudgeScore(
                judge_name=self.name,
                score=score,
                rationale="Fallback: keyword matching (sentence-transformers not available)",
                metadata={"fallback": True},
            )

        # Simple Jaccard similarity on word sets
        output_words = set(output.lower().split())
        ref_words = set(reference.lower().split())
        if not output_words or not ref_words:
            return JudgeScore(
                judge_name=self.name,
                score=0.0,
                rationale="Empty text",
                metadata={"fallback": True},
            )
        intersection = output_words & ref_words
        union = output_words | ref_words
        jaccard = len(intersection) / len(union)
        return JudgeScore(
            judge_name=self.name,
            score=jaccard,
            rationale=f"Fallback Jaccard similarity: {jaccard:.4f}",
            metadata={"fallback": True, "jaccard": jaccard},
        )
