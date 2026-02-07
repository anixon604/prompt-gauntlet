"""BM25-style search tool over a local JSONL corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from promptbench.tools.base import Tool


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Simple BM25 index for search over documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter[str]] = []
        self.avg_dl: float = 0.0
        self.idf: dict[str, float] = {}

    def add_documents(self, documents: list[dict[str, Any]], text_field: str = "text") -> None:
        """Index a list of documents."""
        self.docs = documents
        self.doc_tokens = []
        self.doc_freqs = []

        # Tokenize all docs
        for doc in documents:
            text = str(doc.get(text_field, ""))
            tokens = _tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_freqs.append(Counter(tokens))

        # Compute average document length
        total_tokens = sum(len(t) for t in self.doc_tokens)
        self.avg_dl = total_tokens / len(documents) if documents else 1.0

        # Compute IDF
        n = len(documents)
        df: Counter[str] = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                df[term] += 1

        self.idf = {}
        for term, count in df.items():
            self.idf[term] = math.log((n - count + 0.5) / (count + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict[str, Any], float]]:
        """Search the index and return top-k results with scores."""
        query_tokens = _tokenize(query)
        scores: list[float] = []

        for i, _doc in enumerate(self.docs):
            score = 0.0
            dl = len(self.doc_tokens[i])
            freq = self.doc_freqs[i]

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = freq.get(token, 0)
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                score += idf * numerator / denominator

            scores.append(score)

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[tuple[dict[str, Any], float]] = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append((self.docs[idx], score))

        return results


class SearchTool(Tool):
    """Search over a local JSONL corpus using BM25."""

    def __init__(self, corpus_path: Path | None = None) -> None:
        self._index = BM25Index()
        self._loaded = False
        self._corpus_path = corpus_path

    def load_corpus(self, path: Path | None = None) -> None:
        """Load a JSONL corpus into the search index."""
        p = path or self._corpus_path
        if p is None:
            # Use default bundled corpus
            p = Path(__file__).resolve().parent.parent.parent.parent / "data" / "tool_use" / "corpus.jsonl"
        if not p.exists():
            # Create a minimal default corpus
            self._index.add_documents(self._default_corpus())
            self._loaded = True
            return

        docs: list[dict[str, Any]] = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        self._index.add_documents(docs)
        self._loaded = True

    def _default_corpus(self) -> list[dict[str, Any]]:
        """Minimal built-in corpus for offline testing."""
        return [
            {"id": "1", "text": "Springfield is a city in Illinois with a population of 116,250 as of 2020.", "title": "Springfield, IL"},
            {"id": "2", "text": "The GDP of Springfield IL is approximately 7.6 billion dollars.", "title": "Springfield Economy"},
            {"id": "3", "text": "Python is a programming language created by Guido van Rossum in 1991.", "title": "Python"},
            {"id": "4", "text": "The speed of light is approximately 299,792,458 meters per second.", "title": "Speed of Light"},
            {"id": "5", "text": "The Earth's circumference is approximately 40,075 kilometers.", "title": "Earth"},
            {"id": "6", "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "title": "Water"},
            {"id": "7", "text": "The average human body temperature is 37 degrees Celsius or 98.6 degrees Fahrenheit.", "title": "Body Temperature"},
            {"id": "8", "text": "Mount Everest is 8,849 meters tall, making it the tallest mountain on Earth.", "title": "Mount Everest"},
            {"id": "9", "text": "The Amazon River is approximately 6,400 kilometers long.", "title": "Amazon River"},
            {"id": "10", "text": "Tokyo has a population of approximately 13.96 million people.", "title": "Tokyo"},
        ]

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search a knowledge corpus for relevant information. Returns top matching documents."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Search the corpus."""
        if not self._loaded:
            self.load_corpus()

        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 3)

        if not query:
            raise ValueError("Missing 'query' argument")

        results = self._index.search(query, top_k=top_k)
        if not results:
            return "No results found."

        parts: list[str] = []
        for i, (doc, score) in enumerate(results, 1):
            title = doc.get("title", "Untitled")
            text = doc.get("text", "")
            parts.append(f"[{i}] {title} (score: {score:.2f})\n{text}")

        return "\n\n".join(parts)
