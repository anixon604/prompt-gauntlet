"""Classification scenario: prompting-in-the-dark labeling."""

from __future__ import annotations

import json
import random
from pathlib import Path

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


def _load_dataset() -> list[dict[str, str]]:
    """Load the built-in sentiment classification dataset."""
    data_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "data"
        / "classification"
        / "sentiment.jsonl"
    )
    if data_path.exists():
        docs: list[dict[str, str]] = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs
    # Fall back to built-in minimal dataset
    return _builtin_dataset()


def _builtin_dataset() -> list[dict[str, str]]:
    """Generate a minimal synthetic sentiment dataset."""
    templates = {
        "positive": [
            "I absolutely loved this product, it exceeded all my expectations!",
            "What a wonderful experience, I would highly recommend it.",
            "The service was outstanding and the staff were incredibly helpful.",
            "This is the best purchase I've made this year, truly amazing quality.",
            "I'm so happy with the results, everything worked perfectly.",
            "Fantastic quality and great value for money.",
            "This made my day so much better, truly a delight.",
            "Exceeded my expectations in every way possible.",
            "The team did an incredible job, very professional.",
            "I can't say enough good things about this product.",
        ],
        "negative": [
            "Terrible experience, I would never recommend this to anyone.",
            "The quality was extremely poor and not worth the money.",
            "I'm very disappointed with the service, it was awful.",
            "This product broke after just one day of use, completely useless.",
            "Worst purchase I've ever made, total waste of money.",
            "The customer support was unhelpful and rude.",
            "Nothing worked as advertised, very frustrating.",
            "I regret buying this, it's a complete disaster.",
            "The food was cold and tasteless, terrible restaurant.",
            "Absolutely horrible quality, do not buy this.",
        ],
        "neutral": [
            "The product arrived on time and was as described.",
            "It's an average product, nothing special but it works.",
            "The service was okay, neither great nor terrible.",
            "I received my order today, it matches the description.",
            "It's a standard item that does what it's supposed to do.",
            "The experience was unremarkable, just average overall.",
            "Nothing noteworthy to report, everything was normal.",
            "The product is functional but has no standout features.",
            "A typical purchase, no complaints but no praise either.",
            "It meets basic requirements but doesn't exceed them.",
        ],
    }

    dataset: list[dict[str, str]] = []
    idx = 0
    # Generate ~200 examples by repeating with slight variations
    for label, texts in templates.items():
        for _i, text in enumerate(texts):
            dataset.append({"id": str(idx), "text": text, "label": label})
            idx += 1
            # Add variations
            for suffix in [
                " Overall, my feelings are clear.",
                " This is my honest review.",
            ]:
                if idx < 200:
                    dataset.append({
                        "id": str(idx),
                        "text": text.rstrip(".!") + suffix,
                        "label": label,
                    })
                    idx += 1

    # Pad to 200 if needed
    while len(dataset) < 200:
        src = dataset[len(dataset) % len(templates["positive"])]
        dataset.append({
            "id": str(len(dataset)),
            "text": src["text"] + " (repeated)",
            "label": src["label"],
        })

    return dataset[:200]


class ClassificationPolicy(ScriptedPolicy):
    """Scripted baseline: few-shot prompting strategy for classification."""

    def __init__(self, examples: list[dict[str, str]]) -> None:
        self._examples = examples
        self._phase = "setup"  # setup -> classify -> done
        self._classify_idx = 0
        self._batch_size = 5

    def next_message(
        self,
        messages: list[Message],
        turn: int,
        scenario: Scenario,
    ) -> str:
        if turn == 0:
            # Initial prompt establishing the task
            return (
                "I need you to classify text snippets by sentiment. "
                "For each text I give you, respond with exactly one word: "
                "'positive', 'negative', or 'neutral'. "
                "Do not include any other text in your response."
            )

        # Send batches of texts to classify
        if self._classify_idx < len(self._examples):
            batch = self._examples[
                self._classify_idx : self._classify_idx + self._batch_size
            ]
            self._classify_idx += self._batch_size
            texts = "\n".join(
                f"[{i+1}] {ex['text']}" for i, ex in enumerate(batch)
            )
            return f"Classify these texts:\n{texts}"

        return ""  # Signal done via empty string


@register_scenario
class ClassificationScenario(Scenario):
    """Prompting-in-the-dark classification scenario.

    The prompter cannot see labels; only sees model responses.
    Goal: elicit a consistent labeling policy from the model.
    Score by accuracy on held-out set within a token/turn budget.
    """

    def __init__(self) -> None:
        self._dataset: list[dict[str, str]] = []
        self._train: list[dict[str, str]] = []
        self._test: list[dict[str, str]] = []
        self._predictions: list[str] = []
        self._seed = 0

    @property
    def config(self) -> ScenarioConfig:
        return ScenarioConfig(
            id="classification/sentiment",
            family=TaskFamily.CLASSIFICATION,
            name="Sentiment Classification",
            description=(
                "Classify text snippets by sentiment (positive/negative/neutral) "
                "without seeing ground truth labels. Measured by accuracy on held-out set."
            ),
            budget_tokens=10000,
            budget_turns=20,
        )

    def setup(self, seed: int) -> list[Message]:
        """Initialize with train/test split."""
        self._seed = seed
        self._dataset = _load_dataset()
        self._predictions = []

        # Deterministic shuffle and split
        rng = random.Random(seed)
        shuffled = list(self._dataset)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * 0.8)
        self._train = shuffled[:split]
        self._test = shuffled[split:]

        return [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a classification assistant. You will be given text "
                    "snippets and must classify each by sentiment. Respond with "
                    "exactly one label per text: 'positive', 'negative', or 'neutral'."
                ),
            )
        ]

    def get_tools(self) -> list[ToolSchema]:
        return []  # No tools for classification

    def handle_tool_call(self, call: ToolCallRequest) -> ToolCallResult:
        return ToolCallResult(
            call_id=call.id,
            name=call.name,
            result="Error: No tools available in this scenario.",
            is_error=True,
        )

    def check_termination(
        self, messages: list[Message], turn: int, tokens: int
    ) -> bool:
        # Terminate when we've gone through enough classification rounds
        user_msgs = [m for m in messages if m.role == Role.USER]
        return len(user_msgs) > len(self._test) // 5 + 2

    def grade(self, result: ScenarioResult) -> dict[str, float]:
        """Grade by extracting predictions from assistant messages."""
        # Extract predictions from assistant responses
        predictions: list[str] = []
        for msg in result.messages:
            if msg.role == Role.ASSISTANT:
                # Parse labels from response
                content = msg.content.lower().strip()
                for line in content.split("\n"):
                    line = line.strip().lower()
                    for label in ["positive", "negative", "neutral"]:
                        if label in line:
                            predictions.append(label)
                            break

        # Compare against test set ground truth
        test_labels = [ex["label"] for ex in self._test]
        correct = 0
        total = min(len(predictions), len(test_labels))

        if total == 0:
            return {
                "task_success": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "efficiency": 0.0,
                "predictions_made": 0.0,
            }

        for i in range(total):
            if predictions[i] == test_labels[i]:
                correct += 1

        accuracy = correct / total

        # Consistency: how often the model gives the same label for similar inputs
        label_counts: dict[str, int] = {}
        for p in predictions:
            label_counts[p] = label_counts.get(p, 0) + 1
        consistency = 1.0 - (len(label_counts) / 3.0) if label_counts else 0.0

        # Efficiency: tokens per prediction
        tokens_per_pred = result.total_tokens / max(len(predictions), 1)
        efficiency = max(0.0, 1.0 - tokens_per_pred / 500.0)

        return {
            "task_success": float(accuracy > 0.5),
            "accuracy": accuracy,
            "consistency": max(0.0, consistency),
            "efficiency": efficiency,
            "predictions_made": float(len(predictions)),
        }

    def get_human_brief(self) -> str | None:
        return (
            "OBJECTIVE: Get the model to classify text snippets by sentiment.\n\n"
            "LABELS: The model must respond with exactly one of: positive, negative, neutral.\n\n"
            "HOW: You don't see ground-truth labels. First establish the task (e.g. ask the model "
            "to classify text by sentiment; you can give a few example texts and ask for labels). "
            "Then send more texts and the model must reply with only the label. Success is measured "
            "by accuracy on a hidden test set—the more correct labels within your turn budget, "
            "the better."
        )

    def get_scripted_prompter_policy(self) -> ScriptedPolicy:
        return ClassificationPolicy(self._test)
