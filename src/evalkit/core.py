"""
Core public types: ScoreResult, Evaluator.

The Evaluator is a tiny orchestration layer: given a (prompt, completion,
reference) tuple, it dispatches to one or more Metric implementations
and aggregates their results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalkit.cost import Cost, estimate_cost
from evalkit.providers import LLMProvider, resolve_provider

if TYPE_CHECKING:
    from evalkit.metrics.base import Metric


@dataclass
class MetricScore:
    """One dimension of a score result."""

    name: str
    value: float | None
    reasoning: str = ""
    cost: Cost | None = None


@dataclass
class ScoreResult:
    """Aggregate result from running one or more metrics."""

    scores: dict[str, MetricScore] = field(default_factory=dict)
    overall: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> dict[str, float | None]:
        return {name: s.value for name, s in self.scores.items()}

    @property
    def reasoning(self) -> dict[str, str]:
        return {name: s.reasoning for name, s in self.scores.items()}

    @property
    def cost_usd(self) -> float:
        return round(sum((s.cost.usd if s.cost else 0.0) for s in self.scores.values()), 6)

    @property
    def total_tokens(self) -> tuple[int, int]:
        inp = sum((s.cost.input_tokens if s.cost else 0) for s in self.scores.values())
        out = sum((s.cost.output_tokens if s.cost else 0) for s in self.scores.values())
        return inp, out


class Evaluator:
    """
    The main entry point.

        ev = Evaluator(judge="claude-sonnet-4-6", provider="anthropic")
        result = ev.score(
            prompt="What is X?",
            completion=model_output,
            reference=ground_truth,
            metrics=[Faithfulness(), AnswerRelevance()],
        )
        result.values       # {"faithfulness": 0.87, "answer_relevance": 0.92}
        result.cost_usd     # 0.0021
    """

    def __init__(
        self,
        judge: str = "claude-sonnet-4-6",
        provider: str | LLMProvider = "anthropic",
        max_tokens: int = 1024,
    ) -> None:
        self.judge = judge
        self.provider: LLMProvider = resolve_provider(provider)
        self.max_tokens = max_tokens

    def score(
        self,
        prompt: str,
        completion: str,
        reference: str | None = None,
        context: str | None = None,
        metrics: list["Metric"] | None = None,
        weights: dict[str, float] | None = None,
    ) -> ScoreResult:
        if not metrics:
            raise ValueError("At least one metric is required.")

        result = ScoreResult()
        for m in metrics:
            completion_result = self.provider.complete(
                model=self.judge,
                system=m.system_prompt(),
                user_prompt=m.build_prompt(
                    prompt=prompt,
                    completion=completion,
                    reference=reference,
                    context=context,
                ),
                max_tokens=self.max_tokens,
            )
            score = m.parse_response(completion_result.text)
            score.cost = estimate_cost(
                completion_result.input_tokens,
                completion_result.output_tokens,
                completion_result.model,
            )
            result.scores[m.name] = score

        if weights:
            result.overall = _weighted_average(result.scores, weights)
        elif len(result.scores) > 1:
            # Equal weighting by default.
            result.overall = _equal_average(result.scores)

        return result


def _weighted_average(scores: dict[str, MetricScore], weights: dict[str, float]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for name, w in weights.items():
        s = scores.get(name)
        if s is None or s.value is None:
            continue
        total += s.value * w
        weight_sum += w
    if weight_sum == 0:
        return None
    return round(total / weight_sum, 4)


def _equal_average(scores: dict[str, MetricScore]) -> float | None:
    vals = [s.value for s in scores.values() if s.value is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)
