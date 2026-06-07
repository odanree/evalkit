"""
G-Eval style multi-dimensional rubric judge.

Lifted from beacon (worker/llm/evaluator.py) and generalized: you supply
the dimensions and weights; the judge handles rubric prompting,
weighted averaging, and JSON-parse fallback.

Example:
    judge = GEval(
        dimensions={
            "faithfulness": "Are claims grounded in the source?",
            "task_completion": "Does the output address the task?",
            "quality": "Clarity, specificity, professional tone.",
        },
        weights={"faithfulness": 0.5, "task_completion": 0.3, "quality": 0.2},
    )
    result = evaluator.score(prompt, completion, context=source, metrics=[judge])
    result.scores["geval"].value  # 0-10 weighted overall
"""

from __future__ import annotations

from evalkit.core import MetricScore
from evalkit.metrics.base import Metric, _parse_json_lenient


class GEval(Metric):
    name = "geval"

    def __init__(
        self,
        dimensions: dict[str, str],
        weights: dict[str, float] | None = None,
        task_description: str = "the generated output",
    ) -> None:
        if not dimensions:
            raise ValueError("At least one dimension is required.")
        self.dimensions = dimensions
        self.task_description = task_description

        if weights is None:
            equal = 1.0 / len(dimensions)
            self.weights = {d: equal for d in dimensions}
        else:
            missing = set(dimensions) - set(weights)
            if missing:
                raise ValueError(f"Missing weights for dimensions: {missing}")
            self.weights = weights

    def system_prompt(self) -> str:
        rubric_lines = []
        for dim, desc in self.dimensions.items():
            rubric_lines.append(f"{dim}:\n  {desc}\n  10 = excellent, 0 = poor")
        rubric = "\n\n".join(rubric_lines)

        json_keys = ",\n  ".join(f'"{d}": <float 0-10>' for d in self.dimensions)

        return f"""You are an expert evaluator. You will receive a task description, optional source material, and a generated output.
Score the output on each of the following dimensions, each on a 0.0–10.0 scale:

{rubric}

Respond ONLY with valid JSON — no explanation outside the JSON:
{{
  {json_keys},
  "reasoning": "<one sentence: the most important strength or issue>"
}}"""

    def build_prompt(
        self,
        prompt: str,
        completion: str,
        reference: str | None,
        context: str | None,
    ) -> str:
        parts = [f"Evaluate this {self.task_description}.\n"]
        parts.append(f"=== TASK / PROMPT ===\n{prompt}\n")
        if context:
            parts.append(f"=== SOURCE MATERIAL ===\n{context[:8000]}\n")
        if reference:
            parts.append(f"=== REFERENCE ANSWER ===\n{reference[:4000]}\n")
        parts.append(f"=== GENERATED OUTPUT ===\n{completion[:12000]}\n")
        parts.append("Score on all dimensions and return the JSON.")
        return "\n".join(parts)

    def parse_response(self, raw: str) -> MetricScore:
        data = _parse_json_lenient(raw)
        if data is None:
            return MetricScore(
                name=self.name,
                value=None,
                reasoning="Judge response was not valid JSON.",
            )

        per_dim: dict[str, float] = {}
        for dim in self.dimensions:
            try:
                per_dim[dim] = float(data.get(dim) or 0.0)
            except (TypeError, ValueError):
                per_dim[dim] = 0.0

        overall = sum(per_dim[d] * self.weights[d] for d in self.dimensions)
        weight_sum = sum(self.weights.values())
        overall = round(overall / weight_sum, 2) if weight_sum else None

        return MetricScore(
            name=self.name,
            value=overall,
            reasoning=str(data.get("reasoning", "")),
        )
