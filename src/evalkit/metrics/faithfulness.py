"""
Faithfulness — does the completion only use facts supported by the provided context?

Mirrors RAGAS faithfulness: penalizes claims that cannot be grounded in the
source material. Requires `context`; without it, the judge has nothing
to ground against.
"""

from __future__ import annotations

from evalkit.metrics.base import Metric


class Faithfulness(Metric):
    name = "faithfulness"

    def system_prompt(self) -> str:
        return """You are an expert evaluator scoring FAITHFULNESS.

A completion is faithful when every factual claim it makes is supported by the SOURCE MATERIAL.
Score 0.0–10.0:
  10 = every claim is grounded in the source
  5  = some grounded, some fabricated or unsupported
  0  = no claims grounded; the output invents facts

Respond ONLY with JSON:
{
  "score": <float 0-10>,
  "reasoning": "<one sentence naming the most important grounded or hallucinated claim>"
}"""

    def build_prompt(
        self,
        prompt: str,
        completion: str,
        reference: str | None,
        context: str | None,
    ) -> str:
        if not context:
            return (
                "WARNING: no source material was provided. Score 0 and explain that "
                "faithfulness cannot be measured without context.\n"
                f"=== OUTPUT ===\n{completion[:8000]}"
            )
        return (
            f"=== SOURCE MATERIAL ===\n{context[:10000]}\n\n"
            f"=== TASK / PROMPT ===\n{prompt}\n\n"
            f"=== COMPLETION TO EVALUATE ===\n{completion[:8000]}\n\n"
            "Score faithfulness and return the JSON."
        )
