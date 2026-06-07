"""
AnswerRelevance — does the completion actually answer the prompt?

Independent of factual correctness: a confidently-wrong answer that
directly addresses the question scores high here; a correct-but-tangential
answer scores low.
"""

from __future__ import annotations

from evalkit.metrics.base import Metric


class AnswerRelevance(Metric):
    name = "answer_relevance"

    def system_prompt(self) -> str:
        return """You are an expert evaluator scoring ANSWER RELEVANCE.

A completion is relevant when it directly addresses what was asked, without padding or off-topic content.
Score 0.0–10.0:
  10 = directly and completely addresses the prompt
  5  = partially addresses; significant tangents or omissions
  0  = does not address the prompt at all

Do not judge factual accuracy — only whether the output is on-topic and responsive.

Respond ONLY with JSON:
{
  "score": <float 0-10>,
  "reasoning": "<one sentence on what makes it relevant or off-topic>"
}"""

    def build_prompt(
        self,
        prompt: str,
        completion: str,
        reference: str | None,
        context: str | None,
    ) -> str:
        return (
            f"=== PROMPT ===\n{prompt}\n\n"
            f"=== COMPLETION ===\n{completion[:8000]}\n\n"
            "Score answer relevance and return the JSON."
        )
