# evalkit

A small, opinionated LLM evaluation framework. **G-Eval style judges**, **RAGAS-style metrics**, cost tracking, and a clean Pydantic-shaped API. Extracted from a production LLM pipeline ([beacon](https://github.com/dtle82/job-search-pipeline)) where the same judge logic shipped against five generation phases.

```python
from evalkit import Evaluator, Faithfulness, AnswerRelevance

ev = Evaluator(judge="claude-sonnet-4-6", provider="anthropic")
result = ev.score(
    prompt="What is the SEC's stance on crypto custody?",
    completion=model_output,
    context=source_documents,
    metrics=[Faithfulness(), AnswerRelevance()],
)

result.values     # {"faithfulness": 8.7, "answer_relevance": 9.2}
result.reasoning  # {"faithfulness": "...", "answer_relevance": "..."}
result.cost_usd   # 0.0021
result.overall    # equal-weighted average across metrics
```

## Why

Every team building with LLMs ends up writing the same eval scaffolding: a judge prompt, a JSON parser that has to be lenient about ```json fences, a weighted-average computation, retry logic, and a cost tracker. `evalkit` is that scaffolding factored out cleanly — drop it into a new project and start scoring in five lines.

## What's in the box

- **`Evaluator`** — single entry point; pluggable judge model + provider.
- **`GEval`** — bring-your-own-dimensions weighted rubric judge (the pattern from the [G-Eval paper](https://arxiv.org/abs/2303.16634)).
- **`Faithfulness`** / **`AnswerRelevance`** — RAGAS-style single-dimension metrics, shippable today.
- **Providers** — Anthropic (streaming + retry on overload/rate-limit), OpenAI / OpenAI-compatible, local (Ollama / vLLM / LM Studio).
- **Cost tracker** — per-call USD estimates; pluggable pricing table.
- **Lenient JSON parser** — handles ```json fences, mixed code blocks, and missing keys without throwing.

## Install

```bash
pip install evalkit[anthropic]   # or [openai], or [all]
```

## Roadmap

- [x] Core: `Evaluator`, `ScoreResult`, `MetricScore`
- [x] Providers: Anthropic, OpenAI, Local (Ollama/vLLM)
- [x] Metrics: `Faithfulness`, `AnswerRelevance`, `GEval`
- [ ] Metrics: `ContextPrecision`, `Hallucination`, `Pairwise`
- [ ] Batch + async runner
- [ ] Drift detector (rolling KS test on score distribution)
- [ ] Langfuse / OpenTelemetry exporter
- [ ] VS Code extension — inline eval on selection

## Custom rubrics with GEval

```python
from evalkit import Evaluator, GEval

judge = GEval(
    dimensions={
        "factual_accuracy": "Are claims about case law cited correctly?",
        "legal_reasoning": "Does the analysis follow from the cited authorities?",
        "completeness":    "Does it cover all elements the question requires?",
    },
    weights={"factual_accuracy": 0.5, "legal_reasoning": 0.3, "completeness": 0.2},
    task_description="legal memo",
)

result = Evaluator().score(
    prompt=question,
    completion=memo,
    context=case_law_snippets,
    metrics=[judge],
)
```

## Custom providers

Any class with a `complete(model, system, user_prompt, max_tokens) -> CompletionResult` method works:

```python
from evalkit import Evaluator, CompletionResult

class MyProvider:
    def complete(self, model, system, user_prompt, max_tokens) -> CompletionResult:
        ...

ev = Evaluator(provider=MyProvider())
```

## Origin

The judge prompt structure, weighted-average computation, and JSON fence stripping all started life in [beacon](https://github.com/dtle82/job-search-pipeline)'s `worker/llm/evaluator.py`, scoring resume generation, gap analysis, cover letters, interview prep, and job-fit scoring. After the third "should we extract this?" moment, here we are.

## Development

```bash
git clone https://github.com/dtle82/evalkit
cd evalkit
pip install -e ".[dev,all]"
pytest
ruff check .
mypy src
```

## License

MIT
