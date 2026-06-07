"""evalkit — a small, opinionated LLM evaluation framework."""

from evalkit.core import Evaluator, MetricScore, ScoreResult
from evalkit.cost import Cost, estimate_cost, register_model_pricing
from evalkit.metrics import AnswerRelevance, Faithfulness, GEval, Metric
from evalkit.providers import (
    AnthropicProvider,
    CompletionResult,
    LLMProvider,
    LocalProvider,
    OpenAIProvider,
)

__version__ = "0.1.0"

__all__ = [
    # core
    "Evaluator",
    "ScoreResult",
    "MetricScore",
    # metrics
    "Metric",
    "GEval",
    "Faithfulness",
    "AnswerRelevance",
    # providers
    "LLMProvider",
    "CompletionResult",
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalProvider",
    # cost
    "Cost",
    "estimate_cost",
    "register_model_pricing",
]
