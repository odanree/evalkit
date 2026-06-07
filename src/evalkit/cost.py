"""
Per-call cost + token tracking.

Pricing is loaded from a static table that you can override at runtime.
Costs are USD per 1M tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens. Update as model prices change.
_PRICING: dict[str, tuple[float, float]] = {
    # input_per_million, output_per_million
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}


@dataclass(frozen=True)
class Cost:
    input_tokens: int
    output_tokens: int
    model: str
    usd: float


def register_model_pricing(model: str, input_per_million: float, output_per_million: float) -> None:
    """Register or override pricing for a model id."""
    _PRICING[model] = (input_per_million, output_per_million)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> Cost:
    rates = _PRICING.get(model)
    if rates is None:
        # Local model or unknown — assume free.
        return Cost(input_tokens, output_tokens, model, 0.0)
    in_rate, out_rate = rates
    usd = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return Cost(input_tokens, output_tokens, model, round(usd, 6))
