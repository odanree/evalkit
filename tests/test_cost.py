"""Cost estimator tests."""

from __future__ import annotations

import pytest

from evalkit import estimate_cost, register_model_pricing


def test_known_model_priced_correctly():
    # Sonnet 4.6: $3/$15 per 1M
    c = estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4-6")
    assert c.usd == pytest.approx(18.0)


def test_unknown_model_is_free():
    c = estimate_cost(1_000_000, 1_000_000, "some-local-model")
    assert c.usd == 0.0


def test_register_custom_pricing():
    register_model_pricing("custom-model", 1.0, 2.0)
    c = estimate_cost(1_000_000, 500_000, "custom-model")
    # 1.0 + 0.5*2.0 = 2.0
    assert c.usd == pytest.approx(2.0)
