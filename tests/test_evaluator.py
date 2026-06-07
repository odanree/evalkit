"""Tests for the Evaluator orchestration layer."""

from __future__ import annotations

import json

import pytest

from evalkit import AnswerRelevance, Evaluator, Faithfulness


def test_score_runs_each_metric_once(fake_provider):
    fake_provider.responses = [
        json.dumps({"score": 9.0, "reasoning": "grounded"}),
        json.dumps({"score": 7.0, "reasoning": "on-topic"}),
    ]
    ev = Evaluator(judge="claude-sonnet-4-6", provider=fake_provider)

    result = ev.score(
        prompt="What is SEC custody policy?",
        completion="The SEC requires qualified custodians for client assets.",
        context="Investment Advisers Act Rule 206(4)-2 requires custody by qualified custodians.",
        metrics=[Faithfulness(), AnswerRelevance()],
    )

    assert len(fake_provider.calls) == 2
    assert result.values == {"faithfulness": 9.0, "answer_relevance": 7.0}


def test_overall_is_equal_average_by_default(fake_provider):
    fake_provider.responses = [
        json.dumps({"score": 10.0, "reasoning": ""}),
        json.dumps({"score": 0.0, "reasoning": ""}),
    ]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(
        prompt="q", completion="a",
        metrics=[Faithfulness(), AnswerRelevance()],
    )
    assert result.overall == pytest.approx(5.0)


def test_overall_uses_weights_when_provided(fake_provider):
    fake_provider.responses = [
        json.dumps({"score": 10.0, "reasoning": ""}),
        json.dumps({"score": 0.0, "reasoning": ""}),
    ]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(
        prompt="q",
        completion="a",
        metrics=[Faithfulness(), AnswerRelevance()],
        weights={"faithfulness": 0.75, "answer_relevance": 0.25},
    )
    assert result.overall == pytest.approx(7.5)


def test_invalid_json_response_yields_none_value(fake_provider):
    fake_provider.responses = ["not valid json at all"]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(
        prompt="q", completion="a",
        metrics=[Faithfulness()],
    )
    assert result.values["faithfulness"] is None
    assert "not valid JSON" in result.reasoning["faithfulness"]


def test_strips_json_code_fences(fake_provider):
    fenced = "```json\n" + json.dumps({"score": 8.5, "reasoning": "ok"}) + "\n```"
    fake_provider.responses = [fenced]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(prompt="q", completion="a", metrics=[Faithfulness()])
    assert result.values["faithfulness"] == 8.5


def test_no_metrics_raises(fake_provider):
    ev = Evaluator(provider=fake_provider)
    with pytest.raises(ValueError):
        ev.score(prompt="q", completion="a", metrics=[])


def test_cost_aggregates_across_metrics(fake_provider):
    fake_provider.responses = [
        json.dumps({"score": 5.0, "reasoning": ""}),
        json.dumps({"score": 5.0, "reasoning": ""}),
    ]
    fake_provider.input_tokens = 1000
    fake_provider.output_tokens = 500
    ev = Evaluator(judge="claude-sonnet-4-6", provider=fake_provider)
    result = ev.score(
        prompt="q", completion="a",
        metrics=[Faithfulness(), AnswerRelevance()],
    )
    # Sonnet 4.6 priced at $3/$15 per 1M tokens.
    # Per call: 1000*3/1M + 500*15/1M = 0.003 + 0.0075 = 0.0105
    # Two calls = 0.021
    assert result.cost_usd == pytest.approx(0.021)
    assert result.total_tokens == (2000, 1000)
