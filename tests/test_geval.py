"""Tests for the GEval rubric judge — mirrors beacon's test patterns."""

from __future__ import annotations

import json

import pytest

from evalkit import Evaluator, GEval


def _judge() -> GEval:
    return GEval(
        dimensions={
            "faithfulness": "grounded in source",
            "task_completion": "addresses the task",
            "quality": "clear and professional",
        },
        weights={"faithfulness": 0.5, "task_completion": 0.3, "quality": 0.2},
        task_description="resume",
    )


def test_geval_rejects_empty_dimensions():
    with pytest.raises(ValueError):
        GEval(dimensions={})


def test_geval_rejects_missing_weights():
    with pytest.raises(ValueError):
        GEval(
            dimensions={"a": "...", "b": "..."},
            weights={"a": 1.0},
        )


def test_geval_defaults_to_equal_weights():
    j = GEval(dimensions={"a": "...", "b": "...", "c": "..."})
    assert j.weights == {"a": pytest.approx(1/3), "b": pytest.approx(1/3), "c": pytest.approx(1/3)}


def test_geval_weighted_average(fake_provider):
    fake_provider.responses = [
        json.dumps({
            "faithfulness": 10.0,
            "task_completion": 0.0,
            "quality": 0.0,
            "reasoning": "high faith only",
        })
    ]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(prompt="q", completion="a", metrics=[_judge()])
    # 10*0.5 + 0*0.3 + 0*0.2 = 5.0 (divided by sum of weights = 1.0)
    assert result.values["geval"] == pytest.approx(5.0)


def test_geval_all_tens_gives_ten(fake_provider):
    fake_provider.responses = [
        json.dumps({
            "faithfulness": 10.0,
            "task_completion": 10.0,
            "quality": 10.0,
            "reasoning": "",
        })
    ]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(prompt="q", completion="a", metrics=[_judge()])
    assert result.values["geval"] == pytest.approx(10.0)


def test_geval_invalid_json_yields_none(fake_provider):
    fake_provider.responses = ["this is not json"]
    ev = Evaluator(provider=fake_provider)
    result = ev.score(prompt="q", completion="a", metrics=[_judge()])
    assert result.values["geval"] is None


def test_geval_system_prompt_includes_all_dimensions():
    j = _judge()
    sys = j.system_prompt()
    for dim in ("faithfulness", "task_completion", "quality"):
        assert dim in sys


def test_geval_user_prompt_includes_task_description():
    j = _judge()
    prompt = j.build_prompt("the task", "the output", reference=None, context=None)
    assert "resume" in prompt
    assert "the task" in prompt
    assert "the output" in prompt
