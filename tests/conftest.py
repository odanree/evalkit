"""Shared fixtures: in-memory fake provider so tests never hit the network."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from evalkit.providers import CompletionResult


@dataclass
class FakeProvider:
    """Returns a queued response per call. Records all calls for assertions."""

    responses: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    input_tokens: int = 100
    output_tokens: int = 50

    def complete(
        self, model: str, system: str, user_prompt: str, max_tokens: int
    ) -> CompletionResult:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        text = self.responses.pop(0) if self.responses else '{"score": 8.0, "reasoning": "ok"}'
        return CompletionResult(
            text=text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=model,
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
