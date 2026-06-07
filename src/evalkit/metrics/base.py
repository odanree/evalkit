"""
Metric base class + shared JSON parsing helpers.

All metrics implement three methods:
  system_prompt()        — judge persona + scoring rubric
  build_prompt(...)      — task-specific user message
  parse_response(text)   — pull a MetricScore out of the judge's JSON output
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from evalkit.core import MetricScore

log = logging.getLogger(__name__)


class Metric(ABC):
    name: str = "metric"

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def build_prompt(
        self,
        prompt: str,
        completion: str,
        reference: str | None,
        context: str | None,
    ) -> str: ...

    def parse_response(self, raw: str) -> MetricScore:
        """Default parser: expects {"score": float 0-10, "reasoning": str}."""
        data = _parse_json_lenient(raw)
        if data is None:
            return MetricScore(
                name=self.name,
                value=None,
                reasoning="Judge response was not valid JSON.",
            )
        try:
            value = float(data.get("score") or 0.0)
        except (TypeError, ValueError):
            value = None
        return MetricScore(
            name=self.name,
            value=round(value, 2) if value is not None else None,
            reasoning=str(data.get("reasoning", "")),
        )


def _parse_json_lenient(raw: str) -> dict | None:
    """Strip ```json fences then attempt to parse. Mirrors beacon's evaluator."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("Judge JSON parse failed: %s", e)
        return None
