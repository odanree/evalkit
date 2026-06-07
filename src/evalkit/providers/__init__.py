"""
LLM provider strategy — generalized from the beacon job-search-pipeline.

A provider exposes one method:
    complete(model, system, user_prompt, max_tokens) -> CompletionResult

CompletionResult carries the text plus token usage so the cost tracker
can attribute spend per metric / per evaluator call.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self, model: str, system: str, user_prompt: str, max_tokens: int
    ) -> CompletionResult: ...


class AnthropicProvider:
    """Anthropic Claude API provider with streaming + retry."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self, model: str, system: str, user_prompt: str, max_tokens: int
    ) -> CompletionResult:
        import anthropic as anthropic_lib

        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                # Streaming avoids idle-connection drops on long generations
                # (see beacon postmortem 2026-05-26-sse-stream-outage.md).
                text_parts: list[str] = []
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        text_parts.append(text)
                    final = stream.get_final_message()
                return CompletionResult(
                    text="".join(text_parts),
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                    model=model,
                )
            except anthropic_lib.RateLimitError as e:
                wait = 2 ** (attempt + 1)
                log.warning("Anthropic rate limit attempt %d/4, waiting %ds", attempt + 1, wait)
                last_error = e
                time.sleep(wait)
            except anthropic_lib.InternalServerError as e:
                wait = 15 * (attempt + 1)
                log.warning("Anthropic server error attempt %d/4, waiting %ds", attempt + 1, wait)
                last_error = e
                time.sleep(wait)
            except anthropic_lib.APIStatusError as e:
                if e.status_code == 529 or "overloaded_error" in str(e):
                    wait = 30 * (attempt + 1)
                    log.warning("Anthropic overloaded attempt %d/4, waiting %ds", attempt + 1, wait)
                    last_error = e
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"Anthropic call failed after 4 retries: {last_error}")


class OpenAIProvider:
    """OpenAI API or any compatible endpoint."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url or None)
        return self._client

    def complete(
        self, model: str, system: str, user_prompt: str, max_tokens: int
    ) -> CompletionResult:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                )
                usage = response.usage
                return CompletionResult(
                    text=response.choices[0].message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    model=model,
                )
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2)
        raise RuntimeError(f"OpenAI call failed after 3 retries: {last_error}")


class LocalProvider:
    """OpenAI-compatible local endpoint: Ollama, vLLM, LM Studio."""

    def __init__(self, base_url: str, num_ctx: int = 32768) -> None:
        self._base_url = base_url
        self._num_ctx = num_ctx
        self._is_ollama = "11434" in base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.getenv("LOCAL_API_KEY", "local")
            self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    def complete(
        self, model: str, system: str, user_prompt: str, max_tokens: int
    ) -> CompletionResult:
        client = self._get_client()
        # Disable qwen3 reasoning mode — eats max_tokens before JSON output.
        effective_system = system + "\n/no_think" if "qwen3" in model.lower() else system
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": effective_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    **(
                        {"extra_body": {"options": {"num_ctx": self._num_ctx}}}
                        if self._is_ollama
                        else {}
                    ),
                )
                usage = response.usage
                return CompletionResult(
                    text=response.choices[0].message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    model=model,
                )
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2)
        raise RuntimeError(f"Local provider call failed after 3 retries: {last_error}")


def resolve_provider(spec: str | LLMProvider) -> LLMProvider:
    """Resolve a provider from a string spec or pass through an instance.

    Accepted strings:
      "anthropic"
      "openai"
      "local:http://host:port/v1"
    """
    if isinstance(spec, LLMProvider) and not isinstance(spec, str):
        return spec
    if spec == "anthropic":
        return AnthropicProvider()
    if spec == "openai":
        return OpenAIProvider()
    if isinstance(spec, str) and spec.startswith("local:"):
        return LocalProvider(spec[len("local:"):])
    raise ValueError(f"Unknown provider spec: {spec!r}")
