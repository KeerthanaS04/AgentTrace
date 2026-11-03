"""OpenAI-compatible chat-completions adapter with retry and backoff.

Meta Model API, NVIDIA NIM, OpenRouter, and vLLM all expose this API shape,
so one adapter serves every provider; differences live in `ProviderConfig`.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from agenttrace.gateway.config import ProviderConfig
from agenttrace.gateway.types import Message, ProviderRequestError

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RawCompletion:
    text: str
    tokens_in: int
    tokens_out: int
    model: str
    finish_reason: str | None = None
    # Hidden reasoning tokens (already included in tokens_out).
    reasoning_tokens: int = 0


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        config: ProviderConfig,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = name
        self.config = config
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.timeout_s,
            transport=transport,
        )

    def chat(
        self,
        model: str,
        messages: list[Message],
        params: dict[str, Any],
        *,
        json_mode: bool = False,
    ) -> RawCompletion:
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            **params,
        }
        if json_mode and self.config.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post_with_retry("/chat/completions", body)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError(f"{self.name}: malformed response: {data!r:.300}") from exc
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return RawCompletion(
            text=text,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            model=data.get("model") or model,
            finish_reason=choice.get("finish_reason"),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        )

    def _post_with_retry(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries. Errors and 429s draw on separate budgets: a rate limit is
        waited out (longer, doubling waits) rather than counted as a failing endpoint."""
        retries = rate_limited = 0
        while True:
            try:
                resp = self._client.post(path, json=body)
            except httpx.TransportError as exc:
                if retries >= self.config.max_retries:
                    raise ProviderRequestError(f"{self.name}: {exc}") from exc
                retries += 1
                self._backoff(retries, None, str(exc))
                continue
            if resp.status_code == 200:
                result: dict[str, Any] = resp.json()
                return result
            reason = f"HTTP {resp.status_code}"
            if resp.status_code == 429 and rate_limited < self.config.rate_limit_retries:
                rate_limited += 1
                cap = self.config.rate_limit_max_wait_s
                self._backoff(rate_limited, resp.headers.get("retry-after"), reason, cap, base=2.0)
                continue
            if (
                resp.status_code in RETRYABLE_STATUS
                and resp.status_code != 429
                and retries < self.config.max_retries
            ):
                retries += 1
                self._backoff(retries, resp.headers.get("retry-after"), reason)
                continue
            raise ProviderRequestError(
                f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )

    def _backoff(
        self,
        attempt: int,
        retry_after: str | None,
        reason: str,
        cap: float = 30.0,
        base: float = 1.0,
    ) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        if delay <= 0:
            delay = min(cap, base * 2 ** (attempt - 1)) + random.uniform(0, 0.5)
        log.warning("%s: %s, retry %d in %.1fs", self.name, reason, attempt, delay)
        self._sleep(delay)

    def close(self) -> None:
        self._client.close()
