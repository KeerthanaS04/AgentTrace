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
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError(f"{self.name}: malformed response: {data!r:.300}") from exc
        usage = data.get("usage") or {}
        return RawCompletion(
            text=text,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            model=data.get("model") or model,
        )

    def _post_with_retry(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(path, json=body)
            except httpx.TransportError as exc:
                if attempt == attempts:
                    raise ProviderRequestError(f"{self.name}: {exc}") from exc
                self._backoff(attempt, None, str(exc))
                continue
            if resp.status_code == 200:
                result: dict[str, Any] = resp.json()
                return result
            if resp.status_code in RETRYABLE_STATUS and attempt < attempts:
                self._backoff(attempt, resp.headers.get("retry-after"), f"HTTP {resp.status_code}")
                continue
            raise ProviderRequestError(
                f"{self.name}: HTTP {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, retry_after: str | None, reason: str) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        if delay <= 0:
            delay = min(30.0, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
        log.warning("%s: %s, retry %d in %.1fs", self.name, reason, attempt, delay)
        self._sleep(delay)

    def close(self) -> None:
        self._client.close()
