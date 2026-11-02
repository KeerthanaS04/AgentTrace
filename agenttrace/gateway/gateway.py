"""The model gateway: the only path from AgentTrace code to any LLM.

Resolves roles to (provider, model, params), serves repeat calls from the
content-addressed cache, retries transient failures, records tokens and cost,
and validates structured output against Pydantic schemas.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from agenttrace.gateway.budget import BudgetGuard, ProviderLimit
from agenttrace.gateway.cache import CompletionCache, content_hash
from agenttrace.gateway.config import (
    DEFAULT_CONFIG_PATH,
    TODO_VERIFY,
    GatewayConfig,
    RoleConfig,
    load_config,
)
from agenttrace.gateway.ledger import CostLedger, read_spend
from agenttrace.gateway.provider import OpenAICompatibleProvider
from agenttrace.gateway.structured import extract_json, repair_instruction, schema_instruction
from agenttrace.gateway.types import (
    CompletionResult,
    Message,
    ModelT,
    ProviderUnavailableError,
    StructuredResult,
    UnknownRoleError,
    UnverifiedModelError,
)
from agenttrace.tracing import SpanBuilder, traced

log = logging.getLogger(__name__)

MessageLike = Message | Mapping[str, str]


def _llm_input(a: dict[str, Any]) -> dict[str, Any]:
    messages = [m.model_dump() if isinstance(m, Message) else dict(m) for m in a["messages"]]
    return {"role": a["role"], "messages": messages, "json_mode": a["json_mode"]}


def _record_llm(span: SpanBuilder, result: CompletionResult) -> None:
    span.model = result.model
    span.params = dict(result.params)
    span.tokens_in = result.tokens_in
    span.tokens_out = result.tokens_out
    span.cost_usd = result.cost_usd
    span.output = {
        "text": result.text,
        "provider": result.provider,
        "finish_reason": result.finish_reason,
        "reasoning_tokens": result.reasoning_tokens,
        "cache_hit": result.cache_hit,
        "original_cost_usd": result.original_cost_usd,
    }


# Rough chars-per-token used only for the pre-call budget estimate (errs high).
_CHARS_PER_TOKEN = 3
_DEFAULT_MAX_TOKENS = 1024


def _env_float(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        cache: CompletionCache,
        ledger: CostLedger | None = None,
        budget: BudgetGuard | None = None,
        env: Mapping[str, str] | None = None,
        transports: Mapping[str, httpx.BaseTransport] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.cache = cache
        self.ledger = ledger or CostLedger()
        self.budget = budget or BudgetGuard()
        self._env = env if env is not None else os.environ
        self._transports = dict(transports or {})
        self._sleep = sleep
        self._providers: dict[str, OpenAICompatibleProvider] = {}

    @classmethod
    def from_env(
        cls,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        cache_dir: Path | str | None = None,
    ) -> Gateway:
        """Build a gateway from `.env`, `config/models.yaml`, and `AGENTTRACE_CACHE_DIR`.

        Budget caps come from MAX_USD_TOTAL / MAX_USD_PER_RUN and each provider's
        `budget_usd` / `warn_usd`; lifetime spend is read back from the ledger.
        """
        load_dotenv()
        root = Path(cache_dir or os.environ.get("AGENTTRACE_CACHE_DIR", ".cache"))
        config = load_config(config_path)
        ledger_path = root / "calls.jsonl"
        budget = BudgetGuard(
            max_usd_total=_env_float(os.environ, "MAX_USD_TOTAL"),
            max_usd_per_run=_env_float(os.environ, "MAX_USD_PER_RUN"),
            provider_limits={
                name: ProviderLimit(pc.budget_usd, pc.warn_usd)
                for name, pc in config.providers.items()
            },
            prior_spend=read_spend(ledger_path),
        )
        return cls(
            config,
            cache=CompletionCache(root / "completions.sqlite"),
            ledger=CostLedger(ledger_path),
            budget=budget,
        )

    # ------------------------------------------------------------ resolution

    def resolve(self, role: str) -> RoleConfig:
        try:
            rc = self.config.roles[role]
        except KeyError:
            raise UnknownRoleError(f"no role {role!r} in models config") from None
        if rc.model == TODO_VERIFY:
            raise UnverifiedModelError(
                f"role {role!r} model ID is {TODO_VERIFY}; confirm it from the "
                f"{rc.provider} dashboard and set it in config/models.yaml"
            )
        return rc

    def is_available(self, role: str) -> bool:
        try:
            self._provider(self.resolve(role).provider)
        except (UnverifiedModelError, ProviderUnavailableError):
            return False
        return True

    def _provider(self, name: str) -> OpenAICompatibleProvider:
        if name not in self._providers:
            pc = self.config.providers[name]
            if pc.base_url_env and self._env.get(pc.base_url_env):
                pc = pc.model_copy(update={"base_url": self._env[pc.base_url_env]})
            if pc.base_url == TODO_VERIFY:
                raise UnverifiedModelError(
                    f"{name}: base_url is {TODO_VERIFY}; confirm the endpoint from provider docs"
                )
            api_key = self._env.get(pc.api_key_env, "")
            if not api_key:
                raise ProviderUnavailableError(
                    f"{name}: environment variable {pc.api_key_env} not set"
                )
            self._providers[name] = OpenAICompatibleProvider(
                name, pc, api_key, transport=self._transports.get(name), sleep=self._sleep
            )
        return self._providers[name]

    # ------------------------------------------------------------ calls

    # Every model call, including each structured-output repair, is one "llm" span.
    @traced("llm", "llm", capture=_llm_input, record=_record_llm)
    def complete(
        self,
        role: str,
        messages: Sequence[MessageLike],
        *,
        params: Mapping[str, Any] | None = None,
        json_mode: bool = False,
        tag: str | None = None,
        use_cache: bool = True,
    ) -> CompletionResult:
        rc = self.resolve(role)
        msgs = [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]
        merged = {**rc.params, **(params or {})}
        key = content_hash(
            {
                "provider": rc.provider,
                "model": rc.model,
                "messages": [m.model_dump() for m in msgs],
                "params": merged,
                "json_mode": json_mode,
            }
        )

        cached = self.cache.get(key) if use_cache else None
        if cached is not None:
            result = CompletionResult.model_validate(
                {**cached, "role": role, "cost_usd": 0.0, "cache_hit": True, "latency_s": 0.0}
            )
        else:
            provider = self._provider(rc.provider)
            self.budget.check(rc.provider, self._estimate_cost(rc, msgs, merged))
            started = time.perf_counter()
            raw = provider.chat(rc.model, msgs, merged, json_mode=json_mode)
            latency = time.perf_counter() - started
            cost, known = self._cost(rc, raw.tokens_in, raw.tokens_out)
            self.budget.record(rc.provider, cost)
            result = CompletionResult(
                text=raw.text,
                role=role,
                provider=rc.provider,
                model=rc.model,
                params=merged,
                tokens_in=raw.tokens_in,
                tokens_out=raw.tokens_out,
                reasoning_tokens=raw.reasoning_tokens,
                finish_reason=raw.finish_reason,
                cost_usd=cost,
                original_cost_usd=cost,
                cost_known=known,
                latency_s=latency,
                cache_key=key,
            )
            self.cache.put(key, result.model_dump(exclude={"role", "cache_hit", "latency_s"}))

        self.ledger.record(result, tag=tag)
        if result.truncated and not result.text.strip():
            log.warning(
                "role=%s hit max_tokens=%s with no visible output (%d reasoning tokens); "
                "raise max_tokens or lower reasoning_effort",
                role,
                merged.get("max_tokens"),
                result.reasoning_tokens,
            )
        log.info(
            "%s role=%s model=%s/%s tokens=%d/%d cost=$%.5f",
            "cache_hit" if result.cache_hit else "call",
            role,
            rc.provider,
            rc.model,
            result.tokens_in,
            result.tokens_out,
            result.cost_usd,
        )
        return result

    def complete_structured(
        self,
        role: str,
        messages: Sequence[MessageLike],
        schema: type[ModelT],
        *,
        params: Mapping[str, Any] | None = None,
        max_repairs: int = 2,
        tag: str | None = None,
    ) -> StructuredResult[ModelT]:
        """Call `role` and validate its reply as `schema`.

        On invalid output, retries up to `max_repairs` times with the validation
        error. If it still fails, returns `parse_failure=True` rather than raising.
        """
        convo: list[MessageLike] = [
            Message(role="system", content=schema_instruction(schema)),
            *messages,
        ]
        completions: list[CompletionResult] = []
        errors: list[str] = []
        for _ in range(max_repairs + 1):
            result = self.complete(role, convo, params=params, json_mode=True, tag=tag)
            completions.append(result)
            try:
                value = schema.model_validate(extract_json(result.text))
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(str(exc))
                convo = [
                    *convo,
                    Message(role="assistant", content=result.text),
                    Message(role="user", content=repair_instruction(str(exc))),
                ]
                continue
            return StructuredResult[ModelT](
                value=value,
                parse_failure=False,
                attempts=len(completions),
                errors=errors,
                completions=completions,
            )

        log.warning(
            "parse_failure role=%s schema=%s after %d attempts",
            role,
            schema.__name__,
            len(completions),
        )
        return StructuredResult[ModelT](
            value=None,
            parse_failure=True,
            attempts=len(completions),
            errors=errors,
            completions=completions,
        )

    # ------------------------------------------------------------ helpers

    def _estimate_cost(
        self, rc: RoleConfig, msgs: Sequence[Message], params: Mapping[str, Any]
    ) -> float:
        """Worst-case cost of a call: prompt size plus the full `max_tokens` of output."""
        tokens_in = sum(len(m.content) for m in msgs) // _CHARS_PER_TOKEN + 1
        tokens_out = int(params.get("max_tokens") or _DEFAULT_MAX_TOKENS)
        return self._cost(rc, tokens_in, tokens_out, warn=False)[0]

    def _cost(
        self, rc: RoleConfig, tokens_in: int, tokens_out: int, *, warn: bool = True
    ) -> tuple[float, bool]:
        pricing = self.config.pricing_for(rc.provider, rc.model)
        if pricing is None:
            if warn:
                log.warning("no pricing for %s/%s; cost recorded as 0", rc.provider, rc.model)
            return 0.0, False
        cost = (tokens_in * pricing.input_per_m + tokens_out * pricing.output_per_m) / 1_000_000
        return cost, True

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()
        self.cache.close()
