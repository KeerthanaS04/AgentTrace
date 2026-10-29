from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from agenttrace.gateway import (
    CompletionCache,
    CostLedger,
    Gateway,
    GatewayConfig,
    ProviderRequestError,
    ProviderUnavailableError,
    UnknownRoleError,
    UnverifiedModelError,
    content_hash,
    load_config,
)
from agenttrace.gateway.structured import extract_json

CONFIG: dict[str, Any] = {
    "providers": {
        "meta": {
            "base_url": "https://meta.test/v1",
            "api_key_env": "MODEL_API_KEY",
            "supports_json_mode": True,
            "models": {"spark-x": {"input_per_m": 1.0, "output_per_m": 2.0}},
        },
        "nvidia": {
            "base_url": "https://nvidia.test/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "models": {"nemo-x": {"input_per_m": 0.5, "output_per_m": 0.5}},
        },
        "openrouter": {
            "base_url": "https://openrouter.test/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "models": {},
        },
    },
    "roles": {
        "judge_fast": {"provider": "meta", "model": "spark-x", "params": {"temperature": 0}},
        "judge_nvidia": {"provider": "nvidia", "model": "nemo-x", "params": {"temperature": 0}},
        "judge_oss": {"provider": "openrouter", "model": "oss-x", "params": {"temperature": 0}},
        "judge_unverified": {"provider": "meta", "model": "TODO_VERIFY"},
    },
}
ENV = {"MODEL_API_KEY": "k-meta", "NVIDIA_API_KEY": "k-nv", "OPENROUTER_API_KEY": "k-or"}


class FakeProvider:
    """Records requests and replies from a queue of (status, body) or a default reply."""

    def __init__(self, replies: list[tuple[int, Any]] | None = None, text: str = "hello") -> None:
        self.replies = list(replies or [])
        self.text = text
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.replies:
            status, body = self.replies.pop(0)
            if isinstance(body, Exception):
                raise body
            return httpx.Response(
                status,
                json=body,
                headers=body.pop("_headers", {}) if isinstance(body, dict) else {},
            )
        return httpx.Response(200, json=ok_body(self.text))

    def body(self, i: int = -1) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[i].content)
        return result


def ok_body(text: str, tokens_in: int = 100, tokens_out: int = 50) -> dict[str, Any]:
    return {
        "model": "served-model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


GatewayFactory = Callable[..., tuple[Gateway, dict[str, FakeProvider]]]


@pytest.fixture
def make_gateway(tmp_path: Path) -> GatewayFactory:
    def factory(
        fakes: dict[str, FakeProvider] | None = None,
        env: dict[str, str] | None = None,
        cache_path: Path | None = None,
    ) -> tuple[Gateway, dict[str, FakeProvider]]:
        fakes = fakes or {name: FakeProvider() for name in CONFIG["providers"]}
        gw = Gateway(
            GatewayConfig.model_validate(CONFIG),
            cache=CompletionCache(cache_path or tmp_path / "cache.sqlite"),
            ledger=CostLedger(tmp_path / "calls.jsonl"),
            env=ENV if env is None else env,
            transports={name: httpx.MockTransport(f) for name, f in fakes.items()},
            sleep=lambda _s: None,
        )
        return gw, fakes

    return factory


USER = [{"role": "user", "content": "ping"}]


# ---------------------------------------------------------------- basic calls


@pytest.mark.parametrize(
    "role, provider, key",
    [
        ("judge_fast", "meta", "k-meta"),
        ("judge_nvidia", "nvidia", "k-nv"),
        ("judge_oss", "openrouter", "k-or"),
    ],
)
def test_complete_works_for_each_provider(
    make_gateway: GatewayFactory, role: str, provider: str, key: str
) -> None:
    gw, fakes = make_gateway()
    result = gw.complete(role, USER)
    req = fakes[provider].requests[0]
    assert result.text == "hello"
    assert result.provider == provider
    assert req.headers["authorization"] == f"Bearer {key}"
    assert req.url.path.endswith("/chat/completions")
    assert fakes[provider].body()["temperature"] == 0


def test_second_identical_call_is_cache_hit(
    make_gateway: GatewayFactory, caplog: pytest.LogCaptureFixture
) -> None:
    gw, fakes = make_gateway()
    caplog.set_level("INFO", logger="agenttrace.gateway")
    first = gw.complete("judge_fast", USER)
    second = gw.complete("judge_fast", USER)
    assert len(fakes["meta"].requests) == 1
    assert not first.cache_hit and second.cache_hit
    assert second.text == first.text
    assert second.cost_usd == 0.0 and second.original_cost_usd == first.cost_usd
    assert any("cache_hit" in r.message for r in caplog.records)
    assert gw.ledger.calls == 2 and gw.ledger.cache_hits == 1


def test_cache_persists_across_gateways(make_gateway: GatewayFactory, tmp_path: Path) -> None:
    gw1, fakes1 = make_gateway()
    gw1.complete("judge_fast", USER)
    gw1.close()
    gw2, fakes2 = make_gateway()
    assert gw2.complete("judge_fast", USER).cache_hit
    assert not fakes2["meta"].requests


def test_param_override_changes_cache_key(make_gateway: GatewayFactory) -> None:
    gw, fakes = make_gateway()
    a = gw.complete("judge_fast", USER)
    b = gw.complete("judge_fast", USER, params={"temperature": 0.7})
    assert a.cache_key != b.cache_key
    assert len(fakes["meta"].requests) == 2
    assert fakes["meta"].body()["temperature"] == 0.7


def test_content_hash_is_key_order_independent() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


# ---------------------------------------------------------------- cost


def test_cost_from_pricing(make_gateway: GatewayFactory, tmp_path: Path) -> None:
    gw, _ = make_gateway()
    result = gw.complete("judge_fast", USER)
    # 100 in * $1/M + 50 out * $2/M
    assert result.cost_usd == pytest.approx(0.0002)
    assert result.cost_known
    assert gw.ledger.spent_by_provider() == {"meta": pytest.approx(0.0002)}
    entry = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert entry["cost_usd"] == pytest.approx(0.0002)
    assert "text" not in entry


def test_missing_pricing_records_unknown_cost(make_gateway: GatewayFactory) -> None:
    gw, _ = make_gateway()
    result = gw.complete("judge_oss", USER)
    assert result.cost_usd == 0.0 and not result.cost_known


# ---------------------------------------------------------------- resolution errors


def test_unknown_role(make_gateway: GatewayFactory) -> None:
    gw, _ = make_gateway()
    with pytest.raises(UnknownRoleError):
        gw.complete("nope", USER)


def test_todo_verify_model_refused(make_gateway: GatewayFactory) -> None:
    gw, fakes = make_gateway()
    with pytest.raises(UnverifiedModelError, match="TODO_VERIFY"):
        gw.complete("judge_unverified", USER)
    assert not fakes["meta"].requests
    assert not gw.is_available("judge_unverified")


def test_missing_api_key(make_gateway: GatewayFactory) -> None:
    gw, _ = make_gateway(env={"MODEL_API_KEY": "k"})
    with pytest.raises(ProviderUnavailableError, match="NVIDIA_API_KEY"):
        gw.complete("judge_nvidia", USER)
    assert gw.is_available("judge_fast")
    assert not gw.is_available("judge_nvidia")


def test_role_with_unknown_provider_rejected() -> None:
    bad = {**CONFIG, "roles": {"r": {"provider": "ghost", "model": "m"}}}
    with pytest.raises(ValidationError, match="unknown provider"):
        GatewayConfig.model_validate(bad)


def test_load_config_from_yaml(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    cfg = load_config(path)
    assert cfg.roles["judge_fast"].model == "spark-x"


# ---------------------------------------------------------------- retries


def test_retries_on_429_then_succeeds(make_gateway: GatewayFactory) -> None:
    fake = FakeProvider(
        replies=[
            (429, {"error": "slow down", "_headers": {"retry-after": "2"}}),
            (200, ok_body("ok")),
        ]
    )
    sleeps: list[float] = []
    gw, _ = make_gateway(fakes={"meta": fake})
    gw._sleep = sleeps.append  # captured by providers created after this point
    assert gw.complete("judge_fast", USER).text == "ok"
    assert len(fake.requests) == 2
    assert sleeps == [2.0]


def test_retries_transport_errors(make_gateway: GatewayFactory) -> None:
    fake = FakeProvider(replies=[(0, httpx.ConnectError("boom")), (200, ok_body("ok"))])
    gw, _ = make_gateway(fakes={"meta": fake})
    assert gw.complete("judge_fast", USER).text == "ok"


def test_non_retryable_error_raises_immediately(make_gateway: GatewayFactory) -> None:
    fake = FakeProvider(replies=[(400, {"error": "bad request"})])
    gw, _ = make_gateway(fakes={"meta": fake})
    with pytest.raises(ProviderRequestError) as exc:
        gw.complete("judge_fast", USER)
    assert exc.value.status_code == 400
    assert len(fake.requests) == 1


def test_retries_exhausted(make_gateway: GatewayFactory) -> None:
    fake = FakeProvider(replies=[(503, {})] * 10)
    gw, _ = make_gateway(fakes={"meta": fake})
    with pytest.raises(ProviderRequestError):
        gw.complete("judge_fast", USER)
    assert len(fake.requests) == 5  # 1 + max_retries (default 4)


def test_failed_call_is_not_cached(make_gateway: GatewayFactory) -> None:
    fake = FakeProvider(replies=[(400, {})])
    gw, _ = make_gateway(fakes={"meta": fake})
    with pytest.raises(ProviderRequestError):
        gw.complete("judge_fast", USER)
    assert len(gw.cache) == 0


# ---------------------------------------------------------------- structured output


class Verdict(BaseModel):
    verdict: str
    confidence: float


def test_json_mode_only_where_supported(make_gateway: GatewayFactory) -> None:
    gw, fakes = make_gateway()
    for name in ("meta", "nvidia"):
        fakes[name].text = '{"verdict": "ok", "confidence": 0.9}'
    gw.complete_structured("judge_fast", USER, Verdict)
    gw.complete_structured("judge_nvidia", USER, Verdict)
    assert fakes["meta"].body()["response_format"] == {"type": "json_object"}
    assert "response_format" not in fakes["nvidia"].body()


def test_structured_valid_first_try(make_gateway: GatewayFactory) -> None:
    gw, fakes = make_gateway()
    fakes["meta"].text = '```json\n{"verdict": "faulty", "confidence": 0.8}\n```'
    result = gw.complete_structured("judge_fast", USER, Verdict)
    assert result.value == Verdict(verdict="faulty", confidence=0.8)
    assert not result.parse_failure and result.attempts == 1
    assert fakes["meta"].body()["messages"][0]["role"] == "system"


def test_structured_repairs_with_error_message(make_gateway: GatewayFactory) -> None:
    fake = FakeProvider(
        replies=[
            (200, ok_body('{"verdict": "ok"}')),
            (200, ok_body('{"verdict": "ok", "confidence": 1}')),
        ]
    )
    gw, _ = make_gateway(fakes={"meta": fake})
    result = gw.complete_structured("judge_fast", USER, Verdict)
    assert result.value is not None and result.attempts == 2
    retry_msgs = fake.body(1)["messages"]
    assert retry_msgs[-2]["role"] == "assistant"
    assert "confidence" in retry_msgs[-1]["content"]


def test_structured_parse_failure_does_not_raise(make_gateway: GatewayFactory) -> None:
    gw, fakes = make_gateway()
    fakes["meta"].text = "I refuse to answer in JSON."
    result = gw.complete_structured("judge_fast", USER, Verdict)
    assert result.value is None and result.parse_failure
    assert result.attempts == 3  # first try + 2 repairs
    assert len(result.errors) == 3
    assert result.cost_usd > 0


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here it is: {"a": 1} hope that helps'],
)
def test_extract_json_variants(text: str) -> None:
    assert extract_json(text) == {"a": 1}
