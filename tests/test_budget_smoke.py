from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

import agenttrace.gateway.gateway as gateway_module
from agenttrace.gateway import (
    BudgetExceededError,
    BudgetGuard,
    CompletionCache,
    CostLedger,
    Gateway,
    GatewayConfig,
    ProviderLimit,
    read_spend,
)
from agenttrace.gateway.smoke import format_report, main, run_smoke

REPO = Path(__file__).resolve().parents[1]
USER = [{"role": "user", "content": "ping"}]

# Priced so that a 1000-max_tokens call has a visible worst-case estimate:
# output 1000 tokens * $10/M = $0.01, plus a little input.
CONFIG: dict[str, Any] = {
    "providers": {
        "meta": {
            "base_url": "https://meta.test/v1",
            "api_key_env": "MODEL_API_KEY",
            "budget_usd": 0.05,
            "warn_usd": 0.0001,
            "models": {"m": {"input_per_m": 10.0, "output_per_m": 10.0}},
        },
        "oss": {"base_url": "https://oss.test/v1", "api_key_env": "OSS_KEY", "models": {}},
    },
    "roles": {
        "judge_fast": {"provider": "meta", "model": "m", "params": {"max_tokens": 1000}},
        "judge_oss": {"provider": "oss", "model": "x", "params": {}},
        "judge_todo": {"provider": "meta", "model": "TODO_VERIFY"},
    },
}


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )


def make_gateway(
    tmp_path: Path,
    budget: BudgetGuard | None = None,
    handler: Any = ok,
    env: dict[str, str] | None = None,
) -> tuple[Gateway, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response: httpx.Response = handler(request)
        return response

    gw = Gateway(
        GatewayConfig.model_validate(CONFIG),
        cache=CompletionCache(tmp_path / "cache.sqlite"),
        ledger=CostLedger(tmp_path / "calls.jsonl"),
        budget=budget,
        env=env if env is not None else {"MODEL_API_KEY": "k", "OSS_KEY": "k"},
        transports={p: httpx.MockTransport(recording) for p in CONFIG["providers"]},
        sleep=lambda _s: None,
    )
    return gw, seen


# ---------------------------------------------------------------- BudgetGuard


def test_total_cap() -> None:
    guard = BudgetGuard(max_usd_total=1.0, prior_spend={"a": 0.6, "b": 0.3})
    guard.check("a", 0.09)
    with pytest.raises(BudgetExceededError, match="total budget"):
        guard.check("b", 0.2)


def test_provider_cap_is_independent_of_other_providers() -> None:
    guard = BudgetGuard(
        provider_limits={"meta": ProviderLimit(budget_usd=1.0)},
        prior_spend={"meta": 0.95, "oss": 50.0},
    )
    guard.check("oss", 10.0)  # no cap on oss
    with pytest.raises(BudgetExceededError, match="meta budget"):
        guard.check("meta", 0.1)


def test_run_cap_applies_only_inside_run_scope() -> None:
    guard = BudgetGuard(max_usd_per_run=0.10)
    guard.check("a", 5.0)  # no run scope -> no per-run limit
    with guard.run("r1") as scope:
        guard.record("a", 0.08)
        assert scope.spent_usd == pytest.approx(0.08)
        with pytest.raises(BudgetExceededError, match="run r1"):
            guard.check("a", 0.05)
    with guard.run("r2"):
        guard.check("a", 0.05)  # fresh run, fresh allowance


def test_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="agenttrace.gateway.budget")
    guard = BudgetGuard(provider_limits={"meta": ProviderLimit(budget_usd=10, warn_usd=1.0)})
    guard.record("meta", 0.5)
    assert not caplog.records
    guard.record("meta", 0.6)
    guard.record("meta", 0.6)
    warnings = [r for r in caplog.records if "warning level" in r.message]
    assert len(warnings) == 1


# ---------------------------------------------------------------- ledger replay


def test_read_spend(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    lines = [
        json.dumps({"provider": "meta", "cost_usd": 0.25}),
        "not json",
        json.dumps({"provider": "meta", "cost_usd": 0.5}),
        json.dumps({"provider": "oss", "cost_usd": 0.0}),
        json.dumps({"no_provider": True}),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    assert read_spend(path) == {"meta": pytest.approx(0.75), "oss": 0.0}
    assert read_spend(tmp_path / "missing.jsonl") == {}


# ---------------------------------------------------------------- gateway integration


def test_gateway_refuses_call_before_sending(tmp_path: Path) -> None:
    guard = BudgetGuard(provider_limits={"meta": ProviderLimit(budget_usd=0.005)})
    gw, seen = make_gateway(tmp_path, guard)
    with pytest.raises(BudgetExceededError):
        gw.complete("judge_fast", USER)  # worst case ~$0.01 > $0.005
    assert not seen


def test_gateway_records_actual_spend(tmp_path: Path) -> None:
    guard = BudgetGuard()
    gw, _ = make_gateway(tmp_path, guard)
    result = gw.complete("judge_fast", USER)
    assert guard.spent_by_provider()["meta"] == pytest.approx(result.cost_usd)
    assert result.cost_usd == pytest.approx(12 * 10.0 / 1_000_000)


def test_cache_hits_allowed_when_over_budget(tmp_path: Path) -> None:
    gw, seen = make_gateway(tmp_path, BudgetGuard())
    gw.complete("judge_fast", USER)
    gw.budget = BudgetGuard(max_usd_total=0.0, prior_spend={"meta": 999.0})
    assert gw.complete("judge_fast", USER).cache_hit
    assert len(seen) == 1


def test_per_run_cap_aborts_run(tmp_path: Path) -> None:
    def expensive(request: httpx.Request) -> httpx.Response:
        # Each call really uses its full 1000 output tokens: ~$0.01 actual spend.
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "long"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1000},
            },
        )

    guard = BudgetGuard(max_usd_per_run=0.025)
    gw, seen = make_gateway(tmp_path, guard, handler=expensive)
    with guard.run("r1"), pytest.raises(BudgetExceededError, match="run r1"):
        for i in range(10):
            # distinct prompts -> no cache hits
            gw.complete("judge_fast", [{"role": "user", "content": f"ping {i}"}])
    # $0.01 + $0.01 fit under $0.025; the third call's worst case would not.
    assert len(seen) == 2


def test_from_env_seeds_spend_from_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("MAX_USD_TOTAL", "5")
    monkeypatch.setenv("MAX_USD_PER_RUN", "0.5")
    cfg_path = tmp_path / "models.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "calls.jsonl").write_text(json.dumps({"provider": "meta", "cost_usd": 0.04}))

    gw = Gateway.from_env(cfg_path, cache_dir)
    assert gw.budget.max_usd_total == 5.0
    assert gw.budget.max_usd_per_run == 0.5
    assert gw.budget.spent_by_provider() == {"meta": pytest.approx(0.04)}
    # meta budget_usd is 0.05 in CONFIG: 0.04 spent + ~0.01 worst case is over.
    with pytest.raises(BudgetExceededError, match="meta budget"):
        gw.budget.check("meta", 0.011)
    gw.close()


def test_from_env_rejects_bad_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("MAX_USD_TOTAL", "lots")
    cfg_path = tmp_path / "models.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    with pytest.raises(ValueError, match="MAX_USD_TOTAL"):
        Gateway.from_env(cfg_path, tmp_path)


# ---------------------------------------------------------------- smoke


def test_smoke_reports_ok_skipped_failed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oss.test":
            return httpx.Response(401, json={"error": "bad key"})
        return ok(request)

    gw, _ = make_gateway(tmp_path, BudgetGuard(), handler=handler)
    results = {r.role: r for r in run_smoke(gw)}
    assert results["judge_fast"].status == "ok"
    assert "pong" in results["judge_fast"].detail
    assert results["judge_oss"].status == "failed"
    assert "401" in results["judge_oss"].detail
    assert results["judge_todo"].status == "skipped"
    report = format_report(gw, list(results.values()))
    assert "1 ok, 1 skipped, 1 failed" in report
    assert "cap $0.05" in report


def test_smoke_skips_missing_keys_and_over_budget(tmp_path: Path) -> None:
    guard = BudgetGuard(provider_limits={"meta": ProviderLimit(budget_usd=0.0)})
    gw, seen = make_gateway(tmp_path, guard, env={"MODEL_API_KEY": "k"})
    results = {r.role: r for r in run_smoke(gw)}
    assert results["judge_fast"].status == "skipped"  # over budget
    assert "budget" in results["judge_fast"].detail
    assert results["judge_oss"].status == "skipped"  # no OSS_KEY
    assert "OSS_KEY" in results["judge_oss"].detail
    assert not seen


def test_smoke_bypasses_cache(tmp_path: Path) -> None:
    gw, seen = make_gateway(tmp_path, BudgetGuard(), env={"MODEL_API_KEY": "k"})
    run_smoke(gw)
    run_smoke(gw)
    assert len(seen) == 2  # judge_fast pinged live both times


def test_smoke_main_with_no_keys_skips_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Done-when for P0-T5 on a machine without keys: smoke exits 0 and skips all roles."""
    monkeypatch.setattr(gateway_module, "load_dotenv", lambda: None)
    for key in ("MODEL_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "LOCAL_OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENTTRACE_CACHE_DIR", str(tmp_path))
    monkeypatch.chdir(REPO)
    assert main() == 0
    out = capsys.readouterr().out
    assert "0 ok" in out and "0 failed" in out
    assert "cap $140.00" in out


def test_smoke_flags_empty_reply_from_reasoning(tmp_path: Path) -> None:
    def reasoning_only(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "length", "message": {"content": None}}],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 256,
                    "completion_tokens_details": {"reasoning_tokens": 254},
                },
            },
        )

    gw, _ = make_gateway(
        tmp_path, BudgetGuard(), handler=reasoning_only, env={"MODEL_API_KEY": "k"}
    )
    results = {r.role: r for r in run_smoke(gw)}
    fast = results["judge_fast"]
    assert fast.status == "failed"
    assert "254 reasoning tokens" in fast.detail
    assert fast.cost_usd > 0


def test_reasoning_tokens_and_finish_reason_recorded(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "pong"}}],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 100,
                    "completion_tokens_details": {"reasoning_tokens": 90},
                },
            },
        )

    gw, _ = make_gateway(tmp_path, BudgetGuard(), handler=handler)
    first = gw.complete("judge_fast", USER)
    assert first.reasoning_tokens == 90 and first.finish_reason == "stop"
    assert not first.truncated
    cached = gw.complete("judge_fast", USER)
    assert cached.cache_hit and cached.reasoning_tokens == 90
