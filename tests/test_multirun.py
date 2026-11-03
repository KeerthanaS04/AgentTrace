"""P4-T5: K seeded runs per scenario, stored, with the pass-rate variance reported."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios import multirun, verify
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.multirun import run_all, seeds_for, variance_report
from agenttrace.scenarios.verify import Verification, prune_stale, run_tag, verify_scenario
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()


def find(code: str, ticket: str, **params: Any) -> Scenario:
    for s in DATASET:
        same = s.ground_truth.fault_code == code and s.ticket.ticket_id == ticket
        if same and all(any(f.params.get(k) == v for f in s.faults) for k, v in params.items()):
            return s
    raise LookupError((code, ticket, params))


CONTROL = find("F0", "T-0101")
DECISION = find("F5", "T-0101", field="decision")


def deny_on_seed_2(prompt: str, seed: Any) -> dict[str, Any]:
    """The scripted model, except that with seed 2 the Policy Agent denies."""
    out = default_script(prompt)
    if seed == 2 and prompt.startswith("You are the Policy Agent"):
        out = {**out, "decision": "deny", "refund_amount": None}
    return out


# ---------------------------------------------------------------- the seed reaches every call


def test_the_seed_reaches_every_model_call_and_the_run(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []
    gateway = make_gateway(tmp_path, bodies=bodies)
    _, run = verify_scenario(gateway, CONTROL, ROLES, "default", {}, {"seed": 7})
    assert run is not None and run.seed == 7 and run.run_id == f"{CONTROL.scenario_id}-default-s7"
    assert bodies and all(b["seed"] == 7 for b in bodies)
    llm = [s for s in run.spans if s.kind == "llm"]
    assert llm and all(s.params["seed"] == 7 for s in llm)


def test_no_seed_means_unchanged_requests(tmp_path: Path) -> None:
    # Runs without model_params keep their old cache keys (P4-T4's verified runs).
    bodies: list[dict[str, Any]] = []
    _, run = verify_scenario(make_gateway(tmp_path, bodies=bodies), CONTROL, ROLES, "default", {})
    assert run is not None and run.seed is None and run.run_id == f"{CONTROL.scenario_id}-default"
    assert bodies and not any("seed" in b for b in bodies)


def test_each_seed_is_cached_separately(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []
    gateway = make_gateway(tmp_path, bodies=bodies)
    run_all(gateway, [CONTROL], ROLES, "default", [1, 2])
    first = len(bodies)
    assert {b["seed"] for b in bodies} == {1, 2}
    run_all(gateway, [CONTROL], ROLES, "default", [1, 2])  # all from the cache
    assert len(bodies) == first
    run_all(gateway, [CONTROL], ROLES, "default", [1, 2, 3])  # only seed 3 is new
    assert {b["seed"] for b in bodies[first:]} == {3}


def test_temperature_override(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []
    run_all(make_gateway(tmp_path, bodies=bodies), [CONTROL], ROLES, "default", [1], 0.7)
    assert bodies and all(b["temperature"] == 0.7 and b["seed"] == 1 for b in bodies)


def test_run_tags() -> None:
    assert run_tag(None) == "" and run_tag({}) == ""
    assert run_tag({"seed": 2}) == "-s2"
    assert run_tag({"seed": 2, "temperature": 0.7}) == "-s2-t0.7"
    assert seeds_for(3) == [1, 2, 3] and seeds_for(2, first=5) == [5, 6]
    with pytest.raises(ValueError):
        seeds_for(0)


# ---------------------------------------------------------------- variance


def test_a_seed_dependent_model_shows_up_as_unstable(tmp_path: Path) -> None:
    gateway = make_gateway(tmp_path, seeded=deny_on_seed_2)
    results = run_all(gateway, [CONTROL, DECISION], ROLES, "default", [1, 2, 3])
    assert [v.behaves for v in results[2]] == [False, False]  # wrong control; fault no effect
    assert all(v.behaves for seed in (1, 3) for v in results[seed])
    # Each seed's fault is judged against the clean run of the same seed: there the
    # Policy Agent already denies, so forcing "deny" changes nothing.
    assert results[2][1].reason.startswith("root output unchanged")
    report = variance_report(results)
    assert [r["behaves_rate"] for r in report["runs"]] == [1.0, 0.0, 1.0]
    assert report["unstable"] == [CONTROL.scenario_id, DECISION.scenario_id]
    assert report["never_behaves"] == []
    rows = {s["scenario_id"]: s for s in report["scenarios"]}
    assert rows[CONTROL.scenario_id]["behaves"] == 2 and rows[CONTROL.scenario_id]["k"] == 3


def v(sid: str, code: str, behaves: bool, correct: bool) -> Verification:
    return Verification(sid, code, "T", behaves, "ok" if behaves else "x", final_correct=correct)


def test_variance_arithmetic() -> None:
    results = {
        1: [v("a", "F1", True, False), v("b", "F0", True, True)],
        2: [v("a", "F1", False, True), v("b", "F0", True, True)],
        3: [v("a", "F1", True, False), v("b", "F0", True, True)],
    }
    report = variance_report(results)
    s = report["behaves_rate"]
    assert [r["behaves_rate"] for r in report["runs"]] == [1.0, 0.5, 1.0]
    assert math.isclose(s["mean"], 5 / 6)
    assert math.isclose(s["variance"], 1 / 12)  # sample variance of 1, 0.5, 1
    assert math.isclose(s["std"], math.sqrt(1 / 12))
    assert (s["min"], s["max"]) == (0.5, 1.0)
    assert math.isclose(report["final_correct_rate"]["mean"], 2 / 3)
    assert report["by_code"]["F0"]["std"] == 0.0 and report["by_code"]["F1"]["n"] == 1
    assert report["unstable"] == ["a"]


def test_one_run_has_no_variance() -> None:
    report = variance_report({1: [v("a", "F1", True, False)]})
    assert report["behaves_rate"]["variance"] is None and report["behaves_rate"]["std"] is None


def test_runs_must_cover_the_same_scenarios() -> None:
    with pytest.raises(ValueError, match="different set"):
        variance_report({1: [v("a", "F1", True, False)], 2: [v("b", "F1", True, False)]})


def test_pruning_only_touches_its_own_seed() -> None:
    class FakeStore:
        ids = ["sc-001-default-s1", "sc-150-default-s1", "sc-150-default-s2", "sc-150-default"]

        def run_ids(self) -> list[str]:
            return list(self.ids)

        def delete_run(self, run_id: str) -> None:
            self.ids.remove(run_id)

    store = FakeStore()
    prune_stale(store, {"sc-001-default-s1"}, "default", "-s1")  # type: ignore[arg-type]
    assert store.ids == ["sc-001-default-s1", "sc-150-default-s2", "sc-150-default"]


# ---------------------------------------------------------------- provider errors


def policy_rate_limited_on_seed_3(prompt: str, seed: Any) -> int:
    return 429 if seed == 3 and prompt.startswith("You are the Policy Agent") else 200


def test_rate_limited_runs_are_not_judged_cached_or_stored(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(prompt: str, seed: Any) -> int:
        calls["n"] += 1
        return policy_rate_limited_on_seed_3(prompt, seed)

    gateway = make_gateway(tmp_path, status=flaky)
    cache: dict[str, Any] = {}
    v, run = verify_scenario(gateway, DECISION, ROLES, "default", cache, {"seed": 3})
    assert run is None and not v.behaves
    assert v.provider_error is not None and "HTTP 429" in v.provider_error
    assert v.reason.startswith("provider error in the clean run")
    assert cache == {}  # the broken clean run is not reused for other scenarios
    # A rerun once the limit has passed redoes only the failed calls.
    v2, run2 = verify_scenario(
        make_gateway(tmp_path), DECISION, ROLES, "default", cache, {"seed": 3}
    )
    assert v2.provider_error is None and v2.behaves and run2 is not None


def test_report_leaves_provider_errors_out_and_says_so(tmp_path: Path) -> None:
    gateway = make_gateway(tmp_path, status=policy_rate_limited_on_seed_3)
    report = variance_report(run_all(gateway, [CONTROL, DECISION], ROLES, "default", [1, 2, 3]))
    assert not report["complete"]
    assert {e["seed"] for e in report["provider_errors"]} == {3}
    assert [r["n"] for r in report["runs"]] == [2, 2, 0]
    assert report["runs"][2]["behaves_rate"] is None
    # Rates and stability use only the judged runs.
    assert report["behaves_rate"]["mean"] == 1.0 and report["unstable"] == []
    assert all(s["k"] == 2 and s["provider_errors"] == 1 for s in report["scenarios"])


def test_verify_summary_excludes_provider_errors() -> None:
    from agenttrace.scenarios.verify import summarize

    rows = [
        Verification("a", "F1", "t", True, "ok"),
        Verification("b", "F1", "t", False, "provider error", provider_error="HTTP 429"),
    ]
    s = summarize(rows)
    assert (s["n"], s["behaves"], s["rate"]) == (1, 1, 1.0) and s["provider_errors"] == ["b"]


# ---------------------------------------------------------------- the command


def test_main_stores_k_runs_per_scenario_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}")),
    )
    monkeypatch.setattr(multirun, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(multirun, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "store.db"))
    ids = [CONTROL.scenario_id, DECISION.scenario_id]
    assert multirun.main(["--only", *ids, "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert "2 scenarios x 3 seeds [1, 2, 3]" in out and "unstable scenarios" in out
    assert "stored 6 runs (3 per scenario)" in out
    report = json.loads((tmp_path / "reports" / "multirun_default.json").read_text("utf-8"))
    assert report["k"] == 3 and report["behaves_rate"]["std"] == 0.0
    assert set(report["rows"]) == {"1", "2", "3"}
    from agenttrace.store import TraceStore

    with TraceStore.open(tmp_path / "store.db") as store:
        runs = {r.run_id: r for r in store.runs()}
    assert set(runs) == {f"{i}-default-s{k}" for i in ids for k in (1, 2, 3)}
    assert {r.seed for r in runs.values()} == {1, 2, 3}
    assert {r.scenario_id for r in runs.values()} == set(ids)
