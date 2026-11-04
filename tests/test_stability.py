"""P5-T6: repeated replays are identical (cached) or within tolerance (live)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.gateway.cache import CompletionCache
from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.patcher import patch_fields, plan_patch
from agenttrace.scenarios import patch_check, stability
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.stability import Repeat, compare_repetitions, repeat_replay
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config
from agenttrace.workflow.support import build_support_workflow

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
WORKFLOW = build_support_workflow(ROLES)
F5 = next(
    s
    for s in DATASET
    if s.ground_truth.fault_code == "F5"
    and s.ticket.ticket_id == "T-0101"
    and s.faults[0].params.get("field") == "decision"
)


def recorded(tmp_path: Path, scenario: Scenario = F5) -> Run:
    _, run = verify_scenario(make_gateway(tmp_path / "rec"), scenario, ROLES, "default", {})
    assert run is not None
    return run


def oracle_plan(run: Run) -> Any:
    # The oracle for this scenario: the scripted model's own (approving) decision.
    patch = patch_fields(run, "policy", {"decision": "approve"}, patch_source="oracle")
    return plan_patch(run, WORKFLOW, patch)


# ---------------------------------------------------------------- comparing repetitions


def test_compare_repetitions(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    same = compare_repetitions([run, run, run], [True, True, True], [1.0, 1.0, 1.0])
    assert same["identical"] and same["same_reply"] and same["score_range"] == 0
    reworded = run.model_copy(update={"final_output": {**run.final_output, "reply": "other"}})
    cmp = compare_repetitions([run, reworded], [True, True], [1.0, 0.5])
    assert not cmp["identical"] and cmp["diffs"] == ["final_output"]
    assert cmp["same_decision"] and not cmp["same_reply"] and cmp["score_range"] == 0.5


def test_tolerance_per_mode() -> None:
    base = dict(scenario_id="s", fault_code="F1", seed=1, target="t", reps=3, score_range=0.0)
    reworded = dict(identical=False, same_decision=True, same_recovered=True, same_reply=False)
    assert not Repeat(mode="cached", **base, **reworded).within_tolerance  # cached: exact
    assert Repeat(mode="live", **base, **reworded).within_tolerance  # live: outcome only
    flipped = dict(identical=False, same_decision=False, same_recovered=True, same_reply=False)
    assert not Repeat(mode="live", **base, **flipped).within_tolerance


# ---------------------------------------------------------------- repeating replays


def test_cached_repetitions_are_identical(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    bodies: list[dict[str, Any]] = []
    gateway = make_gateway(tmp_path / "g", bodies=bodies)
    row = repeat_replay(
        gateway,
        F5,
        run,
        oracle_plan(run),
        "policy",
        ROLES,
        mode="cached",
        reps=3,
        scratch=tmp_path,
    )
    assert row.identical and row.within_tolerance and row.decisions == ["approve"] * 3
    assert len(bodies) == 1  # only repetition 1 asked the model (the writer); then cached


def wording_drift() -> Any:
    """A live model whose Writer words the reply differently on every call."""
    n = itertools.count()

    def script(prompt: str) -> dict[str, Any]:
        out = default_script(prompt)
        if prompt.startswith("You are the Response Writer"):
            out = {**out, "reply": f"{out['reply']} (take {next(n)})"}
        return out

    return script


def test_live_repetitions_really_call_the_model(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    bodies: list[dict[str, Any]] = []
    gateway = make_gateway(tmp_path / "g", script=wording_drift(), bodies=bodies)
    row = repeat_replay(
        gateway,
        F5,
        run,
        oracle_plan(run),
        "policy",
        ROLES,
        mode="live",
        reps=3,
        scratch=tmp_path,
    )
    assert len(bodies) == 3  # each repetition's own empty cache: one live writer call each
    assert not row.identical and not row.same_reply and row.diffs
    assert row.same_decision and row.same_recovered and row.within_tolerance


CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)


def test_a_live_decision_flip_is_outside_tolerance(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL)
    # A patch Policy's inputs depend on, so Policy really re-runs (live) each time.
    patch = patch_fields(run, "tool_agent.order_lookup", {"found": False}, patch_source="rule")
    plan = plan_patch(run, WORKFLOW, patch)
    n = itertools.count()

    def flaky_policy(prompt: str) -> dict[str, Any]:
        out = default_script(prompt)
        if prompt.startswith("You are the Policy Agent") and next(n) == 1:
            out = {**out, "decision": "deny", "refund_amount": None}
        return out

    gateway = make_gateway(tmp_path / "g", script=flaky_policy)
    row = repeat_replay(
        gateway,
        CONTROL,
        run,
        plan,
        "tool_agent.order_lookup",
        ROLES,
        mode="live",
        reps=3,
        scratch=tmp_path,
    )
    assert row.decisions == ["approve", "deny", "approve"] and not row.within_tolerance


def test_with_cache_keeps_everything_but_the_cache(tmp_path: Path) -> None:
    base = make_gateway(tmp_path / "g")
    other = base.with_cache(CompletionCache(tmp_path / "x.sqlite"))
    assert other.cache is not base.cache and other.ledger is base.ledger
    assert other.budget is base.budget and other.config is base.config


# ---------------------------------------------------------------- the command


def test_stability_command_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    with TraceStore.open() as store:
        _, run = verify_scenario(
            make_gateway(tmp_path / "r"), F5, ROLES, "default", {}, {"seed": 1}
        )
        assert run is not None
        store.save_run(run)
    reports = tmp_path / "reports"
    reports.mkdir()
    rows = [{"scenario_id": F5.scenario_id, "fault_code": "F5", "behaves": True}]
    (reports / "multirun_default.json").write_text(json.dumps({"rows": {"1": rows}}), "utf-8")
    monkeypatch.setattr(patch_check, "REPORT", reports / "multirun_default.json")
    monkeypatch.setattr(stability, "ROOT", tmp_path)
    monkeypatch.setattr(stability, "REPORT_PATH", reports / "replay_stability.json")
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(
            lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}", script=wording_drift())
        ),
    )
    assert stability.main(["--reps", "3"]) == 0
    out = capsys.readouterr().out
    assert "cached (zero diffs): 1/1 within tolerance" in out
    assert "live (same decision and recovered verdict): 1/1 within tolerance" in out
    assert "same reply text           : 0/1" in out  # live wording drifted, as scripted
    report = json.loads((reports / "replay_stability.json").read_text("utf-8"))
    assert report["tolerance"]["cached"] == "zero diffs between repetitions"
    assert report["summary"]["live"]["within_tolerance"] == 1
    assert {r["mode"] for r in report["rows"]} == {"cached", "live"}
