"""P6-T6: escalation triggers on the right cases; Wonder's ranking and its separate cost."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.diagnosis.escalation import (
    Rules,
    deep_rank,
    final_diagnosis,
    run_agreement,
    should_escalate,
)
from agenttrace.diagnosis.stage_b import attribute
from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios import escalation_check
from agenttrace.scenarios.ground_truth import load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.diagnosis import AnomalyScore, ValidatorFinding
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)
NONE = Rules(low_agreement=None, close_gap=None, max_roots=None, max_hops=None, conflict=False)
USAGE = {"prompt_tokens": 5000, "completion_tokens": 2000}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    _, r = verify_scenario(
        make_gateway(tmp_path_factory.mktemp("e")), CONTROL, ROLES, "default", {}
    )
    assert r is not None
    return r


def sc(
    run: Run,
    local: str,
    verdict: str = "ok",
    inputs: float = 0.9,
    judge: str = "ensemble",
    out: float | None = None,
) -> AnomalyScore:
    o = out if out is not None else {"ok": 0.9, "suspicious": 0.5, "faulty": 0.1}[verdict]
    return AnomalyScore(
        span_id=f"{run.run_id}/{local}",
        judge=judge,
        prompt_version="v",
        output_ok_given_inputs=o,
        inputs_ok=inputs,
        verdict=verdict,  # type: ignore[arg-type]
        evidence=[] if verdict == "ok" else [f"{local}: {verdict}"],
        reasoning="r",
    )


def aggregate(run: Run, flags: dict[str, AnomalyScore]) -> dict[str, AnomalyScore]:
    out = {}
    for s in run.spans:
        if s.kind != "llm":
            local = s.span_id.split("/", 1)[1]
            out[s.span_id] = flags.get(local) or sc(run, local)
    return out


def decide(
    run: Run,
    flags: dict[str, AnomalyScore],
    rules: Rules,
    stored: list[AnomalyScore] | None = None,
    findings: list[ValidatorFinding] | None = None,
) -> list[str]:
    scores = aggregate(run, flags)
    attr = attribute(run, scores, findings or [])
    return should_escalate(run, attr, scores, stored or [], findings or [], rules).reasons


# ---------------------------------------------------------------- the rules


def test_the_config_loads_every_rule() -> None:
    rules = Rules.load()
    assert rules.low_agreement == 0.4 and rules.close_gap == 0.15 and rules.max_roots == 1
    assert rules.max_hops == 3 and rules.conflict and rules.max_usd == 5.0


def test_a_disabled_rule_is_off(tmp_path: Path) -> None:
    path = tmp_path / "e.yaml"
    path.write_text("rules:\n  multiple_roots: {enabled: false, max_roots: 1}\n", encoding="utf-8")
    assert Rules.load(path).max_roots is None


def test_a_clear_single_root_does_not_escalate(run: Run) -> None:
    flags = {"policy": sc(run, "policy", "faulty"), "writer": sc(run, "writer", "suspicious", 0.1)}
    assert decide(run, flags, Rules()) == []


def test_low_agreement(run: Run) -> None:
    disagree = [
        sc(run, local, v1, judge="judge_fast")
        for local, v1 in (
            ("router", "ok"),
            ("policy", "faulty"),
            ("writer", "ok"),
            ("retriever", "faulty"),
        )
    ] + [
        sc(run, local, v2, judge="judge_nvidia")
        for local, v2 in (
            ("router", "faulty"),
            ("policy", "ok"),
            ("writer", "faulty"),
            ("retriever", "ok"),
        )
    ]
    assert run_agreement(disagree) is not None and run_agreement(disagree) < 0  # type: ignore[operator]
    reasons = decide(run, {}, replace(NONE, low_agreement=0.4), stored=disagree)
    assert reasons and reasons[0].startswith("judge agreement kappa")
    agree = [replace_judge(s, "judge_nvidia") for s in disagree[:4]] + disagree[:4]
    assert decide(run, {}, replace(NONE, low_agreement=0.4), stored=agree) == []


def replace_judge(s: AnomalyScore, judge: str) -> AnomalyScore:
    return s.model_copy(update={"judge": judge})


def test_close_top_two(run: Run) -> None:
    close = {
        "router": sc(run, "router", "faulty", out=0.10),
        "policy": sc(run, "policy", "faulty", out=0.15),
    }
    assert decide(run, close, replace(NONE, close_gap=0.15))[0].startswith(
        "top two roots too close"
    )
    apart = {
        "router": sc(run, "router", "faulty", out=0.10),
        "policy": sc(run, "policy", "faulty", out=0.60),
    }
    assert decide(run, apart, replace(NONE, close_gap=0.15)) == []
    # Different evidence tiers are never "close": proof beats a judge's opinion.
    tiers = {
        "router": sc(run, "router", "faulty", out=0.10, judge="stage-a"),
        "policy": sc(run, "policy", "faulty", out=0.12),
    }
    assert decide(run, tiers, replace(NONE, close_gap=0.15)) == []


def test_multiple_roots(run: Run) -> None:
    two = {"router": sc(run, "router", "faulty"), "policy": sc(run, "policy", "faulty")}
    assert decide(run, two, replace(NONE, max_roots=1)) == ["2 root candidates"]
    assert decide(run, {"policy": sc(run, "policy", "faulty")}, replace(NONE, max_roots=1)) == []


def test_long_chain(run: Run) -> None:
    chain = {
        "tool_agent.order_lookup": sc(run, "tool_agent.order_lookup", "faulty", 1.0, "stage-a"),
        "tool_agent": sc(run, "tool_agent", "suspicious"),
        "policy": sc(run, "policy", "suspicious", 0.2),
        "writer": sc(run, "writer", "suspicious", 0.2),
    }  # order_lookup -> tool_agent -> policy -> writer: 3 hops
    assert decide(run, chain, replace(NONE, max_hops=3)) == []
    assert decide(run, chain, replace(NONE, max_hops=2)) == ["propagation chain of 3 hops"]


def test_validator_judge_conflict(run: Run) -> None:
    finding = ValidatorFinding(
        span_id=f"{run.run_id}/tool_agent.account_status",
        check="tool_error_payload",
        severity="error",
        message="account_status returned an error",
    )
    passed = [sc(run, "tool_agent", "ok", judge="judge_fast")]
    reasons = decide(run, {}, replace(NONE, conflict=True), stored=passed, findings=[finding])
    assert reasons == [
        "validators flag tool_agent.account_status (tool_error_payload)"
        " but judge_fast passed its step"
    ]
    flagged = [sc(run, "tool_agent", "suspicious", judge="judge_fast")]
    assert decide(run, {}, replace(NONE, conflict=True), stored=flagged, findings=[finding]) == []


# ---------------------------------------------------------------- Wonder


def wonder(cite: str = "policy", bad_first: bool = False) -> Any:
    calls = itertools.count()

    def script(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the senior investigator"):
            span = "KB-NOPE" if bad_first and next(calls) == 0 else cite
            return {
                "ranking": [
                    {
                        "span": span,
                        "confidence": 0.85,
                        "fault_type": "invalid_decision",
                        "reason": "the decision contradicts the policy it cites",
                    }
                ],
                "narrative": "Policy decided wrongly; the reply followed it.",
            }
        return default_script(prompt)

    return script


def two_root_case(run: Run) -> tuple[Any, dict[str, AnomalyScore]]:
    flags = {
        "router": sc(run, "router", "faulty"),
        "policy": sc(run, "policy", "faulty"),
        "writer": sc(run, "writer", "suspicious", 0.1),
    }
    scores = aggregate(run, flags)
    return attribute(run, scores), scores


def test_wonder_reranks_and_its_cost_is_its_own(run: Run, tmp_path: Path) -> None:
    attr, scores = two_root_case(run)
    decision = should_escalate(run, attr, scores, [], [], Rules())
    assert decision.escalate
    prompts: list[str] = []
    gw = make_gateway(tmp_path, script=wonder(), usage=USAGE, prompts=prompts)
    deep = deep_rank(gw, run, attr, scores, [], [], decision.reasons)
    assert deep.valid and [r.node_id for r in deep.roots] == ["policy"]
    assert deep.cost_usd > 0 and deep.model == "muse-wonder-1.0"
    # Its context: the reasons, the candidates, the graph, nothing about ground truth.
    [prompt] = [p for p in prompts if p.startswith("You are the senior investigator")]
    assert "2 root candidates" in prompt and "1. router" in prompt and "router -> " in prompt
    assert "expected_final" not in prompt and run.scenario_id not in prompt
    diag = final_diagnosis(attr, decision, deep)
    assert diag.escalated_to_deep and "2 root candidates" in (diag.escalation_reason or "")
    assert [r.node_id for r in diag.root_causes] == ["policy"]
    assert diag.cost_usd == deep.cost_usd  # Stage C's own cost
    assert [p.node_id for p in diag.propagated] == ["writer"]  # still explained by policy
    assert diag.narrative.startswith("Policy decided wrongly")


def test_a_bad_citation_is_repaired(run: Run, tmp_path: Path) -> None:
    attr, scores = two_root_case(run)
    deep = deep_rank(
        make_gateway(tmp_path, script=wonder(bad_first=True)), run, attr, scores, [], [], ["x"]
    )
    assert deep.valid and deep.attempts == 2 and "not spans of this run" in deep.errors[0]


def test_without_a_valid_ranking_stage_b_stands(run: Run, tmp_path: Path) -> None:
    attr, scores = two_root_case(run)

    def broken(prompt: str) -> dict[str, Any]:
        return (
            {"nonsense": 1} if prompt.startswith("You are the senior") else default_script(prompt)
        )

    decision = should_escalate(run, attr, scores, [], [], Rules())
    deep = deep_rank(
        make_gateway(tmp_path, script=broken), run, attr, scores, [], [], decision.reasons
    )
    assert not deep.valid
    diag = final_diagnosis(attr, decision, deep)
    assert [r.node_id for r in diag.root_causes] == ["router", "policy"]  # Stage B's
    assert diag.escalated_to_deep and "Stage B stands" in (diag.escalation_reason or "")


# ---------------------------------------------------------------- the command


def store_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    ids = []
    with TraceStore.open() as store:
        for i in range(2):
            _, r = verify_scenario(
                make_gateway(tmp_path / f"r{i}"), CONTROL, ROLES, "default", {}, {"seed": i + 1}
            )
            assert r is not None
            store.save_run(r)
            for s in aggregate(
                r, {"router": sc(r, "router", "faulty"), "policy": sc(r, "policy", "faulty")}
            ).values():
                store.add_anomaly(s)
            ids.append(r.run_id)
    monkeypatch.setattr(escalation_check, "REPORT", tmp_path / "esc.json")
    monkeypatch.setattr(escalation_check, "ROOT", tmp_path)
    return ids


def test_dry_run_decides_without_calling_wonder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = store_runs(tmp_path, monkeypatch)

    def no_gateway(*a: Any, **k: Any) -> Any:
        raise AssertionError("a dry run must not open a gateway")

    monkeypatch.setattr(Gateway, "from_env", classmethod(no_gateway))
    assert escalation_check.main(["--runs", *ids, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("ESC ") == 2 and "escalation triggered: 2/2 runs" in out


def test_the_budget_caps_escalations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = store_runs(tmp_path, monkeypatch)
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(
            lambda cls, *a, **k: make_gateway(
                tmp_path / f"g{next(count)}", script=wonder(), usage=USAGE
            )
        ),
    )
    assert escalation_check.main(["--runs", *ids, "--max-escalations", "1"]) == 0
    out = capsys.readouterr().out
    assert "Wonder: policy" in out and "(budget reached)" in out
    assert "for 1 escalations (1 valid rankings), tracked apart from Stage A" in out
    report = json.loads((tmp_path / "esc.json").read_text("utf-8"))
    assert report["cost_usd"] > 0 and sum(r["escalated"] for r in report["rows"]) == 1
    from agenttrace.store import TraceStore

    with TraceStore.open() as store:
        assert [d.escalated_to_deep for d in store.diagnoses(run_id=ids[0])] == [True]


def test_wonder_naming_two_spans_of_one_node_gives_one_root(run: Run, tmp_path: Path) -> None:
    searches = [x.span_id.split("/", 1)[1] for x in run.spans if x.node_id == "retriever.search"]
    attr, scores = two_root_case(run)

    def twice(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the senior investigator"):
            item = {"confidence": 0.9, "fault_type": "retrieval_wrong", "reason": "wrong hit"}
            return {
                "ranking": [{**item, "span": searches[0]}, {**item, "span": searches[1]}],
                "narrative": "The search returned a wrong document.",
            }
        return default_script(prompt)

    deep = deep_rank(make_gateway(tmp_path, script=twice), run, attr, scores, [], [], ["x"])
    assert [r.node_id for r in deep.roots] == ["retriever.search"]
    assert any("the same fault" in e for e in deep.roots[0].evidence)
