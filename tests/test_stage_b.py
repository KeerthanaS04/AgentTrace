"""P6-T5: Stage B separates root causes from propagated errors, with valid paths.

The ten hand-built cases give a real support run fixed anomaly scores, so the correct
attribution is known exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import make_gateway

from agenttrace.diagnosis.stage_b import (
    aggregate_scores,
    attribute,
    fault_type,
    valid_path,
)
from agenttrace.graph import build_graph
from agenttrace.scenarios import attribution_check
from agenttrace.scenarios.ground_truth import load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.diagnosis import AnomalyScore
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    tmp = tmp_path_factory.mktemp("b")
    _, r = verify_scenario(make_gateway(tmp), CONTROL, ROLES, "default", {})
    assert r is not None
    return r


def score(
    run: Run, local: str, verdict: str = "ok", inputs: float = 0.9, judge: str = "ensemble"
) -> AnomalyScore:
    out = {"ok": 0.9, "suspicious": 0.5, "faulty": 0.1}[verdict]
    return AnomalyScore(
        span_id=f"{run.run_id}/{local}",
        judge=judge,
        prompt_version="v",
        output_ok_given_inputs=out,
        inputs_ok=inputs,
        verdict=verdict,  # type: ignore[arg-type]
        evidence=[] if verdict == "ok" else [f"{local}: {verdict}"],
        reasoning="r",
    )


def scores(run: Run, **flags: Any) -> dict[str, AnomalyScore]:
    """Every judged span ok, except those given as local_id=(verdict, inputs_ok[, judge])."""
    out = {}
    for s in run.spans:
        if s.kind == "llm":
            continue
        local = s.span_id.split("/", 1)[1]
        key = local.replace(".", "__").replace("#", "_")
        spec = flags.get(key)
        out[s.span_id] = score(run, local, *spec) if spec else score(run, local)
    return out


def local(run: Run, ids: list[str]) -> list[str]:
    return [i.split("/", 1)[1] for i in ids]


def roles(run: Run, attr: Any) -> tuple[list[str], dict[str, list[str]]]:
    roots = local(run, [r.span_id for r in attr.roots])
    prop = {p.span_id.split("/", 1)[1]: local(run, p.path) for p in attr.propagated}
    return roots, prop


# ---------------------------------------------------------------- the ten hand-checked cases


def test_1_a_clean_run_has_no_roots(run: Run) -> None:
    attr = attribute(run, scores(run))
    assert attr.roots == [] and attr.propagated == [] and attr.anomalous == []


def test_2_a_bad_tool_call_propagates_through_its_step_and_downstream(run: Run) -> None:
    attr = attribute(
        run,
        scores(
            run,
            tool_agent__order_lookup=("faulty", 1.0, "stage-a"),
            tool_agent=("suspicious", 0.9),  # its inputs are fine; its call is not
            policy=("suspicious", 0.2),
            writer=("suspicious", 0.2),
        ),
    )
    roots, prop = roles(run, attr)
    assert roots == ["tool_agent.order_lookup"]
    assert prop == {
        "tool_agent": ["tool_agent.order_lookup", "tool_agent"],
        "policy": ["tool_agent.order_lookup", "tool_agent", "policy"],
        "writer": ["tool_agent.order_lookup", "tool_agent", "policy", "writer"],
    }
    g = build_graph(run)
    assert all(valid_path(g, p.path) for p in attr.propagated)
    assert all(p.caused_by == [attr.roots[0].span_id] for p in attr.propagated)


def test_3_a_step_with_trusted_inputs_is_its_own_root(run: Run) -> None:
    attr = attribute(run, scores(run, policy=("faulty", 0.9), writer=("suspicious", 0.1)))
    roots, prop = roles(run, attr)
    assert roots == ["policy"] and prop == {"writer": ["policy", "writer"]}


def test_4_two_independent_roots_rank_in_causal_order(run: Run) -> None:
    attr = attribute(
        run,
        scores(run, router=("faulty", 0.9), policy=("faulty", 0.9), writer=("suspicious", 0.1)),
    )
    roots, prop = roles(run, attr)
    assert roots == ["router", "policy"]  # earliest first
    assert prop == {"writer": ["policy", "writer"]}


def test_5_untrusted_inputs_point_upstream(run: Run) -> None:
    attr = attribute(run, scores(run, tool_agent=("faulty", 0.9), policy=("faulty", 0.2)))
    roots, prop = roles(run, attr)
    assert roots == ["tool_agent"] and prop == {"policy": ["tool_agent", "policy"]}


def test_6_an_anomaly_with_nothing_anomalous_upstream_is_a_root(run: Run) -> None:
    attr = attribute(run, scores(run, writer=("suspicious", 0.1)))
    assert roles(run, attr) == (["writer"], {})


def test_7_searches_judged_alone_are_ignored_unless_asked(run: Run) -> None:
    flagged = scores(run, retriever__search=("suspicious", 1.0, "judge_fast"))
    assert attribute(run, flagged).roots == []
    attr = attribute(run, flagged, judge_searches=True)
    # The retriever passes its search's hits on, so it is propagated, not a second root.
    assert roles(run, attr) == (
        ["retriever.search"],
        {"retriever": ["retriever.search", "retriever"]},
    )


def test_8_a_validator_flagged_search_explains_its_retriever(run: Run) -> None:
    attr = attribute(
        run,
        scores(
            run,
            retriever__search=("faulty", 1.0, "stage-a"),
            retriever=("suspicious", 0.9),
            policy=("suspicious", 0.3),
        ),
    )
    roots, prop = roles(run, attr)
    assert roots == ["retriever.search"]
    assert prop["retriever"] == ["retriever.search", "retriever"]
    assert prop["policy"] == ["retriever.search", "retriever", "policy"]


def test_9_more_confident_and_wider_roots_win_ties(run: Run) -> None:
    # Two calls in the same step: causal order first, so the earlier call ranks first.
    attr = attribute(
        run,
        scores(
            run,
            tool_agent__order_lookup=("suspicious", 1.0, "stage-a"),
            tool_agent__account_status=("faulty", 1.0, "stage-a"),
        ),
    )
    roots, prop = roles(run, attr)
    assert roots == ["tool_agent.order_lookup", "tool_agent.account_status"]
    assert attr.roots[0].confidence == 0.5 and attr.roots[1].confidence == 0.9


def test_10_a_call_fault_flows_through_a_step_its_judge_passed(run: Run) -> None:
    # The tool agent was judged ok, but it passes the refund quote on: still propagated.
    attr = attribute(
        run,
        scores(run, tool_agent__refund_api=("faulty", 1.0, "stage-a"), policy=("faulty", 0.2)),
    )
    roots, prop = roles(run, attr)
    assert roots == ["tool_agent.refund_api"]
    assert prop == {
        "tool_agent": ["tool_agent.refund_api", "tool_agent"],
        "policy": ["tool_agent.refund_api", "tool_agent", "policy"],
    }
    diag = attr.to_diagnosis(0.32)
    assert diag.root_causes and diag.propagated
    assert {c for p in diag.propagated for c in p.caused_by} <= {
        r.span_id for r in diag.root_causes
    }
    assert attr.role(diag.root_causes[0].span_id) == "root_cause"
    assert attr.role(f"{run.run_id}/router") == "ok"


# ---------------------------------------------------------------- helpers


def test_fault_types() -> None:
    assert fault_type("retriever.search", "retrieval", ["search_empty"]) == "retrieval_empty"
    assert fault_type("retriever.search", "retrieval", ["hit_score_mismatch"]) == "retrieval_wrong"
    assert fault_type("tool_agent.order_lookup", "tool", ["tool_error_payload"]) == "tool_malformed"
    assert fault_type("tool_agent.order_lookup", "tool", ["tool_consistency"]) == "tool_wrong_value"
    assert fault_type("policy", "agent", []) == "invalid_decision"
    assert fault_type("policy", "agent", ["numbers_supported"]) == "hallucination"
    assert fault_type("writer", "agent", []) == "hallucination"


def test_aggregate_prefers_the_ensemble(run: Run) -> None:
    a = score(run, "policy", "faulty", judge="judge_fast")
    b = score(run, "policy", "ok", judge="ensemble")
    c = score(run, "policy.llm", "faulty", judge="inherited:ensemble")
    best = aggregate_scores([a, b, c])
    assert best == {b.span_id: b}


def test_an_invalid_path_is_caught(run: Run) -> None:
    g = build_graph(run)
    rid = run.run_id
    assert valid_path(g, [f"{rid}/router", f"{rid}/tool_agent"])  # data edge
    assert valid_path(g, [f"{rid}/tool_agent.order_lookup", f"{rid}/tool_agent"])  # call -> step
    assert not valid_path(g, [f"{rid}/writer", f"{rid}/router"])  # backwards


# ---------------------------------------------------------------- the command


def test_attribution_check_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    f4 = next(
        s
        for s in DATASET
        if s.ground_truth.fault_code == "F4" and s.faults[0].injector == "tool_malformed"
    )
    with TraceStore.open() as store:
        for sc in (CONTROL, f4):
            _, r = verify_scenario(
                make_gateway(tmp_path / sc.scenario_id), sc, ROLES, "default", {}
            )
            assert r is not None
            store.save_run(r)
            root = sc.faults[0].target_node_id if sc.faults else None
            for s in r.spans:
                if s.kind == "llm":
                    continue
                here = s.span_id.split("/", 1)[1]
                if here == root:  # Stage A stores the validators' verdict for it
                    store.add_anomaly(score(r, here, "faulty", 1.0, "stage-a"))
                else:
                    store.add_anomaly(score(r, here))  # every judge said ok
    monkeypatch.setattr(attribution_check, "REPORT", tmp_path / "a.json")
    monkeypatch.setattr(attribution_check, "ROOT", tmp_path)
    ids = [f"{CONTROL.scenario_id}-default", f"{f4.scenario_id}-default"]
    assert attribution_check.main(["--runs", *ids]) == 0
    out = capsys.readouterr().out
    assert "controls with no root (no false alarm)   1/1" in out
    # The F4 fault is settled by the validators even though every judge said ok.
    assert "[TOP-1]" in out and "runs whose propagated paths are all valid 2/2" in out
    report = json.loads((tmp_path / "a.json").read_text("utf-8"))
    f4_row = next(r for r in report["rows"] if r["fault_code"] == "F4")
    assert f4_row["top1_node"] and f4_row["roots"][0]["fault_type"] == "tool_malformed"


# ---------------------------------------------------------------- ranking by evidence


def test_proof_outranks_an_earlier_hunch(run: Run) -> None:
    # sc-012 on the dev set: a judge's soft flag on the retriever came first in causal
    # order, but the account_status fault is validator-proven.
    attr = attribute(
        run,
        scores(
            run,
            retriever=("suspicious", 0.9),
            tool_agent__account_status=("faulty", 1.0, "stage-a"),
        ),
    )
    roots, _ = roles(run, attr)
    assert roots == ["tool_agent.account_status", "retriever"]


def test_a_judge_faulty_outranks_an_earlier_judge_suspicious(run: Run) -> None:
    attr = attribute(run, scores(run, router=("suspicious", 0.9), policy=("faulty", 0.9)))
    assert roles(run, attr)[0] == ["policy", "router"]


def test_spans_of_one_node_are_one_root(run: Run) -> None:
    searches = [s.span_id.split("/", 1)[1] for s in run.spans if s.node_id == "retriever.search"]
    assert len(searches) >= 2
    flags = {
        name.replace(".", "__").replace("#", "_"): ("faulty", 1.0, "stage-a") for name in searches
    }
    attr = attribute(run, scores(run, policy=("suspicious", 0.2), **flags))
    roots, prop = roles(run, attr)
    assert roots == ["retriever.search"]
    assert any("the same fault also at" in e for e in attr.roots[0].evidence)
    assert prop["retriever"][0] == "retriever.search"
    assert all(p.caused_by == [attr.roots[0].span_id] for p in attr.propagated)
    g = build_graph(run)
    assert all(valid_path(g, p.path) for p in attr.propagated)
