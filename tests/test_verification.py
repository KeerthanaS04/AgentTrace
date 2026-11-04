"""P7-T1: the verification loop verifies or demotes candidates and stores linked replays."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.diagnosis.verification import (
    RecoveryCheck,
    default_patch_for,
    judged_recovery,
    patch_target,
    verify,
)
from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios import verification_check
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.diagnosis import Diagnosis, RootCause
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

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
F3 = next(
    s for s in DATASET if s.ground_truth.fault_code == "F3" and s.ticket.ticket_id == "T-0101"
)
POLICY_OK = {
    "decision": "approve",
    "refund_amount": 41.05,
    "escalate_to": None,
    "cited_doc_ids": ["KB-REF-001"],
    "justification": "Within 30 days.",
}


def patcher(prompt: str) -> dict[str, Any]:
    """Scripted model; the patch proposal fixes Policy, and fits no other node."""
    if prompt.startswith("You are repairing"):
        return {
            "output": POLICY_OK,
            "rationale": "the decision contradicts KB-REF-001",
            "confidence": 0.8,
        }
    if prompt.startswith("You are auditing"):
        span = re.search(r"^Span: (\S+)", prompt, re.M)
        return {
            "inputs_ok": 0.9,
            "inputs_reasoning": "r",
            "output_ok_given_inputs": 0.9,
            "output_reasoning": "r",
            "verdict": "ok",
            "evidence": [{"ref": span.group(1) if span else "policy", "claim": "fine"}],
        }
    return default_script(prompt)


def recorded(tmp_path: Path, scenario: Scenario = F5) -> Run:
    _, run = verify_scenario(make_gateway(tmp_path / "rec"), scenario, ROLES, "default", {})
    assert run is not None
    return run


def root(run: Run, local: str, confidence: float = 0.8) -> RootCause:
    node = next(s.node_id for s in run.spans if s.span_id == f"{run.run_id}/{local}")
    return RootCause(
        node_id=node,
        span_id=f"{run.run_id}/{local}",
        fault_type="other",
        confidence=confidence,
        evidence=[f"{local}: suspected"],
    )


def diagnosis(run: Run, *locals_: str) -> Diagnosis:
    return Diagnosis(
        diagnosis_id=f"{run.run_id}-stage-b-v1",
        run_id=run.run_id,
        prompt_version="stage-b-v1",
        root_causes=[root(run, x) for x in locals_],
        judge_agreement=0.3,
    )


def correct(original: Run, replay: Run) -> RecoveryCheck:
    ok = replay.final_output.get("decision") == "approve"
    return RecoveryCheck(recovered=ok, score=1.0 if ok else 0.0, detail="truth")


# ---------------------------------------------------------------- the loop


def test_a_wrong_candidate_is_demoted_and_the_right_one_verified(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    assert run.final_output["decision"] == "deny"
    gw = make_gateway(tmp_path / "g", script=patcher)
    out = verify(
        run,
        diagnosis(run, "router", "policy"),
        WORKFLOW,
        make_resources(gw),
        patch_for=default_patch_for(gw, make_resources(gw)),
        recovery=correct,
    )
    router, policy = out.diagnosis.root_causes
    # Router: no rule applies and the model's proposal does not fit it -> no attempt spent.
    assert router.status == "unverified" and router.confidence == 0.4
    assert out.attempts[0].replay_run is None and "no patch" in out.attempts[0].detail
    assert policy.status == "verified" and policy.confidence == 0.8
    assert out.verified is not None and out.verified.node_id == "policy"
    [result] = out.replays
    assert result.diagnosis_id == f"{run.run_id}-stage-b-v1"  # linked to the diagnosis
    assert result.recovered and result.patch_source == "model"
    assert result.patched_span_id == f"{run.run_id}/policy"
    assert result.replay_final["decision"] == "approve"


def test_the_loop_stops_at_the_attempt_cap(tmp_path: Path) -> None:
    run = recorded(tmp_path, F3)
    gw = make_gateway(tmp_path / "g", script=patcher)
    never = lambda o, r: RecoveryCheck(recovered=False, score=0.0)  # noqa: E731
    diag = diagnosis(
        run,
        "tool_agent.order_lookup",
        "tool_agent.account_status",
        "tool_agent.refund_api",
        "retriever.search",
    )
    out = verify(
        run,
        diag,
        WORKFLOW,
        make_resources(gw),
        patch_for=default_patch_for(gw, make_resources(gw)),
        recovery=never,
    )
    assert len(out.replays) == 3  # the cap
    statuses = [r.status for r in out.diagnosis.root_causes]
    assert statuses == ["unverified", "unverified", "unverified", "candidate"]
    assert all(r.confidence == 0.4 for r in out.diagnosis.root_causes[:3])
    assert all(a.patch is not None and a.patch.patch_source == "rule" for a in out.attempts)


def test_the_loop_ends_when_candidates_run_out(tmp_path: Path) -> None:
    run = recorded(tmp_path, F3)
    gw = make_gateway(tmp_path / "g")
    never = lambda o, r: RecoveryCheck(recovered=False, score=0.0)  # noqa: E731
    out = verify(
        run,
        diagnosis(run, "tool_agent.order_lookup"),
        WORKFLOW,
        make_resources(gw),
        patch_for=default_patch_for(gw, make_resources(gw)),
        recovery=never,
    )
    assert len(out.attempts) == 1 and out.verified is None


def test_patch_targets(tmp_path: Path) -> None:
    run = recorded(tmp_path, F3)
    assert patch_target(run, root(run, "tool_agent.order_lookup")) == "tool_agent.order_lookup"
    assert patch_target(run, root(run, "retriever.search")) == "retriever"  # ran several times
    assert patch_target(run, root(run, "policy")) == "policy"


def test_other_faults_are_requested_per_candidate(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    gw = make_gateway(tmp_path / "g", script=patcher)
    seen: list[str] = []

    def faults_for(r: RootCause) -> None:
        seen.append(r.node_id)

    verify(
        run,
        diagnosis(run, "policy"),
        WORKFLOW,
        make_resources(gw),
        patch_for=default_patch_for(gw, make_resources(gw)),
        recovery=correct,
        faults_for=faults_for,
    )
    assert seen == ["policy"]


# ---------------------------------------------------------------- the truth-free check


def test_judged_recovery(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    prompts: list[str] = []
    gw = make_gateway(tmp_path / "g", script=patcher, prompts=prompts)
    check = judged_recovery(gw)
    same = check(run, run)
    assert not same.recovered and "did not change" in same.detail and prompts == []
    out = verify(
        run,
        diagnosis(run, "policy"),
        WORKFLOW,
        make_resources(gw),
        patch_for=default_patch_for(gw, make_resources(gw)),
        recovery=check,
    )
    assert out.verified is not None  # the outcome changed and the judge passes it
    assert out.attempts[0].detail.startswith("the judge passes the new outcome (")


# ---------------------------------------------------------------- the command


def test_verify_diagnoses_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    run = recorded(tmp_path)
    control = recorded(
        tmp_path / "c",
        next(
            s
            for s in DATASET
            if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
        ),
    )
    with TraceStore.open() as store:
        store.save_run(run)
        store.save_run(control)
        store.save_diagnosis(diagnosis(run, "router", "policy"))
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(
            lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}", script=patcher)
        ),
    )
    monkeypatch.setattr(verification_check, "REPORT", tmp_path / "v.json")
    monkeypatch.setattr(verification_check, "ROOT", tmp_path)
    assert verification_check.main(["--runs", control.run_id, run.run_id, "--compare-judge"]) == 0
    out = capsys.readouterr().out
    assert "outcome already correct: skipped" in out
    assert "verified: policy" in out and "[right]" in out
    assert "failing runs verified: 1/1; the verified root is a true root: 1/1" in out
    assert "judged (truth-free) recovery agrees with the truth check: 1/1" in out
    with TraceStore.open() as store:
        [replay] = store.replays(diagnosis_id=f"{run.run_id}-stage-b-v1")
        assert replay.recovered and store.has_run(replay.replay_run_id)
        [diag] = store.diagnoses(run_id=run.run_id)
        assert [r.status for r in diag.root_causes] == ["unverified", "verified"]
    report = json.loads((tmp_path / "v.json").read_text("utf-8"))
    assert report["rows"][0]["verified_right"]


def test_a_target_already_tried_is_not_replayed_again(tmp_path: Path) -> None:
    run = recorded(tmp_path, F3)
    gw = make_gateway(tmp_path / "g")
    never = lambda o, r: RecoveryCheck(recovered=False, score=0.0)  # noqa: E731
    searches = [s.span_id.split("/", 1)[1] for s in run.spans if s.node_id == "retriever.search"]
    diag = diagnosis(run, searches[0], searches[1])  # both patch the retriever step
    out = verify(
        run,
        diag,
        WORKFLOW,
        make_resources(gw),
        patch_for=default_patch_for(gw, make_resources(gw)),
        recovery=never,
    )
    assert len(out.replays) == 1 and "already patched" in out.attempts[1].detail


def test_judged_recovery_accepts_suspicious_but_not_faulty(tmp_path: Path) -> None:
    run = recorded(tmp_path)

    def verdict_for(writer: str) -> Any:
        def script(prompt: str) -> dict[str, Any]:
            out = patcher(prompt)
            if prompt.startswith("You are auditing") and "Span: writer " in prompt:
                out = {
                    **out,
                    "verdict": writer,
                    "evidence": [{"ref": "writer", "claim": "unsure about the timeline"}],
                }
            return out

        return script

    for writer, expected in (("suspicious", True), ("faulty", False)):
        gw = make_gateway(tmp_path / writer, script=verdict_for(writer))
        out = verify(
            run,
            diagnosis(run, "policy"),
            WORKFLOW,
            make_resources(gw),
            patch_for=default_patch_for(gw, make_resources(gw)),
            recovery=judged_recovery(gw),
        )
        assert (out.verified is not None) is expected, writer
