"""P6-T3: Stage A gives every span an AnomalyScore and logs cost per run."""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.diagnosis import stage_a
from agenttrace.diagnosis.stage_a import STAGE, score_run
from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios import devset
from agenttrace.scenarios.ground_truth import FaultInjection, Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)
USAGE = {"prompt_tokens": 1000, "completion_tokens": 200}


def judge(prompt: str) -> dict[str, Any]:
    if prompt.startswith("You are auditing"):
        span = re.search(r"^Span: (\S+)", prompt, re.M)
        assert span is not None
        return {
            "inputs_ok": 0.9,
            "inputs_reasoning": "fine",
            "output_ok_given_inputs": 0.8,
            "output_reasoning": "fine",
            "verdict": "ok",
            "evidence": [{"ref": span.group(1), "claim": "consistent"}],
        }
    return default_script(prompt)


def with_faults(*faults: FaultInjection, code: str = "F4") -> Scenario:
    data = CONTROL.model_dump(mode="json")
    data["scenario_id"] = "sc-test"
    data["faults"] = [f.model_dump(mode="json") for f in faults]
    data["ground_truth"].update(
        fault_code=code,
        root_cause_node_ids=list(dict.fromkeys(f.target_node_id for f in faults)),
        affected_node_ids=[],
    )
    return Scenario.model_validate(data)


def recorded(tmp_path: Path, scenario: Scenario = CONTROL, script: Any = None) -> Run:
    gw = make_gateway(tmp_path / "rec", script=script) if script else make_gateway(tmp_path / "rec")
    _, run = verify_scenario(gw, scenario, ROLES, "default", {})
    assert run is not None
    return run


def score(tmp_path: Path, run: Run, **kw: Any) -> stage_a.StageAResult:
    gateway = make_gateway(tmp_path / "judge", script=kw.pop("script", judge), usage=USAGE)
    return score_run(run, gateway, **kw)


# ---------------------------------------------------------------- every span gets a score


def test_every_span_of_a_clean_run_is_scored(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    res = score(tmp_path, run)
    assert res.complete and len(res.spans) == len(run.spans)
    assert {s.span_id for s in res.spans} == {s.span_id for s in run.spans}
    src = res.sources()
    judged = sum(s.kind in ("agent", "tool", "retrieval") for s in run.spans)
    llm = sum(s.kind == "llm" for s in run.spans)
    assert src == {"input": 1, "judge": judged, "inherited": llm}
    assert res.judge_calls == judged and res.judge_failures == 0
    assert res.cost_usd > 0 and res.tokens_in == judged * USAGE["prompt_tokens"]


def test_model_calls_inherit_their_steps_score(tmp_path: Path) -> None:
    res = score(tmp_path, recorded(tmp_path)).by_span()
    run_id = next(iter(res)).split("/")[0]
    llm = res[f"{run_id}/policy.llm"]
    assert llm.source == "inherited" and list(llm.scores) == ["inherited:judge_fast"]
    copy, step = llm.scores["inherited:judge_fast"], res[f"{run_id}/policy"].scores["judge_fast"]
    assert copy.span_id == f"{run_id}/policy.llm" and copy.verdict == step.verdict
    assert copy.reasoning.startswith("Inherited from step policy")


def test_validator_errors_skip_the_judge(tmp_path: Path) -> None:
    broken = FaultInjection(
        target_node_id="tool_agent.account_status",
        injector="tool_malformed",
        params={"mode": "error", "occurrence": 1},
    )
    run = recorded(tmp_path, with_faults(broken))
    prompts: list[str] = []
    gateway = make_gateway(tmp_path / "j", script=judge, prompts=prompts)
    res = score_run(run, gateway)
    entry = res.by_span()[f"{run.run_id}/tool_agent.account_status"]
    assert entry.source == "validators" and list(entry.scores) == [STAGE]
    s = entry.scores[STAGE]
    assert s.verdict == "faulty" and s.output_ok_given_inputs == 0.0
    assert any("tool_error_payload" in e for e in s.evidence)
    assert not any("Span: tool_agent.account_status " in p for p in prompts)  # never judged
    assert res.short_circuited >= 1


def test_deterministic_inputs_ok_comes_from_the_producers(tmp_path: Path) -> None:
    empty = FaultInjection(target_node_id="retriever.search", injector="retrieval_empty", params={})
    run = recorded(tmp_path, with_faults(empty, code="F2"))
    res = score(tmp_path, run).by_span()
    retriever = res[f"{run.run_id}/retriever"]
    assert retriever.source == "validators"  # docs_nonempty is an error
    # Its input (the router's category) was judged 0.8 output_ok earlier in causal order.
    assert retriever.scores[STAGE].inputs_ok == 0.8


def test_validator_warnings_ride_along_as_evidence(tmp_path: Path) -> None:
    res = score(tmp_path, recorded(tmp_path)).by_span()
    writer = next(v for k, v in res.items() if k.endswith("/writer"))
    assert writer.source == "judge"
    assert any(
        "[validator numbers_supported, warning]" in e for e in writer.scores["judge_fast"].evidence
    )
    assert writer.scores["judge_fast"].verdict == "ok"  # warnings do not override the judge


def test_failed_and_skipped_steps(tmp_path: Path) -> None:
    def broken_tool_agent(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the Tool Agent"):
            return {"not": "valid"}
        return default_script(prompt)

    run = recorded(tmp_path, script=broken_tool_agent)
    res = score(tmp_path, run).by_span()
    assert res[f"{run.run_id}/tool_agent"].source == "failed"
    assert res[f"{run.run_id}/tool_agent"].scores[STAGE].verdict == "faulty"
    for step in ("policy", "writer"):
        entry = res[f"{run.run_id}/{step}"]
        assert entry.source == "skipped" and entry.scores[STAGE].verdict == "suspicious"


def test_a_judge_that_never_answers_validly(tmp_path: Path) -> None:
    def silent_on_policy(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are auditing") and "Span: policy " in prompt:
            return {"verdict": "fine"}
        return judge(prompt)

    run = recorded(tmp_path)
    res = score(tmp_path, run, script=silent_on_policy)
    entry = res.by_span()[f"{run.run_id}/policy"]
    assert entry.source == "no answer" and res.judge_failures == 1 and res.complete
    s = entry.scores["judge_fast"]
    assert (s.verdict, s.inputs_ok, s.output_ok_given_inputs) == ("suspicious", 0.5, 0.5)


def test_several_fast_judges(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    res = score(tmp_path, run, roles=("judge_fast", "judge_nvidia"))
    policy = res.by_span()[f"{run.run_id}/policy"]
    assert set(policy.scores) == {"judge_fast", "judge_nvidia"}
    assert res.judge_calls == 2 * res.sources()["judge"]


# ---------------------------------------------------------------- storage and commands


def test_scores_are_stored_once_per_span_judge_and_prompt(tmp_path: Path) -> None:
    from agenttrace.store import TraceStore

    run = recorded(tmp_path)
    res = score(tmp_path, run)
    with TraceStore.open(tmp_path / "s.db") as store:
        for _ in range(2):  # re-scoring replaces, never duplicates
            for s in res.all_scores():
                store.add_anomaly(s)
        stored = store.anomalies(run_id=run.run_id)
        assert len(stored) == len(res.all_scores())
        assert {s.judge for s in store.anomalies(run_id=run.run_id, judge="judge_fast")} == {
            "judge_fast"
        }


def test_the_stage_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    run = recorded(tmp_path)
    with TraceStore.open() as store:
        store.save_run(run)
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(
            lambda cls, *a, **k: make_gateway(
                tmp_path / f"g{next(count)}", script=judge, usage=USAGE
            )
        ),
    )
    root = tmp_path / "root"
    monkeypatch.setattr(stage_a, "__file__", str(root / "agenttrace" / "diagnosis" / "x.py"))
    assert stage_a.main(["--runs", run.run_id]) == 0
    out = capsys.readouterr().out
    assert "every span scored: 1/1 runs" in out and "per run" in out and "PASS" in out
    report = json.loads((root / "reports" / "stage_a.json").read_text("utf-8"))
    [row] = report["runs"]
    assert row["complete"] and row["cost_usd"] > 0 and row["scored"] == row["spans"]
    with TraceStore.open() as store:
        assert len(store.anomalies(run_id=run.run_id)) >= row["spans"]


def test_the_dev_set_is_stratified_ids_only(tmp_path: Path) -> None:
    by_code: dict[str, list[str]] = {}
    for s in DATASET:
        by_code.setdefault(s.ground_truth.fault_code, []).append(s.scenario_id)
    rows = [
        {"scenario_id": sid, "behaves": True}
        for code in sorted(by_code)
        for sid in by_code[code][:3]
    ]
    report = tmp_path / "multirun.json"
    report.write_text(json.dumps({"rows": {"1": rows}}), encoding="utf-8")
    ids = devset.pick_devset(2, report)
    assert len(ids) == 2 * len(by_code) and all(i.endswith("-default-s1") for i in ids)
    assert all(re.fullmatch(r"sc-\d{3}-default-s1", i) for i in ids)  # nothing but ids
