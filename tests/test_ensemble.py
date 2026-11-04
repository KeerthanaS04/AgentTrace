"""P6-T4: the judge ensemble's weighted vote, per-judge verdicts, and agreement (kappa)."""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.diagnosis import ensemble as ens_mod
from agenttrace.diagnosis.ensemble import (
    ENSEMBLE,
    JUDGES,
    cohens_kappa,
    ensemble,
    fleiss_kappa,
    load_weights,
    pooled_agreement,
    vote,
)
from agenttrace.diagnosis.stage_a import STAGE, score_run
from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios.ground_truth import FaultInjection, Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.diagnosis import AnomalyScore
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)
# Which verdict each judge model gives the Policy Agent; every other span is "ok".
POLICY_VERDICT = {
    "muse-spark-1.3": "ok",
    "nvidia/nemotron-3-ultra-550b-a55b": "faulty",
    "qwen/qwen-2.5-72b-instruct": "suspicious",
}


def judges(prompt: str, model: str) -> dict[str, Any]:
    if not prompt.startswith("You are auditing"):
        return default_script(prompt)
    span = re.search(r"^Span: (\S+)", prompt, re.M)
    assert span is not None
    verdict = POLICY_VERDICT.get(model, "ok") if span.group(1) == "policy" else "ok"
    return {
        "inputs_ok": 0.9,
        "inputs_reasoning": "r",
        "output_ok_given_inputs": {"ok": 0.9, "suspicious": 0.5, "faulty": 0.1}[verdict],
        "output_reasoning": "r",
        "verdict": verdict,
        "evidence": [{"ref": span.group(1), "claim": f"{model} says {verdict}"}],
    }


def recorded(tmp_path: Path, scenario: Scenario = CONTROL) -> Run:
    _, run = verify_scenario(make_gateway(tmp_path / "rec"), scenario, ROLES, "default", {})
    assert run is not None
    return run


def stage(tmp_path: Path, run: Run, **kw: Any) -> Any:
    gateway = make_gateway(tmp_path / "j", by_model=kw.pop("by_model", judges))
    return score_run(run, gateway, roles=JUDGES, **kw)


# ---------------------------------------------------------------- agreement statistics


def test_cohens_kappa() -> None:
    a, b = list("yynny"), list("ynnny")
    assert math.isclose(cohens_kappa(a, b) or 0, (0.8 - 0.48) / 0.52)
    assert cohens_kappa(list("abc"), list("abc")) == 1.0
    assert cohens_kappa(list("aabb"), list("abab")) == 0.0
    assert cohens_kappa(list("aaa"), list("aaa")) == 1.0  # degenerate: full agreement
    assert cohens_kappa([], []) is None
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_fleiss_kappa() -> None:
    assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == 1.0
    assert math.isclose(fleiss_kappa([["a", "a", "b"], ["a", "b", "b"]]) or 0, -1 / 3)
    assert fleiss_kappa([]) is None
    with pytest.raises(ValueError):
        fleiss_kappa([["a", "b"], ["a"]])


def _score(verdict: str, out: float = 0.5) -> AnomalyScore:
    return AnomalyScore(
        span_id="r/x",
        judge="j",
        prompt_version="v",
        output_ok_given_inputs=out,
        inputs_ok=0.8,
        verdict=verdict,  # type: ignore[arg-type]
        evidence=["x: e"],
        reasoning="r",
    )


def test_weighted_vote_and_severity_tie_break() -> None:
    three = {"a": _score("ok", 0.9), "b": _score("faulty", 0.1), "c": _score("suspicious", 0.5)}
    verdict, dis, _, out = vote(three, {"a": 1, "b": 1, "c": 1})
    assert verdict == "faulty" and math.isclose(dis, 2 / 3, abs_tol=1e-3)  # 3-way tie
    assert math.isclose(out, 0.5)
    verdict, dis, _, _ = vote(three, {"a": 3, "b": 1, "c": 1})
    assert verdict == "ok" and math.isclose(dis, 0.4)
    two = {"a": _score("ok"), "b": _score("ok")}
    assert vote(two, {})[:2] == ("ok", 0.0)


def test_weights_config(tmp_path: Path) -> None:
    assert set(load_weights()) == set(JUDGES)
    bad = tmp_path / "e.yaml"
    bad.write_text("weights: {judge_fast: -1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_weights(bad)


# ---------------------------------------------------------------- the ensemble on a run


def test_per_judge_verdicts_aggregate_and_agreement(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    ens = ensemble(stage(tmp_path, run), {j: 1.0 for j in JUDGES})
    policy = ens.spans[f"{run.run_id}/policy"]
    assert policy.votes == {"judge_fast": "ok", "judge_nvidia": "faulty", "judge_oss": "suspicious"}
    assert policy.verdict == "faulty" and policy.score is not None
    assert policy.score.judge == ENSEMBLE and math.isclose(policy.score.output_ok_given_inputs, 0.5)
    assert [s.local_id for s in ens.split_spans()] == ["policy"]
    # Everyone agrees on every other span, so kappa is high but not 1 (one split).
    assert ens.agreement is not None and 0 < ens.agreement < 1 and ens.fleiss is not None
    assert set(ens.pairwise) == {
        "judge_fast|judge_nvidia",
        "judge_fast|judge_oss",
        "judge_nvidia|judge_oss",
    }
    llm = ens.spans[f"{run.run_id}/policy.llm"]
    assert llm.source == "inherited" and llm.verdict == "faulty"


def test_spans_settled_without_judges_keep_their_score_and_skip_agreement(
    tmp_path: Path,
) -> None:
    broken = FaultInjection(
        target_node_id="tool_agent.account_status",
        injector="tool_malformed",
        params={"mode": "error", "occurrence": 1},
    )
    data = CONTROL.model_dump(mode="json")
    data.update(scenario_id="sc-x", faults=[broken.model_dump(mode="json")])
    data["ground_truth"].update(fault_code="F4", root_cause_node_ids=[broken.target_node_id])
    run = recorded(tmp_path, Scenario.model_validate(data))
    ens = ensemble(stage(tmp_path, run))
    tool = ens.spans[f"{run.run_id}/tool_agent.account_status"]
    assert tool.source == "validators" and tool.votes == {} and tool.verdict == "faulty"
    assert tool.score is not None and tool.score.judge == STAGE


def test_a_judge_without_a_valid_reply_does_not_vote(tmp_path: Path) -> None:
    def oss_silent(prompt: str, model: str) -> dict[str, Any]:
        if model.startswith("qwen/") and prompt.startswith("You are auditing"):
            return {"verdict": "fine"}
        return judges(prompt, model)

    run = recorded(tmp_path)
    ens = ensemble(stage(tmp_path, run, by_model=oss_silent))
    policy = ens.spans[f"{run.run_id}/policy"]
    assert set(policy.votes) == {"judge_fast", "judge_nvidia"}  # judge_oss failed: no vote


def test_pooled_agreement_over_runs(tmp_path: Path) -> None:
    runs = [recorded(tmp_path / str(i)) for i in range(2)]
    results = [ensemble(stage(tmp_path / f"s{i}", r)) for i, r in enumerate(runs)]
    pooled = pooled_agreement(results)
    assert set(pooled) == {
        *(f"{a}|{b}" for a, b in itertools.combinations(JUDGES, 2)),
        "fleiss",
        "mean_pairwise",
    }
    assert pooled["mean_pairwise"] is not None


# ---------------------------------------------------------------- the command


def test_the_ensemble_command(
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
            lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}", by_model=judges)
        ),
    )
    monkeypatch.setattr(ens_mod, "ROOT", tmp_path)
    assert ens_mod.main(["--runs", run.run_id]) == 0
    out = capsys.readouterr().out
    assert "judges judge_fast, judge_nvidia, judge_oss" in out
    assert "agreement over all" in out and "fleiss" in out and "PASS" in out
    report = json.loads((tmp_path / "reports" / "ensemble.json").read_text("utf-8"))
    [row] = report["runs"]
    policy = next(s for s in row["spans"] if s["span_id"].endswith("/policy"))
    assert policy["verdict"] == "faulty" and len(policy["votes"]) == 3
    with TraceStore.open() as store:
        assert store.anomalies(run_id=run.run_id, judge=ENSEMBLE)
        assert store.anomalies(run_id=run.run_id, judge="judge_oss")


def test_a_rejected_key_disables_that_judge_instead_of_crashing(tmp_path: Path) -> None:
    from agenttrace.judges.node_judge import judge_span

    def unauthorized(prompt: str, seed: Any) -> int:
        return 401 if prompt.startswith("You are auditing") else 200

    run = recorded(tmp_path)
    bad = make_gateway(tmp_path / "bad", by_model=judges, status=unauthorized)
    res = judge_span(bad, run, "policy", role="judge_oss")
    assert res.score is None and res.auth_error and "401" in (res.provider_error or "")
    # Stage A carries on: the judge is disabled for the run after its first rejection.
    stage_res = score_run(run, bad, roles=("judge_oss",))
    assert "judge_oss" in stage_res.disabled_roles and stage_res.complete
    assert stage_res.model_calls == 0
