"""P5-T4: oracle, rule, and model patch sources, each end to end, tagged on the replay."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.executor import execute_plan
from agenttrace.replay.patcher import patch_fields, plan_patch
from agenttrace.replay.sources import PatchSourceError, model_patch, rule_patch
from agenttrace.scenarios import patch_check
from agenttrace.scenarios.ground_truth import FaultInjection, Scenario, load_dataset
from agenttrace.scenarios.oracle import oracle_patch
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.diagnosis import ReplaceOutputPatch
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
WORKFLOW = build_support_workflow(ROLES)


def find(code: str, injector: str | None = None, **params: Any) -> Scenario:
    for s in DATASET:
        if s.ground_truth.fault_code != code or s.ticket.ticket_id != "T-0101":
            continue
        if injector is None or any(
            f.injector == injector and all(f.params.get(k) == v for k, v in params.items())
            for f in s.faults
        ):
            return s
    raise LookupError((code, injector, params))


def with_faults(*faults: FaultInjection, code: str = "F4") -> Scenario:
    data = find("F0").model_dump(mode="json")
    data["scenario_id"] = "sc-test"
    data["faults"] = [f.model_dump(mode="json") for f in faults]
    data["ground_truth"].update(
        fault_code=code,
        root_cause_node_ids=list(dict.fromkeys(f.target_node_id for f in faults)),
        affected_node_ids=[],
    )
    return Scenario.model_validate(data)


def recorded(tmp_path: Path, scenario: Scenario, **gw: Any) -> Run:
    _, run = verify_scenario(make_gateway(tmp_path / "rec", **gw), scenario, ROLES, "default", {})
    assert run is not None
    return run


def spans(run: Run) -> dict[str, Any]:
    return {s.span_id.split("/", 1)[1]: s for s in run.spans}


def replay(tmp_path: Path, run: Run, patch: Any, **gw: Any) -> Run:
    plan = plan_patch(run, WORKFLOW, patch)
    gateway = make_gateway(tmp_path / "replay", **gw)
    out = execute_plan(run, WORKFLOW, plan, make_resources(gateway))
    assert out.violations() == []
    return out.replay


def clean_value(tmp_path: Path, local: str) -> dict[str, Any]:
    return dict(spans(recorded(tmp_path / "clean", find("F0")))[local].output)


# ---------------------------------------------------------------- rule


def test_rule_recalls_a_tool_that_returned_a_wrong_value(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F3"))  # days_since_delivery set to 45
    assert spans(run)["tool_agent.order_lookup"].output["order"]["days_since_delivery"] == 45
    patch = rule_patch(run, "tool_agent.order_lookup", make_resources(make_gateway(tmp_path)))
    assert patch is not None and patch.patch_source == "rule" and "re-called" in patch.note
    assert patch.output == clean_value(tmp_path, "tool_agent.order_lookup")
    out = replay(tmp_path, run, patch)
    assert out.patch_source == "rule" and out.replay_of == run.run_id


def test_rule_recalls_a_tool_that_returned_an_error(tmp_path: Path) -> None:
    broken = FaultInjection(
        target_node_id="tool_agent.account_status",
        injector="tool_malformed",
        params={"mode": "error", "occurrence": 1},
    )
    run = recorded(tmp_path, with_faults(broken))
    assert "error" in spans(run)["tool_agent.account_status"].output
    patch = rule_patch(run, "tool_agent.account_status", make_resources(make_gateway(tmp_path)))
    assert patch is not None and patch.output["found"] is True
    assert patch.output == clean_value(tmp_path, "tool_agent.account_status")
    replay(tmp_path, run, patch)


def test_rule_reruns_retrieval(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F2", "retrieval_empty"))
    resources = make_resources(make_gateway(tmp_path))
    assert spans(run)["retriever"].output["docs"] == []
    step = rule_patch(run, "retriever", resources)
    assert step is not None and step.output == clean_value(tmp_path, "retriever")
    one = rule_patch(run, "retriever.search", resources)
    assert one is not None and one.output == clean_value(tmp_path, "retriever.search")
    assert one.output["hits"]
    replay(tmp_path, run, step)


def test_rule_recalls_every_tool_of_the_tool_agent(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F3"))
    patch = rule_patch(run, "tool_agent", make_resources(make_gateway(tmp_path)))
    assert patch is not None
    assert patch.output == clean_value(tmp_path, "tool_agent")


def test_no_rule_for_model_judgements(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F0"))
    resources = make_resources(make_gateway(tmp_path))
    for target in ("router", "policy", "writer", "policy.llm", "__input__"):
        assert rule_patch(run, target, resources) is None


# ---------------------------------------------------------------- model


def patcher(prompt: str) -> dict[str, Any]:
    """The scripted model, also answering the patch prompt for the Policy Agent."""
    if prompt.startswith("You are repairing"):
        fixed = {
            "decision": "approve",
            "refund_amount": 41.05,
            "escalate_to": None,
            "cited_doc_ids": ["KB-REF-001"],
            "justification": "Within 30 days.",
        }
        return {
            "output": fixed,
            "rationale": "the decision contradicts KB-REF-001",
            "confidence": 0.8,
        }
    return default_script(prompt)


def test_model_proposes_a_fix_from_the_inputs_only(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F5", "set_field", field="decision"))
    assert run.final_output["decision"] == "deny"
    prompts: list[str] = []
    gateway = make_gateway(tmp_path / "m", script=patcher, prompts=prompts)
    patch = model_patch(run, "policy", gateway, hint="decision looks inconsistent")
    assert patch.patch_source == "model" and patch.output["decision"] == "approve"
    assert "0.80" in patch.note and "KB-REF-001" in patch.note
    asked = [p for p in prompts if p.startswith("You are repairing")]
    assert len(asked) == 1 and "decision looks inconsistent" in asked[0]
    # The model gets the step's contract, inputs, and suspect output, never ground truth.
    assert "Decide approve, deny, or escalate" in asked[0]
    for leak in ("expected_final", "root_cause", "fault_code", "set_field", "F5"):
        assert leak not in asked[0]
    out = replay(tmp_path, run, patch)
    assert out.patch_source == "model" and out.final_output["decision"] == "approve"


def test_model_patch_with_wrong_keys_or_no_answer_fails(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F5", "set_field", field="decision"))

    def wrong_keys(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are repairing"):
            return {"output": {"decision": "approve"}, "rationale": "r", "confidence": 0.5}
        return default_script(prompt)

    with pytest.raises(PatchSourceError, match="returned keys"):
        model_patch(run, "policy", make_gateway(tmp_path / "a", script=wrong_keys))

    def no_json(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are repairing"):
            return {"nonsense": True}
        return default_script(prompt)

    with pytest.raises(PatchSourceError, match="no valid patch"):
        model_patch(run, "policy", make_gateway(tmp_path / "b", script=no_json))


# ---------------------------------------------------------------- oracle


def test_oracle_is_the_clean_runs_output(tmp_path: Path) -> None:
    scenario = find("F3")
    run = recorded(tmp_path, scenario)
    patch = oracle_patch(
        make_gateway(tmp_path / "o"), scenario, run, "tool_agent.order_lookup", ROLES
    )
    assert patch is not None and patch.patch_source == "oracle"
    assert patch.output == clean_value(tmp_path, "tool_agent.order_lookup")
    assert replay(tmp_path, run, patch).patch_source == "oracle"


def test_plans_export_the_schema_patch(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F0"))
    tagged = patch_fields(run, "policy", {"decision": "deny"}, patch_source="model")
    schema = plan_patch(run, WORKFLOW, tagged).schema_patch()
    assert isinstance(schema, ReplaceOutputPatch)
    assert schema.source == "model" and schema.target_span_id == f"{run.run_id}/policy"
    manual = plan_patch(run, WORKFLOW, patch_fields(run, "policy", {"decision": "deny"}))
    with pytest.raises(ValueError, match="no schema form"):
        manual.schema_patch()


# ---------------------------------------------------------------- the command


@pytest.mark.parametrize("source", ["oracle", "rule", "model"])
def test_patch_check_runs_each_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    chosen = [find("F3"), find("F5", "set_field", field="decision")]
    rows = []
    with TraceStore.open() as store:
        for sc in chosen:
            _, run = verify_scenario(
                make_gateway(tmp_path / sc.scenario_id), sc, ROLES, "default", {}, {"seed": 1}
            )
            assert run is not None
            store.save_run(run)
            rows.append(
                {
                    "scenario_id": sc.scenario_id,
                    "fault_code": sc.ground_truth.fault_code,
                    "behaves": True,
                }
            )
    report = tmp_path / "multirun.json"
    report.write_text(json.dumps({"rows": {"1": rows}}), encoding="utf-8")
    monkeypatch.setattr(patch_check, "REPORT", report)
    monkeypatch.setattr(patch_check, "ROOT", tmp_path)
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(
            lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}", script=patcher)
        ),
    )
    assert patch_check.main(["--source", source, "-v"]) == 0
    out = capsys.readouterr().out
    assert f"source {source}" in out and "PASS" in out
    if source == "oracle":
        assert "2/2 replays" in out
    elif source == "rule":  # a tool fault has a rule, a policy decision does not
        assert "1/1 replays" in out and "no rule patch (n/a): 1" in out
    else:  # the scripted patcher answers in Policy's shape: rejected for the tool
        assert "1/1 replays" in out and "1x model patch failed" in out
        assert "recovered the final outcome on: F5" in out
