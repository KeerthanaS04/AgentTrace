"""P5-T3: executing a replay plan changes only the target and its descendants."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from fake_llm import agent_of, make_gateway

from agenttrace.faults import FaultPlan
from agenttrace.gateway.gateway import Gateway
from agenttrace.replay import ReplayCache
from agenttrace.replay.executor import execute_plan
from agenttrace.replay.patcher import Patch, patch_fields, plan_patch
from agenttrace.scenarios import patch_check
from agenttrace.scenarios.ground_truth import FaultInjection, Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation
from agenttrace.tracing import traced
from agenttrace.tracing.context import ReplayMissError
from agenttrace.workflow.agent_config import load_agents_config
from agenttrace.workflow.runtime import (
    AgentOutput,
    FromInput,
    FromStep,
    Step,
    StepContext,
    Workflow,
)
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


CONTROL = find("F0")


def recorded(tmp_path: Path, scenario: Scenario = CONTROL, seed: int | None = None) -> Run:
    params = None if seed is None else {"seed": seed}
    _, run = verify_scenario(
        make_gateway(tmp_path / "record"), scenario, ROLES, "default", {}, params
    )
    assert run is not None
    return run


def resources(tmp_path: Path, calls: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """A gateway with an empty cache, logging which agents' prompts reach the model."""
    return make_resources(make_gateway(tmp_path / "replay", calls=calls))


def local(run: Run) -> dict[str, Any]:
    return {s.span_id.split("/", 1)[1]: s for s in run.spans}


# ---------------------------------------------------------------- patching steps and calls


def test_agent_patch_reruns_only_what_follows(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    plan = plan_patch(run, WORKFLOW, patch_fields(run, "policy", {"decision": "deny"}))
    calls: list[tuple[str, str]] = []
    out = execute_plan(run, WORKFLOW, plan, resources(tmp_path, calls), run_id="rp")
    assert out.violations() == []
    assert out.replay.replay_of == run.run_id and out.replay.run_id == "rp"
    assert out.replay.final_output["decision"] == "deny"
    # Only the writer saw a new prompt; everything before the patch came from the record.
    assert [agent for agent, _ in calls] == ["Response Writer"]
    assert out.live_model_calls == 1
    a, b = local(run), local(out.replay)
    assert b["policy"].patched and b["policy"].explanation is None
    assert b["policy.llm"].output == a["policy.llm"].output  # its own call is frozen
    for name in ("__input__", "router", "router.llm", "retriever", "tool_agent", "tool_agent.llm"):
        assert b[name].output == a[name].output and not b[name].patched
    changed = {p.local_id for p in out.alignment.pairs if p.status != "unchanged"}
    assert changed == {"policy", "writer", "writer.llm"}


def test_tool_patch_reruns_its_step_from_the_patched_call_on(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    plan = plan_patch(
        run, WORKFLOW, Patch(target="tool_agent.order_lookup", output={"found": False})
    )
    calls: list[tuple[str, str]] = []
    out = execute_plan(run, WORKFLOW, plan, resources(tmp_path, calls))
    assert out.violations() == []
    b = local(out.replay)
    assert b["tool_agent.order_lookup"].patched
    assert b["tool_agent.order_lookup"].output == {"found": False}
    # The tool agent's first round asked for the lookup before it existed: frozen.
    assert {agent for agent, _ in calls} <= {"Tool Agent", "Policy Agent", "Response Writer"}
    assert local(run)["tool_agent.llm"].output == b["tool_agent.llm"].output


def test_a_no_op_patch_changes_nothing_but_the_patched_flag(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    plan = plan_patch(run, WORKFLOW, patch_fields(run, "router", {}))
    calls: list[tuple[str, str]] = []
    out = execute_plan(run, WORKFLOW, plan, resources(tmp_path, calls))
    changed = [(p.local_id, p.changes) for p in out.alignment.pairs if p.status != "unchanged"]
    assert changed == [("router", ("patched",))]
    assert calls == [] and out.violations() == []


def test_patching_the_input_reruns_every_step(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    ticket = {**local(run)["__input__"].output["ticket"], "subject": "A different subject"}
    plan = plan_patch(run, WORKFLOW, Patch(target="__input__", output={"ticket": ticket}))
    out = execute_plan(run, WORKFLOW, plan, resources(tmp_path))
    assert local(out.replay)["__input__"].patched
    assert local(out.replay)["__input__"].output["ticket"]["subject"] == "A different subject"
    assert out.violations() == [] and out.live_model_calls > 0


def test_the_replay_is_stored_linked_to_the_original(tmp_path: Path) -> None:
    from agenttrace.store import TraceStore

    run = recorded(tmp_path)
    plan = plan_patch(run, WORKFLOW, patch_fields(run, "policy", {"decision": "deny"}))
    out = execute_plan(run, WORKFLOW, plan, resources(tmp_path), run_id=f"{run.run_id}-patch-r1")
    with TraceStore.open(tmp_path / "s.db") as store:
        store.save_run(run)
        store.save_run(out.replay)  # integrity-checked on save
        linked = store.runs(replay_of=run.run_id)
    assert [r.run_id for r in linked] == [f"{run.run_id}-patch-r1"]
    assert sum(s.patched for s in linked[0].spans) == 1


def test_a_seeded_run_replays_with_its_seed(tmp_path: Path) -> None:
    run = recorded(tmp_path, seed=2)
    plan = plan_patch(run, WORKFLOW, patch_fields(run, "policy", {"decision": "deny"}))
    out = execute_plan(run, WORKFLOW, plan, resources(tmp_path))
    assert out.replay.model_params == {"seed": 2} and out.live_model_calls == 1


# ---------------------------------------------------------------- guards


def test_a_frozen_span_without_a_record_is_an_error(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    plan = plan_patch(run, WORKFLOW, patch_fields(run, "policy", {"decision": "deny"}))
    with pytest.raises(ReplayMissError, match="frozen span router.llm"):
        execute_plan(run, WORKFLOW, plan, resources(tmp_path), cache=ReplayCache())


def test_the_plan_must_be_for_the_run(tmp_path: Path) -> None:
    run = recorded(tmp_path)
    plan = plan_patch(run, WORKFLOW, patch_fields(run, "policy", {"decision": "deny"}))
    other = run.model_copy(update={"run_id": "other"})
    with pytest.raises(ValueError, match="plan is for run"):
        execute_plan(other, WORKFLOW, plan, resources(tmp_path))


# ---------------------------------------------------------------- other faults stay in force


def test_other_faults_persist_through_a_replay(tmp_path: Path) -> None:
    tool = FaultInjection(
        target_node_id="tool_agent.order_lookup",
        injector="tool_wrong_value",
        params={"path": "order.status", "value": "shipped", "occurrence": 1},
    )
    policy = FaultInjection(
        target_node_id="policy", injector="set_field", params={"field": "decision", "value": "deny"}
    )
    data = CONTROL.model_dump(mode="json")
    data["scenario_id"] = "sc-two"
    data["faults"] = [tool.model_dump(mode="json"), policy.model_dump(mode="json")]
    data["ground_truth"].update(
        fault_code="F7",
        root_cause_node_ids=["tool_agent.order_lookup", "policy"],
        affected_node_ids=[],
    )
    two_roots = Scenario.model_validate(data)
    run = recorded(tmp_path, two_roots)
    clean = local(recorded(tmp_path / "clean"))["tool_agent.order_lookup"].output
    plan = plan_patch(run, WORKFLOW, Patch(target="tool_agent.order_lookup", output=clean))
    # Fixing the tool root re-runs Policy, whose own fault must still apply.
    kept = execute_plan(run, WORKFLOW, plan, resources(tmp_path), faults=FaultPlan([policy]))
    assert kept.replay.final_output["decision"] == "deny" and kept.violations() == []
    gone = execute_plan(run, WORKFLOW, plan, resources(tmp_path / "2"))
    assert gone.replay.final_output["decision"] == "approve"  # the overstated counterfactual


class Lookup:
    @traced("tool", "lookup", inject=(lambda r: r, lambda r: r))
    def call(self, key: str) -> dict[str, Any]:
        return {"value": key}


class ThreeLookups:
    name, input_keys, output_keys = "A", ("x",), ("a",)

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        lookup = ctx.resource("lookup", Lookup)
        values = [lookup.call(f"{inputs['x']}{i}")["value"] for i in range(3)]
        return AgentOutput(
            output={"a": values}, explanation=Explanation(rationale="r", confidence=1)
        )


def test_served_calls_still_count_toward_fault_occurrences() -> None:
    workflow = Workflow(
        "three",
        ["x"],
        [Step("A", ThreeLookups(), {"x": FromInput("x")})],
        {"out": FromStep("A", "a")},
    )
    third = FaultInjection(
        target_node_id="A.lookup",
        injector="tool_wrong_value",
        params={"path": "value", "value": "BAD", "occurrence": 3},
    )
    res = {"lookup": Lookup()}
    run = workflow.run({"x": "k"}, run_id="t", resources=res, faults=FaultPlan([third])).to_run()
    assert run.final_output["out"] == ["k0", "k1", "BAD"]
    # Patch the second call: the first is served (frozen), the third re-runs, and the
    # fault must hit it again, as the third call, not the first one executed.
    # A patch is the span's recorded output: this tool has no `record`, so `{"result": ...}`.
    patch = Patch(target="A.lookup#2", output={"result": {"value": "P"}})
    plan = plan_patch(run, workflow, patch)
    out = execute_plan(run, workflow, plan, res, faults=FaultPlan([third]))
    assert out.replay.final_output["out"] == ["k0", "P", "BAD"]
    assert out.violations() == []


# ---------------------------------------------------------------- commands


def patched_gateway(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}")),
    )


def test_cli_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.cli import main
    from agenttrace.store import TraceStore

    run = recorded(tmp_path)
    db = tmp_path / "s.db"
    with TraceStore.open(db) as store:
        store.save_run(run)
    patched_gateway(monkeypatch, tmp_path)
    code = main(["replay", run.run_id, "policy", "--set", "decision=deny", "--db", str(db)])
    out = capsys.readouterr().out
    assert code == 0
    assert "only the target and its descendants changed: yes" in out
    assert 'decision: "approve" -> "deny"' in out
    assert "recovery     : correct -> wrong, score 1.00 -> 0.50" in out
    with TraceStore.open(db) as store:
        assert store.run_ids(replay_of=run.run_id) == [f"{run.run_id}-patch-r1"]
        assert store.replays() == []  # a manual patch has no source: no replay record


def test_patch_check_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    scenarios = [
        find("F5", "set_field", field="decision"),
        find("F2", "retrieval_empty"),
        find("F7"),
    ]
    rows = []
    with TraceStore.open() as store:
        for sc in scenarios:
            store.save_run(recorded(tmp_path / sc.scenario_id, sc, seed=1))
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
    patched_gateway(monkeypatch, tmp_path)
    assert patch_check.main(["-v"]) == 0
    out = capsys.readouterr().out
    assert "only the target and its descendants changed: 4/4 replays" in out
    # A fault that hit every search is patched at the step that holds them.
    assert "patch retriever" in out and "PASS" in out


def test_target_for_a_root_with_several_spans(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F2", "retrieval_empty"))
    assert patch_check.target_for(run, "retriever.search") == "retriever"
    assert patch_check.target_for(run, "policy") == "policy"
    assert patch_check.target_for(run, "nope") is None
    assert agent_of("You are the Router in a customer-support system") == "Router"
