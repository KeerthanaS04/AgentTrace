"""P5-T5: the differ and recovery scoring feed the UI and the evaluation harness."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from fake_llm import make_gateway

from agenttrace.evaluation.outcome import check_outcome
from agenttrace.evaluation.recovery import outcome_score, replay_result, score_recovery
from agenttrace.gateway.gateway import Gateway
from agenttrace.graph.ui import SCHEMA_PATH
from agenttrace.replay.differ import diff_runs, to_ui_replay
from agenttrace.replay.executor import execute_plan
from agenttrace.replay.patcher import Patch, patch_fields, plan_patch
from agenttrace.scenarios import patch_check
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.ground_truth import ExpectedFinal
from agenttrace.schemas.replay import ReplayResult
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation
from agenttrace.tracing import traced
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
CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)
F3 = next(
    s for s in DATASET if s.ground_truth.fault_code == "F3" and s.ticket.ticket_id == "T-0101"
)


def recorded(tmp_path: Path, scenario: Scenario = CONTROL) -> Run:
    _, run = verify_scenario(make_gateway(tmp_path / "rec"), scenario, ROLES, "default", {})
    assert run is not None
    return run


def deny(tmp_path: Path, source: Any = "model") -> tuple[Run, Any]:
    run = recorded(tmp_path)
    patch = patch_fields(run, "policy", {"decision": "deny"}, patch_source=source)
    plan = plan_patch(run, WORKFLOW, patch)
    out = execute_plan(run, WORKFLOW, plan, make_resources(make_gateway(tmp_path / "rp")))
    return run, out


# ---------------------------------------------------------------- the differ


def test_diff_statuses_and_final_changes(tmp_path: Path) -> None:
    run, out = deny(tmp_path)
    diff = diff_runs(run, out.replay)
    assert diff.patched == ["policy"]
    assert {s.local_id for s in diff.spans if s.status == "changed"} == {"writer", "writer.llm"}
    assert diff.counts()["new"] == diff.counts()["missing"] == 0
    assert diff.status("router.llm") == "unchanged"
    assert diff.status("writer") == "changed"
    # The scripted Writer replies the same whatever the decision: its input changed only.
    assert next(s for s in diff.spans if s.local_id == "writer").changes == ["input"]
    keys = {c.key: (c.before, c.after) for c in diff.final_changes}
    assert keys == {"decision": ("approve", "deny")}
    assert diff.patch_source == "model" and not diff.final_identical
    assert set(diff.changed_spans) == {"policy", "writer", "writer.llm"}


class Lookup:
    @traced("tool", "lookup")
    def call(self, key: str) -> dict[str, Any]:
        return {"value": key}


class Count:
    name, input_keys, output_keys = "A", ("x",), ("n",)

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        return AgentOutput(
            output={"n": len(inputs["x"])}, explanation=Explanation(rationale="r", confidence=1)
        )


class LookUpN:
    name, input_keys, output_keys = "B", ("n",), ("values",)

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        box = ctx.resource("lookup", Lookup)
        values = [box.call(str(i))["value"] for i in range(inputs["n"])]
        return AgentOutput(
            output={"values": values}, explanation=Explanation(rationale="r", confidence=1)
        )


def test_new_and_missing_spans() -> None:
    workflow = Workflow(
        "count",
        ["x"],
        [
            Step("A", Count(), {"x": FromInput("x")}),
            Step("B", LookUpN(), {"n": FromStep("A", "n")}),
        ],
        {"out": FromStep("B", "values")},
    )
    res = {"lookup": Lookup()}
    run = workflow.run({"x": "ab"}, run_id="c", resources=res).to_run()  # two lookups
    more = execute_plan(
        run, workflow, plan_patch(run, workflow, Patch(target="A", output={"n": 3})), res
    )
    assert diff_runs(run, more.replay).status("B.lookup#3") == "new"
    fewer = execute_plan(
        run, workflow, plan_patch(run, workflow, Patch(target="A", output={"n": 1})), res
    )
    diff = diff_runs(run, fewer.replay)
    assert diff.status("B.lookup#2") == "missing" and diff.counts()["missing"] == 1


# ---------------------------------------------------------------- recovery scoring


EXPECTED = ExpectedFinal(decision="approve", required_facts=["5-7 business days", "48"])


def test_partial_credit() -> None:
    right = {"decision": "approve", "reply": "Refund of $48 in 5-7 business days."}
    half_facts = {"decision": "approve", "reply": "Refund of $48."}
    wrong_decision = {"decision": "deny", "reply": "Refund of $48 in 5-7 business days."}
    assert outcome_score(check_outcome(EXPECTED, right)) == 1.0
    assert outcome_score(check_outcome(EXPECTED, half_facts)) == 0.75
    assert outcome_score(check_outcome(EXPECTED, wrong_decision)) == 0.5
    none = ExpectedFinal(decision="escalate", required_facts=[])
    assert outcome_score(check_outcome(none, {"decision": "escalate", "reply": "x"})) == 1.0
    assert outcome_score(check_outcome(none, {"decision": "approve", "reply": "x"})) == 0.0


def test_recovery_against_the_clean_run() -> None:
    broken = {"decision": "deny", "reply": "No."}
    fixed = {"decision": "approve", "reply": "Refund of $48 in 5-7 business days."}
    rec = score_recovery(EXPECTED, broken, fixed, clean_final=fixed)
    assert rec.recovered and rec.clean_correct and rec.matches_clean
    assert (rec.before_score, rec.recovery_score) == (0.0, 1.0) and rec.gain == 1.0
    # The clean run itself misses a fact: the replay matches it, though both are wrong.
    clean = {"decision": "approve", "reply": "Refund of $48."}
    rec = score_recovery(EXPECTED, broken, clean, clean_final=clean)
    assert not rec.recovered and rec.clean_correct is False and rec.matches_clean
    assert rec.facts_missing == ["5-7 business days"]
    assert score_recovery(EXPECTED, broken, fixed).clean_correct is None


def test_replay_records_are_built_and_stored(tmp_path: Path) -> None:
    from agenttrace.store import TraceStore

    run, out = deny(tmp_path)
    diff = diff_runs(run, out.replay)
    rec = score_recovery(
        CONTROL.ground_truth.expected_final, run.final_output, out.replay.final_output
    )
    result = replay_result(out, diff, rec, diagnosis_id="d-1")
    assert result.patch_source == "model" and result.patched_span_id == f"{run.run_id}/policy"
    assert result.changed_spans == diff.changed_spans and not result.recovered
    assert result.replay_final["decision"] == "deny" and result.patch.kind == "replace_output"
    with TraceStore.open(tmp_path / "s.db") as store:
        store.add_replay(result)
        assert store.replays(original_run_id=run.run_id) == [result]
    # A record from before P5-T5 (no clean-run fields) still loads.
    doc = result.model_dump(mode="json")
    del doc["clean_correct"], doc["matches_clean"]
    assert ReplayResult.model_validate(doc).clean_correct is None
    _, manual = deny(tmp_path / "m", source=None)
    with pytest.raises(ValueError, match="no schema form"):
        replay_result(manual, diff, rec)


# ---------------------------------------------------------------- the UI


def test_ui_replay_view_validates_against_the_committed_schema(tmp_path: Path) -> None:
    run, out = deny(tmp_path)
    diff = diff_runs(run, out.replay)
    doc = to_ui_replay(out.replay, diff, {"recovered": False, "recovery_score": 0.5})
    payload = json.loads(doc.model_dump_json())
    jsonschema.validate(payload, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    by_local = {n["data"]["local_id"]: n["data"] for n in payload["nodes"]}
    assert by_local["policy"]["replay_status"] == "patched"
    assert by_local["writer"]["replay_status"] == "changed"
    assert by_local["writer"]["replay_changes"] == ["input"]
    assert by_local["router"]["replay_status"] == "unchanged"
    replay = payload["replay"]
    assert replay["original_run_id"] == run.run_id and replay["patched"] == ["policy"]
    assert replay["recovered"] is False and replay["recovery_score"] == 0.5
    assert [c["key"] for c in replay["final_changes"]] == ["decision"]
    assert {item["status"] for item in replay["legend"]} == {
        "patched",
        "changed",
        "new",
        "missing",
        "unchanged",
    }
    # A plain run's graph has no overlay.
    from agenttrace.graph import build_graph
    from agenttrace.graph.ui import to_ui_graph

    assert to_ui_graph(build_graph(run), run).replay is None


# ---------------------------------------------------------------- the commands


def patched_gateway(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}")),
    )


def test_cli_replay_from_a_source_with_ui_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.cli import main
    from agenttrace.store import TraceStore

    run = recorded(tmp_path, F3)
    db = tmp_path / "s.db"
    with TraceStore.open(db) as store:
        store.save_run(run)
    patched_gateway(monkeypatch, tmp_path)
    out_json = tmp_path / "ui" / "replay.json"
    args = [
        "replay",
        run.run_id,
        "tool_agent.order_lookup",
        "--source",
        "rule",
        "--json",
        str(out_json),
        "--db",
        str(db),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "source rule" in out and "patch note   : re-called order_lookup" in out
    assert "clean run    : correct; replay matches it: yes" in out
    assert "stored + replay record" in out
    jsonschema.validate(
        json.loads(out_json.read_text(encoding="utf-8")),
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    )
    with TraceStore.open(db) as store:
        [record] = store.replays(original_run_id=run.run_id)
    assert record.patch_source == "rule" and record.clean_correct and record.matches_clean
    # A source with no patch for the node is refused before anything runs.
    assert main(["replay", run.run_id, "policy", "--source", "rule", "--db", str(db)]) == 2
    assert "no rule patch for policy" in capsys.readouterr().out


def test_patch_check_stores_records_and_reports_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    with TraceStore.open() as store:
        _, run = verify_scenario(
            make_gateway(tmp_path / "r"), F3, ROLES, "default", {}, {"seed": 1}
        )
        assert run is not None
        store.save_run(run)
    report = tmp_path / "multirun.json"
    rows = [{"scenario_id": F3.scenario_id, "fault_code": "F3", "behaves": True}]
    report.write_text(json.dumps({"rows": {"1": rows}}), encoding="utf-8")
    monkeypatch.setattr(patch_check, "REPORT", report)
    monkeypatch.setattr(patch_check, "ROOT", tmp_path)
    patched_gateway(monkeypatch, tmp_path)
    assert patch_check.main(["--source", "rule", "--store"]) == 0
    out = capsys.readouterr().out
    assert "recovery (rule; recoverable = the seed's clean run is correct)" in out
    assert "recovered 100.0% of recoverable replays (1)" in out
    with TraceStore.open() as store:
        [record] = store.replays()
        assert store.has_run(record.replay_run_id)
    assert record.patch_source == "rule" and record.recovered and record.recovery_score == 1.0
