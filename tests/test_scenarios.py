"""P4-T2: the scenario dataset validates, is balanced, and is committed."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pytest

from agenttrace.faults import INJECTORS, FaultPlan
from agenttrace.scenarios.generator import (
    DROPPED,
    EXCLUDED_BASES,
    SPECS,
    Spec,
    affected_steps,
    build_scenarios,
    main,
    render,
)
from agenttrace.scenarios.ground_truth import SCENARIOS_PATH, load_dataset
from agenttrace.schemas.ground_truth import FaultInjection, Scenario
from agenttrace.workflow.clean_eval import load_scenarios
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.support import STEPS, build_support_workflow
from agenttrace.workflow.tools import TOOLS, ToolBox, load_db
from agenttrace.workflow.tools.tools import tool_result_json

DATASET = load_dataset()
BASES = {s.ticket.ticket_id: s for s in load_scenarios()}
WORKFLOW = build_support_workflow()


def by_code() -> Counter[str]:
    return Counter(s.ground_truth.fault_code for s in DATASET)


# ---------------------------------------------------------------- the done-when


def test_committed_dataset_is_current() -> None:
    assert SCENARIOS_PATH.read_text(encoding="utf-8") == render(build_scenarios())
    assert main(["--check"]) == 0


def test_dataset_validates() -> None:
    # load_dataset re-validates every scenario with the strict schemas.
    assert all(isinstance(s, Scenario) for s in DATASET)
    for s in DATASET:
        FaultPlan(s.faults)  # every injector name and occurrence is valid


def test_dataset_is_balanced() -> None:
    counts = by_code()
    assert 80 <= len(DATASET) <= 120
    assert counts["F0"] >= 15
    for code in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"):
        assert counts[code] >= 10, (code, counts[code])
    categories = Counter(s.category for s in DATASET)
    assert set(categories) >= {"refund", "shipping", "account"}
    # Controls and faults both span every category.
    for category in categories:
        codes = {s.ground_truth.fault_code for s in DATASET if s.category == category}
        assert "F0" in codes and len(codes) >= 4, (category, codes)


# ---------------------------------------------------------------- ids and bases


def test_ids_are_opaque_and_unique() -> None:
    ids = [s.scenario_id for s in DATASET]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"sc-\d{3}", i) for i in ids)
    # The order does not group by fault code (ids must not hint at the fault).
    first = [s.ground_truth.fault_code for s in DATASET[:20]]
    assert len(set(first)) >= 5


def test_scenarios_reuse_correct_clean_tickets() -> None:
    for s in DATASET:
        base = BASES[s.ticket.ticket_id]
        assert base.scenario_id not in EXCLUDED_BASES
        assert s.ticket == base.ticket and s.category == base.category
        assert s.ground_truth.expected_final == base.ground_truth.expected_final
    controls = [s for s in DATASET if s.ground_truth.fault_code == "F0"]
    assert {s.ticket.ticket_id for s in controls} == {
        b.ticket.ticket_id for b in BASES.values() if b.scenario_id not in EXCLUDED_BASES
    }


def test_the_workflow_view_hides_faults_and_truth() -> None:
    view = DATASET[0].to_input().model_dump()
    assert "faults" not in view and "ground_truth" not in view


# ---------------------------------------------------------------- faults are applicable


def node_kind(node_id: str) -> str:
    if node_id in STEPS:
        return "agent"
    if node_id == "retriever.search":
        return "retrieval"
    step, call = node_id.split(".", 1)
    assert step == "tool_agent" and call in TOOLS, node_id
    return "tool"


def real_output(s: Scenario, f: FaultInjection) -> dict[str, Any]:
    """The value this fault will meet in a clean run, computed from the fixtures."""
    kind = node_kind(f.target_node_id)
    if kind == "retrieval":
        return {"hits": [{"doc_id": "KB-REF-001", "score": 3.0, "title": "t"}]}
    if kind == "agent":
        return dict.fromkeys(WORKFLOW.steps[f.target_node_id].agent.output_keys, "x")
    box = ToolBox(load_db())
    order = re.search(r"O\d{4}", s.ticket.body)
    name = f.target_node_id.split(".", 1)[1]
    if name == "account_status":
        args: dict[str, Any] = {"customer_id": s.ticket.customer_id}
    elif name == "order_lookup":
        assert order, s.scenario_id
        args = {"order_id": order.group()}
    else:
        assert order, s.scenario_id
        args = {"order_id": order.group(), "amount": 1, "mode": "quote", "reason": "r"}
    return tool_result_json(box.call(name, args))


@pytest.mark.parametrize("scenario", DATASET, ids=[s.scenario_id for s in DATASET])
def test_every_fault_targets_a_real_node_and_applies_to_real_data(scenario: Scenario) -> None:
    kb = load_kb()
    for f in scenario.faults:
        injector = INJECTORS[f.injector]
        assert node_kind(f.target_node_id) in injector.kinds
        params = {k: v for k, v in f.params.items() if k != "occurrence"}
        out = injector(real_output(scenario, f), params)  # no FaultParamError
        assert out is not None
        for key in ("replace", "with"):
            if key in params:
                kb.get(params[key])  # swapped-in documents exist in the KB
        if "field" in params:
            assert params["field"] in WORKFLOW.steps[f.target_node_id].agent.output_keys


# ---------------------------------------------------------------- ground truth


def test_roots_are_exactly_the_fault_targets() -> None:
    for s in DATASET:
        targets = {f.target_node_id for f in s.faults}
        assert set(s.ground_truth.root_cause_node_ids) == targets
        assert not set(s.ground_truth.affected_node_ids) & targets
        if s.ground_truth.fault_code == "F7":
            assert len(targets) == 2
        elif s.ground_truth.fault_code != "F0":
            assert len(targets) == 1


def fi(target: str, **params: Any) -> FaultInjection:
    return FaultInjection(target_node_id=target, injector="set_field", params=params)


def test_affected_steps_follow_the_wiring_per_field() -> None:
    def aff(*faults: FaultInjection) -> list[str]:
        return affected_steps(WORKFLOW, faults)

    assert aff(fi("tool_agent.order_lookup")) == ["tool_agent", "policy", "writer"]
    assert aff(fi("retriever.search")) == ["retriever", "policy", "writer"]
    assert aff(fi("router", field="order_id")) == ["tool_agent", "policy", "writer"]
    assert aff(fi("router", field="category")) == ["retriever", "tool_agent", "policy", "writer"]
    assert aff(fi("policy", field="decision")) == ["writer"]
    # Outputs nobody consumes: the fault cannot spread (F8).
    assert aff(fi("retriever", field="queries")) == []
    assert aff(fi("policy", field="cited_doc_ids")) == []
    # Two roots: the downstream root is not also "affected".
    assert aff(fi("tool_agent.order_lookup"), fi("policy", field="decision")) == [
        "tool_agent",
        "writer",
    ]


def test_dropped_specs_are_documented_and_out_of_the_dataset() -> None:
    def key(spec: Spec) -> str:
        code, base, faults = spec
        return json.dumps([code, base, [f.model_dump(mode="json") for f in faults]])

    assert DROPPED
    kept = {key(s) for s in SPECS}
    for spec, reason in DROPPED:
        assert key(spec) not in kept, spec
        assert reason.strip()
        FaultPlan(spec[2])  # still a valid spec, so it can be re-verified later
    assert len({key(s) for s in SPECS}) == len(SPECS)  # no spec listed twice


def test_generation_is_deterministic() -> None:
    assert render(build_scenarios()) == render(build_scenarios())


def test_coverage_table_lists_every_code_and_category() -> None:
    from agenttrace.scenarios.generator import coverage_table

    table = coverage_table(DATASET)
    assert table.splitlines()[0].split() == ["code", "account", "refund", "shipping", "total"]
    assert table.splitlines()[-1].split()[-1] == str(len(DATASET))
    assert all(f"F{i}" in table for i in range(9))
