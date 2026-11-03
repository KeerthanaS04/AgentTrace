"""P4-T1: each injector mutates the intended node only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import TICKET, make_gateway

from agenttrace.faults import (
    FAULT_CATALOG,
    INJECTORS,
    FaultParamError,
    FaultPlan,
    FaultPlanError,
)
from agenttrace.graph import align_runs, build_graph, downstream, first_divergence
from agenttrace.schemas.ground_truth import FaultInjection
from agenttrace.schemas.run import Run
from agenttrace.workflow.support import build_support_workflow, make_resources


def run_with(tmp_path: Path, faults: list[FaultInjection], run_id: str) -> tuple[Run, FaultPlan]:
    plan = FaultPlan(faults)
    gateway = make_gateway(tmp_path)
    result = build_support_workflow().run(
        {"ticket": TICKET}, run_id=run_id, resources=make_resources(gateway), faults=plan
    )
    return result.to_run(), plan


def fault(target: str, injector: str, **params: Any) -> FaultInjection:
    return FaultInjection(target_node_id=target, injector=injector, params=params)


def assert_only_target_and_downstream(clean: Run, faulty: Run, targets: set[str]) -> None:
    """The run splits at a target, and nothing outside targets + their descendants moved."""
    g = build_graph(clean)
    local = {n: g.nodes[n]["local_id"] for n in g}
    allowed = set(targets)
    for n in g:
        if local[n] in targets:
            allowed |= {local[d] for d in downstream(g, n)}
    al = align_runs(clean, faulty)
    assert al.changed, "the fault changed nothing"
    assert al.changed <= allowed, f"changed outside the target's reach: {al.changed - allowed}"
    d = first_divergence(build_graph(clean), build_graph(faulty))
    assert d is not None and d.local_id in targets and d.reason == "output_changed"


def output_of(run: Run, local_id: str) -> dict[str, Any]:
    return next(s.output for s in run.spans if s.span_id == f"{run.run_id}/{local_id}")


@pytest.fixture
def clean(tmp_path: Path) -> Run:
    return run_with(tmp_path, [], "c")[0]


# ---------------------------------------------------------------- one case per fault class

CASES: list[tuple[str, FaultInjection, set[str]]] = [
    # (fault code, injection, local ids of the spans it should hit)
    (
        "F1",
        fault("retriever.search", "retrieval_wrong_doc", **{"with": "KB-ESC-005", "occurrence": 1}),
        {"retriever.search"},
    ),
    (
        "F2",
        fault("retriever.search", "retrieval_empty"),
        {"retriever.search", "retriever.search#2", "retriever.search#3"},
    ),
    (
        "F2",
        fault("retriever.search", "retrieval_irrelevant", docs=["KB-GEN-002"], occurrence=2),
        {"retriever.search#2"},
    ),
    (
        "F3",
        fault(
            "tool_agent.order_lookup", "tool_wrong_value", path="order.status", value="processing"
        ),
        {"tool_agent.order_lookup"},
    ),
    (
        "F4",
        fault("tool_agent.account_status", "tool_malformed", mode="error"),
        {"tool_agent.account_status"},
    ),
    (
        "F4",
        fault(
            "tool_agent.account_status",
            "tool_malformed",
            mode="missing_fields",
            fields=["refunds_last_90_days"],
        ),
        {"tool_agent.account_status"},
    ),
    (
        "F4",
        fault("tool_agent.refund_api", "tool_malformed", mode="wrong_type", path="max_refundable"),
        {"tool_agent.refund_api"},
    ),
    ("F5", fault("policy", "set_field", field="decision", value="deny"), {"policy"}),
    ("F5", fault("router", "set_field", field="category", value="shipping"), {"router"}),
    (
        "F6",
        fault(
            "policy",
            "hallucinated_claim",
            field="justification",
            claim="Per KB-REF-099, refunds over $40 need a receipt photo.",
        ),
        {"policy"},
    ),
    (
        "F6",
        fault("policy", "hallucinated_claim", field="cited_doc_ids", claim="KB-REF-099"),
        {"policy"},
    ),
]


@pytest.mark.parametrize(
    ("code", "injection", "hit"), CASES, ids=[f"{c[0]}-{c[1].injector}" for c in CASES]
)
def test_injector_mutates_the_intended_node_only(
    tmp_path: Path, clean: Run, code: str, injection: FaultInjection, hit: set[str]
) -> None:
    assert injection.injector in FAULT_CATALOG[code].injectors  # type: ignore[index]
    faulty, plan = run_with(tmp_path, [injection], "f")
    # Applied exactly at the intended span(s), nowhere else.
    assert {a.span_id.removeprefix("f/") for a in plan.applied} == hit
    assert plan.unapplied() == []
    assert_only_target_and_downstream(clean, faulty, hit)
    # The recorded output is exactly the injector's mutation of the clean output.
    params = {k: v for k, v in injection.params.items() if k != "occurrence"}
    for local in hit:
        expected = INJECTORS[injection.injector](output_of(clean, local), params)
        assert output_of(faulty, local) == expected


def test_f7_two_independent_roots(tmp_path: Path, clean: Run) -> None:
    roots = [
        fault("tool_agent.order_lookup", "tool_wrong_value", path="order.status", value="shipped"),
        fault("policy", "set_field", field="escalate_to", value="Payments team"),
    ]
    faulty, plan = run_with(tmp_path, roots, "f")
    assert {a.span_id for a in plan.applied} == {"f/tool_agent.order_lookup", "f/policy"}
    assert_only_target_and_downstream(clean, faulty, {"tool_agent.order_lookup", "policy"})
    # The first split is the earlier root; the later root changed too.
    assert output_of(faulty, "policy")["escalate_to"] == "Payments team"


def test_f8_silent_fault_leaves_the_final_outcome_unchanged(tmp_path: Path, clean: Run) -> None:
    # Nothing downstream reads the customer's email, so the result must not change.
    silent = fault(
        "tool_agent.order_lookup", "tool_wrong_value", path="order.email", value="x@example.com"
    )
    faulty, plan = run_with(tmp_path, [silent], "f")
    assert [a.span_id for a in plan.applied] == ["f/tool_agent.order_lookup"]
    assert_only_target_and_downstream(clean, faulty, {"tool_agent.order_lookup"})
    assert faulty.final_output == clean.final_output


def test_f0_control_changes_nothing(tmp_path: Path, clean: Run) -> None:
    control, plan = run_with(tmp_path, [], "f")
    assert plan.applied == [] and align_runs(clean, control).identical


# ---------------------------------------------------------------- hidden from the trace


def test_the_trace_does_not_reveal_the_fault(tmp_path: Path) -> None:
    faulty, _ = run_with(
        tmp_path, [fault("policy", "set_field", field="decision", value="deny")], "f"
    )
    text = json.dumps(faulty.model_dump(mode="json"))
    for leak in ("set_field", "injector", "target_node_id", "FaultInjection"):
        assert leak not in text
    assert not any(s.patched or s.error for s in faulty.spans)


# ---------------------------------------------------------------- injectors are pure


@pytest.mark.parametrize("name", sorted(INJECTORS))
def test_injectors_do_not_mutate_their_input(name: str) -> None:
    value = {
        "hits": [{"doc_id": "KB-REF-001", "score": 2.0, "title": "t"}],
        "order": {"status": "delivered", "email": "a@b"},
        "decision": "approve",
        "justification": "ok",
    }
    params = {
        "retrieval_wrong_doc": {"with": "KB-ESC-005"},
        "retrieval_irrelevant": {"docs": ["KB-GEN-002"]},
        "tool_wrong_value": {"path": "order.status", "value": "processing"},
        "tool_malformed": {"mode": "missing_fields", "fields": ["order.email"]},
        "set_field": {"field": "decision", "value": "deny"},
        "hallucinated_claim": {"field": "justification", "claim": "made up"},
    }.get(name, {})
    before = json.dumps(value, sort_keys=True)
    out = INJECTORS[name](value, params)
    assert json.dumps(value, sort_keys=True) == before
    assert out != value


def test_injector_details() -> None:
    hits = {
        "hits": [
            {"doc_id": "A", "score": 2.0, "title": "a"},
            {"doc_id": "B", "score": 1.0, "title": "b"},
        ]
    }
    assert (
        INJECTORS["retrieval_wrong_doc"](hits, {"replace": "B", "with": "Z"})["hits"][1]["doc_id"]
        == "Z"
    )
    assert (
        INJECTORS["retrieval_wrong_doc"](hits, {"with": "Z"})["hits"][0]["doc_id"] == "Z"
    )  # top hit
    assert INJECTORS["tool_malformed"]({"x": 1}, {"mode": "error"}) == {
        "error": "upstream timeout",
        "status": 504,
    }
    assert INJECTORS["hallucinated_claim"]({"ids": ["A"]}, {"field": "ids", "claim": "B"}) == {
        "ids": ["A", "B"]
    }


def test_wrong_values_change_a_record_consistently() -> None:
    order = {"order": {"delivered_at": "2026-01-19", "delivered_late": True, "status": "x"}}
    out = INJECTORS["tool_wrong_value"](
        order, {"values": {"order.delivered_at": "2026-01-16", "order.delivered_late": False}}
    )
    assert out == {"order": {"delivered_at": "2026-01-16", "delivered_late": False, "status": "x"}}


def test_claim_can_replace_the_reasoning() -> None:
    value = {"justification": "The true reasoning."}
    appended = INJECTORS["hallucinated_claim"](value, {"field": "justification", "claim": "Lie."})
    assert appended["justification"] == "The true reasoning. Lie."
    replaced = INJECTORS["hallucinated_claim"](
        value, {"field": "justification", "claim": "Lie.", "mode": "replace"}
    )
    assert replaced["justification"] == "Lie."


# ---------------------------------------------------------------- misconfiguration fails loudly


def test_bad_fault_specs_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FaultPlanError, match="unknown injector"):
        FaultPlan([fault("policy", "nope")])
    with pytest.raises(FaultPlanError, match="occurrence"):
        FaultPlan([fault("policy", "set_field", field="decision", value="x", occurrence=0)])
    with pytest.raises(FaultParamError, match="missing params"):
        INJECTORS["set_field"]({"decision": "a"}, {"field": "decision"})
    with pytest.raises(FaultParamError, match="no key 'nope'"):
        INJECTORS["tool_wrong_value"]({"order": {}}, {"path": "order.nope", "value": 1})
    with pytest.raises(FaultParamError, match="unknown mode"):
        INJECTORS["tool_malformed"]({}, {"mode": "explode"})
    with pytest.raises(FaultParamError, match="path and value, or values"):
        INJECTORS["tool_wrong_value"]({"order": {}}, {"path": "order.status"})
    with pytest.raises(FaultParamError, match="values is empty"):
        INJECTORS["tool_wrong_value"]({"order": {}}, {"values": {}})
    with pytest.raises(FaultParamError, match="unknown mode"):
        INJECTORS["hallucinated_claim"]({"j": "x"}, {"field": "j", "claim": "c", "mode": "swap"})
    with pytest.raises(FaultParamError, match="cannot take a replace claim"):
        INJECTORS["hallucinated_claim"](
            {"ids": []}, {"field": "ids", "claim": "c", "mode": "replace"}
        )


def test_wrong_kind_or_bad_path_fails_the_run(tmp_path: Path) -> None:
    with pytest.raises(FaultPlanError, match="targets agent spans"):
        run_with(tmp_path, [fault("retriever.search", "set_field", field="x", value=1)], "a")
    with pytest.raises(FaultParamError):
        run_with(tmp_path, [fault("policy", "set_field", field="nope", value=1)], "b")


def test_unreached_target_is_reported(tmp_path: Path) -> None:
    _, plan = run_with(tmp_path, [fault("no_such_step", "set_field", field="x", value=1)], "u")
    assert [f.target_node_id for f in plan.unapplied()] == ["no_such_step"]


def test_catalog_covers_every_fault_code() -> None:
    assert set(FAULT_CATALOG) == {f"F{i}" for i in range(9)}
    assert FAULT_CATALOG["F0"].injectors == ()
    for code in ("F1", "F2", "F3", "F4", "F5", "F6"):
        assert all(name in INJECTORS for name in FAULT_CATALOG[code].injectors)  # type: ignore[index]
