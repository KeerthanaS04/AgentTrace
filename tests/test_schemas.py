from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

import agenttrace.schemas as schemas
from agenttrace.schemas import (
    AnomalyScore,
    Diagnosis,
    ExplainabilityScore,
    Explanation,
    Fix,
    NodeContract,
    OverrideInputsPatch,
    Patch,
    PropagatedError,
    Ref,
    ReplaceOutputPatch,
    ReplayResult,
    RootCause,
    Run,
    ScenarioInput,
    Span,
    Ticket,
)
from agenttrace.schemas.ground_truth import ExpectedFinal, FaultInjection, GroundTruth, Scenario

T0 = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------- factories


def make_span(span_id: str = "s2", **overrides: Any) -> Span:
    data: dict[str, Any] = {
        "span_id": span_id,
        "run_id": "r1",
        "node_id": "retriever.search",
        "agent": "retriever",
        "kind": "retrieval",
        "input": {"query": "refund window"},
        "input_refs": [Ref(span_id="s1", key="category", content_hash="sha256:ab")],
        "output": {"doc_ids": ["kb-012"]},
        "explanation": Explanation(rationale="matched policy", confidence=0.8),
        "started_at": T0,
        "ended_at": T0 + timedelta(seconds=1),
    }
    data.update(overrides)
    return Span(**data)


def make_root(span_id: str = "s2") -> RootCause:
    return RootCause(
        node_id="retriever.search",
        span_id=span_id,
        fault_type="retrieval_wrong",
        confidence=0.9,
        evidence=[span_id],
    )


def make_patch(**overrides: Any) -> ReplaceOutputPatch:
    data: dict[str, Any] = {
        "target_span_id": "s2",
        "source": "rule",
        "output": {"doc_ids": ["kb-007"]},
    }
    data.update(overrides)
    return ReplaceOutputPatch(**data)


def make_diagnosis(**overrides: Any) -> Diagnosis:
    data: dict[str, Any] = {
        "diagnosis_id": "d1",
        "run_id": "r1",
        "prompt_version": "judge-v1",
        "root_causes": [make_root()],
        "propagated": [
            PropagatedError(
                span_id="s4", node_id="policy.decide", caused_by=["s2"], path=["s2", "s4"]
            )
        ],
        "recommended_fixes": [
            Fix(
                fix_id="f1",
                target_span_id="s2",
                fault_type="retrieval_wrong",
                description="re-query with order date",
                patch=make_patch(),
            )
        ],
        "judge_agreement": 0.62,
    }
    data.update(overrides)
    return Diagnosis(**data)


def make_scenario(**overrides: Any) -> Scenario:
    data: dict[str, Any] = {
        "scenario_id": "sc-001",
        "category": "refund",
        "ticket": Ticket(
            ticket_id="t1", customer_id="c1", subject="Refund", body="Refund order 42"
        ),
        "fixtures": {"orders": [{"order_id": "42", "amount": 59.0}]},
        "faults": [FaultInjection(target_node_id="retriever.search", injector="F1_wrong_doc")],
        "ground_truth": GroundTruth(
            fault_code="F1",
            root_cause_node_ids=["retriever.search"],
            affected_node_ids=["policy.decide", "writer.reply"],
            expected_final=ExpectedFinal(decision="approve", required_facts=["30 days"]),
        ),
    }
    data.update(overrides)
    return Scenario(**data)


VALID_INSTANCES: list[BaseModel] = [
    Ref(span_id="s1", key="k", content_hash="h"),
    Explanation(rationale="r", cited_evidence=["kb-1"], confidence=0.5),
    make_span(),
    NodeContract(
        node_id="router.classify",
        description="classify ticket",
        input_expectations="ticket text",
        output_expectations="one category",
        programmatic_checks=["category_in_enum"],
    ),
    Run(run_id="r1", workflow_version="v1", spans=[make_span()], created_at=T0),
    AnomalyScore(
        span_id="s2",
        judge="judge_fast",
        prompt_version="judge-v1",
        output_ok_given_inputs=0.2,
        inputs_ok=0.9,
        verdict="faulty",
        evidence=["s2", "kb-012"],
        reasoning="wrong policy doc",
    ),
    make_root(),
    make_diagnosis(created_at=T0),
    make_diagnosis(created_at=T0, escalated_to_deep=True, escalation_reason="low agreement"),
    ReplayResult(
        original_run_id="r1",
        replay_run_id="r1-replay",
        diagnosis_id="d1",
        patch=OverrideInputsPatch(
            target_span_id="s2", source="model", input_overrides={"query": "x"}
        ),
        patched_span_id="s2",
        patch_source="model",
        changed_spans=["s2", "s4"],
        recovered=True,
        recovery_score=1.0,
    ),
    ExplainabilityScore(
        scope="span", run_id="r1", span_id="s2", metric="G", value=0.75, computed_at=T0
    ),
    ExplainabilityScore(scope="agent", agent="policy", metric="C", value=0.9, computed_at=T0),
    make_scenario(),
    make_scenario().to_input(),
]


# ---------------------------------------------------------------- round trips


@pytest.mark.parametrize("instance", VALID_INSTANCES, ids=lambda m: type(m).__name__)
def test_json_round_trip(instance: BaseModel) -> None:
    restored = type(instance).model_validate_json(instance.model_dump_json())
    assert restored == instance


@pytest.mark.parametrize("instance", VALID_INSTANCES, ids=lambda m: type(m).__name__)
def test_json_schema_generates(instance: BaseModel) -> None:
    assert type(instance).model_json_schema()["type"] == "object"


def test_patch_union_round_trip_keeps_kind() -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(Patch)
    patch = OverrideInputsPatch(target_span_id="s2", source="rule", input_overrides={"q": 1})
    restored = adapter.validate_json(adapter.dump_json(patch))
    assert isinstance(restored, OverrideInputsPatch)
    assert restored == patch


def test_timestamps_are_normalized_to_utc() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    span = make_span(started_at=T0.astimezone(ist), ended_at=T0.astimezone(ist))
    assert span.started_at.tzinfo == UTC
    assert span.started_at == T0


# ---------------------------------------------------------------- rejections


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Ref(span_id="s1", key="k", content_hash="h", surprise=1)  # type: ignore[call-arg]


def test_assignment_is_validated() -> None:
    root = make_root()
    with pytest.raises(ValidationError):
        root.confidence = 1.5


@pytest.mark.parametrize("confidence", [-0.1, 1.01])
def test_probability_bounds(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Explanation(rationale="r", confidence=confidence)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        make_span(started_at=datetime(2026, 10, 29), ended_at=datetime(2026, 10, 29))


def test_span_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError, match="ended_at"):
        make_span(ended_at=T0 - timedelta(seconds=1))


def test_span_self_parent_rejected() -> None:
    with pytest.raises(ValidationError, match="own parent"):
        make_span(parent_span_id="s2")


def test_span_self_ref_rejected() -> None:
    with pytest.raises(ValidationError, match="own output"):
        make_span(input_refs=[Ref(span_id="s2", key="x", content_hash="h")])


def test_span_bad_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        make_span(kind="banana")


def test_empty_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Ref(span_id="", key="k", content_hash="h")


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValidationError):
        make_span(cost_usd=-0.01)


def test_run_duplicate_span_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Run(run_id="r1", workflow_version="v1", spans=[make_span(), make_span()])


def test_run_foreign_span_rejected() -> None:
    with pytest.raises(ValidationError, match="different run"):
        Run(run_id="r1", workflow_version="v1", spans=[make_span(run_id="r2")])


def test_run_replay_of_self_rejected() -> None:
    with pytest.raises(ValidationError, match="itself"):
        Run(run_id="r1", workflow_version="v1", replay_of="r1")


def test_anomaly_non_ok_without_evidence_rejected() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        AnomalyScore(
            span_id="s2",
            judge="j",
            prompt_version="v1",
            output_ok_given_inputs=0.1,
            inputs_ok=0.9,
            verdict="suspicious",
            reasoning="hmm",
        )


def test_root_cause_without_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        RootCause(node_id="n", span_id="s2", fault_type="other", confidence=0.5, evidence=[])


@pytest.mark.parametrize(
    "path, match",
    [(["s2", "s9"], "end at"), (["s9", "s4"], "start at")],
)
def test_propagated_bad_path_rejected(path: list[str], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        PropagatedError(span_id="s4", node_id="n", caused_by=["s2"], path=path)


def test_patch_kind_payload_mismatch_rejected() -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(Patch)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"kind": "override_inputs", "target_span_id": "s2", "source": "rule", "output": {}}
        )


def test_fix_patch_target_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="target_span_id"):
        Fix(
            fix_id="f1",
            target_span_id="s2",
            fault_type="other",
            description="d",
            patch=make_patch(target_span_id="s3"),
        )


def test_diagnosis_escalation_needs_reason() -> None:
    with pytest.raises(ValidationError, match="escalation_reason"):
        make_diagnosis(escalated_to_deep=True)
    with pytest.raises(ValidationError, match="escalation_reason"):
        make_diagnosis(escalation_reason="why not")


def test_diagnosis_propagated_unknown_root_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown root"):
        make_diagnosis(root_causes=[make_root("s3")], recommended_fixes=[])


def test_diagnosis_fix_on_non_root_rejected() -> None:
    with pytest.raises(ValidationError, match="not a root cause"):
        make_diagnosis(
            root_causes=[make_root("s2"), make_root("s3")],
            recommended_fixes=[
                Fix(
                    fix_id="f9",
                    target_span_id="s5",
                    fault_type="other",
                    description="d",
                    patch=make_patch(target_span_id="s5"),
                )
            ],
        )


def test_diagnosis_kappa_bounds() -> None:
    assert make_diagnosis(judge_agreement=-0.3).judge_agreement == -0.3
    with pytest.raises(ValidationError):
        make_diagnosis(judge_agreement=1.2)


def test_replay_inconsistent_fields_rejected() -> None:
    base: dict[str, Any] = {
        "original_run_id": "r1",
        "replay_run_id": "r2",
        "patch": make_patch(),
        "patched_span_id": "s2",
        "patch_source": "rule",
        "recovered": False,
        "recovery_score": 0.0,
    }
    ReplayResult(**base)
    for bad in ({"replay_run_id": "r1"}, {"patched_span_id": "s9"}, {"patch_source": "oracle"}):
        with pytest.raises(ValidationError):
            ReplayResult(**{**base, **bad})


def test_score_scope_requires_ids() -> None:
    with pytest.raises(ValidationError, match="span_id"):
        ExplainabilityScore(scope="span", run_id="r1", metric="G", value=0.5)
    with pytest.raises(ValidationError, match="agent"):
        ExplainabilityScore(scope="agent", metric="XI", value=0.5)


@pytest.mark.parametrize(
    "fault_code, roots, affected",
    [
        ("F0", ["a"], []),
        ("F0", [], ["b"]),
        ("F7", ["a"], []),
        ("F1", [], []),
        ("F1", ["a", "b"], []),
        ("F3", ["a"], ["a"]),
    ],
)
def test_ground_truth_invalid_label_combos(
    fault_code: str, roots: list[str], affected: list[str]
) -> None:
    with pytest.raises(ValidationError):
        GroundTruth(
            fault_code=fault_code,  # type: ignore[arg-type]
            root_cause_node_ids=roots,
            affected_node_ids=affected,
            expected_final=ExpectedFinal(decision="deny"),
        )


def test_scenario_faults_must_match_roots() -> None:
    with pytest.raises(ValidationError, match="fault targets"):
        make_scenario(faults=[FaultInjection(target_node_id="tool.call", injector="F3")])


# ---------------------------------------------------------------- ground-truth isolation


def test_to_input_strips_hidden_fields() -> None:
    view = make_scenario().to_input()
    assert type(view) is ScenarioInput
    dumped = view.model_dump()
    assert "ground_truth" not in dumped
    assert "faults" not in dumped


def test_scenario_input_rejects_ground_truth() -> None:
    full = make_scenario().model_dump()
    with pytest.raises(ValidationError, match="extra"):
        ScenarioInput.model_validate(full)


def test_ground_truth_not_reexported() -> None:
    for name in ("GroundTruth", "Scenario", "FaultInjection", "ExpectedFinal", "FaultCode"):
        assert not hasattr(schemas, name)
