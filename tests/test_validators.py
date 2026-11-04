"""P6-T2: deterministic validators give hard evidence on faulty spans, with no model call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import make_gateway

from agenttrace.diagnosis.validators import (
    ValidationContext,
    by_span,
    numeric_claims,
    short_circuit,
    validate_run,
)
from agenttrace.scenarios import validator_check
from agenttrace.scenarios.ground_truth import FaultInjection, Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.diagnosis import ValidatorFinding
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
CTX = ValidationContext.default()
CONTROL = next(
    s for s in DATASET if s.ground_truth.fault_code == "F0" and s.ticket.ticket_id == "T-0101"
)


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


def fault(target: str, injector: str, **params: Any) -> FaultInjection:
    if target.startswith("tool_agent."):
        params.setdefault("occurrence", 1)
    return FaultInjection(target_node_id=target, injector=injector, params=params)


def run_of(tmp_path: Path, scenario: Scenario) -> Run:
    _, run = verify_scenario(make_gateway(tmp_path), scenario, ROLES, "default", {})
    assert run is not None
    return run


def findings_at(tmp_path: Path, scenario: Scenario, local: str) -> dict[str, str]:
    """check -> severity of the findings on span `local`."""
    run = run_of(tmp_path, scenario)
    found = by_span(validate_run(run, CTX)).get(f"{run.run_id}/{local}", [])
    return {f.check: f.severity for f in found}


# ---------------------------------------------------------------- clean runs


def test_a_clean_run_has_no_errors(tmp_path: Path) -> None:
    findings = validate_run(run_of(tmp_path, CONTROL), CTX)
    assert [f for f in findings if f.severity == "error"] == []
    # The scripted writer's "5-7 business days" is not in its documents: a real warning.
    assert {f.check for f in findings} <= {"numbers_supported"}


# ---------------------------------------------------------------- F4: malformed tool replies


@pytest.mark.parametrize(
    ("params", "check"),
    [
        ({"mode": "error"}, "tool_error_payload"),
        ({"mode": "missing_fields", "fields": ["plus_since"]}, "tool_record_complete"),
        ({"mode": "wrong_type", "path": "refunds_last_90_days", "value": "N/A"}, "tool_schema"),
    ],
)
def test_malformed_tool_replies_are_errors(
    tmp_path: Path, params: dict[str, Any], check: str
) -> None:
    sc = with_faults(fault("tool_agent.account_status", "tool_malformed", **params))
    got = findings_at(tmp_path, sc, "tool_agent.account_status")
    assert got.get(check) == "error"


def test_missing_nested_fields_are_errors(tmp_path: Path) -> None:
    sc = with_faults(
        fault(
            "tool_agent.order_lookup",
            "tool_malformed",
            mode="missing_fields",
            fields=["order.status", "order.items.0.price"],
        )
    )
    run = run_of(tmp_path, sc)
    [missing] = [f for f in validate_run(run, CTX) if f.check == "tool_record_complete"]
    assert "order.status" in missing.message and "order.items.0.price" in missing.message


# ---------------------------------------------------------------- F3: inconsistent records


def test_a_day_count_that_contradicts_the_dates(tmp_path: Path) -> None:
    sc = with_faults(
        fault(
            "tool_agent.order_lookup",
            "tool_wrong_value",
            path="order.days_since_delivery",
            value=45,
        )
    )
    assert findings_at(tmp_path, sc, "tool_agent.order_lookup") == {"tool_consistency": "error"}


def test_an_account_that_contradicts_itself(tmp_path: Path) -> None:
    sc = with_faults(
        fault(
            "tool_agent.account_status", "tool_wrong_value", path="under_fraud_review", value=True
        )
    )
    assert findings_at(tmp_path, sc, "tool_agent.account_status") == {"tool_consistency": "error"}


def test_tools_that_disagree(tmp_path: Path) -> None:
    sc = with_faults(
        fault("tool_agent.order_lookup", "tool_wrong_value", path="order.total", value=999.0)
    )
    got = findings_at(tmp_path, sc, "tool_agent")
    assert got.get("tool_results_agree") == "error"


# ---------------------------------------------------------------- F1 / F2: retrieval


def test_an_empty_search(tmp_path: Path) -> None:
    sc = with_faults(fault("retriever.search", "retrieval_empty"), code="F2")
    assert findings_at(tmp_path, sc, "retriever.search") == {"search_empty": "warning"}
    assert findings_at(tmp_path / "b", sc, "retriever")["docs_nonempty"] == "error"


def test_irrelevant_hits(tmp_path: Path) -> None:
    sc = with_faults(
        fault("retriever.search", "retrieval_irrelevant", docs=["KB-GEN-001", "KB-SHP-006"]),
        code="F2",
    )
    got = findings_at(tmp_path, sc, "retriever.search")
    assert got.keys() & {"hit_irrelevant", "hit_score_mismatch"}
    assert set(got.values()) == {"error"}


def test_a_superseded_or_swapped_document(tmp_path: Path) -> None:
    stale = with_faults(
        fault(
            "retriever.search",
            "retrieval_wrong_doc",
            replace="KB-REF-001",
            **{"with": "KB-REF-010"},
        ),
        code="F1",
    )
    assert findings_at(tmp_path, stale, "retriever.search") == {"hit_not_indexed": "error"}
    assert findings_at(tmp_path / "b", stale, "retriever")["docs_superseded"] == "error"
    swapped = with_faults(
        fault(
            "retriever.search",
            "retrieval_wrong_doc",
            replace="KB-REF-001",
            **{"with": "KB-REF-008"},
        ),
        code="F1",
    )
    got = findings_at(tmp_path / "c", swapped, "retriever.search")
    assert got.keys() & {"hit_score_mismatch", "hit_irrelevant"}


# ---------------------------------------------------------------- steps


def test_a_router_order_id_the_ticket_does_not_name(tmp_path: Path) -> None:
    sc = with_faults(fault("router", "set_field", field="order_id", value="O1999"), code="F5")
    assert findings_at(tmp_path, sc, "router") == {"order_id_in_ticket": "error"}


def test_unsupported_numbers_in_a_justification(tmp_path: Path) -> None:
    claim = "Per KB-REF-004, the refund is $59.25 within 3-5 business days."
    sc = with_faults(
        fault("policy", "hallucinated_claim", field="justification", claim=claim, mode="replace"),
        code="F6",
    )
    run = run_of(tmp_path, sc)
    flagged = [
        f.message
        for f in validate_run(run, CTX)
        if f.check == "numbers_supported" and f.span_id.endswith("/policy")
    ]
    assert any("$59.25" in m for m in flagged) and any("3-5 business days" in m for m in flagged)


def test_numeric_claims() -> None:
    claims = numeric_claims("Refund $1,234.50 minus 10% within 5‑7 business days or 30 minutes.")
    assert [c for c, _ in claims] == ["$1,234.50", "10%", "5‑7 business days", "30 minutes"]
    assert claims[0][1] == [1234.5] and claims[2][1] == [5.0, 7.0]


def test_findings_are_schema_records_and_short_circuit_on_errors() -> None:
    err = ValidatorFinding(span_id="r/x", check="c", severity="error", message="m")
    warn = err.model_copy(update={"severity": "warning"})
    assert short_circuit([warn, err]) and not short_circuit([warn]) and not short_circuit([])


# ---------------------------------------------------------------- the command


def test_validator_check_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    monkeypatch.setattr(validator_check, "REPORT_PATH", tmp_path / "v.json")
    monkeypatch.setattr(validator_check, "ROOT", tmp_path)
    f2 = next(
        s
        for s in DATASET
        if s.ground_truth.fault_code == "F2" and s.faults[0].injector == "retrieval_empty"
    )
    f4 = next(
        s
        for s in DATASET
        if s.ground_truth.fault_code == "F4"
        and s.faults[0].target_node_id.endswith("account_status")
    )
    with TraceStore.open() as store:
        for sc in (CONTROL, f2, f4):
            _, run = verify_scenario(
                make_gateway(tmp_path / sc.scenario_id), sc, ROLES, "default", {}, {"seed": 1}
            )
            assert run is not None
            store.save_run(run)
    assert validator_check.main([]) == 0
    out = capsys.readouterr().out
    assert "F4 caught by an error: 1/1 (all); F2 caught: 1/1 (most)" in out
    assert "controls (F0, no fault): 1 runs; 0 with an error finding" in out
    report = json.loads((tmp_path / "v.json").read_text("utf-8"))
    assert report["by_code"]["F4"] == {"runs": 1, "caught_error": 1, "caught_any": 1}
