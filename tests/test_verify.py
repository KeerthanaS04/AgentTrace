"""P4-T4: scenario verification judges each scenario against its label."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import make_gateway

from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios import verify
from agenttrace.scenarios.ground_truth import FaultInjection, Scenario, load_dataset
from agenttrace.scenarios.verify import Verification, summarize, verify_scenario
from agenttrace.workflow.agent_config import load_agents_config

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()


def find(code: str, ticket: str, injector: str | None = None, **params: Any) -> Scenario:
    """The dataset scenario with this code, base ticket, and (optionally) fault."""
    for s in DATASET:
        if s.ground_truth.fault_code != code or s.ticket.ticket_id != ticket:
            continue
        if injector is None or any(
            f.injector == injector and all(f.params.get(k) == v for k, v in params.items())
            for f in s.faults
        ):
            return s
    raise LookupError((code, ticket, injector, params))


def with_faults(base: Scenario, *faults: FaultInjection, code: str = "F3") -> Scenario:
    """A hand-made scenario on `base`'s ticket (for verdicts the dataset never hits)."""
    data = base.model_dump(mode="json")
    data["scenario_id"] = "sc-test"
    data["faults"] = [f.model_dump(mode="json") for f in faults]
    data["ground_truth"].update(
        fault_code=code,
        root_cause_node_ids=list(dict.fromkeys(f.target_node_id for f in faults)),
        affected_node_ids=[],
    )
    return Scenario.model_validate(data)


def check(tmp_path: Path, scenario: Scenario) -> Verification:
    v, _ = verify_scenario(make_gateway(tmp_path), scenario, ROLES, "default", {})
    return v


# ---------------------------------------------------------------- verdicts


def test_control_behaves_when_the_clean_decision_is_right(tmp_path: Path) -> None:
    v = check(tmp_path, find("F0", "T-0101"))  # the fake model approves, as expected
    assert v.behaves and v.reason == "ok" and v.applied == []


def test_effective_fault_behaves(tmp_path: Path) -> None:
    v = check(tmp_path, find("F5", "T-0101", "set_field", field="decision"))
    assert v.behaves and v.reason == "ok"
    assert v.applied == ["policy"] and v.root_changed
    assert v.clean_outcome is not None and v.faulty_outcome is not None
    assert v.clean_outcome[1] == "approve" and v.faulty_outcome[1] == "deny"
    assert v.final_correct is False
    assert "writer" in v.changed_steps


def test_silent_fault_behaves_when_the_outcome_is_unchanged(tmp_path: Path) -> None:
    v = check(tmp_path, find("F8", "T-0101", "tool_wrong_value", path="order.email"))
    assert v.behaves and v.reason == "ok (outcome unchanged)"
    assert v.applied == ["tool_agent.order_lookup"] and v.root_changed
    assert v.faulty_outcome == v.clean_outcome


def test_fault_without_effect_is_flagged(tmp_path: Path) -> None:
    # The scripted model ignores its documents, so emptying the search changes nothing.
    v = check(tmp_path, find("F2", "T-0101", "retrieval_empty"))
    assert not v.behaves and v.reason == "no effect on outcome"
    assert v.root_changed  # the fault did land; it just did not matter


def test_root_unchanged_is_flagged(tmp_path: Path) -> None:
    base = find("F0", "T-0101")
    nothing_to_swap = FaultInjection(
        target_node_id="retriever.search",
        injector="retrieval_wrong_doc",
        params={"replace": "KB-XXX-999", "with": "KB-REF-010"},
    )
    v = check(tmp_path, with_faults(base, nothing_to_swap, code="F1"))
    assert not v.behaves and v.reason.startswith("root output unchanged")


def test_unreached_target_is_flagged(tmp_path: Path) -> None:
    base = find("F0", "T-0101")
    fifth_call = FaultInjection(
        target_node_id="tool_agent.order_lookup",
        injector="tool_wrong_value",
        params={"path": "order.status", "value": "shipped", "occurrence": 5},
    )
    v = check(tmp_path, with_faults(base, fifth_call))
    assert not v.behaves and v.reason.startswith("fault not applied")


def test_fault_error_is_reported_not_raised(tmp_path: Path) -> None:
    base = find("F0", "T-0101")
    bad_path = FaultInjection(
        target_node_id="tool_agent.order_lookup",
        injector="tool_wrong_value",
        params={"path": "order.nope", "value": 1, "occurrence": 1},
    )
    v = check(tmp_path, with_faults(base, bad_path))
    assert not v.behaves and v.reason.startswith("fault error:")


def test_clean_run_is_shared_per_ticket(tmp_path: Path) -> None:
    gateway = make_gateway(tmp_path)
    cache: dict[str, Any] = {}
    for s in (find("F0", "T-0101"), find("F5", "T-0101", "set_field", field="decision")):
        verify_scenario(gateway, s, ROLES, "default", cache)
    assert list(cache) == ["T-0101"]


def test_summary_counts() -> None:
    rows = [
        Verification("a", "F1", "t", True, "ok"),
        Verification("b", "F1", "t", False, "no effect on outcome"),
        Verification("c", "F8", "t", True, "ok (outcome unchanged)"),
    ]
    s = summarize(rows)
    assert (s["n"], s["behaves"]) == (3, 2)
    assert s["by_code"] == {"F1": {"n": 2, "behaves": 1}, "F8": {"n": 1, "behaves": 1}}
    assert s["reasons"] == {"no effect on outcome": 1}


def test_full_store_run_prunes_runs_of_old_scenario_ids() -> None:
    class FakeStore:
        ids = ["sc-001-default", "sc-150-default", "sc-150-bolt", "clean-07-x", "T-0101-default"]

        def run_ids(self) -> list[str]:
            return list(self.ids)

        def delete_run(self, run_id: str) -> None:
            self.ids.remove(run_id)

    store = FakeStore()
    verify._prune_stale(store, {"sc-001-default", None}, "default")  # type: ignore[arg-type]
    # Only this profile's scenario runs that the verification did not produce go.
    assert store.ids == ["sc-001-default", "sc-150-bolt", "clean-07-x", "T-0101-default"]


# ---------------------------------------------------------------- the command


def test_main_reports_and_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}")),
    )
    monkeypatch.setattr(verify, "REPORT_PATH", tmp_path / "reports" / "v.json")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "store.db"))
    ids = [
        find("F0", "T-0101").scenario_id,
        find("F5", "T-0101", "set_field", field="decision").scenario_id,
        find("F2", "T-0101", "retrieval_empty").scenario_id,
    ]
    code = verify.main(["--only", *ids, "--store"])
    out = capsys.readouterr().out
    assert code == 1 and "FAIL" in out  # 2 of 3 behave: below the 95% target
    assert "behave as labelled: 2/3" in out and "no effect on outcome" in out
    report = json.loads((tmp_path / "reports" / "v.json").read_text(encoding="utf-8"))
    assert report["behaves"] == 2 and len(report["rows"]) == 3
    from agenttrace.store import TraceStore

    with TraceStore.open(tmp_path / "store.db") as store:
        stored = store.run_ids()
    assert len(stored) == 2  # the two faulty runs (controls have none)
    assert all(store_id.endswith("-default") for store_id in stored)
