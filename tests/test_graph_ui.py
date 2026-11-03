"""P3-T4: the UI graph JSON validates against the committed schema the UI consumes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaError
from test_export import failed_run, foreign_doc
from test_store import traced_run

from agenttrace import cli
from agenttrace.graph import build_graph
from agenttrace.graph.ui import (
    SCHEMA_PATH,
    STATUS_STYLE,
    display_status,
    render_schema,
    to_ui_graph,
)
from agenttrace.schemas.run import Run
from agenttrace.store import TraceStore
from agenttrace.tracing.export import from_otel
from agenttrace.workflow import toy
from agenttrace.workflow.contracts import CONTRACTS

VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def ui_doc(run: Run) -> dict[str, Any]:
    doc = to_ui_graph(build_graph(run, contracts=CONTRACTS), run).model_dump(mode="json")
    # Round-trip through text, as the API would send it.
    parsed: dict[str, Any] = json.loads(json.dumps(doc))
    return parsed


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return traced_run(tmp_path, "r1")


# ---------------------------------------------------------------- the done-when


def test_committed_schema_is_current() -> None:
    """ui/schema/run-graph.schema.json is exactly what the models generate."""
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render_schema()
    Draft202012Validator.check_schema(json.loads(render_schema()))


def test_exported_graph_validates_against_the_schema(tmp_path: Path, run: Run) -> None:
    VALIDATOR.validate(ui_doc(run))
    VALIDATOR.validate(ui_doc(failed_run(tmp_path)))
    VALIDATOR.validate(ui_doc(toy.run_toy("refund my headphones", run_id="t").to_run()))
    (foreign,) = from_otel(foreign_doc())
    VALIDATOR.validate(ui_doc(foreign))


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (lambda d: d.pop("nodes"), "missing nodes"),
        (lambda d: d["nodes"][0]["data"].update(display_status="weird"), "unknown status"),
        (lambda d: d["nodes"][0].update(extra=1), "unknown field"),
        (lambda d: d["edges"][0]["data"].update(kind="other"), "unknown edge kind"),
        (lambda d: d.update(schema_version=2), "future version"),
        (lambda d: d["nodes"][0]["position"].pop("y"), "position without y"),
    ],
)
def test_schema_rejects_malformed_documents(run: Run, mutate: Any, why: str) -> None:
    doc = ui_doc(run)
    mutate(doc)
    with pytest.raises(SchemaError):
        VALIDATOR.validate(doc)


# ---------------------------------------------------------------- content


def test_nodes_and_edges_cover_the_graph(run: Run) -> None:
    g = build_graph(run)
    doc = ui_doc(run)
    ids = [n["id"] for n in doc["nodes"]]
    assert sorted(ids) == sorted(g.nodes) and len(set(ids)) == len(ids)
    assert len(doc["edges"]) == g.number_of_edges()
    assert all(e["source"] in ids and e["target"] in ids for e in doc["edges"])
    assert len({e["id"] for e in doc["edges"]}) == len(doc["edges"])
    assert doc["final_spans"] == ["r1/router", "r1/policy", "r1/writer"]
    finals = {n["id"] for n in doc["nodes"] if n["data"]["is_final"]}
    assert finals == set(doc["final_spans"])


def test_node_data_for_the_span_panel(run: Run) -> None:
    nodes = {n["id"]: n for n in ui_doc(run)["nodes"]}
    policy = nodes["r1/policy"]
    assert policy["type"] == "agent" and policy["data"]["display_status"] == "ok"
    assert policy["data"]["color"] == STATUS_STYLE["ok"][0]
    assert policy["data"]["contract"] == CONTRACTS["policy"].description
    llm = nodes["r1/policy.llm"]
    assert llm["type"] == "llm" and llm["data"]["parent"] == "r1/policy"
    assert llm["data"]["label"] == "policy.llm · nemotron-3.5-lightning-30b-a3b"
    assert llm["data"]["tokens_in"] == 11 and llm["data"]["duration_ms"] >= 0


def test_edges_are_styled_by_kind(run: Run) -> None:
    edges = {e["id"]: e for e in ui_doc(run)["edges"]}
    data = edges["data:r1/policy->r1/writer"]
    assert data["animated"] and data["type"] == "smoothstep"
    assert data["label"] == "decision, refund_amount, escalate_to, justification"
    assert "decision -> decision" in data["data"]["bindings"]
    nested = edges["nested:r1/policy.llm->r1/policy"]
    assert not nested["animated"] and nested["style"]["strokeDasharray"] == "4 4"


def test_layout_columns_follow_causal_depth(run: Run) -> None:
    nodes = {n["data"]["local_id"]: n for n in ui_doc(run)["nodes"]}
    layer = {
        k: nodes[k]["data"]["layer"]
        for k in ("__input__", "router", "retriever", "tool_agent", "policy", "writer")
    }
    assert layer == {
        "__input__": 0,
        "router": 1,
        "retriever": 2,
        "tool_agent": 2,
        "policy": 3,
        "writer": 4,
    }
    # Nested calls sit in their step's column, below it.
    assert nodes["tool_agent.order_lookup"]["data"]["layer"] == 2
    assert nodes["tool_agent.order_lookup"]["position"]["y"] > nodes["tool_agent"]["position"]["y"]
    # Nothing overlaps within a column.
    by_x: dict[float, list[tuple[float, float]]] = {}
    for n in nodes.values():
        x0 = n["position"]["x"] // 300
        by_x.setdefault(x0, []).append((n["position"]["y"], n["position"]["y"] + n["height"]))
    for spans in by_x.values():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_failed_run_statuses(tmp_path: Path) -> None:
    nodes = {n["data"]["local_id"]: n["data"] for n in ui_doc(failed_run(tmp_path))["nodes"]}
    assert nodes["policy"]["display_status"] == "error" and nodes["policy"]["error"]
    assert nodes["writer"]["display_status"] == "skipped"
    assert nodes["writer"]["color"] == STATUS_STYLE["skipped"][0]


def test_display_status_priority() -> None:
    assert display_status({"role": "root_cause", "status": "error"}) == "root_cause"
    assert display_status({"role": "propagated", "verdict": "ok"}) == "propagated"
    assert display_status({"status": "skipped", "verdict": "faulty"}) == "skipped"
    assert display_status({"status": "ok", "verdict": "suspicious"}) == "suspicious"
    assert display_status({"status": "ok"}) == "ok"


def test_diagnosed_nodes_are_colored(run: Run) -> None:
    g = build_graph(run)
    g.nodes["r1/retriever"]["role"] = "root_cause"  # as diagnosis (P7) will set it
    g.nodes["r1/policy"]["role"] = "propagated"
    doc = to_ui_graph(g, run).model_dump(mode="json")
    VALIDATOR.validate(doc)
    status = {n["id"]: n["data"]["display_status"] for n in doc["nodes"]}
    assert status["r1/retriever"] == "root_cause" and status["r1/policy"] == "propagated"
    assert {i["status"] for i in doc["legend"]} == set(STATUS_STYLE)


# ---------------------------------------------------------------- CLI


def test_cli_graph_writes_ui_json(
    tmp_path: Path, run: Run, capsys: pytest.CaptureFixture[str]
) -> None:
    db, out = str(tmp_path / "u.db"), tmp_path / "ui" / "r1.json"
    with TraceStore.open(db) as store:
        store.save_run(run)
    assert cli.main(["graph", "r1", "--json", str(out), "--db", db]) == 0
    assert "UI graph JSON" in capsys.readouterr().out
    VALIDATOR.validate(json.loads(out.read_text(encoding="utf-8")))
