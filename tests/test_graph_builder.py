"""P3-T1: the span graph matches the declared workflow topology on clean runs."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from test_export import failed_run, foreign_doc
from test_integrity import fixture_run
from test_store import traced_run

from agenttrace.graph import (
    GraphError,
    build_graph,
    data_subgraph,
    edges_of_kind,
    node_topology,
)
from agenttrace.schemas.run import Run
from agenttrace.store import TraceStore
from agenttrace.tracing.export import from_otel
from agenttrace.workflow import toy
from agenttrace.workflow.contracts import CONTRACTS
from agenttrace.workflow.support import STEPS, build_support_workflow


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return traced_run(tmp_path, "r1")


# ---------------------------------------------------------------- the done-when


def test_graph_matches_declared_workflow_topology(run: Run) -> None:
    g = build_graph(run, contracts=CONTRACTS)
    assert node_topology(g) == set(build_support_workflow().edges())


def test_toy_graph_matches_its_topology() -> None:
    run = toy.run_toy("I want to return my headphones for a refund", run_id="t").to_run()
    assert node_topology(build_graph(run)) == set(toy.build_toy_workflow().edges())


def test_topology_is_the_same_across_runs_and_models(tmp_path: Path) -> None:
    a = build_graph(traced_run(tmp_path, "a"))
    b = build_graph(traced_run(tmp_path, "b"))
    assert node_topology(a) == node_topology(b)


# ---------------------------------------------------------------- nodes


def test_every_span_is_a_node_with_attributes(run: Run) -> None:
    g = build_graph(run, contracts=CONTRACTS)
    assert set(g.nodes) == {s.span_id for s in run.spans}
    assert g.graph == {"run_id": "r1", "workflow_version": "support@1"}
    policy = g.nodes["r1/policy"]
    assert (policy["node_id"], policy["agent"], policy["kind"]) == ("policy", "policy", "agent")
    assert policy["status"] == "ok" and policy["error"] is None
    assert policy["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert policy["anomaly"] is None and policy["verdict"] is None  # filled in by diagnosis
    llm = g.nodes["r1/policy.llm"]
    assert llm["parent"] == "r1/policy" and llm["tokens_in"] == 11


def test_contracts_attach_by_node_id(run: Run) -> None:
    g = build_graph(run, contracts=CONTRACTS)
    for step in STEPS:
        assert g.nodes[f"r1/{step}"]["contract"] is CONTRACTS[step]
    assert g.nodes["r1/policy.llm"]["contract"] is None
    assert build_graph(run).nodes["r1/policy"]["contract"] is None  # contracts are optional


def test_failed_and_skipped_status(tmp_path: Path) -> None:
    g = build_graph(failed_run(tmp_path))
    assert g.nodes["bad/policy"]["status"] == "error"
    assert g.nodes["bad/writer"]["status"] == "skipped"
    assert g.nodes["bad/router"]["status"] == "ok"
    # Skipped steps consumed nothing, so they have no incoming data edges.
    assert not [u for u, _ in g.in_edges("bad/writer")]


# ---------------------------------------------------------------- edges


def test_data_edges_carry_their_bindings(run: Run) -> None:
    g = build_graph(run)
    # policy feeds the writer four inputs: one edge, four bindings.
    edge = g.edges["r1/policy", "r1/writer"]
    assert edge["kind"] == "data"
    assert sorted(b["input_key"] for b in edge["bindings"]) == [
        "decision",
        "escalate_to",
        "justification",
        "refund_amount",
    ]
    total = sum(len(g.edges[e]["bindings"]) for e in edges_of_kind(g, "data"))
    assert total == sum(len(s.input_refs) for s in run.spans) == 16


def test_nested_edges_point_child_to_parent(run: Run) -> None:
    g = build_graph(run)
    nested = edges_of_kind(g, "nested")
    assert ("r1/tool_agent.order_lookup", "r1/tool_agent") in nested
    assert all(g.nodes[u]["parent"] == v for u, v in nested)
    assert len(nested) == sum(1 for s in run.spans if s.parent_span_id)
    # A tool call's effect reaches the final reply through its step.
    assert nx.has_path(g, "r1/tool_agent.order_lookup", "r1/writer")


def test_nested_edges_can_be_left_out(run: Run) -> None:
    g = build_graph(run, nested=False)
    assert edges_of_kind(g, "nested") == []
    assert set(g.nodes) == {s.span_id for s in run.spans}  # nodes are still all there


def test_data_subgraph_drops_nesting(run: Run) -> None:
    sub = data_subgraph(build_graph(run))
    assert edges_of_kind(sub, "nested") == [] and len(sub) == len(run.spans)
    assert not nx.has_path(sub, "r1/tool_agent.order_lookup", "r1/writer")


def test_graph_is_a_dag_with_the_input_first(run: Run) -> None:
    g = build_graph(run)
    assert nx.is_directed_acyclic_graph(g)
    order = [g.nodes[n]["node_id"] for n in nx.topological_sort(data_subgraph(g))]
    agents = [n for n in order if n in (*STEPS, "__input__")]
    assert agents.index("__input__") < agents.index("router") < agents.index("writer")


# ---------------------------------------------------------------- other sources


def test_stored_and_foreign_runs_build(tmp_path: Path, run: Run) -> None:
    with TraceStore.open(tmp_path / "g.db") as store:
        store.save_run(run)
        assert node_topology(build_graph(store.get_run("r1"))) == node_topology(build_graph(run))
    (foreign,) = from_otel(foreign_doc())
    g = build_graph(foreign)
    assert edges_of_kind(g, "data") == [] and len(edges_of_kind(g, "nested")) == 2


def test_broken_traces_are_refused() -> None:
    with pytest.raises(GraphError, match="missing span"):
        build_graph(fixture_run("orphan_ref"))
    with pytest.raises(GraphError, match="parent"):
        build_graph(fixture_run("orphan_parent"))
    with pytest.raises(GraphError, match="cycle"):
        build_graph(fixture_run("ref_cycle"))
