"""Build a causal graph (networkx DiGraph) from a run's spans (P3-T1).

    g = build_graph(run, contracts=CONTRACTS)

Nodes are spans, keyed by span_id, with attributes:
    node_id, agent, kind, model, status ("ok" / "error" / "skipped"), error, parent,
    tokens_in, tokens_out, cost_usd, contract (NodeContract or None), and placeholders
    anomaly / verdict (None until diagnosis fills them in).

Edges point the way causation flows, and carry `kind`:
    "data"   producer -> consumer, one per input ref (attrs: key, input_key, content_hash).
             This is the workflow's declared data flow (P2-T2).
    "nested" child -> parent: a model / tool / search call inside a step contributes to
             that step's output. Without these, nested spans would be isolated and a fault
             in one tool call could never be traced to the step it broke.
The graph is a DAG for any trace that passes the integrity checks (P2-T5).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import networkx as nx

from agenttrace.schemas.run import Run
from agenttrace.schemas.span import NodeContract, Ref, Span

EdgeKind = Literal["data", "nested"]
SpanStatus = Literal["ok", "error", "skipped"]


class GraphError(ValueError):
    pass


def span_status(span: Span) -> SpanStatus:
    if span.error is None:
        return "ok"
    return "skipped" if span.error.startswith("skipped:") else "error"


def build_graph(
    run: Run,
    *,
    contracts: Mapping[str, NodeContract] | None = None,
    nested: bool = True,
) -> nx.DiGraph:
    """The run as a DAG. `contracts` maps node_id -> NodeContract (attached by node_id)."""
    contracts = contracts or {}
    g: nx.DiGraph = nx.DiGraph(run_id=run.run_id, workflow_version=run.workflow_version)
    for span in run.spans:
        g.add_node(
            span.span_id,
            node_id=span.node_id,
            agent=span.agent,
            kind=span.kind,
            model=span.model,
            status=span_status(span),
            error=span.error,
            parent=span.parent_span_id,
            tokens_in=span.tokens_in,
            tokens_out=span.tokens_out,
            cost_usd=span.cost_usd,
            contract=contracts.get(span.node_id),
            anomaly=None,
            verdict=None,
        )
    for span in run.spans:
        for ref in span.input_refs:
            if ref.span_id not in g:
                raise GraphError(f"{span.span_id}: input ref to missing span {ref.span_id!r}")
            if g.has_edge(ref.span_id, span.span_id):
                # Two inputs from the same producer: keep one edge, list every binding.
                g.edges[ref.span_id, span.span_id]["bindings"].append(_binding(ref))
                continue
            g.add_edge(ref.span_id, span.span_id, kind="data", bindings=[_binding(ref)])
        if nested and span.parent_span_id is not None:
            if span.parent_span_id not in g:
                raise GraphError(f"{span.span_id}: parent {span.parent_span_id!r} is missing")
            g.add_edge(span.span_id, span.parent_span_id, kind="nested")
    if not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        raise GraphError(f"run {run.run_id!r} graph has a cycle: {cycle}")
    return g


def _binding(ref: Ref) -> dict[str, Any]:
    return {"key": ref.key, "input_key": ref.input_key, "content_hash": ref.content_hash}


def edges_of_kind(g: nx.DiGraph, kind: EdgeKind) -> list[tuple[str, str]]:
    return [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == kind]


def data_subgraph(g: nx.DiGraph) -> nx.DiGraph:
    """Only the data-flow edges (all nodes kept)."""
    sub: nx.DiGraph = nx.DiGraph()
    sub.graph.update(g.graph)  # not DiGraph(g.graph): a dict argument is read as edges
    sub.add_nodes_from(g.nodes(data=True))
    sub.add_edges_from((u, v, d) for u, v, d in g.edges(data=True) if d["kind"] == "data")
    return sub


def node_topology(g: nx.DiGraph) -> set[tuple[str, str]]:
    """Data-flow edges as (producer node_id, consumer node_id): comparable across runs."""
    return {(g.nodes[u]["node_id"], g.nodes[v]["node_id"]) for u, v in edges_of_kind(g, "data")}
