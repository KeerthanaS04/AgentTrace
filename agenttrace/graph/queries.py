"""Queries over a run's causal graph (P3-T2).

All functions take a graph from `build_graph` (or any DiGraph whose nodes carry the
attributes they mention), and are deterministic: ties are broken by recorded order
(`seq`), then span id.

    ancestors(g, s)        everything that could have influenced span s
    descendants(g, s)      everything span s could have influenced
    topological_order(g)   causes before effects
    paths_to_final(g, s)   every causal path from s to the run's final output
    critical_path(g)       the chain of spans that determines total time (or cost, tokens)
    first_divergence(a, b) the earliest span whose output differs between two runs
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

Weight = Literal["duration_s", "cost_usd", "tokens"]


def _order_key(g: nx.DiGraph) -> Any:
    return lambda n: (g.nodes[n].get("seq", float("inf")), str(n))


def _check(g: nx.DiGraph, span_id: str) -> None:
    if span_id not in g:
        raise KeyError(f"no span {span_id!r} in the graph")


def ancestors(g: nx.DiGraph, span_id: str) -> set[str]:
    """Spans with a causal path *to* `span_id` (not including it)."""
    _check(g, span_id)
    return set(nx.ancestors(g, span_id))


def descendants(g: nx.DiGraph, span_id: str) -> set[str]:
    """Spans `span_id` has a causal path *to* (not including it)."""
    _check(g, span_id)
    return set(nx.descendants(g, span_id))


def topological_order(g: nx.DiGraph, nodes: Iterable[str] | None = None) -> list[str]:
    """Causes before effects; among independent spans, recorded order first."""
    sub = g if nodes is None else g.subgraph(nodes)
    return list(nx.lexicographical_topological_sort(sub, key=_order_key(g)))


def final_spans(g: nx.DiGraph) -> list[str]:
    """Where the run's result comes from (set by `build_graph`; else the graph's sinks)."""
    finals = g.graph.get("final")
    if finals is not None:
        return list(finals)
    return sorted((n for n in g if g.out_degree(n) == 0), key=_order_key(g))


def paths_to_final(
    g: nx.DiGraph, span_id: str, targets: Iterable[str] | None = None
) -> list[list[str]]:
    """Every simple causal path from `span_id` to a final span (or to `targets`).

    A final span itself has the one-element path [span_id]. Paths are sorted by length,
    then by the recorded order of their nodes, so the shortest route comes first.
    """
    _check(g, span_id)
    key = _order_key(g)
    out: list[list[str]] = []
    for target in targets if targets is not None else final_spans(g):
        _check(g, target)
        if target == span_id:
            out.append([span_id])
        else:
            out.extend(list(p) for p in nx.all_simple_paths(g, span_id, target))
    return sorted(out, key=lambda p: (len(p), [key(n) for n in p]))


def _weight(g: nx.DiGraph, node: str, weight: Weight) -> float:
    attrs = g.nodes[node]
    if weight == "tokens":
        return float(attrs.get("tokens_in", 0) + attrs.get("tokens_out", 0))
    return float(attrs.get(weight, 0.0))


@dataclass(frozen=True)
class CriticalPath:
    spans: list[str]
    total: float
    weight: Weight


def critical_path(
    g: nx.DiGraph, *, weight: Weight = "duration_s", data_only: bool = True
) -> CriticalPath:
    """The heaviest causal chain: the spans that set the run's total time (or cost/tokens).

    With `data_only` (default) the chain is over steps (spans not nested in another), and
    follows data edges only. A step's time already includes its nested calls; its tokens
    and cost do not (they live on the LLM spans), so for those weights each step counts
    its own plus everything nested inside it. With `data_only=False` every span and edge
    counts, each span with only its own weight.
    """
    sub = g
    own = {n: _weight(g, n, weight) for n in g}
    if data_only:
        nested: nx.DiGraph = nx.DiGraph()
        nested.add_nodes_from(g.nodes)
        nested.add_edges_from((u, v) for u, v, d in g.edges(data=True) if d.get("kind") == "nested")
        steps = [n for n in g if nested.out_degree(n) == 0]  # not inside another span
        sub = nx.DiGraph()
        sub.add_nodes_from(steps)
        sub.add_edges_from(
            (u, v)
            for u, v, d in g.edges(data=True)
            if d.get("kind") != "nested" and u in sub and v in sub
        )
        if weight != "duration_s":
            own = {n: own[n] + sum(own[c] for c in nx.ancestors(nested, n)) for n in steps}
    best: dict[str, float] = {}
    via: dict[str, str | None] = {}
    for node in topological_order(sub):
        preds = sorted(sub.predecessors(node), key=_order_key(g))
        top = max(preds, key=lambda p: best[p], default=None)
        best[node] = own[node] + (best[top] if top is not None else 0.0)
        via[node] = top
    if not best:
        return CriticalPath(spans=[], total=0.0, weight=weight)
    order = {n: i for i, n in enumerate(topological_order(sub))}
    end = max(best, key=lambda n: (best[n], -order[n]))
    path = [end]
    while (prev := via[path[-1]]) is not None:
        path.append(prev)
    return CriticalPath(spans=path[::-1], total=best[end], weight=weight)


# ---------------------------------------------------------------- comparing two runs


DivergenceReason = Literal["output_changed", "status_changed", "missing_in_b", "extra_in_b"]


@dataclass(frozen=True)
class Divergence:
    local_id: str  # e.g. "retriever" or "tool_agent.llm#2"
    span_a: str | None
    span_b: str | None
    reason: DivergenceReason


def _local(g: nx.DiGraph) -> dict[str, str]:
    """local_id -> span_id."""
    return {g.nodes[n].get("local_id", n): n for n in g}


def first_divergence(a: nx.DiGraph, b: nx.DiGraph) -> Divergence | None:
    """The earliest span (in run `a`'s causal order) that differs in run `b`.

    Spans are matched by local id (span id without the run prefix), so the same step of
    two runs lines up. A span differs if its status or output hash changed, or it is
    missing from `b`. Because ancestors come first in causal order, the span returned is
    one whose own inputs did not differ: where the runs actually split. If every span
    of `a` matches, the first span that only `b` has is reported; None if identical.
    """
    in_b = _local(b)
    for node in topological_order(a):
        local = a.nodes[node].get("local_id", node)
        other = in_b.get(local)
        if other is None:
            return Divergence(local, node, None, "missing_in_b")
        if a.nodes[node].get("status") != b.nodes[other].get("status"):
            return Divergence(local, node, other, "status_changed")
        if a.nodes[node].get("output_hash") != b.nodes[other].get("output_hash"):
            return Divergence(local, node, other, "output_changed")
    in_a = _local(a)
    for node in topological_order(b):
        local = b.nodes[node].get("local_id", node)
        if local not in in_a:
            return Divergence(local, None, node, "extra_in_b")
    return None
