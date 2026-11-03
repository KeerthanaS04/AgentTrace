"""P3-T2: graph queries, tested on hand-built graphs (answers worked out by hand)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from fake_llm import TICKET, default_script, make_gateway
from test_export import failed_run
from test_store import traced_run

from agenttrace.graph import (
    ancestors,
    build_graph,
    critical_path,
    descendants,
    final_spans,
    first_divergence,
    paths_to_final,
    topological_order,
)
from agenttrace.workflow.support import build_support_workflow, make_resources

# ---------------------------------------------------------------- a hand-built graph
#
#   in --> a --> c --> out          durations: in 0, a 3, b 1, c 2, out 1, d 10, t 2
#    \          ^                   final: out
#     `-> b ----'                   t is nested inside a (t -> a, kind "nested")
#          `--> d                   d is a side branch that never reaches `out`
#   t ==> a

NODES = ["in", "a", "b", "c", "out", "d", "t"]
DATA = [("in", "a"), ("in", "b"), ("a", "c"), ("b", "c"), ("c", "out"), ("b", "d")]
DURATION = {"in": 0, "a": 3, "b": 1, "c": 2, "out": 1, "d": 10, "t": 2}


def hand(
    *,
    final: tuple[str, ...] | None = ("out",),
    outputs: dict[str, str] | None = None,
    status: dict[str, str] | None = None,
    nodes: list[str] = NODES,
    prefix: str = "",
) -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    for seq, n in enumerate(nodes):
        g.add_node(
            prefix + n,
            local_id=n,
            seq=seq,
            duration_s=float(DURATION.get(n, 0)),
            tokens_in=10 * seq,
            tokens_out=1,
            output_hash=(outputs or {}).get(n, f"h-{n}"),
            status=(status or {}).get(n, "ok"),
        )
    for u, v in DATA:
        if u in nodes and v in nodes:
            g.add_edge(prefix + u, prefix + v, kind="data")
    if "t" in nodes:
        g.add_edge(prefix + "t", prefix + "a", kind="nested")
    if final is not None:
        g.graph["final"] = [prefix + f for f in final]
    return g


G = hand()


def test_ancestors_and_descendants() -> None:
    assert ancestors(G, "c") == {"in", "a", "b", "t"}
    assert ancestors(G, "in") == set()
    assert descendants(G, "b") == {"c", "out", "d"}
    assert descendants(G, "t") == {"a", "c", "out"}  # through the nested edge
    assert descendants(G, "out") == set()
    with pytest.raises(KeyError, match="no span 'zz'"):
        ancestors(G, "zz")


def test_topological_order_is_deterministic() -> None:
    # Ready spans are taken in recorded order: in(0); then b(2) before t(6); then d(5)
    # before t(6); only then a (needs t), c, out.
    assert topological_order(G) == ["in", "b", "d", "t", "a", "c", "out"]
    assert topological_order(G, ["c", "a", "in"]) == ["in", "a", "c"]


def test_final_spans() -> None:
    assert final_spans(G) == ["out"]
    # Without a recorded final, the graph's sinks stand in.
    assert final_spans(hand(final=None)) == ["out", "d"]


def test_paths_to_final() -> None:
    assert paths_to_final(G, "in") == [["in", "a", "c", "out"], ["in", "b", "c", "out"]]
    assert paths_to_final(G, "t") == [["t", "a", "c", "out"]]
    assert paths_to_final(G, "d") == []  # a side branch cannot affect the result
    assert paths_to_final(G, "out") == [["out"]]
    assert paths_to_final(G, "b", targets=["d"]) == [["b", "d"]]


def test_critical_path_by_duration() -> None:
    # in-b-d = 0+1+10 = 11 beats in-a-c-out = 0+3+2+1 = 6.
    cp = critical_path(G)
    assert (cp.spans, cp.total, cp.weight) == (["in", "b", "d"], 11.0, "duration_s")
    # Counting nesting adds t to the a-branch: 2+3+2+1 = 8, still < 11.
    assert critical_path(G, data_only=False).spans == ["in", "b", "d"]
    no_d = hand(nodes=[n for n in NODES if n != "d"])
    assert critical_path(no_d).spans == ["in", "a", "c", "out"]
    assert critical_path(no_d, data_only=False).spans == ["t", "a", "c", "out"]


def test_critical_path_by_tokens_and_empty() -> None:
    # tokens = 10*seq + 1: in 1, a 11, b 21, c 31, out 41, d 51, t 61.
    # Step level: a counts its nested call t too (11 + 61 = 72), so
    # in-a-c-out = 1+72+31+41 = 145 beats in-b-c-out = 1+21+31+41 = 94 and in-b-d = 73.
    cp = critical_path(G, weight="tokens")
    assert cp.spans == ["in", "a", "c", "out"] and cp.total == 145
    # Span level (no rollup): t-a-c-out = 61+11+31+41 = 144 beats in-b-c-out = 94.
    span_level = critical_path(G, weight="tokens", data_only=False)
    assert span_level.spans == ["t", "a", "c", "out"] and span_level.total == 144
    assert critical_path(nx.DiGraph()).spans == []


def test_first_divergence_finds_where_runs_split() -> None:
    a = hand(prefix="A/")
    # b's output changed, so c and out changed too: the split is at b, not downstream.
    b = hand(prefix="B/", outputs={"b": "x", "c": "x", "out": "x"})
    d = first_divergence(a, b)
    assert d is not None and (d.local_id, d.span_a, d.span_b, d.reason) == (
        "b",
        "A/b",
        "B/b",
        "output_changed",
    )


def test_first_divergence_reasons() -> None:
    a = hand()
    assert first_divergence(a, hand()) is None
    status = first_divergence(a, hand(status={"c": "error"}))
    assert status is not None and (status.local_id, status.reason) == ("c", "status_changed")
    missing = first_divergence(a, hand(nodes=[n for n in NODES if n != "d"]))
    assert missing is not None and (missing.local_id, missing.reason) == ("d", "missing_in_b")
    extra = first_divergence(hand(nodes=[n for n in NODES if n != "d"]), a)
    assert extra is not None and (extra.local_id, extra.span_a, extra.reason) == (
        "d",
        None,
        "extra_in_b",
    )


# ---------------------------------------------------------------- on real runs


def test_real_run_queries(tmp_path: Path) -> None:
    g = build_graph(traced_run(tmp_path, "r1"))
    assert final_spans(g) == ["r1/router", "r1/policy", "r1/writer"]
    assert {"r1/__input__", "r1/router", "r1/retriever", "r1/policy", "r1/policy.llm"} <= (
        ancestors(g, "r1/writer")
    )
    # A tool call reaches the reply through the Tool Agent and the Policy Agent.
    paths = paths_to_final(g, "r1/tool_agent.order_lookup", targets=["r1/writer"])
    assert paths == [["r1/tool_agent.order_lookup", "r1/tool_agent", "r1/policy", "r1/writer"]]
    order = topological_order(g)
    assert order[0] == "r1/__input__" and order[-1] == "r1/writer"
    assert critical_path(g).spans[0] == "r1/__input__"


def test_failed_run_finals_exclude_the_skipped_writer(tmp_path: Path) -> None:
    g = build_graph(failed_run(tmp_path))
    assert "bad/writer" not in final_spans(g) and "bad/router" in final_spans(g)


def run_support(tmp_path: Path, run_id: str, script: Any = default_script) -> nx.DiGraph:
    gateway = make_gateway(tmp_path, script=script)  # same cache dir across calls
    result = build_support_workflow().run(
        {"ticket": TICKET}, run_id=run_id, resources=make_resources(gateway)
    )
    return build_graph(result.to_run())


def test_cached_rerun_does_not_diverge(tmp_path: Path) -> None:
    """The rerun is served from cache (cache_hit flips to True): not a real difference."""
    a = run_support(tmp_path, "live")
    b = run_support(tmp_path, "cached")
    assert first_divergence(a, b) is None


def test_changed_model_reply_is_located_in_the_llm_call(tmp_path: Path) -> None:
    def deny(prompt: str) -> dict[str, Any]:
        out = default_script(prompt)
        if prompt.startswith("You are the Policy Agent"):
            out = {**out, "decision": "deny", "refund_amount": None}
        return out

    a = run_support(tmp_path / "a", "orig")
    b = run_support(tmp_path / "b", "changed", script=deny)
    d = first_divergence(a, b)
    # Everything upstream matches; the split starts in the Policy Agent's model call.
    assert d is not None and (d.local_id, d.reason) == ("policy.llm", "output_changed")


# ---------------------------------------------------------------- CLI


def test_cli_graph(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agenttrace import cli
    from agenttrace.store import TraceStore

    db = str(tmp_path / "g.db")
    with TraceStore.open(db) as store:
        store.save_run(traced_run(tmp_path, "r1"))
        store.save_run(traced_run(tmp_path, "r2"))
    assert cli.main(["graph", "r1", "--db", db, "--from", "tool_agent.order_lookup"]) == 0
    out = capsys.readouterr().out
    assert "final spans   : router, policy, writer" in out
    assert (
        "causal order  : __input__ -> router -> retriever -> tool_agent -> policy -> writer" in out
    )
    assert "tool_agent.order_lookup -> tool_agent -> policy -> writer" in out
    assert cli.main(["graph", "r1", "--diff", "r2", "--db", db]) == 0
    assert "first divergence vs r2: none" in capsys.readouterr().out
    assert cli.main(["graph", "r1", "--from", "nope", "--db", db]) == 2
    assert cli.main(["graph", "missing", "--db", db]) == 2


def test_token_critical_path_on_a_real_run_is_a_chain_of_steps(tmp_path: Path) -> None:
    """Regression: tokens live on LLM spans, so the step-level path must roll them up
    (it once returned a lone `policy.llm` span)."""
    run = traced_run(tmp_path, "r1")  # every LLM call reports 11 + 7 tokens
    g = build_graph(run)
    cp = critical_path(g, weight="tokens")
    assert all(g.nodes[n]["parent"] is None for n in cp.spans)  # steps only
    assert cp.spans[-1] == "r1/writer" and len(cp.spans) > 1
    llm_calls = {
        step: sum(1 for s in run.spans if s.parent_span_id == f"r1/{step}" and s.kind == "llm")
        for step in ("router", "retriever", "tool_agent", "policy", "writer")
    }
    expected = 18 * sum(
        llm_calls[g.nodes[n]["node_id"]] for n in cp.spans if g.nodes[n]["node_id"] in llm_calls
    )
    assert cp.total == expected
