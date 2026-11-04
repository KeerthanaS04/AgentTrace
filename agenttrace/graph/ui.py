"""Serialize a run's causal graph as JSON for the UI's React Flow canvas (P3-T4).

    doc = to_ui_graph(build_graph(run, contracts=CONTRACTS), run)   # a UiGraph
    doc.model_dump(mode="json")                                     # what the API sends

Nodes and edges use React Flow's field names (`id`, `type`, `position`, `data`;
`source`, `target`, `label`, `animated`, `style`), so the UI can pass them straight in.
Every node gets a `display_status` (root_cause / propagated / suspicious / error /
skipped / ok, in that priority) and its colour; the legend carries the same palette so
the UI never hard-codes it.

Layout hints: top-level steps sit in columns by causal depth (input first), stacked by
recorded order; calls nested in a step sit in the same column just below it.

A replay's graph can carry a diff overlay (P5-T5, `with_replay_overlay`): each node's
`replay_status` (patched / changed / new / unchanged) and what changed, plus a `replay`
block with the final-output changes, spans only the original had, and (when graded)
recovery, with its own legend.

The JSON Schema the UI consumes is generated from these models and committed at
`ui/schema/run-graph.schema.json`:

    uv run python -m agenttrace.graph.ui      # rewrite the schema file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from pydantic import Field

from agenttrace.graph.builder import data_subgraph
from agenttrace.graph.queries import final_spans, topological_order
from agenttrace.schemas.base import StrictModel
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import SpanKind

UI_SCHEMA_VERSION: Literal[1] = 1
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "ui" / "schema" / "run-graph.schema.json"

DisplayStatus = Literal["root_cause", "propagated", "suspicious", "error", "skipped", "ok"]

# Colour-blind-safe palette; root cause and errors are the loudest.
STATUS_STYLE: dict[DisplayStatus, tuple[str, str]] = {
    "root_cause": ("#d62728", "Root cause"),
    "propagated": ("#ff7f0e", "Propagated error"),
    "suspicious": ("#bcbd22", "Suspicious"),
    "error": ("#9467bd", "Failed"),
    "skipped": ("#7f7f7f", "Skipped"),
    "ok": ("#2ca02c", "OK"),
}

ReplayStatus = Literal["patched", "changed", "new", "missing", "unchanged"]

REPLAY_STYLE: dict[ReplayStatus, tuple[str, str]] = {
    "patched": ("#1f77b4", "Patched (output replaced)"),
    "changed": ("#e377c2", "Re-ran, output changed"),
    "new": ("#17becf", "New in the replay"),
    "missing": ("#c7c7c7", "Only in the original"),
    "unchanged": ("#d9d9d9", "Unchanged"),
}

COLUMN_GAP = 300
STEP_WIDTH, STEP_HEIGHT = 240, 64
CALL_WIDTH, CALL_HEIGHT = 220, 40
ROW_GAP = 24


# ---------------------------------------------------------------- the schema


class Position(StrictModel):
    x: float
    y: float


class UiNodeData(StrictModel):
    label: str
    node_id: str
    local_id: str
    kind: SpanKind
    agent: str
    display_status: DisplayStatus
    color: str
    is_final: bool
    parent: str | None = None  # span id of the step this call is nested in
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    error: str | None = None
    verdict: str | None = None
    contract: str | None = None  # the step's contract description, if any
    # Set only on a replay's graph (P5-T5).
    replay_status: ReplayStatus | None = None
    replay_changes: list[str] = Field(default_factory=list)
    layer: int = Field(description="Column: causal depth of the step (0 = workflow input).")
    row: int = Field(description="Position within the column, top to bottom.")


class UiNode(StrictModel):
    id: str
    type: SpanKind  # React Flow node type: the UI registers one component per span kind
    position: Position
    width: int
    height: int
    data: UiNodeData


class UiEdgeData(StrictModel):
    kind: Literal["data", "nested"]
    bindings: list[str] = Field(
        default_factory=list, description="'producer_key -> input_key' for each value passed."
    )


class UiEdge(StrictModel):
    id: str
    source: str
    target: str
    type: Literal["smoothstep", "straight"]
    label: str | None = None
    animated: bool
    style: dict[str, str]
    data: UiEdgeData


class LegendItem(StrictModel):
    status: DisplayStatus
    color: str
    label: str


class ReplayLegendItem(StrictModel):
    status: ReplayStatus
    color: str
    label: str


class UiFinalChange(StrictModel):
    key: str
    before: Any = None
    after: Any = None


class UiReplay(StrictModel):
    """What a replay changed against its original run (P5-T5)."""

    original_run_id: str
    patch_source: str | None = None
    patched: list[str] = Field(default_factory=list)  # local ids
    missing: list[str] = Field(
        default_factory=list, description="Local ids only the original has (not drawn)."
    )
    final_changes: list[UiFinalChange] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    # Set when the replay was graded against a scenario's expected outcome.
    recovered: bool | None = None
    recovery_score: float | None = None
    before_score: float | None = None
    clean_correct: bool | None = None
    legend: list[ReplayLegendItem]


class UiGraph(StrictModel):
    schema_version: Literal[1] = UI_SCHEMA_VERSION
    run_id: str
    scenario_id: str | None = None
    workflow_version: str
    final_correct: bool | None = None
    final_spans: list[str]
    nodes: list[UiNode]
    edges: list[UiEdge]
    legend: list[LegendItem]
    replay: UiReplay | None = None


# ---------------------------------------------------------------- building it


def display_status(attrs: dict[str, Any]) -> DisplayStatus:
    role = attrs.get("role")
    if role in ("root_cause", "propagated"):
        return role
    if attrs.get("status") in ("error", "skipped"):
        return attrs["status"]
    if attrs.get("verdict") in ("suspicious", "faulty"):
        return "suspicious"
    return "ok"


def _layers(g: nx.DiGraph) -> dict[str, int]:
    """Causal depth of each top-level span along data edges (longest path from a root)."""
    steps = [n for n in g if g.nodes[n].get("parent") is None]
    data = data_subgraph(g).subgraph(steps)
    depth: dict[str, int] = {}
    for n in topological_order(data):
        depth[n] = max((depth[p] + 1 for p in data.predecessors(n)), default=0)
    return depth


def _label(attrs: dict[str, Any]) -> str:
    local = str(attrs.get("local_id", ""))
    if attrs.get("kind") == "llm" and attrs.get("model"):
        return f"{local} · {str(attrs['model']).split('/')[-1]}"
    return local


def to_ui_graph(g: nx.DiGraph, run: Run | None = None) -> UiGraph:
    finals = set(final_spans(g))
    layers = _layers(g)
    children: dict[str, list[str]] = {}
    for n in topological_order(g):
        parent = g.nodes[n].get("parent")
        if parent is not None:
            children.setdefault(parent, []).append(n)

    nodes: list[UiNode] = []
    by_layer: dict[int, list[str]] = {}
    for n in topological_order(g):
        if n in layers:
            by_layer.setdefault(layers[n], []).append(n)

    def make(n: str, x: float, y: float, layer: int, row: int, nested: bool) -> UiNode:
        a = g.nodes[n]
        status = display_status(a)
        contract = a.get("contract")
        return UiNode(
            id=n,
            type=a.get("kind", "agent"),
            position=Position(x=x, y=y),
            width=CALL_WIDTH if nested else STEP_WIDTH,
            height=CALL_HEIGHT if nested else STEP_HEIGHT,
            data=UiNodeData(
                label=_label(a),
                node_id=a.get("node_id", n),
                local_id=a.get("local_id", n),
                kind=a.get("kind", "agent"),
                agent=a.get("agent", ""),
                display_status=status,
                color=STATUS_STYLE[status][0],
                is_final=n in finals,
                parent=a.get("parent"),
                model=a.get("model"),
                tokens_in=a.get("tokens_in", 0),
                tokens_out=a.get("tokens_out", 0),
                cost_usd=a.get("cost_usd", 0.0),
                duration_ms=round(a.get("duration_s", 0.0) * 1000, 3),
                error=a.get("error"),
                verdict=a.get("verdict"),
                contract=getattr(contract, "description", None),
                layer=layer,
                row=row,
            ),
        )

    for layer in sorted(by_layer):
        y, row = 0.0, 0
        for step in by_layer[layer]:
            x = layer * COLUMN_GAP
            nodes.append(make(step, x, y, layer, row, nested=False))
            row += 1
            y += STEP_HEIGHT + ROW_GAP
            # Nested calls (and calls nested in those) go just below their step.
            stack = list(reversed(children.get(step, [])))
            while stack:
                child = stack.pop()
                nodes.append(make(child, x + 10, y, layer, row, nested=True))
                row += 1
                y += CALL_HEIGHT + ROW_GAP / 2
                stack.extend(reversed(children.get(child, [])))
            y += ROW_GAP

    edges: list[UiEdge] = []
    for u, v, d in g.edges(data=True):
        kind = d["kind"]
        bindings = [f"{b['key']} -> {b['input_key'] or b['key']}" for b in d.get("bindings", [])]
        edges.append(
            UiEdge(
                id=f"{kind}:{u}->{v}",
                source=u,
                target=v,
                type="smoothstep" if kind == "data" else "straight",
                label=", ".join(b.split(" -> ")[1] for b in bindings) or None,
                animated=kind == "data",
                style={"stroke": "#555555"}
                if kind == "data"
                else {"stroke": "#bbbbbb", "strokeDasharray": "4 4"},
                data=UiEdgeData(kind=kind, bindings=bindings),
            )
        )
    edges.sort(key=lambda e: e.id)

    return UiGraph(
        run_id=g.graph.get("run_id", run.run_id if run else "unknown"),
        scenario_id=run.scenario_id if run else None,
        workflow_version=g.graph.get("workflow_version", run.workflow_version if run else ""),
        final_correct=run.final_correct if run else None,
        final_spans=final_spans(g),
        nodes=nodes,
        edges=edges,
        legend=[LegendItem(status=s, color=c, label=lbl) for s, (c, lbl) in STATUS_STYLE.items()],
    )


def replay_legend() -> list[ReplayLegendItem]:
    return [ReplayLegendItem(status=s, color=c, label=lbl) for s, (c, lbl) in REPLAY_STYLE.items()]


def with_replay_overlay(
    doc: UiGraph,
    statuses: dict[str, tuple[ReplayStatus, list[str]]],
    replay: UiReplay,
) -> UiGraph:
    """A replay's graph with each node's diff status (keyed by local id) and the summary."""
    nodes = []
    for node in doc.nodes:
        status, changes = statuses.get(node.data.local_id, ("unchanged", []))
        data = node.data.model_copy(update={"replay_status": status, "replay_changes": changes})
        nodes.append(node.model_copy(update={"data": data}))
    return doc.model_copy(update={"nodes": nodes, "replay": replay})


# ---------------------------------------------------------------- the schema file


def ui_json_schema() -> dict[str, Any]:
    schema = UiGraph.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://agenttrace.local/schema/run-graph.schema.json"
    schema["title"] = "AgentTrace run graph (React Flow)"
    return schema


def render_schema() -> str:
    return json.dumps(ui_json_schema(), indent=2, sort_keys=True) + "\n"


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_schema(), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"wrote {write_schema()}")
