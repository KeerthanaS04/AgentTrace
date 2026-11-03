"""Align the spans of two runs of the same workflow by logical node (P3-T3).

    alignment = align_runs(original, replay)
    alignment.changed      # local ids whose span differs, e.g. {"policy.llm", "policy", ...}
    alignment.unchanged    # local ids identical in both runs

Spans are matched by logical node and occurrence: the k-th `tool_agent.llm` span of run
A pairs with the k-th of run B, which is exactly their local id (span id minus the run
prefix: "tool_agent.llm#2"), so different run ids never matter. Each pair is compared on
status, input, output (cache-only fields ignored), model, params, and the replay
`patched` flag; `changes` says which. Used for replay diffs (P5) and regression
comparison across runs and models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agenttrace.gateway.cache import content_hash
from agenttrace.graph.builder import build_graph, output_hash, span_status
from agenttrace.graph.queries import topological_order
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span

PairStatus = Literal["unchanged", "changed", "only_a", "only_b"]
Aspect = Literal["status", "input", "output", "model", "params", "patched"]


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class AlignedSpan:
    local_id: str
    node_id: str
    span_a: str | None
    span_b: str | None
    status: PairStatus
    changes: tuple[Aspect, ...] = ()  # empty unless status == "changed"


@dataclass(frozen=True)
class Alignment:
    run_a: str
    run_b: str
    pairs: list[AlignedSpan] = field(default_factory=list)  # in run A's causal order

    def _ids(self, status: PairStatus) -> set[str]:
        return {p.local_id for p in self.pairs if p.status == status}

    @property
    def changed(self) -> set[str]:
        return self._ids("changed")

    @property
    def unchanged(self) -> set[str]:
        return self._ids("unchanged")

    @property
    def only_a(self) -> set[str]:
        return self._ids("only_a")

    @property
    def only_b(self) -> set[str]:
        return self._ids("only_b")

    @property
    def identical(self) -> bool:
        return all(p.status == "unchanged" for p in self.pairs)

    def pair(self, local_id: str) -> AlignedSpan:
        return next(p for p in self.pairs if p.local_id == local_id)

    def node_status(self) -> dict[str, PairStatus]:
        """Per logical node: the shared status if all its occurrences agree, else "changed".

        So a node is "unchanged" only if every occurrence matched; one extra or missing
        call (e.g. a third `tool_agent.llm` in B) makes the node "changed".
        """
        seen: dict[str, set[PairStatus]] = {}
        for p in self.pairs:
            seen.setdefault(p.node_id, set()).add(p.status)
        return {node: next(iter(s)) if len(s) == 1 else "changed" for node, s in seen.items()}


def _local_id(span: Span, run: Run) -> str:
    return span.span_id.removeprefix(f"{run.run_id}/")


def _changes(a: Span, b: Span) -> tuple[Aspect, ...]:
    found: list[Aspect] = []
    if span_status(a) != span_status(b):
        found.append("status")
    if content_hash(a.input) != content_hash(b.input):
        found.append("input")
    if output_hash(a.output) != output_hash(b.output):
        found.append("output")
    if a.model != b.model:
        found.append("model")
    if content_hash(a.params) != content_hash(b.params):
        found.append("params")
    if a.patched != b.patched:
        found.append("patched")
    return tuple(found)


def align_runs(a: Run, b: Run, *, same_workflow: bool = True) -> Alignment:
    """Pair every span of `a` with its counterpart in `b` (see module docstring).

    Pairs follow run A's causal order; spans only in B come last, in B's causal order.
    With `same_workflow`, runs of different workflow versions are refused.
    """
    if same_workflow and a.workflow_version != b.workflow_version:
        raise AlignmentError(
            f"runs are different workflows: {a.workflow_version!r} vs {b.workflow_version!r}"
        )
    spans_a = {_local_id(s, a): s for s in a.spans}
    spans_b = {_local_id(s, b): s for s in b.spans}
    local_a = {s.span_id: _local_id(s, a) for s in a.spans}
    local_b = {s.span_id: _local_id(s, b) for s in b.spans}

    pairs: list[AlignedSpan] = []
    for span_id in topological_order(build_graph(a)):
        local = local_a[span_id]
        sa, sb = spans_a[local], spans_b.get(local)
        if sb is None:
            pairs.append(AlignedSpan(local, sa.node_id, sa.span_id, None, "only_a"))
            continue
        changes = _changes(sa, sb)
        status: PairStatus = "changed" if changes else "unchanged"
        pairs.append(AlignedSpan(local, sa.node_id, sa.span_id, sb.span_id, status, changes))
    for span_id in topological_order(build_graph(b)):
        local = local_b[span_id]
        if local not in spans_a:
            sb = spans_b[local]
            pairs.append(AlignedSpan(local, sb.node_id, None, sb.span_id, "only_b"))
    return Alignment(run_a=a.run_id, run_b=b.run_id, pairs=pairs)
