"""Trace integrity checks: structural problems any trace must not have (P2-T5).

Independent of any workflow, so imported foreign traces are checked too. (Whether a
run's data flow matches *our* workflow's declaration is `workflow/provenance.py`.)

    issues = check_integrity(run)    # [] when the trace is sound
    validate_run(run)                # raises TraceIntegrityError listing every issue

Already enforced by the schemas themselves (so such traces never load): unique span ids,
spans belonging to the run, end >= start, no span its own parent or consuming its own
output.
"""

from __future__ import annotations

from dataclasses import dataclass

from agenttrace.gateway.cache import content_hash
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    span_id: str | None
    message: str

    def __str__(self) -> str:
        where = f"{self.span_id}: " if self.span_id else ""
        return f"[{self.code}] {where}{self.message}"


class TraceIntegrityError(ValueError):
    def __init__(self, run_id: str, issues: list[IntegrityIssue]) -> None:
        self.run_id = run_id
        self.issues = issues
        lines = "\n".join(f"  - {i}" for i in issues)
        super().__init__(f"run {run_id!r} failed {len(issues)} integrity check(s):\n{lines}")


def _find_cycle(edges: dict[str, list[str]]) -> list[str] | None:
    """A cycle in a directed graph, as [a, b, ..., a], or None. Iterative DFS."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(edges, WHITE)
    for start in sorted(edges):
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path = [start]
        color[start] = GREY
        while stack:
            node, i = stack[-1]
            nxt = edges.get(node, [])
            if i < len(nxt):
                stack[-1] = (node, i + 1)
                child = nxt[i]
                if color.get(child, BLACK) == GREY:
                    return path[path.index(child) :] + [child]
                if color.get(child) == WHITE:
                    color[child] = GREY
                    stack.append((child, 0))
                    path.append(child)
            else:
                color[node] = BLACK
                stack.pop()
                path.pop()
    return None


def check_integrity(run: Run) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    spans: dict[str, Span] = {s.span_id: s for s in run.spans}
    if not spans:
        return [IntegrityIssue("empty", None, "the run has no spans")]

    def add(code: str, span: Span, message: str) -> None:
        issues.append(IntegrityIssue(code, span.span_id, message))

    for span in run.spans:
        # Nesting: the parent must exist, and contain the child in time.
        if span.parent_span_id is not None:
            parent = spans.get(span.parent_span_id)
            if parent is None:
                add("orphan_parent", span, f"parent {span.parent_span_id!r} is not in the run")
            elif span.started_at < parent.started_at or span.ended_at > parent.ended_at:
                add(
                    "child_outside_parent",
                    span,
                    f"runs {span.started_at.isoformat()}..{span.ended_at.isoformat()}, outside "
                    f"parent {parent.span_id!r} "
                    f"({parent.started_at.isoformat()}..{parent.ended_at.isoformat()})",
                )

        # Data flow: every ref resolves to a recorded output, unchanged, produced earlier.
        for ref in span.input_refs:
            producer = spans.get(ref.span_id)
            label = f"input ref {ref.span_id!r}.{ref.key}"
            if producer is None:
                add("orphan_ref", span, f"{label}: producer span is not in the run")
                continue
            if ref.key not in producer.output:
                add("ref_key_missing", span, f"{label}: producer recorded no output {ref.key!r}")
            elif content_hash(producer.output[ref.key]) != ref.content_hash:
                add("ref_hash_mismatch", span, f"{label}: hash differs from the recorded output")
            if producer.ended_at > span.started_at:
                add(
                    "ref_from_future",
                    span,
                    f"{label}: consumed at {span.started_at.isoformat()}, but the producer "
                    f"finished at {producer.ended_at.isoformat()}",
                )

    nesting: dict[str, list[str]] = {sid: [] for sid in spans}  # child -> parent
    flow: dict[str, list[str]] = {sid: [] for sid in spans}  # producer -> consumers
    for span in run.spans:
        if span.parent_span_id is not None and span.parent_span_id in spans:
            nesting[span.span_id].append(span.parent_span_id)
        for ref in span.input_refs:
            if ref.span_id in spans:
                flow[ref.span_id].append(span.span_id)
    for code, graph, what in (
        ("parent_cycle", nesting, "spans are nested in a loop"),
        ("ref_cycle", flow, "data flows in a loop"),
    ):
        cycle = _find_cycle(graph)
        if cycle:
            issues.append(IntegrityIssue(code, cycle[0], f"{what}: {' -> '.join(cycle)}"))
    return issues


def validate_run(run: Run) -> Run:
    """Return `run` unchanged if it is sound; otherwise raise listing every issue."""
    issues = check_integrity(run)
    if issues:
        raise TraceIntegrityError(run.run_id, issues)
    return run
