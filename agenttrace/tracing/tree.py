"""Print a run's spans as an indented tree (nesting by `parent_span_id`)."""

from __future__ import annotations

from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span


def _short(span_id: str, run_id: str) -> str:
    return span_id.removeprefix(f"{run_id}/")


def _line(span: Span) -> str:
    ms = (span.ended_at - span.started_at).total_seconds() * 1000
    bits = [f"{span.kind:<9}", f"{span.node_id}"]
    if _short(span.span_id, span.run_id) != span.node_id:
        bits.append(f"[{_short(span.span_id, span.run_id)}]")
    if span.model:
        bits.append(f"model={span.model}")
    if span.tokens_in or span.tokens_out:
        bits.append(f"tokens={span.tokens_in}/{span.tokens_out}")
    if span.cost_usd:
        bits.append(f"${span.cost_usd:.5f}")
    if span.input_refs:
        bits.append(
            "refs=" + ",".join(f"{_short(r.span_id, span.run_id)}.{r.key}" for r in span.input_refs)
        )
    bits.append(f"{ms:.0f}ms")
    if span.error:
        bits.append(f"ERROR {span.error[:80]}")
    return " ".join(bits)


def format_tree(run: Run) -> str:
    kids: dict[str | None, list[Span]] = {}
    for span in run.spans:
        kids.setdefault(span.parent_span_id, []).append(span)
    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for span in kids.get(parent, []):
            lines.append("  " * depth + _line(span))
            walk(span.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)
