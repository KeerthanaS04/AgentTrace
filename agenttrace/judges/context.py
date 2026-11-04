"""What a judge sees about one span (P6-T1).

    ctx = judge_context(run, "policy")
    ctx.citable      span ids the judge may cite;  ctx.doc_ids  policy docs it may cite

For a step (agent span): its contract, its full inputs, output, and self-explanation; the
outputs of the steps it read from (via its input refs), summarized; and the calls made
inside it, summarized. For a call (tool / retrieval span): the tool's description, its
arguments and result, and the enclosing step's inputs, summarized.

Model-call spans are not judged on their own: they repeat their step's prompt, and the
step is judged. Nothing derived from ground truth is shown (no `final_correct`, no
scenario id), and nothing about injected faults exists in a trace to show.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span
from agenttrace.workflow.contracts import CONTRACTS
from agenttrace.workflow.tools import TOOLS

JUDGED_KINDS = ("agent", "tool", "retrieval")
DOC_ID = re.compile(r"\bKB-[A-Z]{3}-\d{3}\b")
MAX_TEXT = 400  # characters per string in a summary
MAX_ITEMS = 8  # list items in a summary

RETRIEVAL_CONTRACT = (
    "Search the policy knowledge base (BM25) for a query.",
    "A short keyword query and the number of hits wanted.",
    "The best-matching policy documents for the query, most relevant first.",
)


class NotJudged(ValueError):
    """The span is not a kind a node judge scores (e.g. a model call)."""


@dataclass
class JudgeContext:
    span_id: str  # local id, e.g. "policy" or "tool_agent.order_lookup"
    node_id: str
    kind: str
    description: str
    input_expectations: str
    output_expectations: str
    inputs: Any
    output: Any
    explanation: Any
    parents: dict[str, Any] = field(default_factory=dict)  # local id -> summarized output
    calls: dict[str, Any] = field(default_factory=dict)
    citable: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)

    def refs(self) -> set[str]:
        return {*self.citable, *self.doc_ids}


def summarize(value: Any, depth: int = 0) -> Any:
    """A shortened copy for context: long strings cut, long lists elided, deep nests flattened."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + " [...]"
    if isinstance(value, list):
        items = [summarize(v, depth + 1) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"[... {len(value) - MAX_ITEMS} more]")
        return items
    if isinstance(value, dict):
        if depth >= 4:
            return json.dumps(value, default=str)[:MAX_TEXT]
        return {k: summarize(v, depth + 1) for k, v in value.items()}
    return value


def _local(run: Run, span_id: str) -> str:
    return span_id.removeprefix(f"{run.run_id}/")


def judge_context(run: Run, target: str) -> JudgeContext:
    """The context for judging span `target` (local or full span id) of `run`."""
    by_id = {s.span_id: s for s in run.spans}
    span = by_id.get(target) or by_id.get(f"{run.run_id}/{target}")
    if span is None:
        raise KeyError(f"run {run.run_id} has no span {target!r}")
    if span.kind not in JUDGED_KINDS:
        raise NotJudged(f"{_local(run, span.span_id)} is a {span.kind} span; judge its step")

    parents: dict[str, Any] = {}
    calls: dict[str, Any] = {}
    if span.kind == "agent":
        contract = CONTRACTS.get(span.node_id)
        description = contract.description if contract else f"Workflow step {span.node_id}."
        expects_in = contract.input_expectations if contract else ""
        expects_out = contract.output_expectations if contract else ""
        for ref in span.input_refs:
            producer = by_id.get(ref.span_id)
            if producer is not None and producer.node_id != "__input__":
                local = _local(run, producer.span_id)
                parents.setdefault(local, {})[ref.key] = summarize(producer.output.get(ref.key))
        for child in run.spans:
            if child.parent_span_id == span.span_id and child.kind in ("tool", "retrieval"):
                calls[_local(run, child.span_id)] = {
                    "input": summarize(child.input),
                    "output": summarize(child.output),
                    "error": child.error,
                }
    else:
        step = by_id[span.parent_span_id] if span.parent_span_id else None
        if span.kind == "tool":
            spec = TOOLS.get(str(span.input.get("tool", "")))
            description = spec.description if spec else f"Tool call {span.node_id}."
            expects_in = "Arguments matching the tool's input schema."
            expects_out = "The tool's true result for these arguments, in its output schema."
        else:
            description, expects_in, expects_out = RETRIEVAL_CONTRACT
        if step is not None:
            parents[_local(run, step.span_id)] = {"input": summarize(step.input)}

    citable = [_local(run, span.span_id), *parents, *calls]
    text = json.dumps([span.input, span.output, parents, calls], default=str)
    return JudgeContext(
        span_id=_local(run, span.span_id),
        node_id=span.node_id,
        kind=span.kind,
        description=description,
        input_expectations=expects_in,
        output_expectations=expects_out,
        inputs=span.input,
        output=span.output,
        explanation=span.explanation.model_dump(mode="json") if span.explanation else None,
        parents=parents,
        calls=calls,
        citable=list(dict.fromkeys(citable)),
        doc_ids=sorted(set(DOC_ID.findall(text))),
    )


def judged_spans(run: Run) -> list[Span]:
    """Spans a node judge scores, in recorded order."""
    return [s for s in run.spans if s.kind in JUDGED_KINDS]
