"""Check a traced run's data flow against the workflow's declared wiring (P2-T2).

For every step span, its `input_refs` must match the declaration input by input:
consumer input `writer.decision` must come from producer `policy`, key `decision`, and
nothing else. Each ref's content hash must equal the hash of the producer span's
recorded output AND of the consumer span's recorded input, so the value that arrived
is provably the value that was produced. Skipped steps consumed nothing.

Returns human-readable problems; an empty list means provenance is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from agenttrace.gateway.cache import content_hash
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span
from agenttrace.workflow.runtime import INPUT_STEP, FromInput, Workflow, step_id


@dataclass(frozen=True)
class DeclaredInput:
    """One declared binding: `consumer.input_key <- producer.key`."""

    consumer: str
    input_key: str
    producer: str
    key: str


def declared_inputs(workflow: Workflow) -> list[DeclaredInput]:
    return [
        DeclaredInput(
            consumer=name,
            input_key=input_key,
            producer=INPUT_STEP if isinstance(src, FromInput) else src.step,
            key=src.key,
        )
        for name in workflow.order
        for input_key, src in workflow.steps[name].inputs.items()
    ]


def _skipped(span: Span) -> bool:
    return bool(span.error and span.error.startswith("skipped:"))


def check_provenance(run: Run, workflow: Workflow) -> list[str]:
    spans = {s.span_id: s for s in run.spans}
    short = len(run.run_id) + 1  # strip "<run_id>/" in messages
    problems: list[str] = []

    want_version = f"{workflow.name}@{workflow.version}"
    if run.workflow_version != want_version:
        problems.append(f"run is {run.workflow_version!r}, workflow is {want_version!r}")
    if step_id(run.run_id, INPUT_STEP) not in spans:
        problems.append("no span for the workflow input")

    declared: dict[str, dict[str, DeclaredInput]] = {}
    for binding in declared_inputs(workflow):
        declared.setdefault(binding.consumer, {})[binding.input_key] = binding

    for name in workflow.order:
        span = spans.get(step_id(run.run_id, name))
        if span is None or span.kind != "agent":
            problems.append(f"{name}: no agent span")
            continue
        if _skipped(span):
            if span.input_refs:
                problems.append(f"{name}: skipped, but has input refs")
            continue

        seen: set[str] = set()
        for ref in span.input_refs:
            if ref.input_key is None:
                problems.append(f"{name}: ref to {ref.span_id[short:]}.{ref.key} names no input")
                continue
            if ref.input_key in seen:
                problems.append(f"{name}.{ref.input_key}: more than one ref")
                continue
            seen.add(ref.input_key)
            d = declared[name].get(ref.input_key)
            if d is None:
                problems.append(f"{name}.{ref.input_key}: not a declared input")
                continue
            where = f"{name}.{ref.input_key}"
            if ref.span_id != step_id(run.run_id, d.producer) or ref.key != d.key:
                problems.append(
                    f"{where}: came from {ref.span_id[short:]}.{ref.key}, "
                    f"declared {d.producer}.{d.key}"
                )
                continue
            producer = spans.get(ref.span_id)
            if producer is None:
                problems.append(f"{where}: producer span {ref.span_id[short:]} is missing")
            elif d.key not in producer.output:
                problems.append(f"{where}: producer {d.producer} recorded no output {d.key!r}")
            elif content_hash(producer.output[d.key]) != ref.content_hash:
                problems.append(f"{where}: ref hash differs from what {d.producer} produced")
            if ref.input_key not in span.input:
                problems.append(f"{where}: not in the span's recorded input")
            elif content_hash(span.input[ref.input_key]) != ref.content_hash:
                problems.append(f"{where}: recorded input differs from the value referenced")
        for missing in sorted(set(declared[name]) - seen):
            problems.append(f"{name}.{missing}: declared input has no ref")
    return problems
