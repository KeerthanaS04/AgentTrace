"""Trace-level schemas: provenance refs, self-explanations, spans, node contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from agenttrace.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    Probability,
    StrictModel,
    UtcDatetime,
)

SpanKind = Literal["agent", "llm", "tool", "retrieval", "decision", "message"]


class Ref(StrictModel):
    """Provenance-carrying pointer to a value produced by an earlier span."""

    # The producer span and the output key it produced the value under.
    span_id: NonEmptyStr
    key: NonEmptyStr
    content_hash: NonEmptyStr
    # The consumer's input this value was bound to (e.g. writer input "decision"), so
    # data flow is exact per input, not just per pair of spans. None if not bound.
    input_key: str | None = None


class Explanation(StrictModel):
    """An agent's structured self-explanation of one step."""

    rationale: str
    cited_evidence: list[str] = Field(default_factory=list)
    confidence: Probability


class Span(StrictModel):
    span_id: NonEmptyStr
    run_id: NonEmptyStr
    # Nesting (which span contains this one). Data flow is `input_refs`, not this.
    parent_span_id: str | None = None
    # Logical node id, stable across runs, e.g. "retriever.search".
    node_id: NonEmptyStr
    agent: NonEmptyStr
    kind: SpanKind
    input: dict[str, Any] = Field(default_factory=dict)
    input_refs: list[Ref] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    explanation: Explanation | None = None
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str | None = None
    started_at: UtcDatetime
    ended_at: UtcDatetime
    tokens_in: NonNegativeInt = 0
    tokens_out: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = 0.0
    error: str | None = None
    # True if this span's output was replaced during counterfactual replay.
    patched: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> Span:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not be before started_at")
        if self.parent_span_id == self.span_id:
            raise ValueError("a span cannot be its own parent")
        if any(ref.span_id == self.span_id for ref in self.input_refs):
            raise ValueError("a span cannot consume its own output")
        return self


class NodeContract(StrictModel):
    """What a workflow step should do; given to judges."""

    node_id: NonEmptyStr
    description: str
    input_expectations: str
    output_expectations: str
    # Names of deterministic validators (schema, range, etc.).
    programmatic_checks: list[str] = Field(default_factory=list)
