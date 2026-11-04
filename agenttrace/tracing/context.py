"""TraceContext: collects the spans of one run.

The active trace and the currently open span live in `ContextVar`s, so any code that
runs inside a step (a model call, a tool call, a search) is recorded as a child of that
step without being passed a tracer. When no trace is active, nothing is recorded.

Note: `ContextVar`s do not carry into `ThreadPoolExecutor` workers on their own; submit
work with `contextvars.copy_context().run` if steps ever run in threads.

Span ids are deterministic: `{run_id}/{node_id}` for the first span of a node in a run,
then `{run_id}/{node_id}#2`, `#3`, ... (e.g. a JSON repair retry is `router.llm#2`), so a
replay of the same run lines its spans up with the original's.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel

from agenttrace.schemas.base import utc_now
from agenttrace.schemas.span import Explanation, Ref, Span, SpanKind

_active_trace: ContextVar[TraceContext | None] = ContextVar("agenttrace_trace", default=None)
_open_span: ContextVar[SpanBuilder | None] = ContextVar("agenttrace_span", default=None)


class HarnessError(RuntimeError):
    """A failure of the evaluation harness (fault plan, replay), not of the workflow. The
    runtime re-raises it instead of recording it as the agent failing."""


class InjectionError(HarnessError):
    """A fault plan is misconfigured. Re-raised by the runtime, so a broken scenario can
    never pass for an agent failure."""


class ReplayMissError(HarnessError):
    """A strict replay needed a node output the replay cache does not have."""


class FaultHook(Protocol):
    """Mutates a node's output as it is produced (implemented by faults.FaultPlan)."""

    def apply(
        self, node_id: str, kind: SpanKind, span_id: str, value: dict[str, Any]
    ) -> dict[str, Any]: ...


class ReplayHook(Protocol):
    """Serves recorded node outputs during replay (implemented by replay.ReplayCache).

    Calls (llm / tool / retrieval) are looked up before they execute; an agent is
    resolved after its body ran, once its nested calls are known.
    """

    def lookup_call(
        self,
        node_id: str,
        kind: SpanKind,
        model: str | None,
        params: Mapping[str, Any],
        span: SpanBuilder,
    ) -> bool: ...

    def record_call(
        self,
        node_id: str,
        kind: SpanKind,
        model: str | None,
        params: Mapping[str, Any],
        span: SpanBuilder,
    ) -> None: ...

    def resolve_agent(
        self,
        trace: TraceContext,
        span: SpanBuilder,
        output: dict[str, Any],
        explanation: Explanation | None,
    ) -> tuple[dict[str, Any], Explanation | None]: ...


def current_trace() -> TraceContext | None:
    return _active_trace.get()


def current_span() -> SpanBuilder | None:
    return _open_span.get()


def jsonable(value: Any) -> Any:
    """A JSON-safe copy of `value` (Pydantic models are dumped; unknown types become str)."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [jsonable(v) for v in value]
    return json.loads(json.dumps(value, default=str))


@dataclass
class SpanBuilder:
    """A span while it is open. Code inside the span fills in output, tokens, etc."""

    span_id: str
    node_id: str
    agent: str
    kind: SpanKind
    parent_span_id: str | None
    started_at: datetime
    input: dict[str, Any] = field(default_factory=dict)
    input_refs: list[Ref] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    explanation: Explanation | None = None
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    prompt_version: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    def build(self, run_id: str, ended_at: datetime) -> Span:
        return Span(
            span_id=self.span_id,
            run_id=run_id,
            parent_span_id=self.parent_span_id,
            node_id=self.node_id,
            agent=self.agent,
            kind=self.kind,
            input=jsonable(self.input),
            input_refs=self.input_refs,
            output=jsonable(self.output),
            explanation=self.explanation,
            model=self.model,
            params=jsonable(self.params),
            prompt_version=self.prompt_version,
            started_at=self.started_at,
            ended_at=ended_at,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost_usd,
            error=self.error,
        )


class TraceContext:
    """Spans of one run. Use as `with TraceContext(run_id) as trace: ...`."""

    def __init__(
        self,
        run_id: str,
        *,
        faults: FaultHook | None = None,
        replay: ReplayHook | None = None,
    ) -> None:
        self.run_id = run_id
        # Fault injection for this run (P4); None in normal runs. Never recorded in spans.
        self.faults = faults
        # Record/replay cache (P5); None in normal runs.
        self.replay = replay
        self._slots: list[SpanBuilder] = []  # in the order spans were opened
        self._done: dict[str, Span] = {}
        self._count: dict[str, int] = {}
        self._tokens: list[Any] = []

    def __enter__(self) -> TraceContext:
        self._tokens.append(_active_trace.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        _active_trace.reset(self._tokens.pop())

    @property
    def spans(self) -> list[Span]:
        """Closed spans, in the order they were opened (a parent before its children)."""
        return [self._done[s.span_id] for s in self._slots if s.span_id in self._done]

    def children(self, span_id: str) -> list[Span]:
        """Closed direct children, in the order they were opened."""
        return [
            self._done[s.span_id]
            for s in self._slots
            if s.parent_span_id == span_id and s.span_id in self._done
        ]

    def _new_span_id(self, node_id: str) -> str:
        n = self._count[node_id] = self._count.get(node_id, 0) + 1
        return f"{self.run_id}/{node_id}" if n == 1 else f"{self.run_id}/{node_id}#{n}"

    @contextmanager
    def span(
        self,
        kind: SpanKind,
        node_id: str,
        *,
        agent: str | None = None,
        input: Mapping[str, Any] | None = None,
        input_refs: Sequence[Ref] = (),
        params: Mapping[str, Any] | None = None,
        prompt_version: str | None = None,
    ) -> Iterator[SpanBuilder]:
        """Open a span under the current one. An exception is recorded, then re-raised."""
        parent = _open_span.get()
        span_id = self._new_span_id(node_id)
        builder = SpanBuilder(
            span_id=span_id,
            node_id=node_id,
            agent=agent or (parent.agent if parent else node_id),
            kind=kind,
            parent_span_id=parent.span_id if parent else None,
            started_at=utc_now(),
            input=dict(input or {}),
            input_refs=list(input_refs),
            params=dict(params or {}),
            prompt_version=prompt_version,
        )
        self._slots.append(builder)
        token = _open_span.set(builder)
        try:
            yield builder
        except Exception as exc:
            builder.error = builder.error or f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _open_span.reset(token)
            if builder.model is None:
                # A span without its own model (an agent) reports the model(s) its LLM calls used.
                models = sorted({c.model for c in self.children(span_id) if c.model})
                builder.model = ", ".join(models) or None
            self._done[span_id] = builder.build(self.run_id, utc_now())

    def instant(
        self,
        kind: SpanKind,
        node_id: str,
        *,
        agent: str | None = None,
        output: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> Span:
        """Record a zero-length span (the workflow input, or a step that was skipped)."""
        with self.span(kind, node_id, agent=agent, params=params) as builder:
            builder.output = dict(output or {})
            builder.error = error
        return self._done[builder.span_id]
