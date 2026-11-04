"""Record/replay cache (P5-T1): node outputs, keyed by what produced them.

A call (llm, tool, retrieval) is keyed, before it runs, by the content hash of

    (node_id, kind, model, params, input)

so the same node, asked the same thing with the same model and params, gets the output
it produced when recorded, without calling anything. For a model call `model` and
`params` are the resolved model and merged params (seed included); for tools and searches
they are empty.

An agent is keyed by the same over its own node id, model(s), step params and inputs,
plus a *fingerprint* (key and output) of each call nested in it, in order. It is resolved
after its body ran: if its nested calls produced exactly what they produced when
recorded, the agent's recorded output is used. So an output changed after the agent
computed it (an injected fault, P4; later a patch, P5-T2) is reproduced too, while an
agent whose calls came out differently (a new seed, an edited input) keeps its new output.

Two modes:
    record (default)  a miss runs live and is added to the cache
    strict            a model-call miss raises ReplayMissError: proof that a replay made
                      no live model call. Tools, searches and agents are deterministic
                      code over their inputs, so a miss there just recomputes (a failed
                      tool call, never recorded, re-raises its own error the same way).

Errors are never recorded, so a failed call runs again. One output per key: the first
recorded wins (the gateway cache already makes a repeated request return the same text).
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenttrace.gateway.cache import CompletionCache, content_hash
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation, Span, SpanKind
from agenttrace.tracing.context import ReplayMissError, SpanBuilder, TraceContext, jsonable

ChildrenOf = Callable[[str], Sequence[Span]]
CALL_KINDS: tuple[SpanKind, ...] = ("llm", "tool", "retrieval")
REPLAYED_KINDS: tuple[SpanKind, ...] = (*CALL_KINDS, "agent")


def node_key(
    node_id: str,
    kind: str,
    model: str | None,
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
    nested: Sequence[str] = (),
) -> str:
    """The replay key: content hash of what determines a node's output."""
    return content_hash(
        {
            "node_id": node_id,
            "kind": kind,
            "model": model,
            "params": jsonable(dict(params)),
            "input": jsonable(dict(inputs)),
            "nested": list(nested),
        }
    )


def span_key(span: Span, children_of: ChildrenOf) -> str:
    """The key a recorded span is filed under (an agent's includes its nested calls)."""
    nested = [fingerprint(c, children_of) for c in children_of(span.span_id)]
    return node_key(span.node_id, span.kind, span.model, span.params, span.input, nested)


def fingerprint(span: Span, children_of: ChildrenOf) -> str:
    """What a span was asked and what it answered: equal fingerprints, equal behaviour."""
    return content_hash({"key": span_key(span, children_of), "output": span.output})


@dataclass(frozen=True)
class Recorded:
    """A node's recorded result: enough to reproduce its span and return value."""

    output: dict[str, Any]
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    explanation: dict[str, Any] | None = None

    @classmethod
    def of(cls, span: Span | SpanBuilder) -> Recorded:
        expl = span.explanation.model_dump(mode="json") if span.explanation else None
        return cls(
            output=jsonable(span.output),
            model=span.model,
            params=jsonable(span.params),
            tokens_in=span.tokens_in,
            tokens_out=span.tokens_out,
            explanation=expl,
        )

    def apply(self, span: SpanBuilder) -> None:
        """Reproduce the recorded span. Nothing was spent, so its cost is 0."""
        span.output = copy.deepcopy(self.output)
        span.model = self.model
        span.params = copy.deepcopy(self.params)
        span.tokens_in = self.tokens_in
        span.tokens_out = self.tokens_out
        span.cost_usd = 0.0
        span.explanation = self.explain()

    def explain(self) -> Explanation | None:
        return None if self.explanation is None else Explanation.model_validate(self.explanation)

    def to_json(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "model": self.model,
            "params": self.params,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "explanation": self.explanation,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Recorded:
        return cls(**doc)


def run_children(run: Run) -> ChildrenOf:
    """children_of for a stored run: direct children in recorded order."""
    by_parent: dict[str, list[Span]] = {}
    for span in run.spans:
        if span.parent_span_id is not None:
            by_parent.setdefault(span.parent_span_id, []).append(span)
    return lambda span_id: by_parent.get(span_id, [])


class ReplayCache:
    """Implements the tracing `ReplayHook`. Pass it as `Workflow.run(replay=...)`.

    In memory; with `path`, also persisted in SQLite (content-addressed, shared by runs).
    """

    def __init__(self, *, strict: bool = False, path: Path | str | None = None) -> None:
        self.strict = strict
        self._memory: dict[str, Recorded] = {}
        self._disk = CompletionCache(path) if path is not None else None
        self.hits: Counter[str] = Counter()
        self.misses: Counter[str] = Counter()

    @classmethod
    def from_runs(
        cls, runs: Iterable[Run], *, strict: bool = False, path: Path | str | None = None
    ) -> ReplayCache:
        cache = cls(strict=strict, path=path)
        for run in runs:
            cache.add_run(run)
        return cache

    def add_run(self, run: Run) -> int:
        """File every successful call and agent span of a recorded run; returns how many
        new keys were added."""
        children_of = run_children(run)
        added = 0
        for span in run.spans:
            if span.kind in REPLAYED_KINDS and span.error is None:
                added += self._put(span_key(span, children_of), Recorded.of(span))
        return added

    def __len__(self) -> int:
        # Everything put is written to disk too, so disk (when used) holds all keys.
        return len(self._memory) if self._disk is None else len(self._disk)

    def get(self, key: str) -> Recorded | None:
        rec = self._memory.get(key)
        if rec is None and self._disk is not None:
            doc = self._disk.get(key)
            if doc is not None:
                rec = self._memory[key] = Recorded.from_json(doc)
        return rec

    def _put(self, key: str, rec: Recorded) -> int:
        if self.get(key) is not None:
            return 0  # first recorded wins
        self._memory[key] = rec
        if self._disk is not None:
            self._disk.put(key, json.loads(json.dumps(rec.to_json())))
        return 1

    def _miss(self, kind: str, node_id: str) -> None:
        self.misses[kind] += 1
        if self.strict and kind == "llm":
            raise ReplayMissError(f"strict replay: no recorded output for {node_id} ({kind})")

    # ------------------------------------------------------------ ReplayHook

    def lookup_call(
        self,
        node_id: str,
        kind: SpanKind,
        model: str | None,
        params: Mapping[str, Any],
        span: SpanBuilder,
    ) -> bool:
        rec = self.get(node_key(node_id, kind, model, params, span.input))
        if rec is None:
            self._miss(kind, node_id)
            return False
        self.hits[kind] += 1
        rec.apply(span)
        return True

    def record_call(
        self,
        node_id: str,
        kind: SpanKind,
        model: str | None,
        params: Mapping[str, Any],
        span: SpanBuilder,
    ) -> None:
        self._put(node_key(node_id, kind, model, params, span.input), Recorded.of(span))

    def resolve_agent(
        self,
        trace: TraceContext,
        span: SpanBuilder,
        output: dict[str, Any],
        explanation: Explanation | None,
    ) -> tuple[dict[str, Any], Explanation | None]:
        children = trace.children(span.span_id)
        # The same model label the closed agent span will carry (see TraceContext.span).
        model = ", ".join(sorted({c.model for c in children if c.model})) or None
        nested = [fingerprint(c, trace.children) for c in children]
        key = node_key(span.node_id, "agent", model, span.params, span.input, nested)
        rec = self.get(key)
        if rec is not None:
            self.hits["agent"] += 1
            return copy.deepcopy(rec.output), rec.explain()
        self._miss("agent", span.node_id)
        expl = explanation.model_dump(mode="json") if explanation else None
        self._put(key, Recorded(output=jsonable(output), model=model, explanation=expl))
        return output, explanation

    # ------------------------------------------------------------ reporting

    def stats(self) -> dict[str, dict[str, int]]:
        kinds = sorted(set(self.hits) | set(self.misses))
        return {k: {"hits": self.hits[k], "misses": self.misses[k]} for k in kinds}

    def close(self) -> None:
        if self._disk is not None:
            self._disk.close()
