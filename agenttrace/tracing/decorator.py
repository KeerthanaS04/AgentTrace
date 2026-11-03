"""`@traced`: record a function call as a span nested under the current one.

    @traced("tool", name=lambda a: a["name"], record=_record_tool)
    def call(self, name, args): ...

The span's node_id is `<parent node_id>.<name>` (e.g. "tool_agent.order_lookup"), so it
is stable across runs. Without an active `TraceContext` the function runs untouched.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from agenttrace.schemas.span import SpanKind
from agenttrace.tracing.context import SpanBuilder, current_span, current_trace

P = ParamSpec("P")
R = TypeVar("R")

Arguments = dict[str, Any]


def traced(
    kind: SpanKind,
    name: str | Callable[[Arguments], str],
    *,
    capture: Callable[[Arguments], dict[str, Any]] | None = None,
    record: Callable[[SpanBuilder, Any], None] | None = None,
    inject: tuple[Callable[[Any], dict[str, Any]], Callable[[dict[str, Any]], Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Trace calls of the decorated function.

    `name` (or `name(arguments)`) is the node name under the parent span. `capture`
    picks the span input from the call's arguments (default: all except `self`).
    `record` copies the return value into the span (default: `{"result": value}`).
    `inject = (to_json, from_json)` makes the node a fault-injection point: when the
    trace carries a fault plan, the return value is converted to JSON, offered to the
    plan, and (if changed) converted back, before it is recorded and returned.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            trace = current_trace()
            if trace is None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
            label = name(arguments) if callable(name) else name
            parent = current_span()
            node_id = f"{parent.node_id}.{label}" if parent else label
            span_input = capture(arguments) if capture else arguments
            with trace.span(kind, node_id, input=span_input) as span:
                result = fn(*args, **kwargs)
                if inject is not None and trace.faults is not None:
                    to_json, from_json = inject
                    before = to_json(result)
                    after = trace.faults.apply(node_id, kind, span.span_id, before)
                    if after is not before:
                        result = from_json(after)
                if record:
                    record(span, result)
                else:
                    span.output = {"result": result}
                return result

        return wrapper

    return decorate
