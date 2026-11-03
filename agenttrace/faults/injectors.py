"""Fault injectors and the fault catalog (P4-T1).

An injector is a pure function `(value, params) -> new value` over a node's output as
JSON: a search's `{"hits": [...]}`, a tool's result dict, or an agent's output dict. It
never mutates its input. Each injector declares which span kinds it may target, so a
fault aimed at the wrong kind of node fails loudly instead of doing nothing.

Fault codes (see FAULT_CATALOG):
    F0 control: no fault                   F5 invalid intermediate decision
    F1 wrong / stale retrieval             F6 hallucinated intermediate claim
    F2 empty or irrelevant retrieval       F7 multi-fault (two roots in one run)
    F3 wrong value in a tool response      F8 silent fault (final outcome unchanged)
    F4 malformed / error tool response
F7 and F8 are scenario-level: F7 is two injections in one scenario, F8 any injection
whose effect never reaches the final outcome (e.g. a tool field nothing reads).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agenttrace.schemas.ground_truth import FaultCode
from agenttrace.schemas.span import SpanKind
from agenttrace.tracing.context import InjectionError

Value = dict[str, Any]
InjectorFn = Callable[[Value, Mapping[str, Any]], Value]

# Reserved parameter: which occurrence of the target node to hit (1-based); default all.
OCCURRENCE = "occurrence"


class FaultParamError(InjectionError, ValueError):
    """An injector was given parameters it cannot apply (bad path, missing field, ...)."""


@dataclass(frozen=True)
class Injector:
    name: str
    kinds: tuple[SpanKind, ...]
    description: str
    fn: InjectorFn
    required: tuple[str, ...] = ()

    def __call__(self, value: Value, params: Mapping[str, Any]) -> Value:
        missing = [p for p in self.required if p not in params]
        if missing:
            raise FaultParamError(f"{self.name}: missing params {missing}")
        return self.fn(copy.deepcopy(value), params)


# ---------------------------------------------------------------- helpers


def _step(node: Any, key: str, path: str) -> Any:
    """One step of a dotted path: a dict key, or a list index ("items.0")."""
    if isinstance(node, dict) and key in node:
        return node[key]
    if isinstance(node, list) and key.isdigit() and int(key) < len(node):
        return node[int(key)]
    raise FaultParamError(f"path {path!r}: no key {key!r}")


def _walk(value: Value, path: str) -> tuple[dict[str, Any], str]:
    """The dict holding the last key of a dotted path, and that key.

    Intermediate steps may index lists, e.g. "order.items.0.final_sale".
    """
    *parents, last = path.split(".")
    node: Any = value
    for key in parents:
        node = _step(node, key, path)
    if not isinstance(node, dict) or last not in node:
        raise FaultParamError(f"path {path!r}: no key {last!r}")
    return node, last


def _hits(value: Value) -> list[dict[str, Any]]:
    hits = value.get("hits")
    if not isinstance(hits, list):
        raise FaultParamError("retrieval output has no hits list")
    return hits


# ---------------------------------------------------------------- retrieval (F1, F2)


def _retrieval_wrong_doc(value: Value, params: Mapping[str, Any]) -> Value:
    """Swap one document for a plausible wrong one (e.g. its superseded version)."""
    hits = _hits(value)
    target = params.get("replace")
    for hit in hits:
        if target is None or hit["doc_id"] == target:
            hit["doc_id"] = params["with"]
            hit["title"] = params.get("title", hit["title"])
            if target is None:
                break  # no target named: replace the top hit only
    return value


def _retrieval_empty(value: Value, params: Mapping[str, Any]) -> Value:
    value["hits"] = []
    return value


def _retrieval_irrelevant(value: Value, params: Mapping[str, Any]) -> Value:
    _hits(value)
    value["hits"] = [
        {"doc_id": doc_id, "score": 1.0, "title": params.get("titles", {}).get(doc_id, doc_id)}
        for doc_id in params["docs"]
    ]
    return value


# ---------------------------------------------------------------- tools (F3, F4)


def _tool_wrong_value(value: Value, params: Mapping[str, Any]) -> Value:
    """Set one field of a tool's result, e.g. order.status or refunds_last_90_days.

    `values` ({path: value}) sets several fields of the same record consistently, so the
    wrong fact is not contradicted by a sibling field (a late delivery rewritten as on
    time must also move `delivered_at`, or the dates still give it away).
    """
    if "values" in params:
        changes = dict(params["values"])
        if not changes:
            raise FaultParamError("tool_wrong_value: values is empty")
    elif "path" in params and "value" in params:
        changes = {params["path"]: params["value"]}
    else:
        raise FaultParamError("tool_wrong_value: needs path and value, or values")
    for path, new in changes.items():
        holder, key = _walk(value, path)
        holder[key] = new
    return value


def _tool_malformed(value: Value, params: Mapping[str, Any]) -> Value:
    """Break the tool's response: an error payload, dropped fields, or a wrong type."""
    mode = params["mode"]
    if mode == "error":
        return {"error": params.get("message", "upstream timeout"), "status": 504}
    if mode == "missing_fields":
        for path in params["fields"]:
            holder, key = _walk(value, path)
            del holder[key]
        return value
    if mode == "wrong_type":
        holder, key = _walk(value, params["path"])
        holder[key] = params.get("value", "N/A")
        return value
    raise FaultParamError(f"tool_malformed: unknown mode {mode!r}")


# ---------------------------------------------------------------- agent outputs (F5, F6)


def _set_field(value: Value, params: Mapping[str, Any]) -> Value:
    """Overwrite one output field, e.g. policy.decision or router.category."""
    field = params["field"]
    if field not in value:
        raise FaultParamError(f"set_field: output has no field {field!r}")
    value[field] = params["value"]
    return value


def _hallucinated_claim(value: Value, params: Mapping[str, Any]) -> Value:
    """Add a fabricated fact: appended to a text field, or added to a list field.

    `mode="replace"` makes the claim the whole text instead: a fabricated rationale. An
    appended claim sits next to the agent's correct reasoning, and readers downstream
    follow the reasoning (P4-T4 measured this); a replaced one has nothing contradicting it.
    """
    field = params["field"]
    mode = params.get("mode", "append")
    if mode not in ("append", "replace"):
        raise FaultParamError(f"hallucinated_claim: unknown mode {mode!r}")
    if field not in value:
        raise FaultParamError(f"hallucinated_claim: output has no field {field!r}")
    current = value[field]
    if isinstance(current, list) and mode == "append":
        value[field] = [*current, params["claim"]]
    elif isinstance(current, str):
        appended = f"{current.rstrip()} {params['claim']}".strip()
        value[field] = params["claim"] if mode == "replace" else appended
    else:
        raise FaultParamError(f"hallucinated_claim: {field!r} cannot take a {mode} claim")
    return value


INJECTORS: dict[str, Injector] = {
    i.name: i
    for i in (
        Injector(
            "retrieval_wrong_doc",
            ("retrieval",),
            "Search returns a plausible but wrong (e.g. superseded) document.",
            _retrieval_wrong_doc,
            required=("with",),
        ),
        Injector(
            "retrieval_empty",
            ("retrieval",),
            "Search returns nothing.",
            _retrieval_empty,
        ),
        Injector(
            "retrieval_irrelevant",
            ("retrieval",),
            "Search returns only unrelated documents.",
            _retrieval_irrelevant,
            required=("docs",),
        ),
        Injector(
            "tool_wrong_value",
            ("tool",),
            "A tool returns a wrong value in one field (or a consistent set of fields).",
            _tool_wrong_value,
        ),
        Injector(
            "tool_malformed",
            ("tool",),
            "A tool returns an error payload, drops fields, or has a wrong type.",
            _tool_malformed,
            required=("mode",),
        ),
        Injector(
            "set_field",
            ("agent",),
            "An agent outputs a wrong value (e.g. wrong decision or routing).",
            _set_field,
            required=("field", "value"),
        ),
        Injector(
            "hallucinated_claim",
            ("agent",),
            "An agent's output includes a fabricated fact (added, or replacing the text).",
            _hallucinated_claim,
            required=("field", "claim"),
        ),
    )
}


@dataclass(frozen=True)
class FaultClass:
    code: FaultCode
    name: str
    injectors: tuple[str, ...]  # usual injectors; empty for F0, any for F7 / F8


FAULT_CATALOG: dict[FaultCode, FaultClass] = {
    c.code: c
    for c in (
        FaultClass("F0", "control (no fault)", ()),
        FaultClass("F1", "wrong / stale retrieval", ("retrieval_wrong_doc",)),
        FaultClass(
            "F2", "empty or irrelevant retrieval", ("retrieval_empty", "retrieval_irrelevant")
        ),
        FaultClass("F3", "wrong value in a tool response", ("tool_wrong_value",)),
        FaultClass("F4", "malformed / error tool response", ("tool_malformed",)),
        FaultClass("F5", "invalid intermediate decision", ("set_field",)),
        FaultClass("F6", "hallucinated intermediate claim", ("hallucinated_claim",)),
        FaultClass("F7", "multi-fault: two independent roots", tuple(INJECTORS)),
        FaultClass("F8", "silent fault: final outcome unchanged", tuple(INJECTORS)),
    )
}
