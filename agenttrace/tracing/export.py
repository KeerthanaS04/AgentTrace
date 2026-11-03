"""Export and import runs as JSON files (native) or OpenTelemetry-style OTLP/JSON.

Native format, one run or a bundle:

    {"format": "agenttrace.run",  "version": 1, "run":  {...Run...}}
    {"format": "agenttrace.runs", "version": 1, "runs": [{...Run...}, ...]}

Import re-validates against the strict schemas, so `import(export(run)) == run`, with
identical JSON.

OTLP/JSON (`to_otel` / `from_otel`) is the layout OpenTelemetry collectors and trace
viewers read: resourceSpans -> scopeSpans -> spans, hex trace/span ids, nanosecond
times, typed attributes. LLM spans use the GenAI semantic-convention names
(`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...). Our own exports also carry
each span's full record (`agenttrace.span`) and the run's (`agenttrace.run` on the
resource), so they round-trip exactly; traces from other systems are mapped from the
standard fields (see `from_otel`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span, SpanKind
from agenttrace.tracing.integrity import validate_run

FORMAT_RUN = "agenttrace.run"
FORMAT_RUNS = "agenttrace.runs"
FORMAT_VERSION = 1
SCOPE_NAME = "agenttrace"

_KINDS: tuple[SpanKind, ...] = ("agent", "llm", "tool", "retrieval", "decision", "message")


class TraceFormatError(ValueError):
    """The file is not a trace this importer understands."""


# ---------------------------------------------------------------- native JSON


def run_to_dict(run: Run) -> dict[str, Any]:
    return {"format": FORMAT_RUN, "version": FORMAT_VERSION, "run": run.model_dump(mode="json")}


def runs_to_dict(runs: Iterable[Run]) -> dict[str, Any]:
    return {
        "format": FORMAT_RUNS,
        "version": FORMAT_VERSION,
        "runs": [r.model_dump(mode="json") for r in runs],
    }


def runs_from_dict(data: Mapping[str, Any]) -> list[Run]:
    """Runs from a native document (single or bundle) or an OTLP/JSON document."""
    if "resourceSpans" in data:
        return from_otel(data)
    fmt, version = data.get("format"), data.get("version")
    if fmt not in (FORMAT_RUN, FORMAT_RUNS):
        raise TraceFormatError(f"unknown trace format {fmt!r}")
    if version != FORMAT_VERSION:
        raise TraceFormatError(
            f"{fmt} version {version!r} is not supported (want {FORMAT_VERSION})"
        )
    if fmt == FORMAT_RUN:
        return [Run.model_validate(data["run"])]
    return [Run.model_validate(r) for r in data["runs"]]


def export_runs(runs: Sequence[Run], path: Path | str, *, otel: bool = False) -> Path:
    """Write runs to `path`: one run as `agenttrace.run`, several as a bundle, or OTLP."""
    if otel:
        doc = to_otel(runs)
    elif len(runs) == 1:
        doc = run_to_dict(runs[0])
    else:
        doc = runs_to_dict(runs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def import_runs(path: Path | str) -> list[Run]:
    """Runs from a file; each is schema-validated and integrity-checked (see integrity.py)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceFormatError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise TraceFormatError(f"{path}: expected a JSON object")
    return [validate_run(run) for run in runs_from_dict(data)]


# ---------------------------------------------------------------- OTLP/JSON


def _hex_id(text: str, n_bytes: int) -> str:
    """Deterministic OTel id: the first `n_bytes` of sha256(text), as hex."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: n_bytes * 2]


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _nanos(value: datetime) -> str:
    # Integer arithmetic (no float timestamps), so microseconds survive exactly.
    # OTLP/JSON encodes 64-bit integers as strings.
    delta = value - _EPOCH
    return str((delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000)


def _from_nanos(value: str | int) -> datetime:
    nanos = int(value)
    return datetime.fromtimestamp(nanos // 1_000_000_000, UTC).replace(
        microsecond=(nanos % 1_000_000_000) // 1000
    )


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        typed: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        typed = {"intValue": str(value)}
    elif isinstance(value, float):
        typed = {"doubleValue": value}
    elif isinstance(value, str):
        typed = {"stringValue": value}
    else:
        # Key order is kept (not sorted): records must come back byte-identical.
        typed = {"stringValue": json.dumps(value, ensure_ascii=False)}
    return {"key": key, "value": typed}


def _attr_value(value: Mapping[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return [_attr_value(v) for v in value["arrayValue"].get("values", [])]
    return None


def _attrs(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {a["key"]: _attr_value(a.get("value", {})) for a in items}


def _otel_span(span: Span, trace_id: str) -> dict[str, Any]:
    attributes = [
        _attr("agenttrace.node_id", span.node_id),
        _attr("agenttrace.kind", span.kind),
        _attr("agenttrace.agent", span.agent),
        _attr("agenttrace.span_id", span.span_id),
        _attr("input.value", span.input),
        _attr("output.value", span.output),
    ]
    if span.kind == "llm":
        attributes += [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.usage.input_tokens", span.tokens_in),
            _attr("gen_ai.usage.output_tokens", span.tokens_out),
        ]
    if span.model:
        attributes.append(_attr("gen_ai.request.model", span.model))
    if span.cost_usd:
        attributes.append(_attr("agenttrace.cost_usd", span.cost_usd))
    # The full record, so our own exports round-trip exactly.
    attributes.append(_attr("agenttrace.span", span.model_dump(mode="json")))
    out: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": _hex_id(span.span_id, 8),
        "name": span.node_id,
        "kind": 3 if span.kind == "llm" else 1,  # SPAN_KIND_CLIENT / SPAN_KIND_INTERNAL
        "startTimeUnixNano": _nanos(span.started_at),
        "endTimeUnixNano": _nanos(span.ended_at),
        "attributes": attributes,
        "status": {"code": 2, "message": span.error} if span.error else {"code": 1},
    }
    if span.parent_span_id:
        out["parentSpanId"] = _hex_id(span.parent_span_id, 8)
    return out


def to_otel(runs: Iterable[Run]) -> dict[str, Any]:
    """OTLP/JSON: one resourceSpans entry (one trace) per run."""
    resource_spans = []
    for run in runs:
        trace_id = _hex_id(run.run_id, 16)
        resource_spans.append(
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", "agenttrace"),
                        _attr("agenttrace.run_id", run.run_id),
                        _attr("agenttrace.run", run.model_dump(mode="json", exclude={"spans"})),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": SCOPE_NAME, "version": str(FORMAT_VERSION)},
                        "spans": [_otel_span(s, trace_id) for s in run.spans],
                    }
                ],
            }
        )
    return {"resourceSpans": resource_spans}


def _json_or_wrap(value: Any) -> dict[str, Any]:
    """OTel input/output attributes are often JSON strings; anything else is wrapped."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {} if value is None else {"value": value}


def _guess_kind(attrs: Mapping[str, Any]) -> SpanKind:
    kind = attrs.get("agenttrace.kind") or attrs.get("openinference.span.kind")
    if isinstance(kind, str) and kind.lower() in _KINDS:
        return kind.lower()  # type: ignore[return-value]
    if isinstance(kind, str) and kind.upper() == "RETRIEVER":
        return "retrieval"
    if isinstance(kind, str) and kind.upper() == "CHAIN":
        return "agent"
    if any(k.startswith("gen_ai.") or k.startswith("llm.") for k in attrs):
        return "llm"
    if "tool.name" in attrs:
        return "tool"
    return "agent"


def _foreign_span(raw: Mapping[str, Any], run_id: str, ids: Mapping[str, str]) -> Span:
    """Map a span from another system using the standard OTel / GenAI fields."""
    attrs = _attrs(raw.get("attributes", []))
    status = raw.get("status") or {}
    error = None
    if status.get("code") in (2, "STATUS_CODE_ERROR"):
        error = status.get("message") or "error"
    name = str(raw.get("name") or "span")
    parent = raw.get("parentSpanId")
    return Span(
        span_id=ids[raw["spanId"]],
        run_id=run_id,
        parent_span_id=ids.get(parent) if parent else None,
        node_id=str(attrs.get("agenttrace.node_id") or name),
        agent=str(attrs.get("agenttrace.agent") or name.split(".")[0]),
        kind=_guess_kind(attrs),
        input=_json_or_wrap(attrs.get("input.value")),
        output=_json_or_wrap(attrs.get("output.value")),
        model=attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"),
        started_at=_from_nanos(raw["startTimeUnixNano"]),
        ended_at=_from_nanos(raw["endTimeUnixNano"]),
        tokens_in=int(attrs.get("gen_ai.usage.input_tokens") or 0),
        tokens_out=int(attrs.get("gen_ai.usage.output_tokens") or 0),
        cost_usd=float(attrs.get("agenttrace.cost_usd") or 0.0),
        error=error,
    )


def from_otel(data: Mapping[str, Any]) -> list[Run]:
    """Runs from OTLP/JSON, one per trace id.

    Spans exported by AgentTrace are restored exactly from their `agenttrace.span`
    record. Foreign spans get span_id `<run_id>/<otel span id>`, node_id = span name,
    kind from `openinference.span.kind` / GenAI attributes, and input/output from
    `input.value` / `output.value`. A foreign run's workflow_version is "otel-import".
    """
    traces: dict[str, dict[str, Any]] = {}
    for resource_spans in data.get("resourceSpans", []):
        res_attrs = _attrs((resource_spans.get("resource") or {}).get("attributes", []))
        for scope_spans in resource_spans.get("scopeSpans", []):
            for raw in scope_spans.get("spans", []):
                entry = traces.setdefault(raw["traceId"], {"res": res_attrs, "spans": []})
                entry["spans"].append(raw)
    if not traces:
        raise TraceFormatError("OTLP document has no spans")

    runs: list[Run] = []
    for trace_id, entry in traces.items():
        res, raw_spans = entry["res"], entry["spans"]
        stored = res.get("agenttrace.run")
        run_fields: dict[str, Any] = json.loads(stored) if isinstance(stored, str) else {}
        run_id = str(run_fields.get("run_id") or res.get("agenttrace.run_id") or trace_id)
        ids = {raw["spanId"]: f"{run_id}/{raw['spanId']}" for raw in raw_spans}
        spans: list[Span] = []
        for raw in raw_spans:
            record = _attrs(raw.get("attributes", [])).get("agenttrace.span")
            if isinstance(record, str):
                spans.append(Span.model_validate_json(record))
            else:
                spans.append(_foreign_span(raw, run_id, ids))
        if not run_fields:
            spans.sort(key=lambda s: s.started_at)
            run_fields = {"run_id": run_id, "workflow_version": "otel-import"}
        runs.append(Run.model_validate({**run_fields, "spans": [s.model_dump() for s in spans]}))
    return runs
