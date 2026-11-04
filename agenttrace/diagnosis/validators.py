"""Deterministic validators (P6-T2): hard evidence about spans, without a model call.

    findings = validate_run(run, ValidationContext.default())
    by_span(findings)["<run>/tool_agent.account_status"]   # [ValidatorFinding, ...]

Per span kind (`error` = the output is malformed or contradicts itself or the record;
`warning` = worth a judge's look):

    tool call      tool_error_payload    the reply is an error payload, not a result    error
                   tool_schema           it does not parse as the tool's output model    error
                   tool_record_complete  a field the tool always sends is missing        error
                   tool_consistency      the record contradicts itself (day counts vs
                                         dates, lateness vs dates, totals vs items, fraud
                                         flag vs account status, Plus flag vs Plus date)  error
    search         search_empty          no hits                                         warning
                   hit_not_indexed       a hit the index does not hold (superseded or
                                         unknown document)                               error
                   hit_score_mismatch    a hit's score is not what the index gives it    error
                   hit_irrelevant        a hit shares no term with the query             error
    router         contract checks; order_id_in_ticket (the ticket names another order)
    retriever      contract checks; docs_superseded; docs_from_searches (every doc came
                   from one of its own searches)
    tool_agent     contract checks; tool_passthrough (results passed on unchanged);
                   tool_results_agree (refund quote vs order and account)
    policy         contract checks; decision_vs_tools (approves what refund_api rejected);
                   numbers_supported (justification numbers are in its inputs)  warning
    writer         contract checks; numbers_supported (reply numbers are in its inputs)

Contract checks are the P1 ones (`workflow.contracts`), reported as errors. Validators
see only the trace and the system's own knowledge base and date; no ground truth.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from agenttrace.schemas.diagnosis import ValidatorFinding
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span
from agenttrace.workflow.contracts import CHECKS, CONTRACTS, ORDER_ID_RE
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.tools import TOOLS, load_db

SCORE_TOLERANCE = 1e-6
MONEY_TOLERANCE = 0.005


@dataclass(frozen=True)
class ValidationContext:
    """What validators may consult besides the trace: the KB index and today's date."""

    index: BM25Index
    today: date

    @classmethod
    def default(cls) -> ValidationContext:
        return cls(index=BM25Index.from_kb(load_kb()), today=load_db().as_of)


class _Collector:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.findings: list[ValidatorFinding] = []

    def local(self, span: Span) -> str:
        return span.span_id.removeprefix(f"{self.run.run_id}/")

    def add(self, span: Span, check: str, severity: str, message: str, *evidence: str) -> None:
        self.findings.append(
            ValidatorFinding(
                span_id=span.span_id,
                check=check,
                severity=severity,  # type: ignore[arg-type]
                message=message,
                evidence=list(dict.fromkeys([self.local(span), *evidence])),
            )
        )


# ---------------------------------------------------------------- tool calls


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The pydantic model inside an annotation (`M`, `M | None`, `list[M]`), if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, UnionType, list):
        for arg in get_args(annotation):
            model = _model_of(arg)
            if model is not None:
                return model
    return None


def _missing_fields(model: type[BaseModel], value: Any, path: str = "") -> list[str]:
    """Fields the real tool always serializes (it dumps every field) that are absent."""
    if isinstance(value, list):
        return [m for i, v in enumerate(value) for m in _missing_fields(model, v, f"{path}{i}.")]
    if not isinstance(value, dict):
        return []
    missing = [f"{path}{name}" for name in model.model_fields if name not in value]
    for name, info in model.model_fields.items():
        inner = _model_of(info.annotation)
        if inner is not None and value.get(name) is not None:
            missing += _missing_fields(inner, value[name], f"{path}{name}.")
    return missing


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _order_consistency(order: Mapping[str, Any], today: date) -> list[str]:
    problems = []
    delivered, promised = _date(order.get("delivered_at")), _date(order.get("promised_delivery"))
    placed = _date(order.get("placed_at"))
    if delivered and order.get("days_since_delivery") != (today - delivered).days:
        problems.append(
            f"days_since_delivery {order.get('days_since_delivery')} but delivered_at "
            f"{delivered} is {(today - delivered).days} days ago"
        )
    if delivered and promised and order.get("delivered_late") != (delivered > promised):
        problems.append(
            f"delivered_late {order.get('delivered_late')} but delivered {delivered}, "
            f"promised {promised}"
        )
    if placed and order.get("days_since_placed") != (today - placed).days:
        problems.append(f"days_since_placed {order.get('days_since_placed')} contradicts placed_at")
    items = order.get("items") or []
    prices = [i.get("price") for i in items if isinstance(i, dict)]
    numbers = [float(p) for p in prices if isinstance(p, int | float)]
    subtotal = order.get("subtotal")
    complete = numbers and len(numbers) == len(prices) and isinstance(subtotal, int | float)
    if complete and abs(sum(numbers) - subtotal) > MONEY_TOLERANCE:  # type: ignore[operator]
        problems.append(f"subtotal {subtotal} but items sum to {round(sum(numbers), 2)}")
    return problems


def _account_consistency(acct: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if acct.get("found") is not True:
        return problems
    suspended = acct.get("account_status") == "suspended"
    if acct.get("under_fraud_review") is not None and acct["under_fraud_review"] != suspended:
        problems.append(
            f"under_fraud_review {acct['under_fraud_review']} but account_status "
            f"{acct.get('account_status')!r}"
        )
    member, since = acct.get("plus_member"), acct.get("plus_since")
    if member is False and since is not None:
        problems.append(f"plus_member False but plus_since {since}")
    if member is True and since is None:
        problems.append("plus_member True but no plus_since")
    if isinstance(acct.get("refunds_last_90_days"), int) and acct["refunds_last_90_days"] < 0:
        problems.append("refunds_last_90_days is negative")
    return problems


def _refund_consistency(quote: Mapping[str, Any]) -> list[str]:
    status, reason = quote.get("status"), quote.get("reject_reason")
    if status == "rejected" and reason is None:
        return ["rejected without a reject_reason"]
    if status in ("quoted", "issued") and reason is not None:
        return [f"{status} but reject_reason {reason!r}"]
    req, cap = quote.get("requested_amount"), quote.get("max_refundable")
    numeric = isinstance(req, int | float) and isinstance(cap, int | float)
    if status == "quoted" and numeric and req > cap + MONEY_TOLERANCE:  # type: ignore[operator]
        return [f"quoted {req} above max_refundable {cap}"]
    return []


def check_tool(c: _Collector, span: Span, today: date) -> None:
    name = str(span.input.get("tool", ""))
    spec = TOOLS.get(name)
    out = span.output
    if spec is None:
        c.add(span, "tool_schema", "error", f"unknown tool {name!r}")
        return
    if "error" in out and not set(out) & set(spec.output_model.model_fields):
        c.add(span, "tool_error_payload", "error", f"{name} returned an error: {out['error']}")
        return
    try:
        spec.output_model.model_validate(out)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        c.add(span, "tool_schema", "error", f"{name} output invalid at {where}: {first['msg']}")
    missing = _missing_fields(spec.output_model, out)
    if missing:
        c.add(span, "tool_record_complete", "error", f"{name} reply lacks {missing}")
    if name == "order_lookup" and isinstance(out.get("order"), dict):
        problems = _order_consistency(out["order"], today)
    elif name == "account_status":
        problems = _account_consistency(out)
    elif name == "refund_api":
        problems = _refund_consistency(out)
    else:
        problems = []
    for p in problems:
        c.add(span, "tool_consistency", "error", f"{name}: {p}")


# ---------------------------------------------------------------- searches


def check_search(c: _Collector, span: Span, index: BM25Index) -> None:
    query = str(span.input.get("query", ""))
    hits = span.output.get("hits")
    if not isinstance(hits, list):
        c.add(span, "search_schema", "error", "search output has no hits list")
        return
    if not hits:
        c.add(span, "search_empty", "warning", f"no hits for {query!r}")
    for hit in hits:
        doc_id = str(hit.get("doc_id"))
        if not index.has(doc_id):
            c.add(span, "hit_not_indexed", "error", f"{doc_id} is not in the index", doc_id)
            continue
        actual = index.score(query, doc_id) or 0.0
        if actual <= 0:
            c.add(
                span, "hit_irrelevant", "error", f"{doc_id} shares no term with {query!r}", doc_id
            )
        elif abs(actual - float(hit.get("score", 0))) > SCORE_TOLERANCE:
            c.add(
                span,
                "hit_score_mismatch",
                "error",
                f"{doc_id} recorded score {hit.get('score')}, index gives {actual}",
                doc_id,
            )


# ---------------------------------------------------------------- steps


def _contract(c: _Collector, span: Span) -> None:
    contract = CONTRACTS.get(span.node_id)
    if contract is None:
        return
    for name in contract.programmatic_checks:
        message = CHECKS[name](span.input, span.output)
        if message:
            c.add(span, name, "error", message)


_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_DURATION = re.compile(
    r"\b(\d+)(?:\s?[-‐-―]\s?(\d+))?\s+(?:business\s+)?"
    r"(?:days?|hours?|minutes?|weeks?|months?|years?)\b",
    re.I,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numeric_claims(text: str) -> list[tuple[str, list[float]]]:
    """(claim text, its numbers) for money, percentages, and durations in `text`."""
    claims = []
    for m in _MONEY.finditer(text):
        claims.append((m.group(0), [float(m.group(1).replace(",", ""))]))
    for m in _PERCENT.finditer(text):
        claims.append((m.group(0), [float(m.group(1))]))
    for m in _DURATION.finditer(text):
        nums = [float(g) for g in m.groups() if g]
        claims.append((m.group(0), nums))
    return claims


# Identifiers and dates carry digits that are not figures: drop them before matching.
_NOT_FIGURES = re.compile(
    r"\bKB-[A-Z]{3}-\d{3}\b|\b\d{4}-\d{2}-\d{2}\b|\b[OCT]-?\d{3,4}\b|\bRF-\S+|\bsc-\d+"
)
_DASH = re.compile(r"\s?[\u2010-\u2015-]\s?")


def _source_text(value: Any) -> str:
    text = json.dumps(value, default=str).replace(",", "")
    return _DASH.sub("-", _NOT_FIGURES.sub(" ", text))


def _numbers(text: str) -> set[float]:
    return {round(float(n), 2) for n in _NUMBER.findall(text)}


def _supported(c: _Collector, span: Span, text: str, sources: Any, what: str) -> None:
    """Flag money, percentages, and durations that the step's inputs do not contain. A
    range ("3-5 business days") must appear as that range, not as two loose numbers."""
    source = _source_text(sources)
    known = _numbers(source)
    for claim, nums in numeric_claims(text):
        if len(nums) == 2:
            ok = f"{nums[0]:g}-{nums[1]:g}" in source
        else:
            ok = all(round(n, 2) in known for n in nums)
        if not ok:
            c.add(
                span,
                "numbers_supported",
                "warning",
                f"{what} states {claim!r}, a figure not in its inputs",
            )


def _tool_calls(span: Span) -> list[dict[str, Any]]:
    calls = span.input.get("tool_calls") or span.output.get("tool_calls") or []
    return [x for x in calls if isinstance(x, dict)]


def _result(calls: Iterable[Mapping[str, Any]], tool: str) -> dict[str, Any] | None:
    for call in calls:
        if call.get("tool") == tool and isinstance(call.get("result"), dict):
            return dict(call["result"])
    return None


def check_router(c: _Collector, span: Span) -> None:
    _contract(c, span)
    ticket = span.input.get("ticket") or {}
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}"
    named = re.findall(r"\bO\d{4}\b", text)
    oid = span.output.get("order_id")
    if named and oid not in named:
        c.add(span, "order_id_in_ticket", "error", f"order_id {oid!r} but the ticket names {named}")
    elif oid and not named and ORDER_ID_RE.match(str(oid)):
        c.add(span, "order_id_in_ticket", "warning", f"order_id {oid!r} not named in the ticket")


def check_retriever(c: _Collector, span: Span, run: Run, index: BM25Index) -> None:
    _contract(c, span)
    docs = [d for d in span.output.get("docs") or [] if isinstance(d, dict)]
    for doc in docs:
        doc_id = str(doc.get("doc_id"))
        if str(doc.get("body", "")).lstrip().lower().startswith("superseded") or not index.has(
            doc_id
        ):
            c.add(span, "docs_superseded", "error", f"{doc_id} is superseded or unindexed", doc_id)
    searches = [s for s in run.spans if s.parent_span_id == span.span_id and s.kind == "retrieval"]
    found = {h.get("doc_id") for s in searches for h in s.output.get("hits") or []}
    extra = [str(d.get("doc_id")) for d in docs if d.get("doc_id") not in found]
    if searches and extra:
        c.add(
            span,
            "docs_from_searches",
            "error",
            f"docs {extra} were not returned by any of its searches",
            *[c.local(s) for s in searches],
        )


def check_tool_agent(c: _Collector, span: Span, run: Run) -> None:
    _contract(c, span)
    calls = _tool_calls(span)
    nested = [s for s in run.spans if s.parent_span_id == span.span_id and s.kind == "tool"]
    answered = [x for x in calls if x.get("result") is not None]
    # A call that raised an input error has no tool span; compare only those that ran.
    for call, tool_span in zip(answered, nested, strict=False):
        if call.get("result") != tool_span.output:
            c.add(
                span,
                "tool_passthrough",
                "error",
                f"the {call.get('tool')} result passed on differs from what the tool returned",
                c.local(tool_span),
            )
    refs = [c.local(s) for s in nested]
    for problem in _disagreements(calls):
        c.add(span, "tool_results_agree", "error", problem, *refs)


def _disagreements(calls: list[dict[str, Any]]) -> list[str]:
    """Contradictions between the refund quote and the order / account records."""
    order = (_result(calls, "order_lookup") or {}).get("order")
    quote, acct = _result(calls, "refund_api"), _result(calls, "account_status")
    if not quote:
        return []
    problems = []
    if isinstance(order, dict):
        cap, total, done = (
            quote.get("max_refundable"),
            order.get("total"),
            order.get("already_refunded"),
        )
        if isinstance(cap, int | float) and isinstance(total, int | float):
            refundable = total - (done if isinstance(done, int | float) else 0)
            if abs(cap - refundable) > MONEY_TOLERANCE:
                problems.append(
                    f"refund_api max_refundable {cap} but the order's total minus refunds is "
                    f"{round(refundable, 2)}"
                )
        if (
            quote.get("reject_reason") == "chargeback_open"
            and order.get("chargeback_filed") is False
        ):
            problems.append(
                "refund_api rejects for an open chargeback, order_lookup says none was filed"
            )
    if (
        acct
        and quote.get("reject_reason") == "account_suspended"
        and acct.get("under_fraud_review") is False
    ):
        problems.append("refund_api rejects a suspended account, account_status says no review")
    return problems


def check_policy(c: _Collector, span: Span) -> None:
    _contract(c, span)
    quote = _result(_tool_calls(span), "refund_api")
    if (
        span.output.get("decision") == "approve"
        and quote
        and quote.get("reject_reason") in ("account_suspended", "chargeback_open")
    ):
        c.add(
            span,
            "decision_vs_tools",
            "error",
            f"approves a refund refund_api rejected ({quote['reject_reason']})",
            "tool_agent",
        )
    sources = {**span.input, "refund_amount": span.output.get("refund_amount")}
    _supported(c, span, str(span.output.get("justification", "")), sources, "the justification")


def check_writer(c: _Collector, span: Span) -> None:
    _contract(c, span)
    _supported(c, span, str(span.output.get("reply", "")), span.input, "the reply")


# ---------------------------------------------------------------- the run


STEP_CHECKS = {"router", "retriever", "tool_agent", "policy", "writer"}


def validate_span(run: Run, span: Span, ctx: ValidationContext) -> list[ValidatorFinding]:
    c = _Collector(run)
    if span.kind == "tool":
        check_tool(c, span, ctx.today)
    elif span.kind == "retrieval":
        check_search(c, span, ctx.index)
    elif span.kind == "agent" and not span.error:
        if span.node_id == "router":
            check_router(c, span)
        elif span.node_id == "retriever":
            check_retriever(c, span, run, ctx.index)
        elif span.node_id == "tool_agent":
            check_tool_agent(c, span, run)
        elif span.node_id == "policy":
            check_policy(c, span)
        elif span.node_id == "writer":
            check_writer(c, span)
    return c.findings


def validate_run(run: Run, ctx: ValidationContext | None = None) -> list[ValidatorFinding]:
    """Every finding for every span of `run`, in recorded order. No model call."""
    ctx = ctx or ValidationContext.default()
    return [f for span in run.spans for f in validate_span(run, span, ctx)]


def by_span(findings: Iterable[ValidatorFinding]) -> dict[str, list[ValidatorFinding]]:
    out: dict[str, list[ValidatorFinding]] = {}
    for f in findings:
        out.setdefault(f.span_id, []).append(f)
    return out


def short_circuit(findings: Iterable[ValidatorFinding]) -> bool:
    """True if hard evidence already marks the span wrong, so a judge can be skipped."""
    return any(f.severity == "error" for f in findings)
