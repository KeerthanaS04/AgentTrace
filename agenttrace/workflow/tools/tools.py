"""Mock support tools: order_lookup, account_status, refund_api.

Each tool is a pure function of (FixtureDB, typed input) -> typed output. "Not found"
and business-rule rejections are normal, typed results, not exceptions; only
malformed arguments raise (ToolInputError).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agenttrace.schemas.base import NonEmptyStr, StrictModel
from agenttrace.tracing import SpanBuilder, traced
from agenttrace.workflow.tools.db import (
    AccountState,
    FixtureDB,
    Item,
    OrderStatus,
    PaymentMethod,
    RefundRecord,
    ShippingMethod,
)

# Look-back window for counting a customer's recent refunds (KB-ESC-003).
RECENT_REFUND_DAYS = 90


class ToolInputError(ValueError):
    """The tool was called with arguments that don't match its input schema."""


def business_days_between(start: date, end: date) -> int:
    """Mon-Fri days after `start` up to and including `end` (0 if end <= start)."""
    days = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


# ---------------------------------------------------------------- order_lookup


class OrderLookupInput(StrictModel):
    order_id: NonEmptyStr


class OrderView(StrictModel):
    order_id: str
    customer_id: str
    email: str
    status: OrderStatus
    placed_at: date
    shipped_at: date | None
    delivered_at: date | None
    promised_delivery: date | None
    shipping_method: ShippingMethod
    shipping_fee: float
    payment_method: PaymentMethod
    card_last4: str | None
    chargeback_filed: bool
    items: list[Item]
    subtotal: float
    total: float
    already_refunded: float
    # Computed from the database's as_of date, so agents never do date math.
    days_since_placed: int
    business_days_since_placed: int
    days_since_delivery: int | None
    delivered_late: bool | None


class OrderLookupResult(StrictModel):
    found: bool
    order: OrderView | None = None


def order_lookup(db: FixtureDB, args: OrderLookupInput) -> OrderLookupResult:
    order = db.order(args.order_id)
    if order is None:
        return OrderLookupResult(found=False)
    late = None
    if order.promised_delivery and order.delivered_at:
        late = order.delivered_at > order.promised_delivery
    view = OrderView(
        order_id=order.order_id,
        customer_id=order.customer_id,
        email=order.email,
        status=order.status,
        placed_at=order.placed_at,
        shipped_at=order.shipped_at,
        delivered_at=order.delivered_at,
        promised_delivery=order.promised_delivery,
        shipping_method=order.shipping_method,
        shipping_fee=order.shipping_fee,
        payment_method=order.payment_method,
        card_last4=order.card_last4,
        chargeback_filed=order.chargeback_filed,
        items=order.items,
        subtotal=order.subtotal,
        total=order.total,
        already_refunded=round(sum(r.amount for r in db.refunds_for_order(order.order_id)), 2),
        days_since_placed=(db.as_of - order.placed_at).days,
        business_days_since_placed=business_days_between(order.placed_at, db.as_of),
        days_since_delivery=(db.as_of - order.delivered_at).days if order.delivered_at else None,
        delivered_late=late,
    )
    return OrderLookupResult(found=True, order=view)


# ---------------------------------------------------------------- account_status


class AccountStatusInput(StrictModel):
    customer_id: NonEmptyStr


class AccountStatusResult(StrictModel):
    found: bool
    customer_id: str
    name: str | None = None
    email: str | None = None
    account_status: AccountState | None = None
    under_fraud_review: bool | None = None
    plus_member: bool | None = None
    plus_since: date | None = None
    refunds_last_90_days: int | None = None


def account_status(db: FixtureDB, args: AccountStatusInput) -> AccountStatusResult:
    customer = db.customer(args.customer_id)
    if customer is None:
        return AccountStatusResult(found=False, customer_id=args.customer_id)
    return AccountStatusResult(
        found=True,
        customer_id=customer.customer_id,
        name=customer.name,
        email=customer.email,
        account_status=customer.account_status,
        under_fraud_review=customer.account_status == "suspended",
        plus_member=customer.plus_member,
        plus_since=customer.plus_since,
        refunds_last_90_days=len(db.refunds_for_customer(customer.customer_id, RECENT_REFUND_DAYS)),
    )


# ---------------------------------------------------------------- refund_api

RefundRejection = Literal[
    "order_not_found",
    "account_suspended",
    "chargeback_open",
    "amount_exceeds_refundable",
]


class RefundApiInput(StrictModel):
    order_id: NonEmptyStr
    amount: float = Field(gt=0)
    # "quote" checks eligibility and limits without moving money; "issue" records the refund.
    mode: Literal["quote", "issue"] = "quote"
    reason: NonEmptyStr


class RefundApiResult(StrictModel):
    status: Literal["quoted", "issued", "rejected"]
    order_id: str
    requested_amount: float
    max_refundable: float | None = None
    already_refunded: float | None = None
    payment_method: PaymentMethod | None = None
    refund_id: str | None = None
    reject_reason: RefundRejection | None = None


def refund_api(db: FixtureDB, args: RefundApiInput) -> RefundApiResult:
    amount = round(args.amount, 2)
    base: dict[str, Any] = {"order_id": args.order_id, "requested_amount": amount}
    order = db.order(args.order_id)
    if order is None:
        return RefundApiResult(status="rejected", reject_reason="order_not_found", **base)

    already = round(sum(r.amount for r in db.refunds_for_order(order.order_id)), 2)
    base |= {
        "max_refundable": round(order.total - already, 2),
        "already_refunded": already,
        "payment_method": order.payment_method,
    }
    customer = db.customer(order.customer_id)
    if customer is not None and customer.account_status == "suspended":
        return RefundApiResult(status="rejected", reject_reason="account_suspended", **base)
    if order.chargeback_filed:
        return RefundApiResult(status="rejected", reject_reason="chargeback_open", **base)
    if amount > base["max_refundable"]:
        return RefundApiResult(status="rejected", reject_reason="amount_exceeds_refundable", **base)
    if args.mode == "quote":
        return RefundApiResult(status="quoted", **base)

    refund_id = f"RF-{order.order_id}-{len(db.refunds_for_order(order.order_id)) + 1}"
    db.refunds.append(
        RefundRecord(
            refund_id=refund_id, order_id=order.order_id, amount=amount, issued_at=db.as_of
        )
    )
    return RefundApiResult(status="issued", refund_id=refund_id, **base)


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[StrictModel]
    output_model: type[StrictModel]
    fn: Callable[[FixtureDB, Any], StrictModel]

    def schema_for_llm(self) -> dict[str, Any]:
        """Name, description, and input JSON Schema, for the Tool Agent's prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "order_lookup",
            "Look up an order by ID: status, dates, items, totals, prior refunds, and "
            "precomputed day counts (days since delivery, business days since placed).",
            OrderLookupInput,
            OrderLookupResult,
            order_lookup,
        ),
        ToolSpec(
            "account_status",
            "Look up a customer account: status (active / locked / suspended), fraud "
            "review flag, Plus membership, and number of refunds in the last 90 days.",
            AccountStatusInput,
            AccountStatusResult,
            account_status,
        ),
        ToolSpec(
            "refund_api",
            "Check (mode='quote') or issue (mode='issue') a refund for an order. Rejects "
            "suspended accounts, open chargebacks, and amounts above what is refundable.",
            RefundApiInput,
            RefundApiResult,
            refund_api,
        ),
    )
}


def _record_tool(span: SpanBuilder, result: StrictModel) -> None:
    span.output = result.model_dump(mode="json")


class ToolBox:
    """Tools bound to one fixture database. Works on a private copy, so `issue` calls
    never leak into the source fixtures or into other runs."""

    def __init__(self, db: FixtureDB) -> None:
        self.db = db.model_copy(deep=True)

    @traced(
        "tool",
        name=lambda a: str(a["name"]),
        capture=lambda a: {"tool": a["name"], "args": a["args"]},
        record=_record_tool,
    )
    def call(self, name: str, args: dict[str, Any]) -> StrictModel:
        spec = TOOLS.get(name)
        if spec is None:
            raise ToolInputError(f"unknown tool {name!r}; available: {sorted(TOOLS)}")
        try:
            parsed: BaseModel = spec.input_model.model_validate(args)
        except ValidationError as exc:
            raise ToolInputError(f"{name}: invalid arguments: {exc}") from exc
        result = spec.fn(self.db, parsed)
        # Re-validate the output so a buggy tool can't return off-schema data.
        return spec.output_model.model_validate(result.model_dump())
