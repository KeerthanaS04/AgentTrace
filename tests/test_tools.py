from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.tools import TOOLS, FixtureDB, ToolBox, ToolInputError, load_db
from agenttrace.workflow.tools.db import DEFAULT_DB_PATH
from agenttrace.workflow.tools.tools import (
    AccountStatusResult,
    OrderLookupResult,
    RefundApiResult,
    business_days_between,
)


@pytest.fixture
def db() -> FixtureDB:
    return load_db()


@pytest.fixture
def tools(db: FixtureDB) -> ToolBox:
    return ToolBox(db)


def lookup(tools: ToolBox, order_id: str) -> OrderLookupResult:
    result = tools.call("order_lookup", {"order_id": order_id})
    assert isinstance(result, OrderLookupResult)
    return result


def account(tools: ToolBox, customer_id: str) -> AccountStatusResult:
    result = tools.call("account_status", {"customer_id": customer_id})
    assert isinstance(result, AccountStatusResult)
    return result


def refund(tools: ToolBox, **args: Any) -> RefundApiResult:
    result = tools.call("refund_api", {"reason": "test", **args})
    assert isinstance(result, RefundApiResult)
    return result


# ---------------------------------------------------------------- fixture DB


def test_default_db_loads_and_matches_kb_shipping_rules(db: FixtureDB) -> None:
    kb = load_kb()
    threshold = kb.get("KB-SHP-001").fact("free_shipping_threshold_usd").value
    std_fee = kb.get("KB-SHP-001").fact("standard_shipping_fee_usd").value
    express_fee = kb.get("KB-SHP-002").fact("express_fee_usd").value
    for order in db.orders:
        if order.shipping_method == "standard":
            expected = 0.0 if order.subtotal >= threshold else std_fee  # type: ignore[operator]
            assert order.shipping_fee == expected, order.order_id
        elif order.shipping_method == "express":
            assert order.shipping_fee == express_fee, order.order_id


def test_default_db_covers_policy_paths(db: FixtureDB) -> None:
    statuses = {o.status for o in db.orders}
    assert {"processing", "shipped", "delivered"} <= statuses
    categories = {i.category for o in db.orders for i in o.items}
    assert {"electronics", "home", "apparel", "gift_card"} <= categories
    assert any(o.chargeback_filed for o in db.orders)
    assert any(i.final_sale for o in db.orders for i in o.items)
    assert {c.account_status for c in db.customers} == {"active", "locked", "suspended"}
    assert any(c.plus_member for c in db.customers)


def _db(**overrides: Any) -> dict[str, Any]:
    data = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d["orders"].append(dict(d["orders"][0])), "duplicate order"),
        (lambda d: d["orders"][0].update(customer_id="C999"), "unknown customer"),
        (lambda d: d["orders"][0].update(email="x@example.com"), "email does not match"),
        (lambda d: d.update(as_of="2025-01-01"), "placed after as_of"),
        (lambda d: d["orders"][0].update(delivered_at="2025-12-29"), "before shipping"),
        (lambda d: d["orders"][0].update(status="processing"), "inconsistent with status"),
        (lambda d: d["orders"][0].update(card_last4=None), "card_last4"),
        (lambda d: d["refunds"][0].update(order_id="O9999"), "unknown order"),
        (lambda d: d["refunds"][0].update(amount=999.0), "exceed order total"),
        (lambda d: d["customers"][0].update(plus_member=True), "plus_since"),
    ],
)
def test_invalid_db_rejected(mutate: Any, match: str) -> None:
    data = _db()
    mutate(data)
    with pytest.raises(ValidationError, match=match):
        FixtureDB.model_validate(data)


# ---------------------------------------------------------------- order_lookup


def test_order_lookup_found_with_computed_days(tools: ToolBox) -> None:
    order = lookup(tools, "O1001").order
    assert order is not None
    assert order.days_since_delivery == 18  # 2026-01-02 -> 2026-01-20
    assert order.total == 48.00
    assert order.already_refunded == 0.0
    assert order.card_last4 == "4417"


def test_order_lookup_not_found(tools: ToolBox) -> None:
    result = lookup(tools, "O9999")
    assert not result.found and result.order is None


def test_order_lookup_unshipped_order(tools: ToolBox) -> None:
    order = lookup(tools, "O1015").order
    assert order is not None and order.status == "processing"
    assert order.days_since_delivery is None
    # Placed Thu 2026-01-08; business days through Tue 2026-01-20 = 8 (> the 5-day policy)
    assert order.business_days_since_placed == 8


def test_order_lookup_late_express(tools: ToolBox) -> None:
    order = lookup(tools, "O1013").order
    assert order is not None and order.delivered_late is True
    assert order.total == pytest.approx(89.00 + 14.99)
    on_time = lookup(tools, "O1001").order
    assert on_time is not None and on_time.delivered_late is None  # no promise recorded


def test_order_lookup_reports_prior_refunds(tools: ToolBox) -> None:
    order = lookup(tools, "O0904").order
    assert order is not None and order.already_refunded == 40.00


# ---------------------------------------------------------------- account_status


def test_account_status_plus_member(tools: ToolBox) -> None:
    acct = account(tools, "C002")
    assert acct.found and acct.plus_member and acct.plus_since == date(2025, 6, 1)
    assert acct.under_fraud_review is False


def test_account_status_suspended_is_fraud_review(tools: ToolBox) -> None:
    acct = account(tools, "C004")
    assert acct.account_status == "suspended" and acct.under_fraud_review is True


def test_account_status_counts_recent_refunds(tools: ToolBox) -> None:
    # C006 has 4 refunds in the last 90 days: more than 3 triggers KB-ESC-003.
    assert account(tools, "C006").refunds_last_90_days == 4
    assert account(tools, "C001").refunds_last_90_days == 0


def test_account_status_not_found(tools: ToolBox) -> None:
    acct = account(tools, "C999")
    assert not acct.found and acct.customer_id == "C999" and acct.account_status is None


# ---------------------------------------------------------------- refund_api


def test_refund_quote_does_not_move_money(tools: ToolBox) -> None:
    quote = refund(tools, order_id="O1001", amount=48.00, mode="quote")
    assert quote.status == "quoted" and quote.max_refundable == 48.00
    assert quote.refund_id is None
    assert lookup(tools, "O1001").order.already_refunded == 0.0  # type: ignore[union-attr]


def test_refund_issue_records_and_limits_next_refund(tools: ToolBox) -> None:
    issued = refund(tools, order_id="O1001", amount=30.00, mode="issue")
    assert issued.status == "issued" and issued.refund_id == "RF-O1001-1"
    assert lookup(tools, "O1001").order.already_refunded == 30.00  # type: ignore[union-attr]
    second = refund(tools, order_id="O1001", amount=20.00, mode="quote")
    assert second.status == "rejected" and second.reject_reason == "amount_exceeds_refundable"
    assert second.max_refundable == 18.00


@pytest.mark.parametrize(
    "order_id, amount, reason",
    [
        ("O9999", 10.0, "order_not_found"),
        ("O1006", 249.0, "account_suspended"),
        ("O1008", 450.0, "chargeback_open"),
        ("O1004", 500.0, "amount_exceeds_refundable"),
    ],
)
def test_refund_rejections(tools: ToolBox, order_id: str, amount: float, reason: str) -> None:
    result = refund(tools, order_id=order_id, amount=amount)
    assert result.status == "rejected" and result.reject_reason == reason


def test_toolbox_works_on_a_copy(db: FixtureDB) -> None:
    ToolBox(db).call(
        "refund_api", {"order_id": "O1001", "amount": 10, "mode": "issue", "reason": "x"}
    )
    assert db.refunds_for_order("O1001") == []
    assert ToolBox(db).db.refunds_for_order("O1001") == []


# ---------------------------------------------------------------- strict I/O


@pytest.mark.parametrize(
    "name, args",
    [
        ("order_lookup", {}),
        ("order_lookup", {"order_id": ""}),
        ("order_lookup", {"order_id": "O1001", "extra": 1}),
        ("account_status", {"customer_id": 42}),
        ("refund_api", {"order_id": "O1001", "amount": -5, "reason": "x"}),
        ("refund_api", {"order_id": "O1001", "amount": 5, "mode": "steal", "reason": "x"}),
        ("refund_api", {"order_id": "O1001", "amount": 5}),
        ("delete_everything", {}),
    ],
)
def test_invalid_tool_calls_raise(tools: ToolBox, name: str, args: dict[str, Any]) -> None:
    with pytest.raises(ToolInputError):
        tools.call(name, args)


def test_tools_are_deterministic(db: FixtureDB) -> None:
    calls = [
        ("order_lookup", {"order_id": "O1013"}),
        ("account_status", {"customer_id": "C006"}),
        ("refund_api", {"order_id": "O1002", "amount": 129, "reason": "x"}),
    ]
    first = [ToolBox(db).call(n, a).model_dump_json() for n, a in calls]
    again = [ToolBox(db).call(n, a).model_dump_json() for n, a in calls]
    assert first == again


def test_every_tool_has_llm_schema() -> None:
    assert set(TOOLS) == {"order_lookup", "account_status", "refund_api"}
    for spec in TOOLS.values():
        schema = spec.schema_for_llm()
        assert schema["name"] == spec.name and schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"].get("additionalProperties") is False


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (date(2026, 1, 8), date(2026, 1, 20), 8),  # Thu -> Tue, spans a weekend
        (date(2026, 1, 16), date(2026, 1, 19), 1),  # Fri -> Mon
        (date(2026, 1, 20), date(2026, 1, 20), 0),
        (date(2026, 1, 20), date(2026, 1, 10), 0),
    ],
)
def test_business_days(start: date, end: date, expected: int) -> None:
    assert business_days_between(start, end) == expected
