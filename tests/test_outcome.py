"""P1-T7: deterministic final-outcome checker, on correct and incorrect outputs.

Reply texts marked "real" are copied from saved clean-ticket runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_llm import make_gateway

from agenttrace.evaluation.outcome import check_outcome, has_fact, normalize_text
from agenttrace.schemas.ground_truth import ExpectedFinal
from agenttrace.workflow import clean_eval
from agenttrace.workflow.agent_config import load_agents_config
from agenttrace.workflow.clean_eval import load_scenarios


def expect(decision: str, *facts: str) -> ExpectedFinal:
    return ExpectedFinal.model_validate({"decision": decision, "required_facts": list(facts)})


# ---------------------------------------------------------------- correct outputs


def test_correct_decision_and_facts() -> None:
    # real (Lightning, clean-02): U+2011 non-breaking hyphen in "5-7"
    reply = (
        "Since the items are opened but in working condition, a 10% restocking fee applies. "
        "Your net refund is $116.10, which will appear in your account within 5\u20117 "
        "business days."
    )
    out = check_outcome(
        expect("approve", "10%", "5-7 business days"), {"decision": "approve", "reply": reply}
    )
    assert out.correct and out.decision_ok
    assert out.facts_found == ["10%", "5-7 business days"] and out.facts_missing == []
    assert out.reason() == ""


def test_no_required_facts_needs_only_the_decision() -> None:
    out = check_outcome(expect("escalate"), {"decision": "escalate", "reply": "Escalated."})
    assert out.correct


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        ("Final Sale", "The item is marked as final sale, which prohibits returns"),  # real
        ("Trust & Safety", "escalating this to our Trust & Safety team"),
        ("Trust & Safety", "handled by our Trust and Safety team"),
        ("5-7 business days", "within 5\u20137 business days"),  # en dash
        ("5-7 business days", "within 5\u20147 business days"),  # em dash
        ("supervisor", "our supervisors will review it"),  # plural
        ("non-refundable", "gift cards are non\u2011refundable in all cases"),
        ("Payments team", "escalate this to our Payments\nteam."),  # line break
        ("79", "You'll receive $79.00 back"),  # real: zero cents
        ("79", "a refund of $79."),  # sentence end
        ("14.99", "We've approved a $14.99 refund"),  # real
        ("January 31", "returns until January 31st"),
        ("January 31", "returns until January 31, 2026"),
        ("140", "Your refund of $140.00 will be processed"),  # real
        ("30 minutes", "will expire after 30\xa0minutes"),  # no-break space
        ("customer's", "the customer\u2019s order"),  # curly apostrophe
    ],
)
def test_fact_found_despite_formatting(fact: str, text: str) -> None:
    assert has_fact(text, fact)


# ---------------------------------------------------------------- incorrect outputs


def test_wrong_decision_is_incorrect_even_with_facts() -> None:
    # real (Lightning, clean-18): approved via a holiday extension that does not apply
    reply = (
        "Since your order was delivered on January 2, 2026, the holiday return extension applies."
    )
    out = check_outcome(expect("deny", "15 days"), {"decision": "approve", "reply": reply})
    assert not out.correct and not out.decision_ok
    assert out.facts_missing == ["15 days"]
    assert out.reason() == "decision approve != deny; reply lacks ['15 days']"


def test_right_decision_with_hallucinated_fact_is_incorrect() -> None:
    # real (Lightning, clean-01): the KB says 5-7 business days
    reply = "A refund of $48.00 will be processed within 5-10 business days."
    out = check_outcome(
        expect("approve", "5-7 business days"), {"decision": "approve", "reply": reply}
    )
    assert out.decision_ok and not out.correct
    assert out.reason() == "reply lacks ['5-7 business days']"


def test_right_decision_wrong_escalation_team_is_incorrect() -> None:
    # real (Ultra, clean-06): fraud review must go to Trust & Safety, not a supervisor
    reply = "we need to escalate this to our Support Supervisor team for further handling."
    out = check_outcome(
        expect("escalate", "Trust & Safety"), {"decision": "escalate", "reply": reply}
    )
    assert out.decision_ok and not out.correct


def test_reply_missing_the_reason_is_incorrect() -> None:
    # real (Lightning, clean-19): approved, but never says why (holiday window to Jan 31)
    reply = "We've approved your refund for order O1017. You'll receive $38.00 back."
    out = check_outcome(expect("approve", "January 31"), {"decision": "approve", "reply": reply})
    assert not out.correct and out.facts_missing == ["January 31"]


def test_failed_run_without_outputs_is_incorrect() -> None:
    out = check_outcome(expect("approve", "140"), {})
    assert not out.correct and out.got_decision is None
    assert out.reason() == "decision missing != approve; reply lacks ['140']"


def test_non_string_outputs_do_not_crash() -> None:
    out = check_outcome(expect("deny", "x"), {"decision": None, "reply": 42})
    assert not out.correct and out.got_decision is None


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        ("79", "a refund of $179.00"),  # part of a larger number
        ("79", "a refund of $79.50"),  # a different amount
        ("79", "order O1079"),  # part of an id
        ("79", "$1,79"),
        ("54", "$54,000"),
        ("10%", "a 110% match"),
        ("140", "$1400"),
        ("30 days", "within 130 days"),
        ("14 days", "14 daysx"),
        ("supervisor", "supervisory"),  # the fact is a whole word
        ("5-7 business days", "5-10 business days"),
        ("5-7 business days", "15-7 business days"),
        ("January 31", "January 3"),
        ("Final Sale", "final"),
    ],
)
def test_fact_not_found_in_lookalikes(fact: str, text: str) -> None:
    assert not has_fact(text, fact)


# ---------------------------------------------------------------- normalization and data


def test_normalize_text() -> None:
    assert normalize_text("  Trust &\xa0Safety\u2014 5\u20117 \u201cOK\u201d ") == (
        'trust and safety- 5-7 "ok"'
    )


def test_clean_eval_reports_final_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner grades each ticket; the fake reply has "5-7 business days" but no "10%"."""
    monkeypatch.setattr(clean_eval, "REPORTS_DIR", tmp_path)
    scenarios = [s for s in load_scenarios() if s.scenario_id in ("clean-01", "clean-02")]
    roles = load_agents_config().profile("default").roles()
    summary = clean_eval.evaluate_profile(
        make_gateway(tmp_path), scenarios, "default", roles, quiet=True
    )
    rows = {r["scenario_id"]: r for r in summary["rows"]}
    assert rows["clean-01"]["correct"] and rows["clean-01"]["final_correct"]
    assert rows["clean-02"]["correct"] and not rows["clean-02"]["final_correct"]
    assert rows["clean-02"]["facts_missing"] == ["10%"]
    assert (summary["correct"], summary["final_correct"]) == (2, 1)
    assert summary["provenance_exact"] == 2  # P2-T2: refs match the declared data flow
    assert (tmp_path / "clean_eval_default.json").exists()


def test_every_clean_ticket_fact_can_be_matched() -> None:
    """Each required fact matches its own text, so no ticket is ungradable."""
    for sc in load_scenarios():
        for fact in sc.ground_truth.expected_final.required_facts:
            assert has_fact(f"Reply: {fact}.", fact), (sc.scenario_id, fact)
