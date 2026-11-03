"""Deterministic final-outcome checker: is a run's final output correct for its scenario?

A run is correct when its decision equals the expected decision and the customer reply
states every required fact. No LLM is involved, so `final_correct` is reproducible.

Facts are matched after normalizing both texts: Unicode compatibility forms (NFKC),
every dash variant to "-" (models write "5-7" with U+2011), curly quotes to straight,
"&" to "and", case, and whitespace. A fact must also stand alone: "79" matches "$79" and
"$79.00" but not "$179", "79.50", or order "O1079"; "January 31" matches "January 31st";
"supervisor" matches "supervisors" but not "supervisory".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from agenttrace.schemas.base import StrictModel
from agenttrace.schemas.ground_truth import ExpectedFinal
from agenttrace.schemas.scenario import Decision

# Hyphen, non-breaking hyphen, figure dash, en/em dash, horizontal bar, minus sign,
# small and fullwidth hyphen-minus.
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
_TRANSLATE = str.maketrans(
    {**{d: "-" for d in _DASHES}, "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)


def normalize_text(text: str) -> str:
    """Canonical form for fact matching (see module docstring)."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = re.sub(r"\s*&\s*", " and ", text)
    return " ".join(text.casefold().split())


def _fact_pattern(fact: str) -> re.Pattern[str]:
    norm = normalize_text(fact)
    pattern = re.escape(norm)
    if norm[:1].isalnum():
        # Not glued to a preceding word or number ("$179", "1.79", "O1079").
        pattern = r"(?<!\w)(?<!\d[.,])" + pattern
    if norm[-1:].isdigit():
        # Ordinals ("31st") are fine; more digits or real decimals ("79.50") are not,
        # but zero cents ("$79.00") are.
        pattern += r"(?:st|nd|rd|th)?(?!\w)(?!,\d)(?!\.(?!00(?!\d))\d)"
    elif norm[-1:].isalnum():
        pattern += r"s?(?!\w)"  # whole word, plural allowed ("supervisors")
    return re.compile(pattern)


def has_fact(text: str, fact: str) -> bool:
    """Whether `text` states `fact`, under the normalization above."""
    return _fact_pattern(fact).search(normalize_text(text)) is not None


class OutcomeCheck(StrictModel):
    correct: bool
    decision_ok: bool
    expected_decision: Decision
    # None when the run produced no decision (e.g. a step failed).
    got_decision: str | None
    facts_found: list[str]
    facts_missing: list[str]

    def reason(self) -> str:
        """One line on why the outcome is wrong ("" when correct)."""
        parts = []
        if not self.decision_ok:
            parts.append(f"decision {self.got_decision or 'missing'} != {self.expected_decision}")
        if self.facts_missing:
            parts.append(f"reply lacks {self.facts_missing}")
        return "; ".join(parts)


def check_outcome(expected: ExpectedFinal, final_output: Mapping[str, Any]) -> OutcomeCheck:
    """Grade a workflow's final output (`decision`, `reply`) against the ground truth."""
    got = final_output.get("decision")
    got_decision = got if isinstance(got, str) else None
    reply = final_output.get("reply")
    reply_text = reply if isinstance(reply, str) else ""
    found = [f for f in expected.required_facts if has_fact(reply_text, f)]
    missing = [f for f in expected.required_facts if f not in found]
    decision_ok = got_decision == expected.decision
    return OutcomeCheck(
        correct=decision_ok and not missing,
        decision_ok=decision_ok,
        expected_decision=expected.decision,
        got_decision=got_decision,
        facts_found=found,
        facts_missing=missing,
    )
