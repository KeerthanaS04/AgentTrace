"""Knowledge-base document model.

Each document carries machine-checkable facts. A fact's `text` appears verbatim in
the body, so a correct customer reply can be checked deterministically (P1-T7) and
a wrong retrieval produces a detectably wrong answer.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from agenttrace.schemas.base import NonEmptyStr, StrictModel

KBTopic = Literal["refund", "shipping", "warranty", "account", "escalation", "general"]

# KB-<topic prefix>-<3 digits>, e.g. KB-REF-001
TOPIC_PREFIX: dict[str, str] = {
    "refund": "REF",
    "shipping": "SHP",
    "warranty": "WAR",
    "account": "ACC",
    "escalation": "ESC",
    "general": "GEN",
}
DOC_ID_RE = re.compile(r"^KB-([A-Z]{3})-\d{3}$")
FACT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces (bodies wrap across lines)."""
    return " ".join(text.split())


class KBFact(StrictModel):
    # snake_case key, e.g. "refund_window_days"
    name: NonEmptyStr
    value: str | int | float | bool
    # Exact phrase from the body that states this fact, e.g. "30 days" (whitespace-insensitive).
    text: NonEmptyStr

    @model_validator(mode="after")
    def _check_name(self) -> KBFact:
        if not FACT_NAME_RE.match(self.name):
            raise ValueError(f"fact name {self.name!r} must be snake_case")
        return self


class KBDoc(StrictModel):
    doc_id: NonEmptyStr
    title: NonEmptyStr
    topic: KBTopic
    status: Literal["active", "superseded"] = "active"
    superseded_by: str | None = None
    effective_date: date
    body: NonEmptyStr
    facts: list[KBFact] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> KBDoc:
        m = DOC_ID_RE.match(self.doc_id)
        if not m:
            raise ValueError(f"doc_id {self.doc_id!r} must look like KB-REF-001")
        if m.group(1) != TOPIC_PREFIX[self.topic]:
            raise ValueError(f"{self.doc_id}: prefix does not match topic {self.topic!r}")
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError(f"{self.doc_id}: superseded_by is required iff status is superseded")
        names = [f.name for f in self.facts]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.doc_id}: duplicate fact names")
        flat_body = normalize_ws(self.body)
        for fact in self.facts:
            if normalize_ws(fact.text) not in flat_body:
                raise ValueError(
                    f"{self.doc_id}: fact {fact.name!r} text {fact.text!r} not in body"
                )
        return self

    def fact(self, name: str) -> KBFact:
        for f in self.facts:
            if f.name == name:
                return f
        raise KeyError(f"{self.doc_id} has no fact {name!r}")
