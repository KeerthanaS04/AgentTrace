from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agenttrace.workflow.kb.loader import DEFAULT_KB_DIR, KnowledgeBase, load_kb
from agenttrace.workflow.kb.models import KBDoc, KBFact, normalize_ws


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return load_kb()


# ---------------------------------------------------------------- Done-when for P1-T1


def test_kb_size_in_range(kb: KnowledgeBase) -> None:
    assert 30 <= len(kb) <= 50


def test_every_doc_has_id_title_body_and_checkable_fact(kb: KnowledgeBase) -> None:
    for doc in kb:
        assert doc.doc_id and doc.title and doc.body.strip()
        assert doc.facts, doc.doc_id
        for fact in doc.facts:
            assert normalize_ws(fact.text) in normalize_ws(doc.body), f"{doc.doc_id}.{fact.name}"


def test_every_topic_is_covered(kb: KnowledgeBase) -> None:
    counts = Counter(d.topic for d in kb.active())
    for topic in ("refund", "shipping", "warranty", "account", "escalation"):
        assert counts[topic] >= 4, f"only {counts[topic]} active {topic} docs"


def test_superseded_docs_point_to_active_replacements(kb: KnowledgeBase) -> None:
    superseded = [d for d in kb if d.status == "superseded"]
    assert len(superseded) >= 3, "need stale docs for F1 (wrong/stale retrieval) faults"
    for doc in superseded:
        assert doc.superseded_by is not None
        replacement = kb.get(doc.superseded_by)
        assert replacement.status == "active"
        # A stale doc must disagree with its replacement on at least one shared fact,
        # otherwise retrieving it would not produce a detectably wrong answer.
        shared = {f.name for f in doc.facts} & {f.name for f in replacement.facts}
        assert shared, f"{doc.doc_id} shares no fact names with {replacement.doc_id}"
        assert any(doc.fact(n).value != replacement.fact(n).value for n in shared)


def test_cross_references_resolve(kb: KnowledgeBase) -> None:
    import re

    for doc in kb:
        for ref in re.findall(r"KB-[A-Z]{3}-\d{3}", doc.body):
            assert ref in kb, f"{doc.doc_id} references missing {ref}"


# ---------------------------------------------------------------- policy consistency


def test_escalation_threshold_matches_agent_refund_limit(kb: KnowledgeBase) -> None:
    limit = kb.get("KB-REF-005").fact("agent_approval_limit_usd").value
    assert kb.get("KB-ESC-001").fact("escalate_refund_over_usd").value == limit


def test_plus_windows_extend_standard_windows(kb: KnowledgeBase) -> None:
    std = kb.get("KB-REF-001").fact("return_window_days").value
    elec = kb.get("KB-REF-002").fact("electronics_return_window_days").value
    plus = kb.get("KB-REF-011")
    assert isinstance(std, int) and isinstance(elec, int)
    assert plus.fact("plus_return_window_days").value == 2 * std
    assert plus.fact("plus_electronics_window_days").value == 2 * elec


def test_confusable_refund_windows_differ(kb: KnowledgeBase) -> None:
    # Standard, electronics, Plus, and stale windows must all differ, so retrieving
    # the wrong one changes the decision for some order dates.
    windows = {
        kb.get("KB-REF-001").fact("return_window_days").value,
        kb.get("KB-REF-002").fact("electronics_return_window_days").value,
        kb.get("KB-REF-011").fact("plus_return_window_days").value,
        kb.get("KB-REF-010").fact("return_window_days").value,
    }
    assert len(windows) == 4


def test_fact_names_unique_per_active_topic(kb: KnowledgeBase) -> None:
    # Among active docs, a fact name means one thing. Stale docs may reuse names on purpose.
    seen: dict[str, str] = {}
    for doc in kb.active():
        for fact in doc.facts:
            assert fact.name not in seen, f"{fact.name} in {seen.get(fact.name)} and {doc.doc_id}"
            seen[fact.name] = doc.doc_id


# ---------------------------------------------------------------- validation


def _doc(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "doc_id": "KB-REF-099",
        "title": "T",
        "topic": "refund",
        "effective_date": "2025-01-01",
        "body": "Returns are accepted within 30 days.",
        "facts": [{"name": "window", "value": 30, "text": "30 days"}],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"doc_id": "REF-1"}, "must look like"),
        ({"doc_id": "KB-SHP-099"}, "prefix does not match"),
        ({"facts": []}, "at least 1"),
        ({"facts": [{"name": "w", "value": 30, "text": "45 days"}]}, "not in body"),
        ({"facts": [{"name": "Window", "value": 30, "text": "30 days"}]}, "snake_case"),
        ({"status": "superseded"}, "superseded_by"),
        ({"superseded_by": "KB-REF-001"}, "superseded_by"),
        (
            {
                "facts": [
                    {"name": "w", "value": 30, "text": "30 days"},
                    {"name": "w", "value": 30, "text": "30 days"},
                ]
            },
            "duplicate fact",
        ),
    ],
)
def test_invalid_docs_rejected(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        KBDoc.model_validate(_doc(**overrides))


def test_duplicate_doc_ids_rejected() -> None:
    doc = KBDoc.model_validate(_doc())
    with pytest.raises(ValueError, match="duplicate doc_ids"):
        KnowledgeBase([doc, doc])


def test_dangling_superseded_by_rejected() -> None:
    stale = KBDoc.model_validate(_doc(status="superseded", superseded_by="KB-REF-098"))
    with pytest.raises(ValueError, match="does not exist"):
        KnowledgeBase([stale])


def test_loader_reports_file_and_entry(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump([_doc(), _doc(doc_id="nope")]))
    with pytest.raises(ValueError, match=r"bad.yaml entry 1"):
        load_kb(tmp_path)


def test_fact_text_may_wrap_across_lines() -> None:
    doc = KBDoc.model_validate(_doc(body="Returns are accepted within 30\n  days of delivery."))
    assert doc.fact("window").value == 30


def test_fact_lookup(kb: KnowledgeBase) -> None:
    fact = kb.get("KB-REF-001").fact("return_window_days")
    assert isinstance(fact, KBFact) and fact.value == 30
    with pytest.raises(KeyError):
        kb.get("KB-REF-001").fact("nope")
    with pytest.raises(KeyError):
        kb.get("KB-XXX-000")


def test_default_kb_dir_points_at_repo_data() -> None:
    assert DEFAULT_KB_DIR.name == "kb" and DEFAULT_KB_DIR.parent.name == "data"
