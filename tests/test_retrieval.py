from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agenttrace.workflow.kb.index import BM25Index, expand_query, recall_at_k, stem, tokenize
from agenttrace.workflow.kb.loader import KnowledgeBase, load_kb
from agenttrace.workflow.kb.models import KBDoc

EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"


def _queries(name: str) -> list[tuple[str, list[str]]]:
    raw = yaml.safe_load((EVAL_DIR / name).read_text(encoding="utf-8"))
    return [(q["query"], q["relevant"]) for q in raw]


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return load_kb()


@pytest.fixture(scope="module")
def index(kb: KnowledgeBase) -> BM25Index:
    return BM25Index.from_kb(kb)


# ---------------------------------------------------------------- Done-when for P1-T2


def test_dev_set_top3_recall_at_least_0_9(index: BM25Index) -> None:
    queries = _queries("retrieval_queries.yaml")
    assert len(queries) >= 20
    recall, misses = recall_at_k(index, queries, k=3)
    assert recall >= 0.9, f"recall@3={recall:.3f}, misses: {misses}"


def test_heldout_set_regression_floor(index: BM25Index) -> None:
    # Held-out recall@3 was 0.80 (n=10) when first measured. This is a floor to catch
    # regressions, not a target; the held-out set must never be used for tuning.
    recall, misses = recall_at_k(index, _queries("retrieval_queries_heldout.yaml"), k=3)
    assert recall >= 0.7, f"recall@3={recall:.3f}, misses: {misses}"


def test_eval_sets_reference_active_docs(kb: KnowledgeBase) -> None:
    for name in ("retrieval_queries.yaml", "retrieval_queries_heldout.yaml"):
        for query, relevant in _queries(name):
            for doc_id in relevant:
                assert doc_id in kb, f"{name}: {doc_id} not in KB ({query})"
                assert kb.get(doc_id).status == "active", f"{name}: {doc_id} is superseded"


def test_eval_sets_do_not_overlap() -> None:
    dev = {q for q, _ in _queries("retrieval_queries.yaml")}
    held = {q for q, _ in _queries("retrieval_queries_heldout.yaml")}
    assert not dev & held


# ---------------------------------------------------------------- search behavior


def test_search_returns_sorted_hits_with_titles(index: BM25Index, kb: KnowledgeBase) -> None:
    hits = index.search("electronics restocking fee", k=3)
    assert hits[0].doc_id == "KB-REF-002"
    assert hits[0].title == kb.get("KB-REF-002").title
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_is_deterministic(index: BM25Index, kb: KnowledgeBase) -> None:
    query = "refund for a damaged item"
    first = index.search(query, k=5)
    assert all(index.search(query, k=5) == first for _ in range(5))
    # Rebuilding from docs in a different order gives the same ranking.
    rebuilt = BM25Index(list(reversed(kb.active())))
    assert rebuilt.search(query, k=5) == first


def test_ties_break_by_doc_id() -> None:
    body = "Returns are accepted within 30 days."
    docs = [
        KBDoc.model_validate(
            {
                "doc_id": doc_id,
                "title": "Same",
                "topic": "refund",
                "effective_date": "2025-01-01",
                "body": body,
                "facts": [{"name": "w", "value": 30, "text": "30 days"}],
            }
        )
        for doc_id in ("KB-REF-003", "KB-REF-001", "KB-REF-002")
    ]
    hits = BM25Index(docs).search("returns accepted", k=3)
    assert [h.doc_id for h in hits] == ["KB-REF-001", "KB-REF-002", "KB-REF-003"]


def test_superseded_docs_excluded_by_default(kb: KnowledgeBase) -> None:
    query = "return policy 14 days store credit only 2023"
    default = BM25Index.from_kb(kb)
    full = BM25Index.from_kb(kb, include_superseded=True)
    assert len(full) == len(kb) and len(default) == len(kb.active())
    assert "KB-REF-010" not in {h.doc_id for h in default.search(query, k=10)}
    assert full.search(query, k=1)[0].doc_id == "KB-REF-010"


def test_no_match_returns_empty(index: BM25Index) -> None:
    assert index.search("zzzz qqqq", k=3) == []
    assert index.search("the and of", k=3) == []  # stopwords only


def test_k_validation(index: BM25Index) -> None:
    with pytest.raises(ValueError):
        index.search("refund", k=0)
    assert len(index.search("refund", k=100)) <= len(index)


def test_empty_index_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Index([])


# ---------------------------------------------------------------- text processing


@pytest.mark.parametrize(
    "words",
    [
        ("refund", "refunds", "refunded"),
        ("damage", "damaged"),
        ("phone", "phones"),
        ("ship", "shipped", "shipping"),
        ("warranty", "warranties"),
        ("cancel", "cancelled"),
        ("return", "returned", "returns"),
    ],
)
def test_stem_merges_related_forms(words: tuple[str, ...]) -> None:
    assert len({stem(w) for w in words}) == 1, {w: stem(w) for w in words}


@pytest.mark.parametrize("word", ["address", "status", "business", "pass", "tv", "is"])
def test_stem_leaves_non_plurals_alone(word: str) -> None:
    assert stem(word).rstrip("e") == word.rstrip("e")


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("Is my $450 refund approved?!") == ["450", "refund", "approv"]


def test_query_expansion_bridges_vocabulary_gaps() -> None:
    terms = expand_query(tokenize("my laptop arrived cracked"))
    assert stem("electronics") in terms
    assert stem("damaged") in terms and stem("defective") in terms
