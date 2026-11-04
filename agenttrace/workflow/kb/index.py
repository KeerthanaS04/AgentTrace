"""Deterministic BM25 retrieval over the knowledge base.

Implemented in-house (no numpy / rank-bm25) so scoring and tie-breaking are fully
under our control: the same query always returns the same ranking, which keeps
recorded runs replayable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import Field

from agenttrace.schemas.base import NonEmptyStr, StrictModel
from agenttrace.tracing import SpanBuilder, traced
from agenttrace.workflow.kb.loader import KnowledgeBase
from agenttrace.workflow.kb.models import KBDoc

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Tokens are split on non-alphanumerics, so "i'm" arrives as "i" + "m".
STOPWORDS = frozenset(
    {
        "a", "about", "after", "an", "and", "any", "are", "as", "at", "be", "been",
        "before", "but", "by", "can", "could", "did", "do", "does", "for", "from",
        "get", "got", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
        "its", "just", "m", "me", "my", "of", "on", "or", "our", "s", "so", "t", "than",
        "that", "the", "their", "them", "then", "there", "this", "to", "too", "us",
        "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
        "would", "you", "your", "yours",
    }
)  # fmt: skip
# Title terms are counted this many times, so a title match outweighs a passing mention.
TITLE_WEIGHT = 2


def stem(token: str) -> str:
    """Tiny suffix stripper so related forms share a term.

    refunds/refunded -> refund, damage/damaged -> damag, phone/phones -> phon,
    ship/shipped/shipping -> ship, warranty/warranties -> warranty.
    """
    t = token
    if len(t) <= 3:
        return t
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if not t.endswith(suffix) or len(t) - len(suffix) < 3:
            continue
        if suffix == "s" and t.endswith(("ss", "us", "is")):
            continue
        t = t[: -len(suffix)] + repl
        # shipp -> ship, cancell -> cancel (but keep "pass", "fizz")
        if suffix in ("ing", "ed") and len(t) >= 4 and t[-1] == t[-2] and t[-1] not in "sz":
            t = t[:-1]
        break
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# Query-side expansion for vocabulary gaps between how customers write and how
# policies are worded. Kept general (categories and synonyms), not per-query.
_EXPANSIONS_RAW: dict[str, tuple[str, ...]] = {
    # damage words -> the policy terms
    "broken": ("damaged", "defective"),
    "broke": ("damaged", "defective"),
    "cracked": ("damaged", "defective"),
    "smashed": ("damaged", "defective"),
    "shattered": ("damaged", "defective"),
    "dented": ("damaged",),
    "faulty": ("defective",),
    # products -> the category policies are written against
    **{
        p: ("electronics",)
        for p in (
            "laptop",
            "phone",
            "smartphone",
            "headphones",
            "tablet",
            "tv",
            "television",
            "monitor",
            "speaker",
            "camera",
            "charger",
            "console",
            "smartwatch",
        )
    },
    # approval wording
    "approve": ("approval",),
    # money / refund wording
    "money": ("refund",),
    "reimburse": ("refund",),
    # packages
    "parcel": ("package",),
}
EXPANSIONS: dict[str, tuple[str, ...]] = {
    stem(k): tuple(stem(v) for v in vs) for k, vs in _EXPANSIONS_RAW.items()
}


def expand_query(terms: list[str]) -> list[str]:
    out = list(terms)
    for t in terms:
        out.extend(EXPANSIONS.get(t, ()))
    return out


class SearchHit(StrictModel):
    doc_id: NonEmptyStr
    score: float = Field(gt=0)
    title: str


def _record_hits(span: SpanBuilder, hits: list[SearchHit]) -> None:
    span.output = {"hits": [h.model_dump(mode="json") for h in hits]}


class BM25Index:
    def __init__(
        self,
        docs: Iterable[KBDoc],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._docs = sorted(docs, key=lambda d: d.doc_id)
        if not self._docs:
            raise ValueError("cannot build an index over zero documents")
        self._tf: list[Counter[str]] = []
        for doc in self._docs:
            terms = tokenize(doc.title) * TITLE_WEIGHT + tokenize(doc.body)
            self._tf.append(Counter(terms))
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._docs)
        # Lucene-style IDF: always positive, even for terms in most documents.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_kb(cls, kb: KnowledgeBase, *, include_superseded: bool = False) -> BM25Index:
        docs = kb.docs if include_superseded else kb.active()
        return cls(docs)

    def __len__(self) -> int:
        return len(self._docs)

    def _bm25(self, q_terms: Counter[str], i: int) -> float:
        tf, length = self._tf[i], self._len[i]
        score = 0.0
        norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
        for term, q_count in q_terms.items():
            f = tf.get(term)
            if f:
                score += q_count * self._idf[term] * f * (self.k1 + 1) / (f + norm)
        # Round so tiny float noise can't reorder ties across platforms.
        return round(score, 9)

    def has(self, doc_id: str) -> bool:
        """Whether the doc is indexed (superseded docs are not, by default)."""
        return any(d.doc_id == doc_id for d in self._docs)

    def score(self, query: str, doc_id: str) -> float | None:
        """The BM25 score `search` gives `doc_id` for `query` (None if not indexed), so a
        recorded hit can be checked against the index (P6-T2)."""
        q_terms = Counter(expand_query(tokenize(query)))
        for i, doc in enumerate(self._docs):
            if doc.doc_id == doc_id:
                return self._bm25(q_terms, i)
        return None

    @traced(
        "retrieval",
        "search",
        record=_record_hits,
        inject=(
            lambda hits: {"hits": [h.model_dump(mode="json") for h in hits]},
            lambda doc: [SearchHit.model_validate(h) for h in doc["hits"]],
        ),
    )
    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        """Top-k docs by BM25 score. Ties break by doc_id; zero-score docs are dropped."""
        if k < 1:
            raise ValueError("k must be >= 1")
        q_terms = Counter(expand_query(tokenize(query)))
        scored: list[tuple[float, str, str]] = []
        for i, doc in enumerate(self._docs):
            score = self._bm25(q_terms, i)
            if score > 0:
                scored.append((score, doc.doc_id, doc.title))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [SearchHit(doc_id=d, score=s, title=t) for s, d, t in scored[:k]]


def recall_at_k(
    index: BM25Index, queries: Iterable[tuple[str, list[str]]], k: int = 3
) -> tuple[float, list[tuple[str, list[str], list[str]]]]:
    """Mean recall@k over (query, relevant_doc_ids) pairs, plus the misses for debugging."""
    total = 0.0
    n = 0
    misses: list[tuple[str, list[str], list[str]]] = []
    for query, relevant in queries:
        got = [h.doc_id for h in index.search(query, k)]
        hit = len(set(relevant) & set(got)) / len(relevant)
        total += hit
        n += 1
        if hit < 1:
            misses.append((query, relevant, got))
    if n == 0:
        raise ValueError("no queries")
    return total / n, misses
