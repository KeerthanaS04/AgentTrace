"""Print retrieval recall on the dev and held-out query sets.

Usage: python tasks.py retrieval   (or python -m agenttrace.workflow.kb.report)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from agenttrace.workflow.kb.index import BM25Index, recall_at_k
from agenttrace.workflow.kb.loader import load_kb

EVAL_DIR = Path(__file__).resolve().parents[3] / "data" / "eval"
QUERY_SETS = {
    "dev": "retrieval_queries.yaml",
    "held-out": "retrieval_queries_heldout.yaml",
}
KS = (1, 3, 5)
# P1-T2 Done-when: dev-set recall@3 >= 0.9.
TARGET = ("dev", 3, 0.9)


def load_queries(path: Path) -> list[tuple[str, list[str]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(q["query"], q["relevant"]) for q in raw]


def main() -> int:
    kb = load_kb()
    index = BM25Index.from_kb(kb)
    print(f"Index: {len(index)} active docs (of {len(kb)} total)\n")

    header = f"{'set':<10} {'n':>3}  " + "  ".join(f"recall@{k}" for k in KS)
    print(header)
    print("-" * len(header))
    results: dict[str, dict[int, float]] = {}
    misses_at_3: dict[str, list[tuple[str, list[str], list[str]]]] = {}
    for name, filename in QUERY_SETS.items():
        queries = load_queries(EVAL_DIR / filename)
        results[name] = {}
        for k in KS:
            recall, misses = recall_at_k(index, queries, k)
            results[name][k] = recall
            if k == 3:
                misses_at_3[name] = misses
        row = "  ".join(f"{results[name][k]:>8.3f}" for k in KS)
        print(f"{name:<10} {len(queries):>3}  {row}")

    for name, misses in misses_at_3.items():
        if not misses:
            continue
        print(f"\nMisses at k=3 ({name}):")
        for query, relevant, got in misses:
            print(f"  query:    {query}")
            print(f"  expected: {', '.join(relevant)}")
            print(f"  got:      {', '.join(got) or '(nothing)'}\n")

    set_name, k, threshold = TARGET
    value = results[set_name][k]
    verdict = "PASS" if value >= threshold else "FAIL"
    print(f"Target: {set_name} recall@{k} >= {threshold} -> {value:.3f} {verdict}")
    return 0 if value >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
