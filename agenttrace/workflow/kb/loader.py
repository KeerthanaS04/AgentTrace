"""Load and validate the knowledge base from `data/kb/*.yaml`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import yaml

from agenttrace.workflow.kb.models import KBDoc

DEFAULT_KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"


class KnowledgeBase:
    def __init__(self, docs: list[KBDoc]) -> None:
        counts = Counter(d.doc_id for d in docs)
        dupes = sorted(doc_id for doc_id, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate doc_ids: {dupes}")
        self._docs = {d.doc_id: d for d in docs}
        for doc in docs:
            if doc.superseded_by is None:
                continue
            target = self._docs.get(doc.superseded_by)
            if target is None:
                raise ValueError(f"{doc.doc_id}: superseded_by {doc.superseded_by} does not exist")
            if target.status != "active":
                raise ValueError(f"{doc.doc_id}: superseded_by {target.doc_id} is not active")
            if target.topic != doc.topic:
                raise ValueError(f"{doc.doc_id}: replacement {target.doc_id} has a different topic")

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[KBDoc]:
        return iter(self._docs.values())

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._docs

    def get(self, doc_id: str) -> KBDoc:
        try:
            return self._docs[doc_id]
        except KeyError:
            raise KeyError(f"no KB doc {doc_id!r}") from None

    @property
    def docs(self) -> list[KBDoc]:
        return list(self._docs.values())

    def active(self) -> list[KBDoc]:
        return [d for d in self._docs.values() if d.status == "active"]


def load_kb(kb_dir: Path | str = DEFAULT_KB_DIR) -> KnowledgeBase:
    docs: list[KBDoc] = []
    for path in sorted(Path(kb_dir).glob("*.yaml")):
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or []
        for i, entry in enumerate(raw):
            try:
                docs.append(KBDoc.model_validate(entry))
            except ValueError as exc:
                raise ValueError(f"{path.name} entry {i}: {exc}") from exc
    return KnowledgeBase(docs)
