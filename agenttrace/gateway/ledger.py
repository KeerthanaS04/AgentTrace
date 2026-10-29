"""Per-call cost and token accounting, appended to a JSONL log."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from agenttrace.gateway.types import CompletionResult


class CostLedger:
    def __init__(self, log_path: Path | str | None = None) -> None:
        self._path = Path(log_path) if log_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._by_provider: dict[str, float] = defaultdict(float)
        self.calls = 0
        self.cache_hits = 0

    def record(self, result: CompletionResult, *, tag: str | None = None) -> None:
        with self._lock:
            self.calls += 1
            self.cache_hits += int(result.cache_hit)
            self._by_provider[result.provider] += result.cost_usd
            if self._path:
                entry = {
                    "ts": datetime.now(UTC).isoformat(),
                    "tag": tag,
                    **result.model_dump(exclude={"text"}),
                }
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")

    @property
    def total_usd(self) -> float:
        return sum(self._by_provider.values())

    def spent_by_provider(self) -> dict[str, float]:
        with self._lock:
            return dict(self._by_provider)
