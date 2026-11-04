"""The diagnosis dev set (P6-T3): a small, stratified set of runs for Phase 6 / 7 work.

    python tasks.py devset                 # 2 verified runs per fault code F1-F8 + 2 controls
    python tasks.py devset --per-code 3

Picked from the P4-T5 multi-run report: for each fault code, the first `--per-code` seeded
runs whose fault behaved as labelled (for F0: whose control decision was right), in
report order. Written as run ids only, with no labels, to reports/diagnosis_devset.json,
so the diagnosis side can read which runs to work on without learning anything about
their answers (the ids are opaque).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from agenttrace.scenarios.ground_truth import load_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "multirun_default.json"
DEVSET_PATH = ROOT / "reports" / "diagnosis_devset.json"
CODES = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")


def pick_devset(per_code: int = 2, report: Path | None = None) -> list[str]:
    rows = json.loads((report or REPORT).read_text(encoding="utf-8"))["rows"]
    codes = {s.scenario_id: s.ground_truth.fault_code for s in load_dataset()}
    taken: Counter[str] = Counter()
    picked: list[str] = []
    for seed, seed_rows in rows.items():
        for row in seed_rows:
            code = codes.get(row["scenario_id"])
            if code is None or not row["behaves"] or taken[code] >= per_code:
                continue
            taken[code] += 1
            picked.append(f"{row['scenario_id']}-default-s{seed}")
    return sorted(picked)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-code", type=int, default=2)
    args = parser.parse_args(argv)
    if not REPORT.exists():
        print(f"{REPORT.relative_to(ROOT)} not found: run `python tasks.py multirun` first")
        return 2
    ids = pick_devset(args.per_code)
    DEVSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVSET_PATH.write_text(
        json.dumps({"per_code": args.per_code, "run_ids": ids}, indent=1), encoding="utf-8"
    )
    print(f"diagnosis dev set: {len(ids)} runs ({args.per_code} per code, F0-F8)")
    print(f"written to {DEVSET_PATH.relative_to(ROOT).as_posix()} (run ids only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
