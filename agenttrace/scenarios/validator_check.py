"""Validator check (P6-T2): which faults do the deterministic validators catch, for free?

    python tasks.py validator-check            # every stored seeded scenario run
    python tasks.py validator-check -v         # also list the checks that fire on controls

Runs: the stored seeded runs `sc-*-default-s{k}` (P4-T5), faulty and control. Every run is
validated (`diagnosis.validators`, no model call). A fault counts as caught when a
finding lands on one of its root-cause spans (any occurrence of the root node):

    caught (error)   an error-severity finding on a root span: hard evidence
    caught (any)     an error or a warning on a root span

Done-when (P6-T2): every F4 fault caught by an error, most F2 faults caught. Controls
(F0) measure false alarms: runs with an error, and runs with only warnings, plus which
checks fire on them. This module reads ground truth only to score the validators; the
validators never see it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agenttrace.diagnosis.validators import ValidationContext, validate_run
from agenttrace.scenarios.ground_truth import load_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "validator_check.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suffix", default="-default-s", help="stored runs: sc-*<suffix><k>")
    parser.add_argument("-v", "--verbose", action="store_true", help="checks firing on controls")
    args = parser.parse_args(argv)

    from agenttrace.store import TraceStore

    scenarios = {s.scenario_id: s for s in load_dataset()}
    ctx = ValidationContext.default()
    rows: list[dict[str, Any]] = []
    fired: dict[str, Counter[str]] = {"faulty": Counter(), "control": Counter()}
    with TraceStore.open() as store:
        ids = [i for i in store.run_ids() if i.startswith("sc-") and args.suffix in i]
        for run_id in ids:
            run = store.get_run(run_id)
            sc = scenarios.get(run.scenario_id or "")
            if sc is None or run.replay_of is not None:
                continue
            truth = sc.ground_truth
            findings = validate_run(run, ctx)
            roots = set(truth.root_cause_node_ids)
            node_of = {s.span_id: s.node_id for s in run.spans}
            at_root = [f for f in findings if node_of.get(f.span_id) in roots]
            kind = "control" if truth.fault_code == "F0" else "faulty"
            fired[kind].update(f"{f.check} ({f.severity})" for f in findings)
            rows.append(
                {
                    "run_id": run_id,
                    "fault_code": truth.fault_code,
                    "roots": sorted(roots),
                    "caught_error": any(f.severity == "error" for f in at_root),
                    "caught_any": bool(at_root),
                    "errors": sum(f.severity == "error" for f in findings),
                    "warnings": sum(f.severity == "warning" for f in findings),
                    "root_checks": sorted({f.check for f in at_root}),
                    "findings": [f.model_dump(mode="json") for f in findings],
                }
            )
    if not rows:
        print("no stored seeded scenario runs: run `python tasks.py multirun` first")
        return 2

    print(f"Validator check: {len(rows)} stored seeded runs, no model calls\n")
    print(f"  {'code':<5} {'runs':>5} {'caught (error)':>15} {'caught (any)':>13}  root checks")
    by_code: dict[str, dict[str, int]] = {}
    for code in sorted({r["fault_code"] for r in rows if r["fault_code"] != "F0"}):
        group = [r for r in rows if r["fault_code"] == code]
        err = sum(bool(r["caught_error"]) for r in group)
        anyc = sum(bool(r["caught_any"]) for r in group)
        checks = Counter(c for r in group for c in r["root_checks"])
        top = ", ".join(f"{c} {n}" for c, n in checks.most_common(3)) or "-"
        by_code[code] = {"runs": len(group), "caught_error": err, "caught_any": anyc}
        print(
            f"  {code:<5} {len(group):>5} {err:>11}/{len(group):<3} "
            f"{anyc:>9}/{len(group):<3}  {top}"
        )

    controls = [r for r in rows if r["fault_code"] == "F0"]
    with_error = sum(bool(r["errors"]) for r in controls)
    only_warn = sum(not r["errors"] and bool(r["warnings"]) for r in controls)
    print(
        f"\ncontrols (F0, no fault): {len(controls)} runs; "
        f"{with_error} with an error finding (false alarm), {only_warn} with warnings only"
    )
    if args.verbose or with_error:
        for check, n in fired["control"].most_common():
            print(f"  on controls: {check} x{n}")

    f4, f2 = by_code.get("F4", {}), by_code.get("F2", {})
    f4_ok = bool(f4) and f4["caught_error"] == f4["runs"]
    f2_ok = bool(f2) and f2["caught_any"] * 2 > f2["runs"]
    print(
        f"\nF4 caught by an error: {f4.get('caught_error', 0)}/{f4.get('runs', 0)} "
        f"({'all' if f4_ok else 'NOT all'}); F2 caught: {f2.get('caught_any', 0)}/"
        f"{f2.get('runs', 0)} ({'most' if f2_ok else 'NOT most'})"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "by_code": by_code,
                "controls": {
                    "runs": len(controls),
                    "with_error": with_error,
                    "warnings_only": only_warn,
                },
                "fired": {k: dict(v) for k, v in fired.items()},
                "rows": rows,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"report: {REPORT_PATH.relative_to(ROOT).as_posix()}")
    print("PASS" if f4_ok and f2_ok else "FAIL")
    return 0 if f4_ok and f2_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
