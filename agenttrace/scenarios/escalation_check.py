"""Escalation check (P6-T6): which dev-set runs escalate to Muse Wonder, and does it help?

    python tasks.py escalate --dry-run      # which runs escalate and why (free, no calls)
    python tasks.py escalate                # escalate them (Muse Wonder; Meta credits)
    python tasks.py escalate --max-usd 2    # a tighter budget than config/escalation.yaml

For each dev-set run: Stage B from the stored ensemble scores and validators (free), the
escalation rules (config/escalation.yaml), and for escalated runs, within the budget,
Wonder's final ranking. Diagnoses are stored (`store.save_diagnosis`). Printed per run:
the reasons, Stage B's top root, Wonder's top root, the true roots; then how often the top
root is right before and after escalation, controls with no root, and Wonder's cost, which
is kept apart from Stage A's.

Reads ground truth only to score; the escalation logic never sees it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from agenttrace.diagnosis.escalation import (
    DeepResult,
    Rules,
    deep_rank,
    final_diagnosis,
    should_escalate,
)
from agenttrace.diagnosis.stage_b import aggregate_scores, attribute
from agenttrace.diagnosis.validators import ValidationContext, validate_run
from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios.ground_truth import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DEVSET = ROOT / "reports" / "diagnosis_devset.json"
REPORT = ROOT / "reports" / "escalation.json"


def _hit(node: str | None, roots: set[str]) -> bool:
    return node is not None and (node in roots or node in {r.split(".", 1)[0] for r in roots})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", help="run ids (default: the dev set)")
    parser.add_argument("--dry-run", action="store_true", help="decide only; no Wonder call")
    parser.add_argument("--max-escalations", type=int, help="override the config budget")
    parser.add_argument("--max-usd", type=float, help="override the config budget")
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args(argv)

    from agenttrace.store import TraceStore

    if args.runs:
        run_ids = list(args.runs)
    elif DEVSET.exists():
        run_ids = json.loads(DEVSET.read_text(encoding="utf-8"))["run_ids"]
    else:
        print(f"{DEVSET.relative_to(ROOT)} not found: run `python tasks.py devset` first")
        return 2
    rules = Rules.load()
    max_n = args.max_escalations if args.max_escalations is not None else rules.max_escalations
    max_usd = args.max_usd if args.max_usd is not None else rules.max_usd
    truth = {s.scenario_id: s.ground_truth for s in load_dataset()}
    ctx = ValidationContext.default()
    gateway = None if args.dry_run else Gateway.from_env()
    spent, done = 0.0, 0
    rows = []
    reasons_count: Counter[str] = Counter()
    mode = "dry run: no Wonder calls" if args.dry_run else f"budget {max_n} runs / ${max_usd:.2f}"
    print(f"Escalation: {len(run_ids)} runs ({mode})\n")
    with TraceStore.open() as store:
        for run_id in run_ids:
            if not store.has_run(run_id):
                continue
            run = store.get_run(run_id)
            stored = store.anomalies(run_id=run_id)
            if not stored:
                print(f"  {run_id}: no stored scores (run `python tasks.py ensemble`)")
                continue
            scores = aggregate_scores(stored)
            findings = validate_run(run, ctx)
            attr = attribute(run, scores, findings)
            decision = should_escalate(run, attr, scores, stored, findings, rules)
            for r in decision.reasons:
                reasons_count[r.split(" (")[0].split(" kappa")[0].split(" 0.")[0]] += 1
            deep: DeepResult | None = None
            skipped = ""
            if decision.escalate and gateway is not None:
                if done >= max_n or spent >= max_usd:
                    skipped = "budget reached"
                else:
                    deep = deep_rank(gateway, run, attr, scores, stored, findings, decision.reasons)
                    spent += deep.cost_usd
                    done += 1
            diag = final_diagnosis(attr, decision, deep)
            if not args.no_store and not args.dry_run:
                store.save_diagnosis(diag, replace=True)
            gt = truth[run.scenario_id or ""]
            roots = set(gt.root_cause_node_ids)
            before = attr.roots[0].node_id if attr.roots else None
            after = diag.root_causes[0].node_id if diag.root_causes else None
            row = {
                "run_id": run_id,
                "fault_code": gt.fault_code,
                "true_roots": sorted(roots),
                "reasons": decision.reasons,
                "agreement": decision.agreement,
                "escalated": deep is not None,
                "wonder_valid": deep.valid if deep else None,
                "skipped": skipped,
                "stage_b_top": before,
                "final_top": after,
                "final_roots": [r.node_id for r in diag.root_causes],
                "before_hit": _hit(before, roots),
                "after_hit": _hit(after, roots),
                "cost_usd": round(deep.cost_usd, 6) if deep else 0.0,
                "narrative": diag.narrative[:300],
            }
            rows.append(row)
            esc = "ESC" if decision.escalate else "   "
            wonder = ""
            if deep is not None:
                wonder = f" -> Wonder: {after or '(no root)'}{'' if deep.valid else ' [invalid]'}"
            elif skipped:
                wonder = f" ({skipped})"
            print(
                f"  {esc} {run_id:<20} {gt.fault_code}  stage B: {before or '(no root)'}{wonder}"
                f"   truth: {', '.join(sorted(roots)) or '(none)'}",
                flush=True,
            )
            for r in decision.reasons:
                print(f"        - {r}")
    if gateway is not None:
        gateway.close()
    if not rows:
        return 2

    faulty = [r for r in rows if r["fault_code"] != "F0"]
    controls = [r for r in rows if r["fault_code"] == "F0"]
    esc_rows = [r for r in rows if r["reasons"]]
    print(f"\nescalation triggered: {len(esc_rows)}/{len(rows)} runs")
    for reason, n in reasons_count.most_common():
        print(f"  {n:>3}x {reason}")
    b_hits = sum(r["before_hit"] for r in faulty)
    print(f"top-1 root right (node or its step), Stage B: {b_hits}/{len(faulty)}")
    if not args.dry_run:
        a_hits = sum(r["after_hit"] for r in faulty)
        print(f"top-1 root right after escalation:           {a_hits}/{len(faulty)}")
        clean_b = sum(r["stage_b_top"] is None for r in controls)
        clean_a = sum(r["final_top"] is None for r in controls)
        n_c = len(controls)
        print(f"controls with no root: Stage B {clean_b}/{n_c}, after {clean_a}/{n_c}")
        valid = sum(bool(r["wonder_valid"]) for r in rows if r["escalated"])
        n = sum(r["escalated"] for r in rows)
        print(
            f"Wonder (Stage C) cost: ${spent:.4f} for {n} escalations "
            f"({valid} valid rankings), tracked apart from Stage A"
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"dry_run": args.dry_run, "cost_usd": spent, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"report: {REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
