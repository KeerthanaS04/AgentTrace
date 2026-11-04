"""Attribution check (P6-T5): Stage B's roots and propagated errors against ground truth.

    python tasks.py attribution-check                  # the diagnosis dev set
    python tasks.py attribution-check --judge-searches # let judges flag searches too
    python tasks.py attribution-check --show 10        # the first 10 runs in full

Uses the stored Stage A / ensemble scores (run `python tasks.py ensemble` first) and the
validators, so it makes no model call. Per run it prints the true roots next to Stage B's
ranked roots and propagated errors, for hand-checking, and scores:

    top-1 node      the top-ranked root is a true root node (e.g. tool_agent.order_lookup)
    top-1 step      ... or the step that contains one (retriever for retriever.search)
    any root        some ranked root matches a true root (node or step)
    control clean   F0 runs: no root at all (no false alarm)
    paths valid     every propagated error's path follows the run's graph

Reads ground truth (harness only); Stage B itself never does.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from agenttrace.diagnosis.stage_b import aggregate_scores, attribute, valid_path
from agenttrace.diagnosis.validators import ValidationContext, validate_run
from agenttrace.graph import build_graph
from agenttrace.scenarios.ground_truth import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DEVSET = ROOT / "reports" / "diagnosis_devset.json"
REPORT = ROOT / "reports" / "attribution_check.json"


def _step(node_id: str) -> str:
    return node_id.split(".", 1)[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", help="run ids (default: the dev set)")
    parser.add_argument("--judge-searches", action="store_true", help="judges may flag searches")
    parser.add_argument("--show", type=int, default=10, help="print this many runs in full")
    args = parser.parse_args(argv)

    from agenttrace.store import TraceStore

    if args.runs:
        run_ids = list(args.runs)
    elif DEVSET.exists():
        run_ids = json.loads(DEVSET.read_text(encoding="utf-8"))["run_ids"]
    else:
        print(f"{DEVSET.relative_to(ROOT)} not found: run `python tasks.py devset` first")
        return 2
    truth = {s.scenario_id: s.ground_truth for s in load_dataset()}
    ctx = ValidationContext.default()
    rows = []
    with TraceStore.open() as store:
        for run_id in run_ids:
            if not store.has_run(run_id):
                continue
            run = store.get_run(run_id)
            stored = store.anomalies(run_id=run_id)
            if not stored:
                print(f"  {run_id}: no stored scores (run `python tasks.py ensemble`)")
                continue
            gt = truth[run.scenario_id or ""]
            attr = attribute(
                run,
                aggregate_scores(stored),
                validate_run(run, ctx),
                judge_searches=args.judge_searches,
            )
            g = build_graph(run)
            local = {n: g.nodes[n].get("local_id", n) for n in g}
            roots = [r.node_id for r in attr.roots]
            true_roots = set(gt.root_cause_node_ids)
            true_steps = {_step(n) for n in true_roots}
            node_hit = bool(roots) and roots[0] in true_roots
            step_hit = bool(roots) and (roots[0] in true_roots or roots[0] in true_steps)
            any_hit = any(r in true_roots or r in true_steps for r in roots)
            prop_steps = {_step(p.node_id) for p in attr.propagated}
            rows.append(
                {
                    "run_id": run_id,
                    "fault_code": gt.fault_code,
                    "true_roots": sorted(true_roots),
                    "true_affected": list(gt.affected_node_ids),
                    "roots": [
                        {
                            "span": local[r.span_id],
                            "fault_type": r.fault_type,
                            "confidence": r.confidence,
                        }
                        for r in attr.roots
                    ],
                    "propagated": [
                        {"span": local[p.span_id], "path": [local[x] for x in p.path]}
                        for p in attr.propagated
                    ],
                    "top1_node": node_hit,
                    "top1_step": step_hit,
                    "any_root": any_hit,
                    "control_clean": gt.fault_code == "F0" and not roots,
                    "paths_valid": all(valid_path(g, p.path) for p in attr.propagated),
                    "propagated_steps": sorted(prop_steps),
                }
            )
    if not rows:
        return 2

    mode = "judges and validators" if args.judge_searches else "validators only"
    print(f"Attribution check: {len(rows)} runs (searches flagged by {mode})\n")
    for row in rows[: args.show]:
        verdict = "TOP-1" if row["top1_node"] else "step" if row["top1_step"] else "miss"
        if row["fault_code"] == "F0":
            verdict = "clean" if row["control_clean"] else "false alarm"
        print(f"{row['run_id']}  {row['fault_code']}  [{verdict}]")
        print(f"  true roots   : {', '.join(row['true_roots']) or '(none: control)'}")
        print(f"  true affected: {', '.join(row['true_affected']) or '-'}")
        ranked = [f"{r['span']} ({r['fault_type']}, {r['confidence']:.2f})" for r in row["roots"]]
        print(f"  stage B roots: {'; '.join(ranked) or '(none)'}")
        for p in row["propagated"]:
            print(f"    propagated : {p['span']:<26} via {' -> '.join(p['path'])}")
        print()

    faulty = [r for r in rows if r["fault_code"] != "F0"]
    controls = [r for r in rows if r["fault_code"] == "F0"]
    paths = sum(r["paths_valid"] for r in rows)
    print("summary:")
    for key, label in (
        ("top1_node", "top-1 root = true root node"),
        ("top1_step", "top-1 root = true root or its step"),
        ("any_root", "a true root among the ranked roots"),
    ):
        print(f"  {label:<40} {sum(r[key] for r in faulty)}/{len(faulty)}")
    print(
        f"  {'controls with no root (no false alarm)':<40} "
        f"{sum(r['control_clean'] for r in controls)}/{len(controls)}"
    )
    print(f"  {'runs whose propagated paths are all valid':<40} {paths}/{len(rows)}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"judge_searches": args.judge_searches, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"report: {REPORT.relative_to(ROOT).as_posix()}")
    return 0 if paths == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
