"""Verification check (P7-T1): run the verification loop on the dev set's failing runs.

    python tasks.py verify-diagnoses                    # rule/model patches, truth recovery
    python tasks.py verify-diagnoses --compare-judge    # also the truth-free judged check
    python tasks.py verify-diagnoses --source oracle    # oracle patches (upper bound)

For each dev-set run whose final outcome is wrong (controls and silent faults are
skipped: there is nothing to recover), the stored diagnosis (Wonder's if it was
escalated, else Stage B's, recomputed if none is stored) goes through the loop:

    patch     rule if one applies, else the model (`--source oracle`: the clean run's)
    replay    with the scenario's other faults kept in force (P5-T3), so patching a wrong
              candidate leaves the real fault in place
    recovery  the deterministic checker against the expected outcome (harness);
              `--compare-judge` also runs the truth-free judged check on every replay
              and reports how often the two agree

The replays, their ReplayResult rows (diagnosis_id set), and the updated diagnosis are
stored. Reads ground truth (harness only).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agenttrace.diagnosis.stage_b import aggregate_scores, attribute
from agenttrace.diagnosis.validators import ValidationContext, validate_run
from agenttrace.diagnosis.verification import (
    PatchFor,
    RecoveryCheck,
    default_patch_for,
    judged_recovery,
    verify,
)
from agenttrace.evaluation.outcome import check_outcome
from agenttrace.evaluation.recovery import outcome_score
from agenttrace.faults import FaultPlan
from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.patcher import Patch
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.oracle import oracle_patch
from agenttrace.schemas.diagnosis import Diagnosis, RootCause
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import DEFAULT_PROFILE, load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

ROOT = Path(__file__).resolve().parents[2]
DEVSET = ROOT / "reports" / "diagnosis_devset.json"
REPORT = ROOT / "reports" / "verification.json"


def stored_diagnosis(store: Any, run: Run, ctx: ValidationContext) -> Diagnosis | None:
    """Wonder's diagnosis if there is one, else Stage B's (recomputed from stored scores)."""
    stored = store.diagnoses(run_id=run.run_id)
    for prefer in ("stage-c", "stage-b"):
        for d in stored:
            if prefer in d.diagnosis_id:
                return d
    scores = store.anomalies(run_id=run.run_id)
    if not scores:
        return None
    attr = attribute(run, aggregate_scores(scores), validate_run(run, ctx))
    return attr.to_diagnosis(None)


def oracle_for(gateway: Gateway, scenario: Scenario, roles: dict[str, str]) -> PatchFor:
    def patch_for(run: Run, target: str) -> Patch | None:
        return oracle_patch(gateway, scenario, run, target, roles)

    return patch_for


def truth_recovery(scenario: Scenario) -> Any:
    expected = scenario.ground_truth.expected_final

    def check(original: Run, replay: Run) -> RecoveryCheck:
        c = check_outcome(expected, replay.final_output)
        return RecoveryCheck(
            recovered=c.correct, score=round(outcome_score(c), 4), detail=c.reason() or "correct"
        )

    return check


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--source", choices=["default", "oracle"], default="default")
    parser.add_argument("--compare-judge", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
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
    scenarios = {s.scenario_id: s for s in load_dataset()}
    gateway = Gateway.from_env()
    roles = load_agents_config().profile(DEFAULT_PROFILE).roles()
    workflow = build_support_workflow(roles)
    resources = make_resources(gateway)
    ctx = ValidationContext.default()
    judge = judged_recovery(gateway, ctx) if args.compare_judge else None
    rows = []
    agree = compared = 0
    judge_cost = 0.0
    print(
        f"Verification loop: {len(run_ids)} runs, patches {args.source}, up to "
        f"{args.max_attempts} attempts\n"
    )
    with TraceStore.open() as store:
        for run_id in run_ids:
            if not store.has_run(run_id):
                continue
            run = store.get_run(run_id)
            sc = scenarios[run.scenario_id or ""]
            gt = sc.ground_truth
            if check_outcome(gt.expected_final, run.final_output).correct:
                print(f"  {run_id:<20} {gt.fault_code}  outcome already correct: skipped")
                continue
            diag = stored_diagnosis(store, run, ctx)
            if diag is None or not diag.root_causes:
                print(f"  {run_id:<20} {gt.fault_code}  no diagnosis / no root: skipped")
                continue

            def faults_for(root: RootCause, sc: Scenario = sc) -> Any:
                others = FaultPlan([f for f in sc.faults if f.target_node_id != root.node_id])
                return others if others.faults else None

            patch_for: PatchFor = (
                oracle_for(gateway, sc, roles)
                if args.source == "oracle"
                else default_patch_for(gateway, resources)
            )
            out = verify(
                run,
                diag,
                workflow,
                resources,
                patch_for=patch_for,
                recovery=truth_recovery(sc),
                faults_for=faults_for,
                max_attempts=args.max_attempts,
            )
            judged: list[bool] = []
            judged_detail: list[str] = []
            if judge is not None:
                for a in out.attempts:
                    if a.replay_run is None:
                        continue
                    j = judge(run, a.replay_run)
                    judge_cost += j.cost_usd
                    judged.append(j.recovered)
                    judged_detail.append(j.detail)
                    if j.recovered != a.recovered:
                        print(f"        judged check disagrees on {a.target}: {j.detail}")
                    compared += 1
                    agree += j.recovered == a.recovered
            if not args.no_store:
                for a in out.attempts:
                    if a.replay_run is not None and a.result is not None:
                        store.save_run(a.replay_run, replace=True)
                        store.add_replay(a.result)
                store.save_diagnosis(out.diagnosis, replace=True)
            verified = out.verified
            true_roots = set(gt.root_cause_node_ids)
            steps = {r.split(".", 1)[0] for r in true_roots}
            hit = verified is not None and (
                verified.node_id in true_roots or verified.node_id in steps
            )
            tries = "; ".join(
                f"{a.target} [{a.patch.patch_source if a.patch else '-'}] "
                f"{'RECOVERED' if a.recovered else 'no'}"
                for a in out.attempts
            )
            print(f"  {run_id:<20} {gt.fault_code}  {tries}")
            print(
                f"        verified: {verified.node_id if verified else '(none)'}   "
                f"truth: {', '.join(sorted(true_roots))}{'   [right]' if hit else ''}"
            )
            rows.append(
                {
                    "run_id": run_id,
                    "fault_code": gt.fault_code,
                    "diagnosis_id": out.diagnosis.diagnosis_id,
                    "attempts": [
                        {
                            "target": a.target,
                            "source": a.patch.patch_source if a.patch else None,
                            "recovered": a.recovered,
                            "detail": a.detail,
                        }
                        for a in out.attempts
                    ],
                    "replays": sum(a.replay_run is not None for a in out.attempts),
                    "verified": verified.node_id if verified else None,
                    "verified_right": hit,
                    "judged": judged,
                    "judged_detail": judged_detail,
                }
            )
    gateway.close()
    if not rows:
        return 2
    n = len(rows)
    with_v = sum(r["verified"] is not None for r in rows)
    right = sum(r["verified_right"] for r in rows)
    replays = sum(r["replays"] for r in rows)
    print(
        f"\nfailing runs verified: {with_v}/{n}; the verified root is a true root: {right}/{with_v}"
    )
    print(
        f"replays: {replays} ({replays / n:.1f} per run, at most {args.max_attempts}); "
        f"every attempt stored as a ReplayResult linked to its diagnosis"
    )
    if compared:
        print(
            f"judged (truth-free) recovery agrees with the truth check: {agree}/{compared} "
            f"replays (judge cost ${judge_cost:.4f})"
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"source": args.source, "rows": rows}, indent=1), encoding="utf-8")
    print(f"report: {REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
