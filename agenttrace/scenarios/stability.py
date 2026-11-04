"""Replay stability (P5-T6): the same replay, repeated, gives the same result.

    python tasks.py replay-stability                  # 3 reps, cached and live, 3 runs per code
    python tasks.py replay-stability --mode cached    # free: no model call after the first rep
    python tasks.py replay-stability --per-code 5 --reps 3 --source rule

Replays: verified seeded runs (P4-T5 report), up to `--per-code` per fault code, each
root cause patched from `--source` (oracle by default, so every code is covered). The
patch is computed once, with the normal gateway, and held fixed: any variation is the
replay's own.

Two modes, `--reps` repetitions each (default 3):

    cached  the normal gateway. Repetition 1 may call the model for re-run spans whose
            prompts are new; later repetitions are served from the gateway cache.
    live    every repetition gets its own empty completion cache, so every re-run model
            call goes to the provider (frozen spans still come from the record). Ledger
            and budget are the normal ones.

Tolerance (fixed before the first run, findings §4):

    cached  IDENTICAL: zero diffs between repetitions: every span's status, input,
            output, model, params, and patched flag (`align_runs`), and the final output.
            Cache bookkeeping in a model call's output (`cache_hit`, `original_cost_usd`)
            is not a result and is ignored: repetition 1 may have made the call that the
            later ones are served from the cache.
    live    decision and `recovered` verdict identical in every repetition. Wording may
            differ: the share of replays whose reply text is identical, and the spread of
            `recovery_score`, are reported but are not pass criteria.

The report goes to reports/replay_stability.json.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agenttrace.evaluation.recovery import score_recovery
from agenttrace.faults import FaultPlan
from agenttrace.gateway.cache import CompletionCache
from agenttrace.gateway.gateway import Gateway
from agenttrace.graph import align_runs
from agenttrace.replay.executor import execute_plan
from agenttrace.replay.patcher import PatchError, ReplayPlan, plan_patch
from agenttrace.replay.sources import PATCHER_ROLE, PatchSourceError
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.patch_check import make_patch, target_for, verified_runs
from agenttrace.schemas.run import PatchSource, Run
from agenttrace.workflow.agent_config import DEFAULT_PROFILE, load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "replay_stability.json"
Mode = str  # "cached" | "live"


@dataclass
class Repeat:
    """One replay, repeated: what stayed the same across the repetitions."""

    scenario_id: str
    fault_code: str
    seed: int
    target: str
    mode: Mode
    reps: int
    identical: bool  # zero diffs between every repetition and the first
    same_decision: bool
    same_recovered: bool
    same_reply: bool
    score_range: float  # max - min recovery_score over the repetitions
    diffs: list[str] = field(default_factory=list)  # spans differing from repetition 1
    decisions: list[str | None] = field(default_factory=list)
    live_calls: int = 0

    @property
    def within_tolerance(self) -> bool:
        if self.mode == "cached":
            return self.identical
        return self.same_decision and self.same_recovered


def compare_repetitions(
    runs: Sequence[Run], recovered: Sequence[bool], scores: Sequence[float]
) -> dict[str, object]:
    """How a set of repeated replays of one plan agree (repetition 1 is the reference)."""
    first = runs[0]
    diffs: set[str] = set()
    for run in runs[1:]:
        diffs |= {p.local_id for p in align_runs(first, run).pairs if p.status != "unchanged"}
        if run.final_output != first.final_output:
            diffs.add("final_output")
    replies = {json.dumps(r.final_output.get("reply")) for r in runs}
    decisions = [r.final_output.get("decision") for r in runs]
    return {
        "identical": not diffs,
        "diffs": sorted(diffs),
        "same_decision": len(set(decisions)) == 1,
        "same_recovered": len(set(recovered)) == 1,
        "same_reply": len(replies) == 1,
        "score_range": round(max(scores) - min(scores), 4),
        "decisions": decisions,
    }


def _rerun_gateway(base: Gateway, mode: Mode, scratch: Path, rep: int) -> Gateway:
    """The gateway one repetition runs on (see module doc)."""
    if mode == "cached":
        return base
    return base.with_cache(CompletionCache(scratch / f"live-{rep}.sqlite"))


def repeat_replay(
    base: Gateway,
    scenario: Scenario,
    faulty: Run,
    plan: ReplayPlan,
    root: str,
    roles: dict[str, str],
    *,
    mode: Mode,
    reps: int,
    scratch: Path,
) -> Repeat:
    workflow = build_support_workflow(roles)
    others = FaultPlan([f for f in scenario.faults if f.target_node_id != root])
    runs: list[Run] = []
    recovered: list[bool] = []
    scores: list[float] = []
    live = 0
    for rep in range(1, reps + 1):
        gateway = _rerun_gateway(base, mode, scratch, rep)
        out = execute_plan(
            faulty,
            workflow,
            plan,
            make_resources(gateway),
            run_id=f"{faulty.run_id}-stab-{mode}-{rep}",
            faults=others if others.faults else None,
        )
        if gateway is not base:
            gateway.cache.close()
        live += out.live_model_calls
        rec = score_recovery(
            scenario.ground_truth.expected_final, faulty.final_output, out.replay.final_output
        )
        runs.append(out.replay)  # compared by local span id, so run ids don't matter
        recovered.append(rec.recovered)
        scores.append(rec.recovery_score)
    cmp = compare_repetitions(runs, recovered, scores)
    return Repeat(
        scenario_id=scenario.scenario_id,
        fault_code=scenario.ground_truth.fault_code,
        seed=int(faulty.seed or 0),
        target=plan.target_local,
        mode=mode,
        reps=reps,
        live_calls=live,
        **cmp,  # type: ignore[arg-type]
    )


def pick_runs(per_code: int, codes: Sequence[str] | None) -> list[tuple[str, int]]:
    """Up to `per_code` verified (scenario, seed) runs per fault code, in report order."""
    scenarios = {s.scenario_id: s for s in load_dataset()}
    taken: Counter[str] = Counter()
    picked = []
    for sid, seed in verified_runs():
        if sid not in scenarios:
            continue
        code = scenarios[sid].ground_truth.fault_code
        if (codes and code not in codes) or taken[code] >= per_code:
            continue
        taken[code] += 1
        picked.append((sid, seed))
    return picked


def summarize(rows: Sequence[Repeat]) -> dict[str, object]:
    out: dict[str, object] = {}
    for mode in sorted({r.mode for r in rows}):
        group = [r for r in rows if r.mode == mode]
        n = len(group)
        out[mode] = {
            "replays": n,
            "within_tolerance": sum(r.within_tolerance for r in group),
            "identical": sum(r.identical for r in group),
            "same_decision": sum(r.same_decision for r in group),
            "same_recovered": sum(r.same_recovered for r in group),
            "same_reply": sum(r.same_reply for r in group),
            "max_score_range": max((r.score_range for r in group), default=0.0),
            "live_model_calls": sum(r.live_calls for r in group),
            "outside_tolerance": [
                f"{r.scenario_id}-s{r.seed} {r.target}" for r in group if not r.within_tolerance
            ],
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["cached", "live", "both"], default="both")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per replay")
    parser.add_argument("--per-code", type=int, default=3, help="verified runs per fault code")
    parser.add_argument("--code", nargs="*", help="only these fault codes")
    parser.add_argument("--source", choices=["oracle", "rule", "model"], default="oracle")
    parser.add_argument("--role", default=PATCHER_ROLE, help="model role for --source model")
    parser.add_argument("-v", "--verbose", action="store_true", help="a line per replay")
    args = parser.parse_args(argv)
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    source: PatchSource = args.source
    modes = ["cached", "live"] if args.mode == "both" else [args.mode]

    if not (ROOT / "reports" / "multirun_default.json").exists():
        print("reports/multirun_default.json not found: run `python tasks.py multirun` first")
        return 2
    scenarios = {s.scenario_id: s for s in load_dataset()}
    todo = pick_runs(args.per_code, args.code)
    base = Gateway.from_env()
    roles = load_agents_config().profile(DEFAULT_PROFILE).roles()
    workflow = build_support_workflow(roles)

    from agenttrace.store import TraceStore

    rows: list[Repeat] = []
    skipped: Counter[str] = Counter()
    print(
        f"Replay stability: {len(todo)} verified runs, {source} patches, "
        f"{args.reps} repetitions, modes {', '.join(modes)}\n"
    )
    with TraceStore.open() as store, tempfile.TemporaryDirectory() as tmp:
        for sid, seed in todo:
            run_id = f"{sid}-{DEFAULT_PROFILE}-s{seed}"
            if not store.has_run(run_id):
                skipped["not stored"] += 1
                continue
            faulty, scenario = store.get_run(run_id), scenarios[sid]
            for root in scenario.ground_truth.root_cause_node_ids:
                target = target_for(faulty, root)
                if target is None:
                    skipped["root never ran"] += 1
                    continue
                try:
                    patch = make_patch(source, base, scenario, faulty, target, roles, args.role)
                    if patch is None:
                        skipped[f"no {source} patch"] += 1
                        continue
                    plan = plan_patch(faulty, workflow, patch)
                except (PatchError, PatchSourceError):
                    skipped[f"{source} patch failed"] += 1
                    continue
                for mode in modes:
                    row = repeat_replay(
                        base,
                        scenario,
                        faulty,
                        plan,
                        root,
                        roles,
                        mode=mode,
                        reps=args.reps,
                        scratch=Path(tmp),
                    )
                    rows.append(row)
                    if args.verbose or not row.within_tolerance:
                        flag = "ok " if row.within_tolerance else "OUT"
                        print(
                            f"  {flag} {mode:<6} {run_id:<20} {row.target:<24} "
                            f"decisions {row.decisions}  diffs {row.diffs[:4]}",
                            flush=True,
                        )
    base.close()

    summary = summarize(rows)
    for mode, s in summary.items():
        assert isinstance(s, dict)
        n = s["replays"]
        rule = "zero diffs" if mode == "cached" else "same decision and recovered verdict"
        print(f"\n{mode} ({rule}): {s['within_tolerance']}/{n} within tolerance")
        print(f"  identical (zero diffs)    : {s['identical']}/{n}")
        print(f"  same decision             : {s['same_decision']}/{n}")
        print(f"  same recovered verdict    : {s['same_recovered']}/{n}")
        print(f"  same reply text           : {s['same_reply']}/{n}")
        print(f"  largest recovery_score spread: {s['max_score_range']}")
        print(f"  model calls sent to the provider or gateway cache: {s['live_model_calls']}")
    if skipped:
        print("\nskipped: " + "; ".join(f"{n}x {why}" for why, n in skipped.items()))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "source": source,
                "reps": args.reps,
                "tolerance": {
                    "cached": "zero diffs between repetitions",
                    "live": "identical decision and recovered verdict in every repetition",
                },
                "summary": summary,
                "rows": [asdict(r) | {"within_tolerance": r.within_tolerance} for r in rows],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {REPORT_PATH.relative_to(ROOT).as_posix()}")
    ok = bool(rows) and all(r.within_tolerance for r in rows)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
