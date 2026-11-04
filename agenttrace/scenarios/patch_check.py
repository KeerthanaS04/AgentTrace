"""Patch check (P5-T3, P5-T4): patch a known-bad node from a patch source and replay.

    python tasks.py patch-check                     # oracle patches, every verified run
    python tasks.py patch-check --source rule       # deterministic fixes (P5-T4)
    python tasks.py patch-check --source model --code F5 F6 --limit 20   # model proposals
    python tasks.py patch-check --code F3 -v        # one fault type, a line per replay
    python tasks.py patch-check --store             # also save the replays (replay_of set)

Runs: the stored seeded runs `sc-xxx-default-s{k}` whose fault behaved as labelled in
`reports/multirun_default.json` (run `python tasks.py multirun` first). For each root
cause of the scenario:

    target       the root node's span; when the fault hit several spans of that node
                 (a retrieval fault hits every search), the step that contains them
    replacement  from `--source` (see replay.sources): `oracle` = what the span produced
                 in the clean run of the same seed (scenarios.oracle); `rule` = a
                 deterministic fix, where one applies to the node; `model` = a proposal
                 by the `patcher` role (`--role` to change; it spends that provider's
                 credits, one call per patch)
    other faults kept in force (F7's second root), so the counterfactual is faithful

Each replay must change only the target (patched) and its descendants: frozen spans
unchanged, and anything new or missing inside a re-run step. Also reported, as context
for P5-T5: whether the final outcome is correct after the patch, and how many model
calls the replay could not serve from the record or the gateway cache. Every replay is
tagged with its `patch_source`; a source with no patch for a node counts as "n/a".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agenttrace.evaluation.recovery import replay_result, score_recovery
from agenttrace.faults import FaultPlan
from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.differ import diff_runs
from agenttrace.replay.executor import execute_plan
from agenttrace.replay.patcher import Patch, PatchError, plan_patch
from agenttrace.replay.sources import PATCHER_ROLE, PatchSourceError, model_patch, rule_patch
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.oracle import clean_run, oracle_patch
from agenttrace.schemas.replay import ReplayResult
from agenttrace.schemas.run import PatchSource, Run
from agenttrace.workflow.agent_config import DEFAULT_PROFILE, load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "multirun_default.json"


@dataclass
class PatchCheck:
    scenario_id: str
    fault_code: str
    seed: int
    root: str
    source: str = "oracle"
    target: str = ""
    verdict: str = ""  # descendants-only / VIOLATION / n/a (no patch from the source) / skipped
    note: str = ""  # how the source made the patch
    detail: str = ""
    recovered: bool | None = None
    score: float | None = None  # recovery_score after the patch
    before_score: float | None = None
    clean_correct: bool | None = None  # recovery was possible (the seed's clean run is right)
    matches_clean: bool | None = None
    changed: int = 0
    live_calls: int = 0


def verified_runs(report_path: Path | None = None) -> list[tuple[str, int]]:
    """(scenario id, seed) of every seeded run whose fault behaved as labelled."""
    report = json.loads((report_path or REPORT).read_text(encoding="utf-8"))
    return [
        (row["scenario_id"], int(seed))
        for seed, rows in report["rows"].items()
        for row in rows
        if row["behaves"] and row["fault_code"] != "F0"
    ]


def target_for(run: Run, root: str) -> str | None:
    """The span to patch for a root node: its only span, else the step around its spans."""
    spans = [s for s in run.spans if s.node_id == root]
    if not spans:
        return None
    local = spans[0].span_id.removeprefix(f"{run.run_id}/")
    return local if len(spans) == 1 else root.split(".", 1)[0]


def make_patch(
    source: PatchSource,
    gateway: Gateway,
    scenario: Scenario,
    faulty: Run,
    target: str,
    roles: dict[str, str],
    role: str = PATCHER_ROLE,
) -> Patch | None:
    """The replacement for `target` from one source (None: the source has none)."""
    if source == "oracle":
        return oracle_patch(gateway, scenario, faulty, target, roles)
    if source == "rule":
        return rule_patch(faulty, target, make_resources(gateway))
    return model_patch(faulty, target, gateway, role=role)


def check_root(
    gateway: Gateway,
    scenario: Scenario,
    seed: int,
    faulty: Run,
    root: str,
    roles: dict[str, str],
    source: PatchSource = "oracle",
    role: str = PATCHER_ROLE,
) -> tuple[PatchCheck, Run | None, ReplayResult | None]:
    truth = scenario.ground_truth
    pc = PatchCheck(scenario.scenario_id, truth.fault_code, seed, root, source)
    target = target_for(faulty, root)
    if target is None:
        pc.verdict, pc.detail = "skipped", f"{root} never ran in the faulty run"
        return pc, None, None
    pc.target = target
    workflow = build_support_workflow(roles)
    try:
        patch = make_patch(source, gateway, scenario, faulty, target, roles, role)
        if patch is None:
            pc.verdict, pc.detail = "n/a", f"no {source} patch for {target}"
            return pc, None, None
        pc.note = patch.note
        plan = plan_patch(faulty, workflow, patch)
    except (PatchError, PatchSourceError) as exc:
        pc.verdict, pc.detail = "n/a", f"{source} patch failed: {exc}"
        return pc, None, None
    others = FaultPlan([f for f in scenario.faults if f.target_node_id != root])
    out = execute_plan(
        faulty,
        workflow,
        plan,
        make_resources(gateway),
        run_id=f"{faulty.run_id}-{source}-{root}",
        faults=others if others.faults else None,
    )
    problems = out.violations()
    if out.replay.patch_source != source:
        problems.append(f"replay tagged {out.replay.patch_source!r}, not {source!r}")
    pc.verdict = "VIOLATION" if problems else "descendants-only"
    pc.detail = "; ".join(problems)
    diff = diff_runs(faulty, out.replay)
    clean = clean_run(gateway, scenario, faulty, roles)
    rec = score_recovery(
        truth.expected_final, faulty.final_output, out.replay.final_output, clean.final_output
    )
    pc.recovered, pc.score, pc.before_score = rec.recovered, rec.recovery_score, rec.before_score
    pc.clean_correct, pc.matches_clean = rec.clean_correct, rec.matches_clean
    pc.changed = len(diff.changed_spans)
    pc.live_calls = out.live_model_calls
    return pc, out.replay, replay_result(out, diff, rec)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["oracle", "rule", "model"], default="oracle")
    parser.add_argument("--role", default=PATCHER_ROLE, help="model role for --source model")
    parser.add_argument("--code", nargs="*", help="only these fault codes")
    parser.add_argument("--only", nargs="*", help="only these scenario ids")
    parser.add_argument("--limit", type=int, help="at most this many runs (cost control)")
    parser.add_argument("--store", action="store_true", help="save the replays")
    parser.add_argument("-v", "--verbose", action="store_true", help="a line per replay")
    args = parser.parse_args(argv)
    source: PatchSource = args.source

    if not REPORT.exists():
        print(f"{REPORT.relative_to(ROOT)} not found: run `python tasks.py multirun` first")
        return 2
    scenarios = {s.scenario_id: s for s in load_dataset()}
    todo = [
        (sid, seed)
        for sid, seed in verified_runs()
        if sid in scenarios
        and (not args.code or scenarios[sid].ground_truth.fault_code in args.code)
        and (not args.only or sid in args.only)
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
    gateway = Gateway.from_env()
    roles = load_agents_config().profile(DEFAULT_PROFILE).roles()

    from agenttrace.store import TraceStore

    rows: list[PatchCheck] = []
    with TraceStore.open() as store:
        n_roots = sum(len(scenarios[s].ground_truth.root_cause_node_ids) for s, _ in todo)
        via = f" (role {args.role})" if source == "model" else ""
        print(
            f"Patch check, source {source}{via}: {len(todo)} verified seeded runs, "
            f"{n_roots} root-cause patches\n"
        )
        for sid, seed in todo:
            run_id = f"{sid}-{DEFAULT_PROFILE}-s{seed}"
            if not store.has_run(run_id):
                rows.append(PatchCheck(sid, "?", seed, "-", verdict="skipped", detail="not stored"))
                continue
            faulty = store.get_run(run_id)
            for root in scenarios[sid].ground_truth.root_cause_node_ids:
                pc, replay, result = check_root(
                    gateway, scenarios[sid], seed, faulty, root, roles, source, args.role
                )
                rows.append(pc)
                if args.store and replay is not None and result is not None:
                    store.save_run(replay, replace=True)
                    store.add_replay(result)
                if args.verbose or pc.verdict == "VIOLATION":
                    rec = {True: "recovered", False: "not recovered", None: "-"}[pc.recovered]
                    print(
                        f"  {pc.verdict:<16} {run_id:<20} patch {pc.target or root:<24} "
                        f"{pc.changed:>2} changed  {rec}  {pc.detail or pc.note}",
                        flush=True,
                    )
                elif len(rows) % 25 == 0:
                    print(f"  {len(rows)} patches checked", flush=True)
    gateway.close()

    judged = [r for r in rows if r.verdict in ("descendants-only", "VIOLATION")]
    ok = sum(r.verdict == "descendants-only" for r in judged)
    na = [r for r in rows if r.verdict == "n/a"]
    print(f"\nonly the target and its descendants changed: {ok}/{len(judged)} replays")
    print(f"no {source} patch (n/a): {len(na)}")
    skipped = Counter(r.detail for r in rows if r.verdict == "skipped")
    if skipped:
        print(
            f"skipped {sum(skipped.values())}: "
            + "; ".join(f"{n}x {d}" for d, n in skipped.items())
        )
    print(f"\nby fault code ({source} patch of one root at a time):")
    print(
        f"  {'code':<5} {'patches':>8} {'n/a':>5} {'replayed':>9} {'descendants-only':>17} "
        f"{'final correct after':>20}"
    )
    fixed: list[str] = []
    for code in sorted({r.fault_code for r in rows if r.verdict != "skipped"}):
        group = [r for r in rows if r.fault_code == code and r.verdict != "skipped"]
        done = [r for r in group if r in judged]
        good = sum(r.verdict == "descendants-only" for r in done)
        n_rec = sum(bool(r.recovered) for r in done)
        n_na = sum(r.verdict == "n/a" for r in group)
        if n_rec:
            fixed.append(code)
        print(
            f"  {code:<5} {len(group):>8} {n_na:>5} {len(done):>9} {good:>17} "
            f"{n_rec:>14}/{len(done)}"
        )

    # Recovery (P5-T5), scored against the clean run of the same seed: a patch can only
    # recover what the workflow gets right without the fault.
    print(f"\nrecovery ({source}; recoverable = the seed's clean run is correct):")
    print(
        f"  {'code':<5} {'recovered':>10} {'recoverable':>12} {'rec./recoverable':>17} "
        f"{'matches clean':>14} {'score before -> after':>22}"
    )
    for code in sorted({r.fault_code for r in judged}):
        done = [r for r in judged if r.fault_code == code]
        able = [r for r in done if r.clean_correct]
        rec_able = sum(bool(r.recovered) for r in able)
        same = sum(bool(r.matches_clean) for r in done)
        before = sum(r.before_score or 0 for r in done) / len(done)
        after = sum(r.score or 0 for r in done) / len(done)
        print(
            f"  {code:<5} {sum(bool(r.recovered) for r in done):>10} {len(able):>12} "
            f"{rec_able:>11}/{len(able):<5} {same:>9}/{len(done):<4} "
            f"{before:>12.2f} -> {after:.2f}"
        )
    able_all = [r for r in judged if r.clean_correct]
    if able_all:
        rate = sum(bool(r.recovered) for r in able_all) / len(able_all)
        print(f"  all   recovered {rate:.1%} of recoverable replays ({len(able_all)})")
    live = sum(r.live_calls for r in judged)
    print(f"model calls not served from the record: {live} (gateway cache or live)")
    print(f"{source} patches recovered the final outcome on: {', '.join(fixed) or 'none'}")
    reasons = Counter(r.detail.split(":")[0] for r in na)
    if reasons:
        print("n/a reasons: " + "; ".join(f"{n}x {d}" for d, n in reasons.most_common(5)))
    passed = bool(judged) and ok == len(judged)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
