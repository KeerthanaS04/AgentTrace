"""Patch check (P5-T3): patching a known-bad node changes only its descendants.

    python tasks.py patch-check                 # every verified seeded run (P4-T5)
    python tasks.py patch-check --code F3 -v    # one fault type, a line per replay
    python tasks.py patch-check --store         # also save the replays (replay_of set)

Runs: the stored seeded runs `sc-xxx-default-s{k}` whose fault behaved as labelled in
`reports/multirun_default.json` (run `python tasks.py multirun` first). For each root
cause of the scenario:

    target       the root node's span; when the fault hit several spans of that node
                 (a retrieval fault hits every search), the step that contains them
    replacement  what that span produced in the clean run of the same seed (an oracle
                 patch, recomputed through the gateway cache)
    other faults kept in force (F7's second root), so the counterfactual is faithful

Each replay must change only the target (patched) and its descendants: frozen spans
unchanged, and anything new or missing inside a re-run step. Also reported, as context
for P5-T5: whether the final outcome is correct after the patch, and how many model
calls the replay could not serve from the record or the gateway cache.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agenttrace.evaluation.outcome import check_outcome
from agenttrace.faults import FaultPlan
from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.executor import execute_plan
from agenttrace.replay.patcher import Patch, PatchError, plan_patch
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.verify import _run
from agenttrace.schemas.run import Run
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
    target: str = ""
    verdict: str = ""  # descendants-only / VIOLATION / skipped
    detail: str = ""
    recovered: bool | None = None
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


def check_root(
    gateway: Gateway, scenario: Scenario, seed: int, faulty: Run, root: str, roles: dict[str, str]
) -> tuple[PatchCheck, Run | None]:
    truth = scenario.ground_truth
    pc = PatchCheck(scenario.scenario_id, truth.fault_code, seed, root)
    target = target_for(faulty, root)
    if target is None:
        pc.verdict, pc.detail = "skipped", f"{root} never ran in the faulty run"
        return pc, None
    pc.target = target
    params = {"seed": seed}
    clean, _ = _run(
        gateway, scenario, roles, f"{scenario.ticket.ticket_id}-default-clean-s{seed}", None, params
    )
    clean_span = next((s for s in clean.spans if s.span_id == f"{clean.run_id}/{target}"), None)
    if clean_span is None or clean_span.error is not None:
        pc.verdict, pc.detail = "skipped", f"no clean output for {target}"
        return pc, None
    workflow = build_support_workflow(roles)
    try:
        plan = plan_patch(
            faulty,
            workflow,
            Patch(target=target, output=clean_span.output, patch_source="oracle"),
        )
    except PatchError as exc:
        pc.verdict, pc.detail = "skipped", str(exc)
        return pc, None
    others = FaultPlan([f for f in scenario.faults if f.target_node_id != root])
    out = execute_plan(
        faulty,
        workflow,
        plan,
        make_resources(gateway),
        run_id=f"{faulty.run_id}-oracle-{root}",
        faults=others if others.faults else None,
    )
    problems = out.violations()
    pc.verdict = "VIOLATION" if problems else "descendants-only"
    pc.detail = "; ".join(problems)
    pc.recovered = check_outcome(truth.expected_final, out.replay.final_output).correct
    pc.changed = sum(p.status != "unchanged" for p in out.alignment.pairs)
    pc.live_calls = out.live_model_calls
    return pc, out.replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code", nargs="*", help="only these fault codes")
    parser.add_argument("--only", nargs="*", help="only these scenario ids")
    parser.add_argument("--store", action="store_true", help="save the replays")
    parser.add_argument("-v", "--verbose", action="store_true", help="a line per replay")
    args = parser.parse_args(argv)

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
    gateway = Gateway.from_env()
    roles = load_agents_config().profile(DEFAULT_PROFILE).roles()

    from agenttrace.store import TraceStore

    rows: list[PatchCheck] = []
    with TraceStore.open() as store:
        n_roots = sum(len(scenarios[s].ground_truth.root_cause_node_ids) for s, _ in todo)
        print(f"Patch check: {len(todo)} verified seeded runs, {n_roots} root-cause patches\n")
        for sid, seed in todo:
            run_id = f"{sid}-{DEFAULT_PROFILE}-s{seed}"
            if not store.has_run(run_id):
                rows.append(PatchCheck(sid, "?", seed, "-", verdict="skipped", detail="not stored"))
                continue
            faulty = store.get_run(run_id)
            for root in scenarios[sid].ground_truth.root_cause_node_ids:
                pc, replay = check_root(gateway, scenarios[sid], seed, faulty, root, roles)
                rows.append(pc)
                if args.store and replay is not None:
                    store.save_run(replay, replace=True)
                if args.verbose or pc.verdict == "VIOLATION":
                    rec = {True: "recovered", False: "not recovered", None: "-"}[pc.recovered]
                    print(
                        f"  {pc.verdict:<16} {run_id:<20} patch {pc.target or root:<24} "
                        f"{pc.changed:>2} changed  {rec}  {pc.detail}",
                        flush=True,
                    )
                elif len(rows) % 25 == 0:
                    print(f"  {len(rows)} patches checked", flush=True)
    gateway.close()

    judged = [r for r in rows if r.verdict != "skipped"]
    ok = sum(r.verdict == "descendants-only" for r in judged)
    print(f"\nonly the target and its descendants changed: {ok}/{len(judged)} replays")
    skipped = Counter(r.detail for r in rows if r.verdict == "skipped")
    if skipped:
        print(
            f"skipped {sum(skipped.values())}: "
            + "; ".join(f"{n}x {d}" for d, n in skipped.items())
        )
    print("\nby fault code (oracle patch of one root at a time):")
    print(f"  {'code':<5} {'replays':>8} {'descendants-only':>17} {'final correct after':>20}")
    for code in sorted({r.fault_code for r in judged}):
        group = [r for r in judged if r.fault_code == code]
        good = sum(r.verdict == "descendants-only" for r in group)
        n_rec = sum(bool(r.recovered) for r in group)
        print(f"  {code:<5} {len(group):>8} {good:>17} {n_rec:>14}/{len(group)}")
    live = sum(r.live_calls for r in judged)
    print(f"model calls not served from the record: {live} (gateway cache or live)")
    print("PASS" if judged and ok == len(judged) else "FAIL")
    return 0 if judged and ok == len(judged) else 1


if __name__ == "__main__":
    raise SystemExit(main())
