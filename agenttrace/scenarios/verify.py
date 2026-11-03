"""Scenario verification (P4-T4): does every fault do what its label says?

    python tasks.py verify                       # all scenarios, default model profile
    python tasks.py verify --only sc-003 -v      # one scenario, with details
    python tasks.py verify --store               # also save each scenario's run (faulty,
                                                 # or the control); a full run drops stale ones

For each scenario the ticket is run twice with the same models: clean (no fault; from
the gateway cache after the first time) and with the scenario's fault plan. A scenario
"behaves as labelled" when:

    F0       no fault is applied and the clean decision is correct
    F1..F7   every fault is applied, the root span's output changes, and the final
             outcome changes and is wrong
    F8       every fault is applied and the root span's output changes, but the final
             outcome is identical to the clean run

The outcome compared is (run completed, decision, required facts missing), so a fault
that only rewords the reply does not count as an effect (findings §6, P3-T3). Prints one
row per scenario, a per-code summary, and PASS/FAIL against TARGET; full details go to
reports/scenario_verification.json.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenttrace.evaluation.outcome import OutcomeCheck, check_outcome
from agenttrace.faults import FaultPlan
from agenttrace.gateway.gateway import Gateway
from agenttrace.graph import align_runs
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.schemas.run import Run
from agenttrace.tracing.context import InjectionError
from agenttrace.workflow.agent_config import DEFAULT_PROFILE, load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

if TYPE_CHECKING:
    from agenttrace.store import TraceStore

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "scenario_verification.json"
TARGET = 0.95

Outcome = tuple[bool, str | None, tuple[str, ...]]


@dataclass
class Verification:
    scenario_id: str
    fault_code: str
    ticket_id: str
    behaves: bool
    reason: str
    applied: list[str] = field(default_factory=list)  # span ids the faults hit
    root_changed: bool = False
    clean_outcome: Outcome | None = None
    faulty_outcome: Outcome | None = None
    final_correct: bool | None = None
    changed_steps: list[str] = field(default_factory=list)
    labelled_affected: list[str] = field(default_factory=list)
    faulty_run_id: str | None = None  # the stored run: faulty, or for F0 the control
    # A model call failed at the provider (rate limit, outage, budget): the run says
    # nothing about the scenario, so it is not judged, cached, or stored. Rerun it.
    provider_error: str | None = None


# Gateway errors that are about the infrastructure, not the model's behaviour.
PROVIDER_ERRORS = ("ProviderRequestError", "ProviderUnavailableError", "BudgetExceededError")


def provider_error(run: Run) -> str | None:
    """The first provider failure recorded in a run's spans, if any."""
    for span in run.spans:
        if span.error and span.error.split(":", 1)[0] in PROVIDER_ERRORS:
            return span.error
    return None


def outcome_of(run: Run, check: OutcomeCheck) -> Outcome:
    completed = all(s.error is None for s in run.spans if s.parent_span_id is None)
    return (completed, check.got_decision, tuple(sorted(check.facts_missing)))


def run_tag(model_params: Mapping[str, Any] | None) -> str:
    """Run-id suffix for repeated runs: "" for the plain run, "-s2" for seed 2, and
    "-s2-t0.7" when the temperature is overridden too."""
    params = dict(model_params or {})
    tag = f"-s{params['seed']}" if "seed" in params else ""
    if "temperature" in params:
        tag += f"-t{params['temperature']:g}"
    return tag


def _run(
    gateway: Gateway,
    scenario: Scenario,
    roles: dict[str, str],
    run_id: str,
    plan: FaultPlan | None,
    model_params: Mapping[str, Any] | None = None,
) -> tuple[Run, OutcomeCheck]:
    workflow = build_support_workflow(roles)
    ticket = scenario.to_input().ticket.model_dump(mode="json")
    result = workflow.run(
        {"ticket": ticket},
        run_id=run_id,
        resources=make_resources(gateway),
        faults=plan,
        model_params=model_params,
    )
    check = check_outcome(scenario.ground_truth.expected_final, result.outputs)
    run = result.to_run(scenario_id=scenario.scenario_id, final_correct=check.correct)
    return run, check


def verify_scenario(
    gateway: Gateway,
    scenario: Scenario,
    roles: dict[str, str],
    profile: str,
    clean_cache: dict[str, tuple[Run, OutcomeCheck]],
    model_params: Mapping[str, Any] | None = None,
) -> tuple[Verification, Run | None]:
    """Run one scenario clean and faulty, and judge it against its label.

    Returns the scenario's own run for storing: the faulty run, or for a control (F0) the
    control run itself. Both runs use the same `model_params`, so a seeded repeat compares
    like with like.
    """
    truth = scenario.ground_truth
    sid, code, ticket_id = scenario.scenario_id, truth.fault_code, scenario.ticket.ticket_id
    tag = run_tag(model_params)
    clean_key = f"{ticket_id}{tag}"
    v = Verification(
        sid,
        code,
        ticket_id,
        behaves=False,
        reason="",
        labelled_affected=list(truth.affected_node_ids),
    )
    if clean_key not in clean_cache:
        clean_run = _run(
            gateway, scenario, roles, f"{ticket_id}-{profile}-clean{tag}", None, model_params
        )
        if (err := provider_error(clean_run[0])) is not None:
            return _provider_failed(v, err, "clean"), None  # not cached: retried next time
        clean_cache[clean_key] = clean_run
    clean, clean_check = clean_cache[clean_key]
    v.clean_outcome = outcome_of(clean, clean_check)
    if code == "F0":
        # Same calls as the clean run, so served from the cache; stored under the
        # control's own id so diagnosis can be scored on controls (false positives).
        control, check = _run(gateway, scenario, roles, f"{sid}-{profile}{tag}", None, model_params)
        if (err := provider_error(control)) is not None:
            return _provider_failed(v, err, "control"), None
        v.faulty_run_id = control.run_id
        v.final_correct = check.correct
        v.behaves = check.decision_ok
        v.reason = "ok" if v.behaves else f"control is wrong: {check.reason()}"
        return v, control

    plan = FaultPlan(scenario.faults)
    try:
        faulty, check = _run(gateway, scenario, roles, f"{sid}-{profile}{tag}", plan, model_params)
    except InjectionError as exc:
        v.reason = f"fault error: {exc}"
        return v, None
    if (err := provider_error(faulty)) is not None:
        return _provider_failed(v, err, "faulty"), None
    v.faulty_run_id = faulty.run_id
    v.applied = [a.span_id.removeprefix(f"{faulty.run_id}/") for a in plan.applied]
    v.faulty_outcome = outcome_of(faulty, check)
    v.final_correct = check.correct

    alignment = align_runs(clean, faulty)
    roots = set(truth.root_cause_node_ids)
    changed_roots = {p.node_id for p in alignment.pairs if p.status == "changed"} & roots
    v.root_changed = changed_roots == roots
    v.changed_steps = sorted(
        {p.node_id for p in alignment.pairs if p.status != "unchanged" and "." not in p.node_id}
    )

    if plan.unapplied():
        missing = ", ".join(f.target_node_id for f in plan.unapplied())
        v.reason = f"fault not applied (target never ran: {missing})"
    elif not v.root_changed:
        v.reason = "root output unchanged (the fault had nothing to change)"
    elif code == "F8":
        same = v.faulty_outcome == v.clean_outcome
        v.behaves = same
        v.reason = "ok (outcome unchanged)" if same else "silent fault changed the outcome"
    else:
        moved = v.faulty_outcome != v.clean_outcome
        v.behaves = moved and not check.correct
        if v.behaves:
            v.reason = "ok"
        elif not moved:
            v.reason = "no effect on outcome"
        else:
            v.reason = "outcome changed but is still correct"
    return v, faulty


def _provider_failed(v: Verification, error: str, which: str) -> Verification:
    v.provider_error = error
    v.reason = f"provider error in the {which} run (not judged; rerun): {error[:120]}"
    return v


def judged(rows: list[Verification]) -> list[Verification]:
    """Rows with a verdict: those not cut short by a provider error."""
    return [r for r in rows if r.provider_error is None]


def summarize(rows: list[Verification]) -> dict[str, Any]:
    failed = [r.scenario_id for r in rows if r.provider_error is not None]
    rows = judged(rows)
    by_code: dict[str, dict[str, int]] = {}
    for code in sorted({r.fault_code for r in rows}):
        group = [r for r in rows if r.fault_code == code]
        by_code[code] = {"n": len(group), "behaves": sum(r.behaves for r in group)}
    n = len(rows)
    ok = sum(r.behaves for r in rows)
    return {
        "n": n,
        "behaves": ok,
        "rate": ok / n if n else 0.0,
        "target": TARGET,
        "by_code": by_code,
        "reasons": dict(Counter(r.reason for r in rows if not r.behaves)),
        "provider_errors": failed,  # not judged: the rate covers the other n rows
    }


def prune_stale(store: TraceStore, current: set[str | None], profile: str, tag: str = "") -> None:
    """Delete stored scenario runs (of this profile and run tag) a full pass did not produce.

    Editing the dataset reshuffles scenario ids, so a run saved under an old id would
    claim the wrong scenario (and the wrong ground truth) in later diagnosis.
    """
    pattern = re.compile(rf"sc-\d+-{re.escape(profile)}{re.escape(tag)}")
    for run_id in store.run_ids():
        if pattern.fullmatch(run_id) and run_id not in current:
            store.delete_run(run_id)


def _fmt(o: Outcome | None) -> str:
    if o is None:
        return "-"
    completed, decision, missing = o
    text = decision or "none"
    if not completed:
        text += " (failed)"
    return text + (f" lacks {list(missing)}" if missing else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="profile in config/agents.yaml")
    parser.add_argument("--only", nargs="*", help="scenario ids to verify")
    parser.add_argument("--code", nargs="*", help="only these fault codes, e.g. F3 F8")
    parser.add_argument("--store", action="store_true", help="save scenario runs to the store")
    parser.add_argument("-v", "--verbose", action="store_true", help="details for every row")
    args = parser.parse_args(argv)

    scenarios = load_dataset()
    if args.only:
        scenarios = [s for s in scenarios if s.scenario_id in set(args.only)]
    if args.code:
        scenarios = [s for s in scenarios if s.ground_truth.fault_code in set(args.code)]
    gateway = Gateway.from_env()
    agents = load_agents_config()
    agents.check_roles(gateway.config)
    roles = agents.profile(args.profile).roles()

    store = None
    if args.store:
        from agenttrace.store import TraceStore

        store = TraceStore.open()

    print(f"Scenario verification: {len(scenarios)} scenarios, profile {args.profile!r}\n")
    print(
        f"{'scenario':<9} {'code':<5} {'ticket':<7} {'ok':<3} {'clean':<22} {'faulty':<28} reason"
    )
    rows: list[Verification] = []
    clean_cache: dict[str, tuple[Run, OutcomeCheck]] = {}
    for sc in scenarios:
        v, faulty = verify_scenario(gateway, sc, roles, args.profile, clean_cache)
        rows.append(v)
        print(
            f"{v.scenario_id:<9} {v.fault_code:<5} {v.ticket_id:<7} {'Y' if v.behaves else 'N':<3} "
            f"{_fmt(v.clean_outcome):<22} {_fmt(v.faulty_outcome):<28} {v.reason}",
            flush=True,
        )
        if args.verbose and v.fault_code != "F0":
            print(f"          applied at {v.applied}; root changed: {v.root_changed}")
            print(f"          steps changed {v.changed_steps}; labelled {v.labelled_affected}")
        if store is not None and faulty is not None:
            # Stable id `{scenario}-{profile}`: re-verifying replaces the stored run.
            store.save_run(faulty, replace=True)
    gateway.close()
    if store is not None:
        if not (args.only or args.code):
            prune_stale(store, {r.faulty_run_id for r in rows}, args.profile)
        store.close()

    s = summarize(rows)
    print("\nby fault code:")
    for code, c in s["by_code"].items():
        print(f"  {code}: {c['behaves']}/{c['n']}")
    if s["reasons"]:
        print("not as labelled, by reason:")
        for reason, count in sorted(s["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {reason}")
    print(f"\nbehave as labelled: {s['behaves']}/{s['n']} = {s['rate']:.0%}  (target {TARGET:.0%})")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"profile": args.profile, **s, "rows": [asdict(r) for r in rows]}, indent=1),
        encoding="utf-8",
    )
    print(f"full report: {REPORT_PATH.relative_to(ROOT).as_posix()}")
    if s["provider_errors"]:
        print(
            f"INCOMPLETE: {len(s['provider_errors'])} scenarios hit provider errors and were "
            "not judged; rerun the same command (finished calls come from the cache)"
        )
        return 1
    print("PASS" if s["rate"] >= TARGET else "FAIL")
    return 0 if s["rate"] >= TARGET else 1


if __name__ == "__main__":
    raise SystemExit(main())
