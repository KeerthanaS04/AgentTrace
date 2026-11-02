"""Run the support workflow on the 20 clean tickets and report decision accuracy.

    python tasks.py clean                            # profile "default" (config/agents.yaml)
    python tasks.py clean --profile bolt             # another model profile
    python tasks.py clean --profile default bolt -q  # compare profiles side by side
    python tasks.py clean --only clean-03 -v         # one ticket, with the full reasoning
    python tasks.py clean --only clean-03 --trace    # one ticket's span tree (P2-T1)

Which model each agent uses comes from config/agents.yaml (profile -> step -> role) and
config/models.yaml (role -> model), never from code. Prints one row per ticket, then
accuracy and PASS/FAIL against the 90% target; with several profiles, a comparison
table. Decision accuracy is the P1-T5 target; "final" is the stricter P1-T7 outcome check
(right decision AND the reply states every required fact). Full step records and each
ticket's traced `Run` go to reports/clean_eval_<profile>.json. Repeat runs are served
from the gateway cache, so they are free and identical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from agenttrace.evaluation.outcome import check_outcome
from agenttrace.gateway.gateway import Gateway
from agenttrace.schemas.ground_truth import Scenario
from agenttrace.tracing.tree import format_tree
from agenttrace.workflow.agent_config import DEFAULT_PROFILE, load_agents_config
from agenttrace.workflow.contracts import run_checks
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.provenance import check_provenance
from agenttrace.workflow.runtime import WorkflowResult
from agenttrace.workflow.support import build_support_workflow, make_resources
from agenttrace.workflow.tools import load_db

ROOT = Path(__file__).resolve().parents[2]
CLEAN_TICKETS = ROOT / "data" / "eval" / "clean_tickets.yaml"
REPORTS_DIR = ROOT / "reports"
TARGET = 0.9


def load_scenarios(path: Path = CLEAN_TICKETS) -> list[Scenario]:
    return [Scenario.model_validate(s) for s in yaml.safe_load(path.read_text(encoding="utf-8"))]


def violations(result: WorkflowResult) -> list[str]:
    out: list[str] = []
    for rec in result.steps:
        if rec.status == "ok":
            out += [f"{rec.step}.{v}" for v in run_checks(rec.step, rec.inputs, rec.output)]
        else:
            out.append(f"{rec.step}: {rec.status} ({rec.error})")
    return out


def explain(result: WorkflowResult) -> str:
    """What each step saw and decided, for reading a miss."""
    tools = result.step("tool_agent").output.get("tool_calls", [])
    docs = result.step("retriever").output.get("docs", [])
    policy = result.step("policy")
    return "\n".join(
        [
            f"    tools : {[(c['tool'], c['args']) for c in tools]}",
            f"    docs  : {[d['doc_id'] for d in docs]}",
            f"    policy: {policy.output.get('justification') or policy.error}",
            f"    reply : {result.outputs.get('reply', '-')}\n",
        ]
    )


def run_one(
    gateway: Gateway, scenario: Scenario, roles: dict[str, str], profile: str
) -> tuple[WorkflowResult, float]:
    kb = load_kb()
    workflow = build_support_workflow(roles)
    resources = make_resources(gateway, db=load_db(), kb=kb, index=BM25Index.from_kb(kb))
    # The workflow sees only the ticket: never the category label or ground truth.
    ticket = scenario.to_input().ticket.model_dump(mode="json")
    started = time.perf_counter()
    result = workflow.run(
        {"ticket": ticket}, run_id=f"{scenario.scenario_id}-{profile}", resources=resources
    )
    return result, time.perf_counter() - started


def evaluate_profile(
    gateway: Gateway,
    scenarios: list[Scenario],
    profile: str,
    roles: dict[str, str],
    *,
    verbose: bool = False,
    quiet: bool = False,
    trace: bool = False,
) -> dict[str, Any]:
    """Run every scenario on one profile; print rows unless quiet; return the summary."""
    if not quiet:
        print(
            f"{'scenario':<10} {'expected':<9} {'got':<9} {'ok':<3} {'final':<6}"
            f"{'category':<17} {'secs':>5}  contract violations / reply problems"
        )
    rows: list[dict[str, Any]] = []
    workflow = build_support_workflow(roles)
    for sc in scenarios:
        result, secs = run_one(gateway, sc, roles, profile)
        expected = sc.ground_truth.expected_final.decision
        got = result.outputs.get("decision", "-")
        cat = result.outputs.get("category", "-")
        found = violations(result)
        outcome = check_outcome(sc.ground_truth.expected_final, result.outputs)
        ok = outcome.decision_ok
        run = result.to_run(scenario_id=sc.scenario_id, final_correct=outcome.correct)
        provenance = check_provenance(run, workflow)
        if not quiet:
            cat_text = cat if cat == sc.category else f"{cat}(want {sc.category})"
            problems = list(found)
            if outcome.facts_missing:
                problems.append(f"reply lacks {outcome.facts_missing}")
            problems += [f"provenance: {p}" for p in provenance]
            print(
                f"{sc.scenario_id:<10} {expected:<9} {got:<9} {'Y' if ok else 'N':<3} "
                f"{'Y' if outcome.correct else 'N':<6}{cat_text:<17} {secs:5.1f}  "
                f"{'; '.join(problems) if problems else '-'}"
            )
            if verbose and (not ok or len(scenarios) == 1):
                print(explain(result))
        if trace:
            print(format_tree(run) + "\n")
        rows.append(
            {
                "scenario_id": sc.scenario_id,
                "expected": expected,
                "got": got,
                "correct": ok,
                "final_correct": outcome.correct,
                "facts_missing": outcome.facts_missing,
                "category_expected": sc.category,
                "category_got": cat,
                "completed": result.status == "ok",
                "seconds": round(secs, 2),
                "violations": found,
                "provenance_problems": provenance,
                "result": result.model_dump(mode="json", exclude={"spans"}),
                "run": run.model_dump(mode="json"),  # the full trace (P2-T1)
            }
        )
    n = len(rows)
    summary = {
        "profile": profile,
        "roles": roles,
        "models": {step: gateway.resolve(role).model for step, role in roles.items()},
        "n": n,
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "final_correct": sum(r["final_correct"] for r in rows),
        "category_correct": sum(r["category_expected"] == r["category_got"] for r in rows),
        "completed": sum(r["completed"] for r in rows),
        "with_violations": sum(bool(r["violations"]) for r in rows),
        "provenance_exact": sum(not r["provenance_problems"] for r in rows),
        "seconds": round(sum(r["seconds"] for r in rows), 1),
        "rows": rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"clean_eval_{profile}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary


def print_summary(s: dict[str, Any]) -> None:
    n = s["n"]
    print(f"\ndecision accuracy : {s['correct']}/{n} = {s['accuracy']:.0%}  (target {TARGET:.0%})")
    print(f"final outcome     : {s['final_correct']}/{n}  (right decision + required facts)")
    print(f"router category   : {s['category_correct']}/{n}")
    print(f"runs completed    : {s['completed']}/{n}  (no step failed)")
    print(f"tickets with contract violations: {s['with_violations']}/{n}")
    print(f"provenance exact  : {s['provenance_exact']}/{n}  (refs match declared data flow)")
    print(f"full step records : reports/clean_eval_{s['profile']}.json")
    print("PASS" if s["accuracy"] >= TARGET else "FAIL")


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    print("\n=== Profile comparison ===")
    head = ("profile", "decision", "final", "router", "completed", "violations", "secs")
    print(f"{head[0]:<10}" + "".join(f"{h:>12}" for h in head[1:]) + "  models")
    for s in summaries:
        n = s["n"]
        stats = (
            f"{s['correct']}/{n} {s['accuracy']:.0%}",
            f"{s['final_correct']}/{n}",
            f"{s['category_correct']}/{n}",
            f"{s['completed']}/{n}",
            f"{s['with_violations']}/{n}",
            f"{s['seconds']:.1f}",
        )
        models = ", ".join(sorted(set(s["models"].values())))
        print(f"{s['profile']:<10}" + "".join(f"{c:>12}" for c in stats) + f"  {models}")
    ids = [r["scenario_id"] for r in summaries[0]["rows"]]
    print("\nx = wrong decision, ~ = right decision but the reply lacks a required fact")
    print(f"{'scenario':<10} {'expected':<9} " + " ".join(f"{s['profile']:<11}" for s in summaries))
    for i, sid in enumerate(ids):
        expected = summaries[0]["rows"][i]["expected"]
        cells = []
        for s in summaries:
            row = s["rows"][i]
            mark = "" if row["final_correct"] else " ~" if row["correct"] else " x"
            cells.append(f"{row['got'] + mark:<11}")
        print(f"{sid:<10} {expected:<9} " + " ".join(cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile", nargs="+", default=[DEFAULT_PROFILE], help="profile(s) in config/agents.yaml"
    )
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="print reasoning for misses")
    parser.add_argument("-q", "--quiet", action="store_true", help="skip per-ticket rows")
    parser.add_argument("-t", "--trace", action="store_true", help="print each ticket's span tree")
    args = parser.parse_args(argv)

    scenarios = load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s.scenario_id in set(args.only)]
    gateway = Gateway.from_env()
    agents = load_agents_config()
    agents.check_roles(gateway.config)

    summaries: list[dict[str, Any]] = []
    for name in args.profile:
        profile = agents.profile(name)
        roles = profile.roles()
        print(f"\n### Profile {name!r}: {profile.description}")
        for step, role in roles.items():
            rc = gateway.resolve(role)
            print(f"    {step:<11} -> {role:<14} {rc.provider}: {rc.model}")
        print(f"Clean-ticket eval: {len(scenarios)} tickets\n")
        summary = evaluate_profile(
            gateway,
            scenarios,
            name,
            roles,
            verbose=args.verbose,
            quiet=args.quiet,
            trace=args.trace,
        )
        print_summary(summary)
        summaries.append(summary)
    gateway.close()

    if len(summaries) > 1:
        print_comparison(summaries)
    return 0 if all(s["accuracy"] >= TARGET for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
