"""Run the support workflow on the 20 clean tickets and report decision accuracy.

    python tasks.py clean                       # agent_default on every step
    python tasks.py clean --role agent_alt_2    # another model role on every step
    python tasks.py clean --only clean-03 -v    # one ticket, with the full reasoning

Prints one row per ticket (expected vs. got, contract violations, time), then accuracy
and PASS/FAIL against the 90% target. Full step records go to reports/clean_eval.json.
Repeat runs are served from the gateway cache, so they are free and identical.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from agenttrace.gateway.gateway import Gateway
from agenttrace.schemas.ground_truth import Scenario
from agenttrace.workflow.contracts import run_checks
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.runtime import WorkflowResult
from agenttrace.workflow.support import STEPS, build_support_workflow, make_resources
from agenttrace.workflow.tools import load_db

ROOT = Path(__file__).resolve().parents[2]
CLEAN_TICKETS = ROOT / "data" / "eval" / "clean_tickets.yaml"
REPORT_PATH = ROOT / "reports" / "clean_eval.json"
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


def run_one(gateway: Gateway, scenario: Scenario, role: str) -> tuple[WorkflowResult, float]:
    kb = load_kb()
    workflow = build_support_workflow({step: role for step in STEPS})
    resources = make_resources(gateway, db=load_db(), kb=kb, index=BM25Index.from_kb(kb))
    # The workflow sees only the ticket: never the category label or ground truth.
    ticket = scenario.to_input().ticket.model_dump(mode="json")
    started = time.perf_counter()
    result = workflow.run(
        {"ticket": ticket}, run_id=f"{scenario.scenario_id}-{role}", resources=resources
    )
    return result, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", default="agent_default", help="model role for every step")
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="print reasoning for misses")
    args = parser.parse_args(argv)

    scenarios = load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s.scenario_id in set(args.only)]
    gateway = Gateway.from_env()
    model = gateway.resolve(args.role)
    print(f"Clean-ticket eval: {len(scenarios)} tickets, role={args.role}")
    print(f"model: {model.model} (provider: {model.provider})\n")
    print(
        f"{'scenario':<10} {'expected':<9} {'got':<9} {'ok':<3} {'category':<17} {'secs':>5}"
        "  contract violations"
    )

    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        result, secs = run_one(gateway, sc, args.role)
        expected = sc.ground_truth.expected_final.decision
        got = result.outputs.get("decision", "-")
        cat = result.outputs.get("category", "-")
        cat_text = cat if cat == sc.category else f"{cat}(want {sc.category})"
        found = violations(result)
        ok = got == expected
        print(
            f"{sc.scenario_id:<10} {expected:<9} {got:<9} {'Y' if ok else 'N':<3} {cat_text:<17} "
            f"{secs:5.1f}  {'; '.join(found) if found else '-'}"
        )
        if args.verbose and (not ok or args.only):
            print(explain(result))
        rows.append(
            {
                "scenario_id": sc.scenario_id,
                "expected": expected,
                "got": got,
                "correct": ok,
                "category_expected": sc.category,
                "category_got": cat,
                "seconds": round(secs, 2),
                "violations": found,
                "result": result.model_dump(mode="json"),
            }
        )
    gateway.close()

    correct = sum(r["correct"] for r in rows)
    accuracy = correct / len(rows) if rows else 0.0
    cat_ok = sum(r["category_expected"] == r["category_got"] for r in rows)
    print(f"\ndecision accuracy : {correct}/{len(rows)} = {accuracy:.0%}  (target {TARGET:.0%})")
    print(f"router category   : {cat_ok}/{len(rows)}")
    print(
        f"tickets with contract violations: {sum(bool(r['violations']) for r in rows)}/{len(rows)}"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {"role": args.role, "model": model.model, "accuracy": accuracy, "rows": rows}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"full step records : {REPORT_PATH.relative_to(ROOT)}")
    passed = accuracy >= TARGET
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
