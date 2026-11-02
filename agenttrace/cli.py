"""Command-line entry point: `agenttrace run|runs|show|diagnose|replay|score|eval|serve`.

    agenttrace run --scenario clean-07            # run one scenario, store its trace
    agenttrace run --scenario clean-07 --trace    # ... and print the span tree
    agenttrace runs [--scenario clean-07]         # list stored runs
    agenttrace show clean-07-default-r1           # reload a stored run, print its tree
    agenttrace show clean-07-default-r1 --node policy   # one span, as JSON

The store is SQLite at .cache/agenttrace.db (override with --db or $AGENTTRACE_DB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agenttrace import __version__

# Subcommands that print a notice until their task lands.
NOT_YET: dict[str, str] = {
    "diagnose": "P7-T4",
    "replay": "P5-T3",
    "score": "P8-T7",
    "eval": "P9-T7",
    "serve": "P10-T1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenttrace")
    parser.add_argument("--version", action="version", version=f"agenttrace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_db(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--db", type=Path, help="SQLite file (default .cache/agenttrace.db)")
        return p

    run = with_db(sub.add_parser("run", help="run a scenario and store its trace"))
    run.add_argument("--scenario", required=True, help="scenario id, e.g. clean-07")
    run.add_argument("--profile", default="default", help="model profile in config/agents.yaml")
    run.add_argument("--trace", action="store_true", help="print the span tree")

    runs = with_db(sub.add_parser("runs", help="list stored runs"))
    runs.add_argument("--scenario", help="only runs of this scenario")

    show = with_db(sub.add_parser("show", help="reload a stored run and print it"))
    show.add_argument("run_id")
    show.add_argument("--node", help="print this node's span(s) as JSON instead of the tree")

    for name in NOT_YET:
        sub.add_parser(name)
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    from agenttrace.evaluation.outcome import check_outcome
    from agenttrace.gateway.gateway import Gateway
    from agenttrace.store import TraceStore
    from agenttrace.tracing.tree import format_tree
    from agenttrace.workflow.agent_config import load_agents_config
    from agenttrace.workflow.clean_eval import load_scenarios, run_one
    from agenttrace.workflow.provenance import check_provenance
    from agenttrace.workflow.support import build_support_workflow

    scenarios = {s.scenario_id: s for s in load_scenarios()}
    if args.scenario not in scenarios:
        print(f"unknown scenario {args.scenario!r}; known: {', '.join(sorted(scenarios))}")
        return 2
    scenario = scenarios[args.scenario]
    gateway = Gateway.from_env()
    agents = load_agents_config()
    agents.check_roles(gateway.config)
    roles = agents.profile(args.profile).roles()

    with TraceStore.open(args.db) as store:
        run_id = store.next_run_id(f"{scenario.scenario_id}-{args.profile}")
        result, secs = run_one(gateway, scenario, roles, args.profile, run_id=run_id)
        gateway.close()
        outcome = check_outcome(scenario.ground_truth.expected_final, result.outputs)
        run = result.to_run(scenario_id=scenario.scenario_id, final_correct=outcome.correct)
        provenance = check_provenance(run, build_support_workflow(roles))
        store.save_run(run)
        url = store.url

    if args.trace:
        print(format_tree(run) + "\n")
    print(f"run       : {run.run_id}  ({len(run.spans)} spans, {secs:.1f}s)")
    print(
        f"decision  : {result.outputs.get('decision', '-')}  (expected {outcome.expected_decision})"
    )
    print(f"final     : {'correct' if outcome.correct else 'wrong: ' + outcome.reason()}")
    print(f"provenance: {'exact' if not provenance else '; '.join(provenance)}")
    print(f"stored in : {url}")
    return 0 if result.status == "ok" else 1


def cmd_runs(args: argparse.Namespace) -> int:
    from agenttrace.store import TraceStore

    with TraceStore.open(args.db) as store:
        rows = store.summaries(scenario_id=args.scenario)
    if not rows:
        print("no stored runs")
        return 0
    print(f"{'run_id':<28} {'scenario':<10} {'final':<6} {'spans':>5}  created (UTC)")
    for r in rows:
        final = {True: "Y", False: "N", None: "-"}[r["final_correct"]]
        print(
            f"{r['run_id']:<28} {r['scenario_id'] or '-':<10} {final:<6} {r['spans']:>5}  "
            f"{r['created_at'][:19]}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from agenttrace.store import TraceStore
    from agenttrace.tracing.tree import format_tree

    with TraceStore.open(args.db) as store:
        if not store.has_run(args.run_id):
            print(f"no stored run {args.run_id!r} (see `agenttrace runs`)")
            return 2
        if args.node:
            spans = store.spans(run_id=args.run_id, node_id=args.node)
            if not spans:
                print(f"run {args.run_id!r} has no node {args.node!r}")
                return 2
            for span in spans:
                print(span.model_dump_json(indent=2))
            return 0
        run = store.get_run(args.run_id)
    correct = {True: "correct", False: "wrong", None: "not graded"}[run.final_correct]
    print(f"run {run.run_id}  scenario={run.scenario_id}  {run.workflow_version}  final {correct}")
    print(format_tree(run))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in NOT_YET:
        print(f"`agenttrace {args.command}` is not implemented yet ({NOT_YET[args.command]}).")
        return 1
    handler = {"run": cmd_run, "runs": cmd_runs, "show": cmd_show}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
