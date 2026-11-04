"""Command-line entry point: `agenttrace run|runs|show|export|import|diagnose|...`.

    agenttrace run --scenario clean-07            # run one scenario, store its trace
    agenttrace run --scenario clean-07 --trace    # ... and print the span tree
    agenttrace runs [--scenario clean-07]         # list stored runs
    agenttrace show clean-07-default-r1           # reload a stored run, print its tree
    agenttrace show clean-07-default-r1 --node policy   # one span, as JSON
    agenttrace export clean-07-default-r1 -o run.json   # one or more runs to a JSON file
    agenttrace export --scenario clean-07 --otel -o otel.json   # OTLP/JSON instead
    agenttrace import run.json                    # load runs (native or OTLP) into the store
    agenttrace graph clean-07-default-r1          # final spans, causal order, critical path
    agenttrace graph RUN --from tool_agent.order_lookup   # its paths to the final output
    agenttrace graph RUN_A --diff RUN_B           # where two runs first diverge
    agenttrace diff RUN_A RUN_B                   # each span: same / changed (how) / only in one
    agenttrace plan RUN policy --set decision='"deny"'   # replay plan for a patch: frozen /
                                                  # patched / re-run spans (P5-T2)

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

    export = with_db(sub.add_parser("export", help="write stored runs to a JSON file"))
    export.add_argument("run_ids", nargs="*", help="run ids (or use --scenario)")
    export.add_argument("--scenario", help="every stored run of this scenario")
    export.add_argument("-o", "--out", type=Path, required=True, help="output file")
    export.add_argument("--otel", action="store_true", help="OpenTelemetry OTLP/JSON format")

    imp = with_db(sub.add_parser("import", help="load runs from a JSON file into the store"))
    imp.add_argument("file", type=Path)
    imp.add_argument("--replace", action="store_true", help="overwrite runs already stored")

    graph = with_db(sub.add_parser("graph", help="causal-graph queries on a stored run"))
    graph.add_argument("run_id")
    graph.add_argument("--from", dest="source", help="span (local id) to trace to the final")
    graph.add_argument("--diff", help="another run: report where the two first diverge")
    graph.add_argument(
        "--weight", choices=["duration_s", "cost_usd", "tokens"], default="duration_s"
    )
    graph.add_argument("--json", type=Path, help="also write the UI graph JSON (React Flow)")

    diff = with_db(sub.add_parser("diff", help="align two stored runs span by span"))
    diff.add_argument("run_a")
    diff.add_argument("run_b")
    diff.add_argument("--all", action="store_true", help="also list unchanged spans")

    plan = with_db(sub.add_parser("plan", help="replay plan for patching one span"))
    plan.add_argument("run_id")
    plan.add_argument("target", help="span to patch (local id, e.g. policy, router.llm#2)")
    plan.add_argument(
        "--set",
        dest="fields",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="change one output field (VALUE is JSON, else a string); repeatable",
    )
    plan.add_argument("--output", type=Path, help="JSON file: the full replacement output")
    plan.add_argument("--profile", default="default", help="model profile for the workflow")

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


def cmd_export(args: argparse.Namespace) -> int:
    from agenttrace.store import TraceStore
    from agenttrace.tracing.export import export_runs

    with TraceStore.open(args.db) as store:
        ids = list(args.run_ids)
        if args.scenario:
            ids += [i for i in store.run_ids(scenario_id=args.scenario) if i not in ids]
        if not ids:
            print("nothing to export: give run ids or --scenario")
            return 2
        missing = [i for i in ids if not store.has_run(i)]
        if missing:
            print(f"no stored run(s): {', '.join(missing)}")
            return 2
        runs = [store.get_run(i) for i in ids]
    path = export_runs(runs, args.out, otel=args.otel)
    fmt = "OTLP/JSON" if args.otel else "agenttrace JSON"
    n_spans = sum(len(r.spans) for r in runs)
    print(f"exported {len(runs)} run(s), {n_spans} spans, as {fmt} to {path}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from agenttrace.store import TraceStore
    from agenttrace.tracing.export import TraceFormatError, import_runs

    try:
        runs = import_runs(args.file)
    except (OSError, TraceFormatError, ValueError) as exc:
        print(f"cannot import {args.file}: {exc}")
        return 2
    with TraceStore.open(args.db) as store:
        clashes = [r.run_id for r in runs if store.has_run(r.run_id)]
        if clashes and not args.replace:
            print(f"already stored (use --replace): {', '.join(clashes)}")
            return 2
        for run in runs:
            store.save_run(run, replace=args.replace)
    for run in runs:
        print(f"imported {run.run_id}  ({len(run.spans)} spans, {run.workflow_version})")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from agenttrace.graph import (
        build_graph,
        critical_path,
        final_spans,
        first_divergence,
        paths_to_final,
        topological_order,
    )
    from agenttrace.graph.ui import to_ui_graph
    from agenttrace.store import TraceStore
    from agenttrace.workflow.contracts import CONTRACTS

    with TraceStore.open(args.db) as store:
        for run_id in (args.run_id, args.diff):
            if run_id and not store.has_run(run_id):
                print(f"no stored run {run_id!r} (see `agenttrace runs`)")
                return 2
        run = store.get_run(args.run_id)
        g = build_graph(run, contracts=CONTRACTS)
        other = build_graph(store.get_run(args.diff)) if args.diff else None

    def local(span_id: str) -> str:
        return str(g.nodes[span_id]["local_id"])

    agents = [n for n in topological_order(g) if g.nodes[n]["parent"] is None]
    print(f"run {args.run_id}: {g.number_of_nodes()} spans, {g.number_of_edges()} edges")
    print(f"final spans   : {', '.join(local(n) for n in final_spans(g))}")
    print(f"causal order  : {' -> '.join(local(n) for n in agents)}  (top-level spans)")
    cp = critical_path(g, weight=args.weight)
    print(
        f"critical path : {' -> '.join(local(n) for n in cp.spans)}  ({args.weight} {cp.total:g})"
    )

    if args.source:
        by_local = {local(n): n for n in g}
        if args.source not in by_local:
            print(f"no span {args.source!r} in {args.run_id}")
            return 2
        paths = paths_to_final(g, by_local[args.source])
        print(f"\npaths from {args.source} to the final output: {len(paths)}")
        for path in paths:
            print("  " + " -> ".join(local(n) for n in path))
    if other is not None:
        d = first_divergence(g, other)
        print(f"\nfirst divergence vs {args.diff}: ", end="")
        print("none (same outputs everywhere)" if d is None else f"{d.local_id} ({d.reason})")
    if args.json:
        doc = to_ui_graph(g, run)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
        print(f"\nUI graph JSON: {len(doc.nodes)} nodes, {len(doc.edges)} edges -> {args.json}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from agenttrace.graph import AlignmentError, align_runs, first_divergence
    from agenttrace.graph import build_graph as graph_of
    from agenttrace.store import TraceStore

    with TraceStore.open(args.db) as store:
        for run_id in (args.run_a, args.run_b):
            if not store.has_run(run_id):
                print(f"no stored run {run_id!r} (see `agenttrace runs`)")
                return 2
        a, b = store.get_run(args.run_a), store.get_run(args.run_b)
    try:
        al = align_runs(a, b)
    except AlignmentError as exc:
        print(f"cannot align: {exc}")
        return 2
    print(f"{args.run_a}  vs  {args.run_b}")
    print(
        f"unchanged {len(al.unchanged)}, changed {len(al.changed)}, "
        f"only in A {len(al.only_a)}, only in B {len(al.only_b)}\n"
    )
    labels = {"changed": "CHANGED", "only_a": "only A", "only_b": "only B", "unchanged": "same"}
    for p in al.pairs:
        if p.status == "unchanged" and not args.all:
            continue
        what = f"  ({', '.join(p.changes)})" if p.changes else ""
        print(f"  {labels[p.status]:<8} {p.local_id}{what}")
    if al.identical:
        print("  every span matches")
    d = first_divergence(graph_of(a), graph_of(b))
    print(f"\nfirst divergence: {'none' if d is None else f'{d.local_id} ({d.reason})'}")
    return 0


def _field(text: str) -> tuple[str, object]:
    import json

    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise ValueError(f"--set needs KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def cmd_plan(args: argparse.Namespace) -> int:
    import json

    from agenttrace.replay.patcher import Patch, PatchError, patch_fields, plan_patch
    from agenttrace.store import TraceStore
    from agenttrace.workflow.agent_config import load_agents_config
    from agenttrace.workflow.support import build_support_workflow

    with TraceStore.open(args.db) as store:
        if not store.has_run(args.run_id):
            print(f"no stored run {args.run_id!r} (see `agenttrace runs`)")
            return 2
        run = store.get_run(args.run_id)
    workflow = build_support_workflow(load_agents_config().profile(args.profile).roles())
    try:
        if args.output:
            output = json.loads(args.output.read_text(encoding="utf-8"))
            patch = Patch(target=args.target, output=output)
        else:
            patch = patch_fields(run, args.target, dict(_field(f) for f in args.fields))
        plan = plan_patch(run, workflow, patch)
    except (PatchError, ValueError, OSError) as exc:
        print(f"cannot plan: {exc}")
        return 2

    where = f"{plan.target_kind}, step {plan.target_step}"
    print(f"run {run.run_id}: patch {plan.target_local} ({where})")
    print(f"changed keys : {', '.join(plan.changed_keys) or 'none (the patch is a no-op)'}")
    print(f"frozen steps : {', '.join(plan.frozen_steps) or '-'}")
    print(f"re-run steps : {', '.join(plan.rerun_steps) or '-'}\n")
    labels = {"frozen": "frozen", "patched": "PATCHED", "rerun": "re-run"}
    for span in run.spans:
        print(f"  {labels[plan.action(span.span_id)]:<8} {plan.local(span.span_id)}")
    print(f"\n{len(plan.frozen)} frozen, 1 patched, {len(plan.rerun)} to re-run")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in NOT_YET:
        print(f"`agenttrace {args.command}` is not implemented yet ({NOT_YET[args.command]}).")
        return 1
    handler = {
        "run": cmd_run,
        "runs": cmd_runs,
        "show": cmd_show,
        "export": cmd_export,
        "import": cmd_import,
        "graph": cmd_graph,
        "diff": cmd_diff,
        "plan": cmd_plan,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
