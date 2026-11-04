"""Replay check (P5-T1): do stored runs replay with zero diffs and no live calls?

    python tasks.py replay-check                    # every stored support-workflow run
    python tasks.py replay-check --prefix sc-001    # runs whose id starts with this
    python tasks.py replay-check --limit 20 -v

Each run is replayed *strictly* through the support workflow, with a replay cache filed
from that run alone: a model call without a recorded output is an error, not a live
call, so a passing replay provably made no model call. The model profile is inferred
from the models the run's agents used. A run passes when its replay has zero diffs:
every span's output, input, model, params, status, and explanation, and the final output,
are exactly equal (`exact_diffs`), and `align_runs` finds every span unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence

from agenttrace.gateway.gateway import Gateway
from agenttrace.gateway.types import GatewayError
from agenttrace.graph import align_runs
from agenttrace.replay.replay import exact_diffs, replay_run
from agenttrace.schemas.run import Run
from agenttrace.tracing.context import ReplayMissError
from agenttrace.workflow.agent_config import AgentsConfig, load_agents_config
from agenttrace.workflow.support import STEPS, build_support_workflow, make_resources

SUPPORT_VERSION = "support@1"


def infer_profile(run: Run, agents: AgentsConfig, gateway: Gateway) -> str | None:
    """The first profile whose role -> model mapping matches the models the run used."""
    used = {step: model for step, model in run.agent_models.items() if step in STEPS}
    if not used:
        return None
    for name, profile in agents.profiles.items():
        roles = profile.roles()
        try:
            models = {step: gateway.resolve(roles[step]).model for step in used}
        except GatewayError:  # an unverified or unknown role: not this profile
            continue
        if models == used:
            return name
    return None


def check_run(
    run: Run, gateway: Gateway, agents: AgentsConfig
) -> tuple[str, str, Mapping[str, Mapping[str, int]]]:
    """(verdict, detail, cache stats). Verdict: identical / diff / miss / skipped."""
    if run.workflow_version != SUPPORT_VERSION:
        return "skipped", f"workflow {run.workflow_version}", {}
    profile = infer_profile(run, agents, gateway)
    if profile is None:
        return "skipped", f"no profile matches models {sorted(set(run.agent_models.values()))}", {}
    workflow = build_support_workflow(agents.profile(profile).roles())
    try:
        result, cache = replay_run(run, workflow, make_resources(gateway), strict=True)
    except ReplayMissError as exc:
        return "miss", str(exc), {}
    replayed = result.to_run(scenario_id=run.scenario_id)
    diffs = exact_diffs(run, replayed)
    aligned = align_runs(run, replayed)
    stats = cache.stats()
    if diffs or not aligned.identical:
        changed = diffs or sorted(aligned.changed)
        return "diff", f"profile {profile}: {', '.join(changed[:6])}", stats
    return "identical", f"profile {profile}, {len(run.spans)} spans", stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="", help="only run ids starting with this")
    parser.add_argument("--limit", type=int, help="at most this many runs")
    parser.add_argument("--db", help="SQLite store (default .cache/agenttrace.db)")
    parser.add_argument("-v", "--verbose", action="store_true", help="a line per run")
    args = parser.parse_args(argv)

    from agenttrace.store import TraceStore

    gateway = Gateway.from_env()
    agents = load_agents_config()
    verdicts: Counter[str] = Counter()
    served: Counter[str] = Counter()
    live = 0
    problems: list[str] = []
    with TraceStore.open(args.db) as store:
        ids = [i for i in store.run_ids() if i.startswith(args.prefix)]
        if args.limit is not None:
            ids = ids[: args.limit]
        print(f"Replay check: {len(ids)} stored runs, strict (any live model call is an error)\n")
        for n, run_id in enumerate(ids, start=1):
            run = store.get_run(run_id)
            verdict, detail, stats = check_run(run, gateway, agents)
            verdicts[verdict] += 1
            for kind, s in stats.items():
                served[kind] += s["hits"]
                live += s["misses"]
            if verdict in ("diff", "miss"):
                problems.append(f"  {verdict:<9} {run_id}: {detail}")
            if args.verbose:
                print(f"  {verdict:<9} {run_id}  {detail}")
            elif n % 50 == 0:
                print(f"  {n}/{len(ids)}", flush=True)
    gateway.close()

    checked = verdicts["identical"] + verdicts["diff"] + verdicts["miss"]
    print(f"\nreplayed {checked} runs ({verdicts['skipped']} skipped: not the support workflow")
    print("or no matching model profile)")
    print(f"  identical (zero diffs): {verdicts['identical']}/{checked}")
    print(f"  with diffs            : {verdicts['diff']}")
    print(f"  missing a recording   : {verdicts['miss']}")
    by_kind = ", ".join(f"{k} {v}" for k, v in sorted(served.items()))
    print(f"outputs served from the replay cache: {by_kind or 'none'}")
    print(f"recomputed instead (cache misses; never a model call in strict mode): {live}")
    if problems:
        print("\nproblems:")
        print("\n".join(problems))
    ok = checked > 0 and verdicts["identical"] == checked
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
