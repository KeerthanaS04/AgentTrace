"""Multi-run protocol (P4-T5): K seeded runs per scenario, and how much the results vary.

    python tasks.py multirun                     # K=3 (seeds 1, 2, 3), default profile
    python tasks.py multirun --k 5 --code F4     # more runs, one fault type
    python tasks.py multirun --temperature 0.7   # sample instead of the configured decoding

Run k passes `seed=k` to every model call in the run (each role keeps its configured
params, temperature 0 included, unless `--temperature` overrides it). The seed is part of
the gateway's cache key, so each seed is a separate, individually cached run: repeating
this command is free, and adding a seed only pays for the new one.

Every run is judged with the P4-T4 criteria (`verify_scenario`) against the clean run of
the *same* seed, and stored as `{scenario}-{profile}-s{seed}`. Reported:

    per run       behaves-as-labelled rate and final-correct rate over all scenarios
    across runs   mean, sample variance and std of those rates, min-max
    per code      mean and std of the behaves rate
    per scenario  how many of the K runs behaved / were correct; scenarios whose result
                  differs between seeds are "unstable"

Full details go to reports/multirun_<profile>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agenttrace.evaluation.outcome import OutcomeCheck
from agenttrace.gateway.gateway import Gateway
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.verify import (
    Verification,
    judged,
    prune_stale,
    run_tag,
    verify_scenario,
)
from agenttrace.schemas.run import Run
from agenttrace.workflow.agent_config import DEFAULT_PROFILE, load_agents_config

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
DEFAULT_K = 3

Results = dict[int, list[Verification]]  # seed -> one Verification per scenario


def seeds_for(k: int, first: int = 1) -> list[int]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return list(range(first, first + k))


def model_params(seed: int, temperature: float | None) -> dict[str, Any]:
    params: dict[str, Any] = {"seed": seed}
    if temperature is not None:
        params["temperature"] = temperature
    return params


def run_all(
    gateway: Gateway,
    scenarios: Sequence[Scenario],
    roles: dict[str, str],
    profile: str,
    seeds: Sequence[int],
    temperature: float | None = None,
    on_run: Callable[[int, Verification, Run | None], None] | None = None,
) -> Results:
    """Every scenario once per seed. Seeds run one after another, so an interrupted
    run leaves whole seeds behind (and the cache makes the rerun cheap)."""
    results: Results = {}
    clean_cache: dict[str, tuple[Run, OutcomeCheck]] = {}
    for seed in seeds:
        params = model_params(seed, temperature)
        rows: list[Verification] = []
        for sc in scenarios:
            v, run = verify_scenario(gateway, sc, roles, profile, clean_cache, params)
            rows.append(v)
            if on_run is not None:
                on_run(seed, v, run)
        results[seed] = rows
    return results


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    """Mean, sample variance / std (None with fewer than two runs), min and max."""
    if not values:
        return dict.fromkeys(("mean", "variance", "std", "min", "max"))
    many = len(values) > 1
    return {
        "mean": statistics.fmean(values),
        "variance": statistics.variance(values) if many else None,
        "std": statistics.stdev(values) if many else None,
        "min": min(values),
        "max": max(values),
    }


def variance_report(results: Results) -> dict[str, Any]:
    """Rates per run and their spread across runs. Runs cut short by a provider error
    (`Verification.provider_error`) carry no verdict: they are left out of every rate and
    listed under `provider_errors`, so the report says how incomplete it is."""
    seeds = list(results)
    if not seeds:
        raise ValueError("no runs to report on")
    ids = [v.scenario_id for v in results[seeds[0]]]
    for seed in seeds:
        if [v.scenario_id for v in results[seed]] != ids:
            raise ValueError(f"seed {seed} ran a different set of scenarios")
    errors = [
        {"seed": seed, "scenario_id": v.scenario_id}
        for seed in seeds
        for v in results[seed]
        if v.provider_error is not None
    ]

    runs: list[dict[str, Any]] = []
    behaves_rates: list[float] = []  # per judged run (runs with no verdicts are skipped)
    correct_rates: list[float] = []
    for seed in seeds:
        rows = judged(results[seed])
        n = len(rows)
        behaves = sum(v.behaves for v in rows)
        correct = sum(bool(v.final_correct) for v in rows)
        if n:
            behaves_rates.append(behaves / n)
            correct_rates.append(correct / n)
        runs.append(
            {
                "seed": seed,
                "n": n,
                "provider_errors": len(results[seed]) - n,
                "behaves": behaves,
                "behaves_rate": behaves / n if n else None,
                "final_correct": correct,
                "final_correct_rate": correct / n if n else None,
            }
        )

    by_code: dict[str, dict[str, Any]] = {}
    for code in sorted({v.fault_code for v in results[seeds[0]]}):
        rates = []
        for seed in seeds:
            group = [v for v in judged(results[seed]) if v.fault_code == code]
            if group:
                rates.append(sum(v.behaves for v in group) / len(group))
        n = sum(v.fault_code == code for v in results[seeds[0]])
        by_code[code] = {"n": n, **_spread(rates)}

    per_scenario = []
    for i, sid in enumerate(ids):
        everything = [results[seed][i] for seed in seeds]
        across = judged(everything)
        k = len(across)
        behaved = sum(v.behaves for v in across)
        correct = sum(bool(v.final_correct) for v in across)
        per_scenario.append(
            {
                "scenario_id": sid,
                "fault_code": everything[0].fault_code,
                "ticket_id": everything[0].ticket_id,
                "k": k,
                "provider_errors": len(everything) - k,
                "behaves": behaved,
                "final_correct": correct,
                # The same verdict, and the same final correctness, on every judged seed.
                "stable": behaved in (0, k) and correct in (0, k),
                "reasons": sorted({v.reason for v in across if not v.behaves}),
            }
        )

    return {
        "k": len(seeds),
        "seeds": seeds,
        "n_scenarios": len(ids),
        "complete": not errors,
        "provider_errors": errors,
        "runs": runs,
        "behaves_rate": _spread(behaves_rates),
        "final_correct_rate": _spread(correct_rates),
        "by_code": by_code,
        "scenarios": per_scenario,
        "unstable": [s["scenario_id"] for s in per_scenario if not s["stable"]],
        "never_behaves": [s["scenario_id"] for s in per_scenario if s["k"] and s["behaves"] == 0],
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def _pp(x: float | None) -> str:
    """A std of rates, in percentage points."""
    return "-" if x is None else f"{x * 100:.1f} pp"


def print_report(report: Mapping[str, Any], verbose: bool = False) -> None:
    print(f"\nper run ({report['n_scenarios']} scenarios each):")
    print(f"  {'seed':<6}{'behaves as labelled':<24}final correct")
    for r in report["runs"]:
        behaves = f"{r['behaves']}/{r['n']} = {_pct(r['behaves_rate'])}"
        correct = f"{r['final_correct']}/{r['n']} = {_pct(r['final_correct_rate'])}"
        lost = (
            f"   ({r['provider_errors']} not judged: provider errors)"
            if r["provider_errors"]
            else ""
        )
        print(f"  {r['seed']:<6}{behaves:<24}{correct}{lost}")

    print(f"\nacross the {report['k']} runs:")
    for label, key in (
        ("behaves as labelled", "behaves_rate"),
        ("final correct", "final_correct_rate"),
    ):
        s = report[key]
        var = "-" if s["variance"] is None else f"{s['variance']:.6f}"
        print(
            f"  {label:<20} mean {_pct(s['mean'])}  std {_pp(s['std'])}  variance {var}  "
            f"range {_pct(s['min'])}-{_pct(s['max'])}"
        )

    print("\nbehaves as labelled, by fault code (mean over runs, std):")
    for code, c in report["by_code"].items():
        print(f"  {code}  n={c['n']:<3} mean {_pct(c['mean']):>6}  std {_pp(c['std'])}")

    shown = [s for s in report["scenarios"] if verbose or not s["stable"] or s["behaves"] < s["k"]]
    unstable = len(report["unstable"])
    print(
        f"\nunstable scenarios (result differs between seeds): {unstable}/{report['n_scenarios']}"
    )
    if shown:
        print(f"  {'scenario':<9} {'code':<5} {'ticket':<7} {'behaves':<8} {'correct':<8} reasons")
        for s in shown:
            k = s["k"]
            print(
                f"  {s['scenario_id']:<9} {s['fault_code']:<5} {s['ticket_id']:<7} "
                f"{s['behaves']}/{k:<6} {s['final_correct']}/{k:<6} "
                f"{'; '.join(s['reasons']) or '-'}"
            )
    never = report["never_behaves"]
    print(f"never behave as labelled (0/{report['k']}): {', '.join(never) or 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="runs per scenario")
    parser.add_argument("--first-seed", type=int, default=1, help="seeds are first..first+k-1")
    parser.add_argument(
        "--temperature", type=float, help="override every role's temperature (default: config)"
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="profile in config/agents.yaml")
    parser.add_argument("--only", nargs="*", help="scenario ids")
    parser.add_argument("--code", nargs="*", help="only these fault codes, e.g. F3 F8")
    parser.add_argument("--no-store", action="store_true", help="do not save runs to the store")
    parser.add_argument("-v", "--verbose", action="store_true", help="every scenario's row")
    args = parser.parse_args(argv)

    seeds = seeds_for(args.k, args.first_seed)
    scenarios = load_dataset()
    if args.only:
        scenarios = [s for s in scenarios if s.scenario_id in set(args.only)]
    if args.code:
        scenarios = [s for s in scenarios if s.ground_truth.fault_code in set(args.code)]
    if not scenarios:
        print("no scenarios selected")
        return 1
    gateway = Gateway.from_env()
    agents = load_agents_config()
    agents.check_roles(gateway.config)
    roles = agents.profile(args.profile).roles()

    store = None
    if not args.no_store:
        from agenttrace.store import TraceStore

        store = TraceStore.open()

    temp = "configured" if args.temperature is None else f"{args.temperature:g}"
    print(
        f"Multi-run: {len(scenarios)} scenarios x {len(seeds)} seeds {seeds}, "
        f"profile {args.profile!r}, temperature {temp}"
    )
    stored: dict[int, set[str | None]] = {seed: set() for seed in seeds}
    done: dict[int, int] = dict.fromkeys(seeds, 0)

    def on_run(seed: int, v: Verification, run: Run | None) -> None:
        done[seed] += 1
        if store is not None and run is not None:
            store.save_run(run, replace=True)  # stable id: re-running replaces it
            stored[seed].add(run.run_id)
        if args.verbose:
            ok = "Y" if v.behaves else "N"
            print(f"  seed {seed}  {v.scenario_id}  {v.fault_code}  {ok}  {v.reason}")
        elif done[seed] % 10 == 0 or done[seed] == len(scenarios):
            print(f"  seed {seed}: {done[seed]}/{len(scenarios)}", flush=True)

    results = run_all(gateway, scenarios, roles, args.profile, seeds, args.temperature, on_run)
    gateway.close()
    if store is not None:
        if not (args.only or args.code):
            for seed in seeds:
                tag = run_tag(model_params(seed, args.temperature))
                prune_stale(store, stored[seed], args.profile, tag)
        store.close()

    report = variance_report(results)
    print_report(report, args.verbose)
    suffix = "" if args.temperature is None else f"_t{args.temperature:g}"
    path = REPORT_DIR / f"multirun_{args.profile}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {str(seed): [asdict(v) for v in vs] for seed, vs in results.items()}
    payload = {"profile": args.profile, "temperature": args.temperature, **report, "rows": rows}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"full report: {path.relative_to(ROOT).as_posix()}")
    if store is not None:
        each = f" ({len(seeds)} per scenario)" if report["complete"] else ""
        print(f"stored {sum(map(len, stored.values()))} runs{each}")
    if not report["complete"]:
        print(
            f"INCOMPLETE: {len(report['provider_errors'])} runs hit provider errors (e.g. rate "
            "limits) and were not judged or stored; rerun the same command: finished calls "
            "come from the cache, so only those runs are redone"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
