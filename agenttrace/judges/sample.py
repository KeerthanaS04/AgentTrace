"""Judge sample (P6-T1): do the judge prompts give schema-valid output on 20 spans per model?

    python tasks.py judge-sample                        # judge_fast, judge_nvidia, judge_oss
    python tasks.py judge-sample --roles judge_nvidia   # one judge (free NVIDIA tier)
    python tasks.py judge-sample -n 10 -v

The sample: `-n` spans (default 20) drawn with a fixed seed from the stored seeded runs
(`sc-*-default-s1`, faulty and control alike), stratified by node so every step, tool
calls and searches all appear. Which runs are faulty plays no part in the draw.

Each judge model scores every sampled span with the current node-judge prompt. Valid =
the reply parsed as a JudgeReply and cited only ids from the context, so it became an
AnomalyScore (repairs allowed, up to 2). A role whose provider has no key or whose model
is unverified is reported as skipped. Report: reports/judge_sample.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agenttrace.gateway.gateway import Gateway
from agenttrace.gateway.types import GatewayError
from agenttrace.judges.context import judged_spans
from agenttrace.judges.node_judge import JudgeResult, judge_span
from agenttrace.schemas.run import Run

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "judge_sample.json"
SEED = 20261104
DEFAULT_ROLES = ("judge_fast", "judge_nvidia", "judge_oss")
# Spans per stratum in a sample of 20 (scaled for other sizes).
STRATA: dict[str, int] = {
    "router": 2,
    "retriever": 2,
    "tool_agent": 2,
    "policy": 5,
    "writer": 3,
    "tool": 4,
    "retrieval": 2,
}


def stratum(node_id: str, kind: str) -> str:
    return node_id if kind == "agent" else kind


def draw_sample(runs: Sequence[Run], n: int = 20, seed: int = SEED) -> list[tuple[str, str]]:
    """(run id, local span id) pairs: a seeded, stratified draw of judged spans."""
    pools: dict[str, list[tuple[str, str]]] = {k: [] for k in STRATA}
    for run in sorted(runs, key=lambda r: r.run_id):
        for span in judged_spans(run):
            key = stratum(span.node_id, span.kind)
            if key in pools:
                pools[key].append((run.run_id, span.span_id.removeprefix(f"{run.run_id}/")))
    rng = random.Random(seed)
    total = sum(STRATA.values())
    picked: list[tuple[str, str]] = []
    for key, quota in STRATA.items():
        want = max(1, round(quota * n / total))
        picked += rng.sample(pools[key], min(want, len(pools[key])))
    rng.shuffle(picked)
    return picked[:n]


def write_report(doc: dict[str, Any]) -> None:
    """Write the report, keeping other judges' results on the same sample.

    Running a subset of judges (`--roles judge_nvidia`) must not erase the results of the
    judges it did not run: when the stored report has the same sample, their entries are
    carried over; a different sample replaces the report.
    """
    if REPORT_PATH.exists():
        old = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if old.get("sample") == doc["sample"]:
            for key in ("summary", "results"):
                doc[key] = {**old.get(key, {}), **doc[key]}
            merged = {**old.get("skipped", {}), **doc["skipped"]}
            doc["skipped"] = {k: v for k, v in merged.items() if k not in doc["results"]}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roles", nargs="*", default=list(DEFAULT_ROLES))
    parser.add_argument("-n", type=int, default=20, help="spans per judge model")
    parser.add_argument("--prefix", default="sc-", help="stored runs to draw from")
    parser.add_argument("--suffix", default="-default-s1", help="(seed 1 runs by default)")
    parser.add_argument("-v", "--verbose", action="store_true", help="a line per judgement")
    args = parser.parse_args(argv)

    from agenttrace.store import TraceStore

    with TraceStore.open() as store:
        ids = [i for i in store.run_ids() if i.startswith(args.prefix) and i.endswith(args.suffix)]
        runs = {i: store.get_run(i) for i in ids}
    if not runs:
        print(f"no stored runs match {args.prefix}*{args.suffix}")
        return 2
    sample = draw_sample(list(runs.values()), args.n)
    gateway = Gateway.from_env()
    spans = {s.span_id: s for run in runs.values() for s in run.spans}
    kinds = Counter(
        stratum(spans[f"{r}/{local}"].node_id, spans[f"{r}/{local}"].kind) for r, local in sample
    )
    print(f"Judge sample: {len(sample)} spans from {len(runs)} runs ({dict(kinds)})\n")

    results: dict[str, list[JudgeResult]] = {}
    skipped: dict[str, str] = {}
    for role in args.roles:
        if not gateway.is_available(role):
            try:
                rc = gateway.resolve(role)
                skipped[role] = f"{rc.provider}: no API key or model not verified"
            except GatewayError as exc:  # unknown or unverified role
                skipped[role] = str(exc)
            continue
        rows: list[JudgeResult] = []
        for run_id, local in sample:
            res = judge_span(gateway, runs[run_id], local, role=role)
            if res.auth_error:
                # The provider rejected the credentials: no other span will do better.
                skipped[role] = f"provider rejected the key ({res.provider_error})"
                rows = []
                break
            rows.append(res)
            if args.verbose or not res.valid:
                verdict = res.score.verdict if res.score else "INVALID"
                last = f"  {res.errors[-1][:160]}" if res.errors and not res.valid else ""
                print(
                    f"  {role:<13} {run_id:<20} {local:<26} {verdict:<10} "
                    f"tries {res.attempts}{last}",
                    flush=True,
                )
        if rows:
            results[role] = rows
    gateway.close()

    summary = {}
    for role, rows in results.items():
        valid = [r for r in rows if r.score is not None]
        verdicts = Counter(r.score.verdict for r in valid if r.score)
        summary[role] = {
            "model": rows[0].model if rows else "",
            "spans": len(rows),
            "valid": len(valid),
            "first_try": sum(r.valid and r.attempts == 1 for r in rows),
            "verdicts": dict(verdicts),
            "tokens_in": sum(r.tokens_in for r in rows),
            "tokens_out": sum(r.tokens_out for r in rows),
            "cost_usd": round(sum(r.cost_usd for r in rows), 4),
            "latency_s": round(sum(r.latency_s for r in rows), 1),
        }
        s = summary[role]
        print(f"\n{role} ({s['model']}): {s['valid']}/{s['spans']} schema-valid")
        print(f"  valid on the first try: {s['first_try']}/{s['spans']}")
        print(f"  verdicts: {dict(verdicts) or '-'}")
        print(
            f"  tokens {s['tokens_in']} in / {s['tokens_out']} out, ${s['cost_usd']:.4f}, "
            f"{s['latency_s']}s of model time"
        )
    for role, why in skipped.items():
        print(f"\n{role}: skipped ({why})")

    doc = {
        "sample": [{"run_id": r, "span": s} for r, s in sample],
        "summary": summary,
        "skipped": skipped,
        "results": {
            role: [
                asdict(r) | {"score": r.score.model_dump(mode="json") if r.score else None}
                for r in rows
            ]
            for role, rows in results.items()
        },
    }
    write_report(doc)
    print(f"\nreport: {REPORT_PATH.relative_to(ROOT).as_posix()}")
    ok = bool(results) and all(s["valid"] == s["spans"] for s in summary.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
