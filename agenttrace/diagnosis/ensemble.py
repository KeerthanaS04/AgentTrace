"""Judge ensemble (P6-T4): several judges' verdicts -> one verdict per span, and agreement.

    stage = score_run(run, gateway, roles=JUDGES)        # Stage A with every judge
    ens = ensemble(stage, load_weights())
    ens.spans["<run>/policy"].verdict                    # aggregate verdict
    ens.spans["<run>/policy"].votes                      # {judge: verdict}
    ens.agreement                                        # mean pairwise Cohen's kappa

Aggregation (config/ensemble.yaml): weighted vote over the judges that gave a valid
verdict on the span; ties go to the more severe verdict (faulty > suspicious > ok);
`inputs_ok` / `output_ok_given_inputs` are the weighted means. Spans settled without a
judge (validators, input, failed / skipped steps) keep their deterministic score and take
no part in agreement; model-call spans inherit their step's aggregate.

Agreement, over the spans every pair (or all) of judges rated:

    pairwise   Cohen's kappa per judge pair (3 categories: ok / suspicious / faulty)
    overall    Fleiss' kappa across all judges, and the mean pairwise kappa (the number
               `Diagnosis.judge_agreement` holds)
    per node   `disagreement` = 1 - (weight of the winning verdict / total weight):
               0 when the judges agree, up to 2/3 for a three-way split
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agenttrace.diagnosis.stage_a import STAGE, SpanScores, StageAResult
from agenttrace.schemas.diagnosis import AnomalyScore, Verdict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "ensemble.yaml"
JUDGES = ("judge_fast", "judge_nvidia", "judge_oss")
ENSEMBLE = "ensemble"
SEVERITY: dict[str, int] = {"ok": 0, "suspicious": 1, "faulty": 2}
VERDICTS: tuple[Verdict, ...] = ("ok", "suspicious", "faulty")


def load_weights(path: Path = CONFIG_PATH) -> dict[str, float]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = {str(k): float(v) for k, v in (doc.get("weights") or {}).items()}
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{path}: judge weights must be non-negative")
    return weights


# ---------------------------------------------------------------- agreement statistics


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same items (None with no items).

    With no expected disagreement (both raters always give the same single category),
    kappa is undefined; it is reported as 1.0 if they agree everywhere, else 0.0.
    """
    if len(a) != len(b):
        raise ValueError("raters must rate the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float | None:
    """Fleiss' kappa: each item rated by the same number (>= 2) of raters."""
    items = [list(r) for r in ratings if r]
    if not items:
        return None
    raters = len(items[0])
    if raters < 2 or any(len(r) != raters for r in items):
        raise ValueError("every item needs the same number (>= 2) of ratings")
    n = len(items)
    cats = sorted({c for r in items for c in r})
    p_items = []
    totals: Counter[str] = Counter()
    for r in items:
        counts = Counter(r)
        totals.update(counts)
        p_items.append((sum(k * k for k in counts.values()) - raters) / (raters * (raters - 1)))
    p_bar = sum(p_items) / n
    p_e = sum((totals[c] / (n * raters)) ** 2 for c in cats)
    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return (p_bar - p_e) / (1 - p_e)


# ---------------------------------------------------------------- aggregation


@dataclass
class EnsembleSpan:
    span_id: str
    local_id: str
    kind: str
    source: str  # "ensemble", or the Stage A source of a span settled without judges
    votes: dict[str, str] = field(default_factory=dict)  # judge -> verdict (valid replies)
    verdict: str = "ok"
    disagreement: float = 0.0
    score: AnomalyScore | None = None  # the aggregate (judge "ensemble") or deterministic


@dataclass
class EnsembleResult:
    run_id: str
    judges: tuple[str, ...]
    spans: dict[str, EnsembleSpan] = field(default_factory=dict)
    pairwise: dict[str, float | None] = field(default_factory=dict)  # "a|b" -> kappa
    fleiss: float | None = None

    @property
    def agreement(self) -> float | None:
        """Mean pairwise Cohen's kappa (None if no pair rated a common span)."""
        values = [k for k in self.pairwise.values() if k is not None]
        return sum(values) / len(values) if values else None

    def split_spans(self) -> list[EnsembleSpan]:
        """Judged spans where the judges did not all give the same verdict."""
        return [s for s in self.spans.values() if len(set(s.votes.values())) > 1]


def vote(
    scores: Mapping[str, AnomalyScore], weights: Mapping[str, float]
) -> tuple[Verdict, float, float, float]:
    """(verdict, disagreement, inputs_ok, output_ok) from the judges' scores."""
    tally: dict[str, float] = dict.fromkeys(VERDICTS, 0.0)
    total = 0.0
    inputs = output = 0.0
    for judge, score in scores.items():
        w = weights.get(judge, 1.0)
        tally[score.verdict] += w
        total += w
        inputs += w * score.inputs_ok
        output += w * score.output_ok_given_inputs
    if total == 0:
        raise ValueError("no judge with a positive weight voted")
    best = max(VERDICTS, key=lambda v: (tally[v], SEVERITY[v]))
    return best, round(1 - tally[best] / total, 4), inputs / total, output / total


def _aggregate(
    entry: SpanScores, judges: Sequence[str], weights: Mapping[str, float]
) -> EnsembleSpan:
    out = EnsembleSpan(entry.span_id, entry.local_id, entry.kind, entry.source)
    valid = {
        j: s
        for j, s in entry.scores.items()
        if j in judges and j not in entry.failed_roles and weights.get(j, 1.0) > 0
    }
    if not valid:
        # Settled without a (valid) judge: keep the deterministic / single score.
        out.score = entry.scores.get(STAGE) or next(iter(entry.scores.values()), None)
        out.verdict = out.score.verdict if out.score else "suspicious"
        return out
    out.source = "ensemble"
    out.votes = {j: s.verdict for j, s in valid.items()}
    verdict, out.disagreement, inputs_ok, output_ok = vote(valid, weights)
    out.verdict = verdict
    evidence = list(dict.fromkeys(e for s in valid.values() for e in s.evidence))
    tally = ", ".join(f"{j} {v}" for j, v in out.votes.items())
    out.score = AnomalyScore(
        span_id=entry.span_id,
        judge=ENSEMBLE,
        prompt_version=next(iter(valid.values())).prompt_version,
        output_ok_given_inputs=round(output_ok, 4),
        inputs_ok=round(inputs_ok, 4),
        verdict=verdict,
        evidence=evidence if (evidence or verdict == "ok") else [f"{entry.local_id}: {tally}"],
        reasoning=f"Weighted vote ({tally}); disagreement {out.disagreement:.2f}.",
    )
    return out


def ensemble(
    stage: StageAResult,
    weights: Mapping[str, float] | None = None,
    judges: Sequence[str] | None = None,
) -> EnsembleResult:
    weights = dict(weights) if weights is not None else load_weights()
    judges = tuple(judges or stage.roles)
    result = EnsembleResult(stage.run_id, judges)
    for entry in stage.spans:
        if entry.source == "inherited":
            continue
        result.spans[entry.span_id] = _aggregate(entry, judges, weights)
    # Model calls inherit their step's aggregate.
    for entry in stage.spans:
        if entry.source != "inherited":
            continue
        step = entry.span_id.rsplit(".", 1)[0]
        parent = result.spans.get(step)
        inherited = EnsembleSpan(entry.span_id, entry.local_id, entry.kind, "inherited")
        if parent is not None and parent.score is not None:
            inherited.verdict = parent.verdict
            inherited.score = parent.score.model_copy(
                update={"span_id": entry.span_id, "judge": f"inherited:{parent.score.judge}"}
            )
        result.spans[entry.span_id] = inherited
    rated = [s for s in result.spans.values() if s.source == "ensemble"]
    for a, b in itertools.combinations(judges, 2):
        both = [s for s in rated if a in s.votes and b in s.votes]
        result.pairwise[f"{a}|{b}"] = cohens_kappa(
            [s.votes[a] for s in both], [s.votes[b] for s in both]
        )
    full = [[s.votes[j] for j in judges] for s in rated if all(j in s.votes for j in judges)]
    result.fleiss = fleiss_kappa(full) if len(judges) >= 2 and full else None
    return result


def pooled_agreement(results: Sequence[EnsembleResult]) -> dict[str, float | None]:
    """Agreement over every judged span of several runs (more stable than per run)."""
    if not results:
        return {}
    judges = results[0].judges
    rated = [s for r in results for s in r.spans.values() if s.source == "ensemble"]
    out: dict[str, float | None] = {}
    for a, b in itertools.combinations(judges, 2):
        both = [s for s in rated if a in s.votes and b in s.votes]
        out[f"{a}|{b}"] = cohens_kappa([s.votes[a] for s in both], [s.votes[b] for s in both])
    full = [[s.votes[j] for j in judges] for s in rated if all(j in s.votes for j in judges)]
    out["fleiss"] = fleiss_kappa(full) if len(judges) >= 2 and full else None
    pairs = [v for k, v in out.items() if k != "fleiss" and v is not None]
    out["mean_pairwise"] = sum(pairs) / len(pairs) if pairs else None
    return out


# ---------------------------------------------------------------- the command


def _fmt(k: float | None) -> str:
    return "  n/a" if k is None else f"{k:+.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    """`python tasks.py ensemble`: every judge on the dev set, aggregated, with agreement."""
    import argparse
    import json

    from agenttrace.diagnosis.stage_a import DEVSET_NAME, score_run
    from agenttrace.diagnosis.validators import ValidationContext
    from agenttrace.gateway.gateway import Gateway
    from agenttrace.store import TraceStore

    parser = argparse.ArgumentParser(description="Judge ensemble with agreement statistics")
    parser.add_argument("--runs", nargs="*", help="run ids (default: the diagnosis dev set)")
    parser.add_argument("--devset", type=Path, default=ROOT / "reports" / DEVSET_NAME)
    parser.add_argument("--judges", nargs="*", default=list(JUDGES))
    parser.add_argument("--limit", type=int, help="at most this many runs")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="every split span")
    args = parser.parse_args(argv)

    run_ids = list(args.runs or [])
    if not run_ids:
        if not args.devset.exists():
            print(f"{args.devset} not found: run `python tasks.py devset` or pass --runs")
            return 2
        run_ids = json.loads(args.devset.read_text(encoding="utf-8"))["run_ids"]
    if args.limit is not None:
        run_ids = run_ids[: args.limit]
    gateway = Gateway.from_env()
    judges = tuple(j for j in args.judges if gateway.is_available(j))
    missing = [j for j in args.judges if j not in judges]
    if len(judges) < 2:
        print(f"need at least two available judges; unavailable: {missing}")
        return 2
    weights = load_weights()
    ctx = ValidationContext.default()
    print(f"Judge ensemble: {len(run_ids)} runs, judges {', '.join(judges)}")
    if missing:
        print(f"  skipped (no key or unverified model): {', '.join(missing)}")
    print(f"  weights: {', '.join(f'{j} {weights.get(j, 1.0):g}' for j in judges)}\n")
    print(f"  {'run':<22} {'judged':>6} {'split':>5} {'mean kappa':>11} {'fleiss':>7} {'cost':>9}")

    results: list[EnsembleResult] = []
    cost_by_role: Counter[str] = Counter()
    calls_by_role: Counter[str] = Counter()
    rows = []
    with TraceStore.open() as store:
        for run_id in run_ids:
            if not store.has_run(run_id):
                print(f"  {run_id:<22} not stored")
                continue
            stage = score_run(store.get_run(run_id), gateway, roles=judges, ctx=ctx)
            if stage.disabled_roles:
                for role, why in stage.disabled_roles.items():
                    print(f"  {role} disabled: {why}")
                print("  fix the credentials and re-run (finished judgements are cached)")
                gateway.close()
                return 2
            ens = ensemble(stage, weights, judges)
            results.append(ens)
            cost_by_role.update(stage.cost_by_role)
            calls_by_role.update(stage.calls_by_role)
            if not args.no_store:
                for score in stage.all_scores():
                    store.add_anomaly(score)
                for span in ens.spans.values():
                    if span.score is not None and span.source == "ensemble":
                        store.add_anomaly(span.score)
            judged = [s for s in ens.spans.values() if s.source == "ensemble"]
            print(
                f"  {run_id:<22} {len(judged):>6} {len(ens.split_spans()):>5} "
                f"{_fmt(ens.agreement):>11} {_fmt(ens.fleiss):>7} ${stage.cost_usd:>8.4f}",
                flush=True,
            )
            if args.verbose:
                for s in ens.split_spans():
                    votes = ", ".join(f"{j}={v}" for j, v in s.votes.items())
                    print(f"      {s.local_id:<30} -> {s.verdict:<10} ({votes})")
            rows.append(
                {
                    "run_id": run_id,
                    "agreement": ens.agreement,
                    "fleiss": ens.fleiss,
                    "pairwise": ens.pairwise,
                    "cost_usd": round(stage.cost_usd, 6),
                    "spans": [
                        {
                            "span_id": s.span_id,
                            "source": s.source,
                            "votes": s.votes,
                            "verdict": s.verdict,
                            "disagreement": s.disagreement,
                        }
                        for s in ens.spans.values()
                    ],
                }
            )
    gateway.close()
    if not results:
        return 2

    pooled = pooled_agreement(results)
    rated = [s for r in results for s in r.spans.values() if s.source == "ensemble"]
    print(f"\nagreement over all {len(rated)} judged spans:")
    for key, value in pooled.items():
        print(f"  {key:<34} {_fmt(value)}")
    print("\nverdicts per judge (judged spans):")
    for j in (*judges, ENSEMBLE):
        if j == ENSEMBLE:
            c = Counter(s.verdict for s in rated)
        else:
            c = Counter(s.votes[j] for s in rated if j in s.votes)
        print(f"  {j:<14} " + "  ".join(f"{v} {c.get(v, 0):>3}" for v in VERDICTS))
    by_node = Counter(s.local_id.split("#")[0] for r in results for s in r.split_spans())
    print("\nwhere judges split most (node: split spans):")
    print("  " + ", ".join(f"{n} {k}" for n, k in by_node.most_common(8)))
    print(
        "\ncost by judge: "
        + ", ".join(
            f"{j} ${cost_by_role.get(j, 0):.4f} ({calls_by_role.get(j, 0)} calls)" for j in judges
        )
    )
    complete = all(
        r.agreement is not None
        and all(s.votes and s.score for s in r.spans.values() if s.source == "ensemble")
        for r in results
    )
    path = ROOT / "reports" / "ensemble.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"judges": judges, "weights": weights, "pooled": pooled, "runs": rows}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {path.relative_to(ROOT).as_posix()}")
    print("PASS" if complete else "FAIL")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
