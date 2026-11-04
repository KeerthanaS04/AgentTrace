"""Stage A (P6-T3): an AnomalyScore for every span of a run, from the cheapest source.

    result = score_run(run, gateway, roles=("judge_fast",))
    result.scores["<run>/policy"]["judge_fast"]     # AnomalyScore
    result.cost_usd, result.judge_calls, result.short_circuited

Spans are visited in causal order (`graph.topological_order`), and each gets its score
from the first source that applies:

    input       the workflow input: given, not judged (ok)
    failed      a step that raised: faulty, evidence = its error
    skipped     a step skipped because something upstream failed: suspicious
    validators  a validator ERROR on the span (P6-T2): faulty, no model call
    judge       steps, tool calls and searches: each role in `roles` (P6-T1 prompt);
                validator warnings are added to its evidence
    no answer   a judge that never gave a valid reply: suspicious, 0.5 / 0.5, and the
                failure counted (never a guessed verdict)
    inherited   model-call spans, which repeat their step's prompt: a copy of the step's
                score under judge "inherited:<role>"

For deterministic scores, `inputs_ok` is the lowest `output_ok_given_inputs` among the
steps the span read from, which have been scored already (causal order). Nothing here
sees ground truth. Cost and tokens are the judge calls' (validators are free).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agenttrace.diagnosis.validators import ValidationContext, validate_run
from agenttrace.gateway.gateway import Gateway
from agenttrace.graph import build_graph, topological_order
from agenttrace.judges.context import JUDGED_KINDS
from agenttrace.judges.node_judge import judge_span
from agenttrace.judges.registry import load_prompt
from agenttrace.schemas.diagnosis import AnomalyScore, ValidatorFinding, Verdict
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span

STAGE = "stage-a"  # the `judge` of deterministic scores (validators, input, failures)
DEFAULT_ROLES = ("judge_fast",)


@dataclass
class SpanScores:
    span_id: str
    local_id: str
    kind: str
    source: str  # input / failed / skipped / validators / judge / no answer / inherited
    scores: dict[str, AnomalyScore] = field(default_factory=dict)  # judge -> score
    findings: list[ValidatorFinding] = field(default_factory=list)


@dataclass
class StageAResult:
    run_id: str
    roles: tuple[str, ...]
    spans: list[SpanScores] = field(default_factory=list)  # causal order
    judge_calls: int = 0  # judgements requested (one per judged span and role)
    model_calls: int = 0  # gateway calls, repairs included
    judge_failures: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def by_span(self) -> dict[str, SpanScores]:
        return {s.span_id: s for s in self.spans}

    @property
    def short_circuited(self) -> int:
        return sum(s.source == "validators" for s in self.spans)

    def all_scores(self) -> list[AnomalyScore]:
        return [score for s in self.spans for score in s.scores.values()]

    def sources(self) -> dict[str, int]:
        return dict(Counter(s.source for s in self.spans))

    def verdicts(self, judge: str | None = None) -> dict[str, int]:
        return dict(
            Counter(
                score.verdict
                for s in self.spans
                for j, score in s.scores.items()
                if judge is None or j == judge
            )
        )

    @property
    def complete(self) -> bool:
        return all(s.scores for s in self.spans)


def _det(
    span: Span,
    verdict: Verdict,
    inputs_ok: float,
    output_ok: float,
    evidence: Iterable[str],
    reasoning: str,
) -> AnomalyScore:
    return AnomalyScore(
        span_id=span.span_id,
        judge=STAGE,
        prompt_version=STAGE,
        output_ok_given_inputs=output_ok,
        inputs_ok=inputs_ok,
        verdict=verdict,
        evidence=list(evidence),
        reasoning=reasoning,
    )


def _finding_evidence(f: ValidatorFinding) -> str:
    ref = f.evidence[0] if f.evidence else f.span_id.split("/", 1)[-1]
    return f"{ref}: [validator {f.check}, {f.severity}] {f.message}"


def score_run(
    run: Run,
    gateway: Gateway,
    *,
    roles: Sequence[str] = DEFAULT_ROLES,
    ctx: ValidationContext | None = None,
    version: str | None = None,
) -> StageAResult:
    ctx = ctx or ValidationContext.default()
    _, prompt_version = load_prompt("node_judge", version)
    findings: dict[str, list[ValidatorFinding]] = {}
    for f in validate_run(run, ctx):
        findings.setdefault(f.span_id, []).append(f)
    by_id = {s.span_id: s for s in run.spans}
    order = topological_order(build_graph(run))
    result = StageAResult(run_id=run.run_id, roles=tuple(roles))
    done: dict[str, SpanScores] = {}

    def local(span: Span) -> str:
        return span.span_id.removeprefix(f"{run.run_id}/")

    def inputs_ok(span: Span) -> float:
        """Lowest output score among the steps this span read from (1.0 if none)."""
        values = [
            score.output_ok_given_inputs
            for ref in span.input_refs
            if ref.span_id in done
            for score in done[ref.span_id].scores.values()
        ]
        return min(values, default=1.0)

    llm_spans: list[Span] = []
    for span_id in order:
        span = by_id[span_id]
        mine = findings.get(span_id, [])
        entry = SpanScores(span_id, local(span), span.kind, "", findings=mine)
        errors = [f for f in mine if f.severity == "error"]
        here = local(span)
        if span.kind == "message":
            entry.source = "input"
            entry.scores[STAGE] = _det(span, "ok", 1.0, 1.0, [], "The workflow input is given.")
        elif span.kind == "llm":
            llm_spans.append(span)
            continue
        elif span.kind == "agent" and (span.error or "").startswith("skipped"):
            entry.source = "skipped"
            entry.scores[STAGE] = _det(
                span,
                "suspicious",
                0.0,
                0.5,
                [f"{here}: {span.error}"],
                "The step did not run: a step it depends on failed.",
            )
        elif span.kind == "agent" and span.error:
            entry.source = "failed"
            entry.scores[STAGE] = _det(
                span,
                "faulty",
                inputs_ok(span),
                0.0,
                [f"{here}: {span.error[:300]}"],
                "The step raised an error instead of producing its output.",
            )
        elif errors:
            entry.source = "validators"
            entry.scores[STAGE] = _det(
                span,
                "faulty",
                inputs_ok(span),
                0.0,
                [_finding_evidence(f) for f in mine],
                "Deterministic checks failed: " + ", ".join(sorted({f.check for f in errors})),
            )
        elif span.kind in JUDGED_KINDS:
            entry.source = "judge"
            for role in roles:
                res = judge_span(gateway, run, span.span_id, role=role, version=version)
                result.judge_calls += 1
                result.model_calls += res.attempts
                result.cost_usd += res.cost_usd
                result.tokens_in += res.tokens_in
                result.tokens_out += res.tokens_out
                if res.score is None:
                    result.judge_failures += 1
                    entry.source = "no answer" if len(roles) == 1 else entry.source
                    entry.scores[role] = AnomalyScore(
                        span_id=span.span_id,
                        judge=role,
                        prompt_version=prompt_version,
                        output_ok_given_inputs=0.5,
                        inputs_ok=0.5,
                        verdict="suspicious",
                        evidence=[f"{here}: {role} gave no valid judgement"],
                        reasoning=f"No valid reply after {res.attempts} tries: "
                        + (res.errors[-1][:300] if res.errors else "no reply"),
                    )
                    continue
                score = res.score
                if mine:  # validator warnings: extra evidence, the verdict stays the judge's
                    score = score.model_copy(
                        update={"evidence": [*score.evidence, *map(_finding_evidence, mine)]}
                    )
                entry.scores[role] = score
        else:
            entry.source = "validators"
            entry.scores[STAGE] = _det(span, "ok", 1.0, 1.0, [], "Not a judged span kind.")
        done[span_id] = entry
        result.spans.append(entry)

    # Model calls inherit their step's scores (they repeat the step's prompt).
    for span in llm_spans:
        step = by_id.get(span.parent_span_id or "")
        parent = done.get(step.span_id) if step else None
        entry = SpanScores(span.span_id, local(span), "llm", "inherited")
        for judge, score in (parent.scores if parent else {}).items():
            entry.scores[f"inherited:{judge}"] = score.model_copy(
                update={
                    "span_id": span.span_id,
                    "judge": f"inherited:{judge}",
                    "reasoning": f"Inherited from step {parent.local_id if parent else '?'}: "
                    + score.reasoning,
                }
            )
        result.spans.append(entry)
    return result


# ---------------------------------------------------------------- the command

REPORT_NAME = "stage_a.json"
DEVSET_NAME = "diagnosis_devset.json"


def main(argv: Sequence[str] | None = None) -> int:
    """`python tasks.py stage-a`: score runs, store every score, log cost per run."""
    import argparse
    import json
    from pathlib import Path

    from agenttrace.store import TraceStore

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Stage A: an AnomalyScore for every span")
    parser.add_argument("--runs", nargs="*", help="run ids (default: the diagnosis dev set)")
    parser.add_argument("--devset", type=Path, default=root / "reports" / DEVSET_NAME)
    parser.add_argument("--roles", nargs="*", default=list(DEFAULT_ROLES))
    parser.add_argument("--limit", type=int, help="score at most this many runs")
    parser.add_argument("--no-store", action="store_true", help="do not store the scores")
    parser.add_argument("-v", "--verbose", action="store_true", help="a line per span")
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
    ctx = ValidationContext.default()
    roles = tuple(args.roles)
    print(f"Stage A: {len(run_ids)} runs, judges {', '.join(roles)}\n")
    header = f"  {'run':<22} {'spans':>5} {'judged':>6} {'valid.':>6} {'inher.':>6} "
    print(header + f"{'fail':>4}  {'ok/susp/faulty':<15} {'cost':>8} {'tokens':>13}")
    rows = []
    with TraceStore.open() as store:
        for run_id in run_ids:
            if not store.has_run(run_id):
                print(f"  {run_id:<22} not stored")
                continue
            res = score_run(store.get_run(run_id), gateway, roles=roles, ctx=ctx)
            if not args.no_store:
                for score in res.all_scores():
                    store.add_anomaly(score)
            src = res.sources()
            v = res.verdicts(roles[0])
            mix = f"{v.get('ok', 0)}/{v.get('suspicious', 0)}/{v.get('faulty', 0)}"
            print(
                f"  {run_id:<22} {len(res.spans):>5} {src.get('judge', 0):>6} "
                f"{res.short_circuited:>6} {src.get('inherited', 0):>6} "
                f"{res.judge_failures:>4}  {mix:<15} ${res.cost_usd:>7.4f} "
                f"{res.tokens_in:>6}/{res.tokens_out:<6}",
                flush=True,
            )
            if args.verbose:
                for s in res.spans:
                    first = next(iter(s.scores.values()), None)
                    verdict = first.verdict if first else "-"
                    print(f"      {s.local_id:<30} {s.source:<11} {verdict}")
            rows.append(
                {
                    "run_id": run_id,
                    "spans": len(res.spans),
                    "scored": sum(bool(s.scores) for s in res.spans),
                    "complete": res.complete,
                    "sources": src,
                    "verdicts": res.verdicts(),
                    "judge_calls": res.judge_calls,
                    "model_calls": res.model_calls,
                    "judge_failures": res.judge_failures,
                    "short_circuited": res.short_circuited,
                    "cost_usd": round(res.cost_usd, 6),
                    "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out,
                }
            )
    gateway.close()
    if not rows:
        return 2
    n = len(rows)
    complete = sum(r["complete"] for r in rows)
    cost = sum(r["cost_usd"] for r in rows)
    calls = sum(r["judge_calls"] for r in rows)
    saved = sum(r["short_circuited"] for r in rows)
    print(f"\nevery span scored: {complete}/{n} runs")
    print(
        f"cost: ${cost:.4f} total, ${cost / n:.4f} per run; {calls} judgements "
        f"({sum(r['model_calls'] for r in rows)} model calls); {saved} spans settled by "
        f"validators, {sum(r['judge_failures'] for r in rows)} judge failures"
    )
    path = root / "reports" / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"roles": roles, "runs": rows}, indent=1), encoding="utf-8")
    print(f"report: {path.relative_to(root).as_posix()}")
    print("PASS" if complete == n else "FAIL")
    return 0 if complete == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
