"""Stage C (P6-T6): escalate uncertain diagnoses to Muse Wonder for a final ranking.

    decision = should_escalate(attr, stored, findings, rules)    # reasons, no model call
    if decision.escalate:
        deep = deep_rank(gateway, run, attr, scores, stored, findings, decision.reasons)
    diagnosis = final_diagnosis(attr, deep, decision, agreement)

Rules (config/escalation.yaml; any one fires):

    low_agreement             the run's mean pairwise Cohen's kappa < min_kappa
    close_top_two             top two roots in the same evidence tier, confidences within
                              max_gap of each other
    multiple_roots            more than max_roots root candidates
    long_chain                a propagated error's path is longer than max_hops hops
    validator_judge_conflict  a validator ERROR on a call inside a step that a judge
                              called ok

Wonder (role `judge_deep`, prompt `escalation-v1`) gets a compact view: the step graph,
every span's first-pass verdict and the judges' votes, the failed deterministic checks,
the ranked candidates and propagated errors, and the top candidates' inputs and outputs.
It returns its own ranking (0-5 roots, each citing a span it was shown; checked the way
the node judge's citations are) and a narrative. Its cost is kept apart from Stage A's
(`DeepResult.cost_usd`, ledger tag `escalation/<run>`), and a batch stops escalating at
the budget in the config. If Wonder never gives a valid ranking, the Stage B diagnosis
stands and the failure is recorded. Nothing here sees ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from agenttrace.diagnosis.ensemble import cohens_kappa
from agenttrace.diagnosis.stage_b import Attribution, _tier
from agenttrace.gateway.gateway import Gateway
from agenttrace.graph import build_graph, topological_order
from agenttrace.judges.context import summarize
from agenttrace.judges.registry import load_prompt
from agenttrace.schemas.base import NonEmptyStr, Probability
from agenttrace.schemas.diagnosis import (
    AnomalyScore,
    Diagnosis,
    FaultType,
    PropagatedError,
    RootCause,
    ValidatorFinding,
)
from agenttrace.schemas.run import Run
from agenttrace.workflow.agents.base import LLMReply

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "escalation.yaml"
DEEP_ROLE = "judge_deep"
DETERMINISTIC = {"stage-a", "ensemble"}  # stored score kinds that are not one judge's vote
_ALLOWED: ContextVar[frozenset[str] | None] = ContextVar("deep_allowed", default=None)


# ---------------------------------------------------------------- rules


@dataclass(frozen=True)
class Rules:
    low_agreement: float | None = 0.4
    close_gap: float | None = 0.15
    max_roots: int | None = 1
    max_hops: int | None = 3
    conflict: bool = True
    max_escalations: int = 20
    max_usd: float = 5.0

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> Rules:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        r = doc.get("rules") or {}
        b = doc.get("budget") or {}

        def on(name: str) -> dict[str, Any] | None:
            rule = r.get(name) or {}
            return rule if rule.get("enabled", True) else None

        la, ct, mr, lc = (
            on("low_agreement"),
            on("close_top_two"),
            on("multiple_roots"),
            on("long_chain"),
        )
        return cls(
            low_agreement=float(la["min_kappa"]) if la else None,
            close_gap=float(ct["max_gap"]) if ct else None,
            max_roots=int(mr["max_roots"]) if mr else None,
            max_hops=int(lc["max_hops"]) if lc else None,
            conflict=on("validator_judge_conflict") is not None,
            max_escalations=int(b.get("max_escalations", 20)),
            max_usd=float(b.get("max_usd", 5.0)),
        )


def judge_votes(stored: Iterable[AnomalyScore]) -> dict[str, dict[str, str]]:
    """span id -> {judge: verdict} for the individual judges' valid replies."""
    votes: dict[str, dict[str, str]] = {}
    for s in stored:
        if s.judge in DETERMINISTIC or s.judge.startswith("inherited:"):
            continue
        if any("gave no valid judgement" in e for e in s.evidence):
            continue  # Stage A's placeholder for a failed judge
        votes.setdefault(s.span_id, {})[s.judge] = s.verdict
    return votes


def run_agreement(stored: Iterable[AnomalyScore]) -> float | None:
    """Mean pairwise Cohen's kappa over the spans each pair of judges rated."""
    votes = judge_votes(stored)
    judges = sorted({j for v in votes.values() for j in v})
    kappas = []
    for i, a in enumerate(judges):
        for b in judges[i + 1 :]:
            both = [v for v in votes.values() if a in v and b in v]
            k = cohens_kappa([v[a] for v in both], [v[b] for v in both])
            if k is not None:
                kappas.append(k)
    return sum(kappas) / len(kappas) if kappas else None


@dataclass
class Decision:
    reasons: list[str] = field(default_factory=list)
    agreement: float | None = None

    @property
    def escalate(self) -> bool:
        return bool(self.reasons)


def should_escalate(
    run: Run,
    attr: Attribution,
    scores: Mapping[str, AnomalyScore],
    stored: Sequence[AnomalyScore],
    findings: Iterable[ValidatorFinding],
    rules: Rules,
) -> Decision:
    d = Decision(agreement=run_agreement(stored))
    k, floor = d.agreement, rules.low_agreement
    if floor is not None and k is not None and k < floor:
        d.reasons.append(f"judge agreement kappa {k:.2f} < {floor}")
    roots = attr.roots
    if rules.close_gap is not None and len(roots) >= 2:
        a, b = roots[0], roots[1]
        same_tier = _tier(scores[a.span_id]) == _tier(scores[b.span_id])
        if same_tier and abs(a.confidence - b.confidence) < rules.close_gap:
            d.reasons.append(
                f"top two roots too close ({a.node_id} {a.confidence:.2f}, "
                f"{b.node_id} {b.confidence:.2f})"
            )
    if rules.max_roots is not None and len(roots) > rules.max_roots:
        d.reasons.append(f"{len(roots)} root candidates")
    if rules.max_hops is not None:
        longest = max((len(p.path) - 1 for p in attr.propagated), default=0)
        if longest > rules.max_hops:
            d.reasons.append(f"propagation chain of {longest} hops")
    if rules.conflict:
        votes = judge_votes(stored)
        parent = {s.span_id: s.parent_span_id for s in run.spans}
        for f in findings:
            step = parent.get(f.span_id)
            if f.severity != "error" or step is None:
                continue
            ok = sorted(j for j, v in votes.get(step, {}).items() if v == "ok")
            if ok:
                local = f.span_id.split("/", 1)[1]
                d.reasons.append(
                    f"validators flag {local} ({f.check}) but {', '.join(ok)} passed its step"
                )
                break
    return d


# ---------------------------------------------------------------- Wonder


class DeepRoot(LLMReply):
    span: NonEmptyStr = Field(description="A span id listed in the context.")
    confidence: Probability
    fault_type: FaultType
    reason: NonEmptyStr


class DeepReply(LLMReply):
    ranking: list[DeepRoot] = Field(default_factory=list, max_length=5)
    narrative: NonEmptyStr

    @model_validator(mode="after")
    def _cites_only_the_context(self) -> DeepReply:
        allowed = _ALLOWED.get()
        if allowed is not None:
            unknown = [r.span for r in self.ranking if r.span not in allowed]
            if unknown:
                raise ValueError(
                    f"ranking cites {unknown}, which are not spans of this run; "
                    f"use only: {sorted(allowed)}"
                )
        return self


@dataclass
class DeepResult:
    roots: list[RootCause]
    narrative: str
    valid: bool
    attempts: int
    model: str = ""
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


def _local(run: Run, span_id: str) -> str:
    return span_id.removeprefix(f"{run.run_id}/")


def render_context(
    run: Run,
    attr: Attribution,
    scores: Mapping[str, AnomalyScore],
    stored: Sequence[AnomalyScore],
    findings: Iterable[ValidatorFinding],
    reasons: Sequence[str],
    template: str,
) -> tuple[str, set[str]]:
    """Wonder's prompt, and the span ids it may cite."""
    g = build_graph(run)
    loc = {n: g.nodes[n].get("local_id", n) for n in g}
    edges = sorted(
        {f"{loc[u]} -> {loc[v]}" for u, v, d in g.edges(data=True) if d.get("kind") == "data"}
    )
    calls: dict[str, list[str]] = {}
    for n in topological_order(g):
        p = g.nodes[n].get("parent")
        if p is not None and g.nodes[n].get("kind") in ("tool", "retrieval"):
            calls.setdefault(loc[p], []).append(loc[n])
    graph = "\n".join([*edges, *(f"{s} contains: {', '.join(c)}" for s, c in calls.items())])
    votes = judge_votes(stored)
    lines, allowed = [], set()
    for n in topological_order(g):
        if g.nodes[n].get("kind") in ("llm", "message"):
            continue
        allowed.add(loc[n])
        s = scores.get(n)
        if s is None:
            continue
        v = ", ".join(f"{j}={x}" for j, x in sorted(votes.get(n, {}).items())) or s.judge
        lines.append(
            f"- {loc[n]} ({g.nodes[n].get('kind')}): {s.verdict}, inputs_ok "
            f"{s.inputs_ok:.2f}, output_ok {s.output_ok_given_inputs:.2f} [{v}]"
        )
    found = [f"- {_local(run, f.span_id)}: [{f.severity} {f.check}] {f.message}" for f in findings]
    cands = [
        f"{i}. {_local(run, r.span_id)} ({r.fault_type}, confidence {r.confidence:.2f}); "
        f"evidence: {' | '.join(r.evidence[:3])}"
        for i, r in enumerate(attr.roots, 1)
    ]
    props = [
        f"- {_local(run, p.span_id)} via {' -> '.join(_local(run, x) for x in p.path)}"
        for p in attr.propagated
    ]
    by_id = {s.span_id: s for s in run.spans}
    details = []
    for r in attr.roots[:4]:
        sp = by_id[r.span_id]
        details.append(
            f"### {_local(run, r.span_id)}\ninputs: "
            f"{json.dumps(summarize(sp.input), default=str)}\noutput: "
            f"{json.dumps(summarize(sp.output), default=str)}"
        )
    prompt = template.format(
        reasons="\n".join(f"- {x}" for x in reasons),
        graph=graph or "(no data edges)",
        spans="\n".join(lines),
        findings="\n".join(found) or "(none)",
        candidates="\n".join(cands) or "(none)",
        propagated="\n".join(props) or "(none)",
        details="\n\n".join(details) or "(none)",
    )
    return prompt, allowed


def deep_rank(
    gateway: Gateway,
    run: Run,
    attr: Attribution,
    scores: Mapping[str, AnomalyScore],
    stored: Sequence[AnomalyScore],
    findings: Sequence[ValidatorFinding],
    reasons: Sequence[str],
    *,
    role: str = DEEP_ROLE,
) -> DeepResult:
    template, _ = load_prompt("escalation")
    prompt, allowed = render_context(run, attr, scores, stored, findings, reasons, template)
    token = _ALLOWED.set(frozenset(allowed))
    try:
        result = gateway.complete_structured(
            role, [{"role": "user", "content": prompt}], DeepReply, tag=f"escalation/{run.run_id}"
        )
    finally:
        _ALLOWED.reset(token)
    calls = result.completions
    out = DeepResult(
        roots=[],
        narrative="",
        valid=result.value is not None,
        attempts=result.attempts,
        model=calls[-1].model if calls else "",
        errors=result.errors,
        cost_usd=sum(c.cost_usd for c in calls),
        tokens_in=sum(c.tokens_in for c in calls),
        tokens_out=sum(c.tokens_out for c in calls),
    )
    if result.value is None:
        return out
    first = {r.span_id: r for r in attr.roots}
    node_of = {s.span_id: s.node_id for s in run.spans}
    by_node: dict[str, RootCause] = {}
    for item in result.value.ranking:
        span_id = f"{run.run_id}/{item.span}"
        node = node_of[span_id]
        if node in by_node:
            # One root per node (as in Stage B): a second span of it is the same fault.
            kept = by_node[node]
            kept.evidence.append(f"{item.span}: the same fault ({item.reason})")
            continue
        earlier = first.get(span_id)
        by_node[node] = RootCause(
            node_id=node,
            span_id=span_id,
            fault_type=item.fault_type,
            confidence=item.confidence,
            evidence=[f"{item.span}: {item.reason}", *(earlier.evidence if earlier else [])],
        )
        out.roots.append(by_node[node])
    out.narrative = result.value.narrative
    return out


def final_diagnosis(
    attr: Attribution,
    decision: Decision,
    deep: DeepResult | None,
) -> Diagnosis:
    """Stage B's diagnosis, or Wonder's ranking when it was escalated and answered."""
    if deep is None or not deep.valid:
        diag = attr.to_diagnosis(decision.agreement, cost_usd=deep.cost_usd if deep else 0.0)
        if decision.escalate:
            note = "" if deep is None else "; Wonder gave no valid ranking, Stage B stands"
            diag = diag.model_copy(
                update={
                    "escalated_to_deep": True,
                    "escalation_reason": "; ".join(decision.reasons) + note,
                    "narrative": (deep.errors[-1][:500] if deep and deep.errors else ""),
                }
            )
        return diag
    roots = {r.span_id for r in deep.roots}
    propagated = []
    for p in attr.propagated:
        if p.span_id in roots:
            continue
        kept = [c for c in p.caused_by if c in roots]
        if kept and p.path[0] in kept:
            propagated.append(PropagatedError(**{**p.model_dump(), "caused_by": kept}))
    return Diagnosis(
        diagnosis_id=f"{attr.run_id}-stage-c-v1",
        run_id=attr.run_id,
        prompt_version="escalation-v1",
        root_causes=deep.roots,
        propagated=propagated,
        escalated_to_deep=True,
        escalation_reason="; ".join(decision.reasons),
        judge_agreement=max(-1.0, min(1.0, decision.agreement or 0.0)),
        narrative=deep.narrative,
        cost_usd=deep.cost_usd,
    )
