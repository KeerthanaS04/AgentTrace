"""Stage B (P6-T5): causal attribution, from anomaly scores to root causes vs propagated errors.

    attr = attribute(run, scores, findings)        # no model call
    attr.roots          # RootCause, ranked best first
    attr.propagated     # PropagatedError: caused_by roots, path root -> ... -> span

The graph is the run's (P3): data edges producer -> consumer, and calls nested in steps.
What *feeds* a span:

    a step   the steps it read from (input refs) and the calls made inside it
    a call   (tool / search) the steps its enclosing step read from

Algorithm (spans in causal order, so a span's feeders are decided before it):

    1. anomalous    aggregate verdict faulty or suspicious. Search spans count as
                    anomalous only if a validator flagged them, unless
                    `judge_searches=True` (the P6-T4 ensemble split most on searches:
                    a soft call the validators make exactly). A step with an anomalous
                    call inside it is anomalous too: it passes the call's result on.
    2. root         anomalous, and nothing anomalous feeds it; or its data inputs are
                    judged trustworthy (`inputs_ok` >= `trusted_inputs`) and no call
                    inside it is anomalous, so its own output is at fault.
    3. propagated   anomalous and explained by an anomalous feeder: a call inside it
                    (a step's calls are its material, whatever `inputs_ok` says), or a
                    step it read from when its inputs are judged untrustworthy. Its
                    roots are the explaining feeders' roots; its path runs from one root
                    through the explaining chain to it.
    4. ranking      roots by evidence tier first: deterministic proof (validators, a
                    failed step) > a judge's "faulty" > a judge's "suspicious"; then the
                    spec's order: earliest in causal order, higher anomaly confidence,
                    more affected descendants. Confidence = 1 - output_ok_given_inputs.
                    (The tier is a deviation from the spec's order: on the dev set a weak
                    upstream hunch otherwise outranked proven faults, P6-T5 findings.)
    5. one per node several spans of the same node as roots (e.g. every search a
                    retrieval fault hit) are one root cause: the first span stands for
                    the node, the others are listed in its evidence and their
                    propagated errors are credited to it.

Model-call spans are left out (they carry their step's score). The fault type of a root
is read off its node kind and validator checks; nothing here sees ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import networkx as nx

from agenttrace.graph import build_graph, downstream, topological_order
from agenttrace.schemas.diagnosis import (
    AnomalyScore,
    Diagnosis,
    FaultType,
    PropagatedError,
    RootCause,
    ValidatorFinding,
)
from agenttrace.schemas.run import Run

PROMPT_VERSION = "stage-b-v1"
TRUSTED_INPUTS = 0.5  # inputs_ok at or above this: the judge trusts the span's data inputs
ANOMALOUS = ("faulty", "suspicious")

_TOOL_MALFORMED = {"tool_error_payload", "tool_schema", "tool_record_complete"}
_RETRIEVAL_WRONG = {"hit_not_indexed", "hit_score_mismatch", "hit_irrelevant", "docs_superseded"}


@dataclass
class Attribution:
    run_id: str
    anomalous: list[str] = field(default_factory=list)  # span ids, causal order
    roots: list[RootCause] = field(default_factory=list)  # ranked
    propagated: list[PropagatedError] = field(default_factory=list)
    explained_by: dict[str, list[str]] = field(default_factory=dict)  # span -> feeders used

    def role(self, span_id: str) -> str:
        if any(r.span_id == span_id for r in self.roots):
            return "root_cause"
        if any(p.span_id == span_id for p in self.propagated):
            return "propagated"
        return "anomalous" if span_id in self.anomalous else "ok"

    def to_diagnosis(self, judge_agreement: float | None, *, cost_usd: float = 0.0) -> Diagnosis:
        return Diagnosis(
            diagnosis_id=f"{self.run_id}-{PROMPT_VERSION}",
            run_id=self.run_id,
            prompt_version=PROMPT_VERSION,
            root_causes=self.roots,
            propagated=self.propagated,
            judge_agreement=max(-1.0, min(1.0, judge_agreement or 0.0)),
            cost_usd=cost_usd,
        )


def fault_type(node_id: str, kind: str, checks: Iterable[str]) -> FaultType:
    """The diagnosis taxonomy's fault type for a root, from its node and validator checks."""
    checks = set(checks)
    if kind == "retrieval" or node_id == "retriever":
        if checks & {"search_empty", "docs_nonempty"}:
            return "retrieval_empty"
        return "retrieval_wrong"
    if kind == "tool" or node_id == "tool_agent":
        return "tool_malformed" if checks & _TOOL_MALFORMED else "tool_wrong_value"
    if node_id in ("router", "policy") and "numbers_supported" not in checks:
        return "invalid_decision"
    if node_id in ("policy", "writer"):
        return "hallucination"
    return "other"


def _feeders(g: nx.DiGraph, span_id: str) -> tuple[list[str], list[str]]:
    """(data feeders, calls inside it) for a span, per the module doc."""
    attrs = g.nodes[span_id]
    calls = [
        child
        for child, _, d in g.in_edges(span_id, data=True)
        if d.get("kind") == "nested" and g.nodes[child].get("kind") in ("tool", "retrieval")
    ]
    if attrs.get("parent") is None:
        data = [u for u, _, d in g.in_edges(span_id, data=True) if d.get("kind") == "data"]
        return data, calls
    step = attrs["parent"]
    data = [u for u, _, d in g.in_edges(step, data=True) if d.get("kind") == "data"]
    return data, []


def attribute(
    run: Run,
    scores: Mapping[str, AnomalyScore],
    findings: Iterable[ValidatorFinding] = (),
    *,
    trusted_inputs: float = TRUSTED_INPUTS,
    judge_searches: bool = False,
) -> Attribution:
    """Roots vs propagated errors from one aggregate score per span (see module doc)."""
    g = build_graph(run)
    checks: dict[str, set[str]] = {}
    for f in findings:
        checks.setdefault(f.span_id, set()).add(f.check)
    order = [n for n in topological_order(g) if g.nodes[n].get("kind") != "llm"]
    position = {n: i for i, n in enumerate(order)}
    result = Attribution(run.run_id)

    def is_anomalous(span_id: str) -> bool:
        score = scores.get(span_id)
        if score is None or score.verdict not in ANOMALOUS:
            return False
        if g.nodes[span_id].get("kind") == "retrieval" and not judge_searches:
            return score.judge == "stage-a"  # only a validator-settled (deterministic) flag
        return True

    roots_of: dict[str, list[str]] = {}  # span -> root span ids explaining it
    path_of: dict[str, list[str]] = {}  # span -> root ... span
    root_ids: list[str] = []
    for span_id in order:
        data, calls = _feeders(g, span_id)
        bad_calls = [c for c in calls if c in roots_of]
        # A step passes its calls' results on (its contract; validators check it), so an
        # anomalous call makes its step anomalous even if the step's judge said ok.
        if not is_anomalous(span_id) and not bad_calls:
            continue
        result.anomalous.append(span_id)
        score = scores.get(span_id)
        bad_data = [d for d in data if d in roots_of]
        explaining = list(bad_calls)
        if not bad_calls and bad_data and score is not None and score.inputs_ok < trusted_inputs:
            explaining = bad_data
        if not explaining:
            root_ids.append(span_id)
            roots_of[span_id] = [span_id]
            path_of[span_id] = [span_id]
            continue
        explaining.sort(key=lambda n: position[n])
        result.explained_by[span_id] = explaining
        roots_of[span_id] = list(dict.fromkeys(r for e in explaining for r in roots_of[e]))
        path_of[span_id] = [*path_of[explaining[0]], span_id]

    # One root per node: the earliest span of the node stands for the others.
    rep_of: dict[str, str] = {}
    first_of_node: dict[str, str] = {}
    for r in root_ids:
        node = g.nodes[r].get("node_id", r)
        rep_of[r] = first_of_node.setdefault(node, r)
    reps = list(dict.fromkeys(rep_of[r] for r in root_ids))
    members = {k: [r for r in root_ids if rep_of[r] == k] for k in reps}
    reach = {k: set().union(*(downstream(g, r) for r in members[k])) for k in reps}

    def tier(r: str) -> int:
        best = min(members[r], key=lambda m: _tier(scores[m]))
        return _tier(scores[best])

    def confidence(r: str) -> float:
        return max(1 - scores[m].output_ok_given_inputs for m in members[r])

    def rank(r: str) -> tuple[int, int, float, int]:
        return (tier(r), position[r], -confidence(r), -len(reach[r]))

    for r in sorted(reps, key=rank):
        s = scores[r]
        attrs = g.nodes[r]
        evidence = list(dict.fromkeys(e for m in members[r] for e in scores[m].evidence))
        evidence = evidence or [f"{attrs.get('local_id', r)}: {s.verdict}"]
        others = [g.nodes[m].get("local_id", m) for m in members[r] if m != r]
        if others:
            evidence.append(
                f"{attrs.get('local_id', r)}: the same fault also at {', '.join(others)}"
            )
        result.roots.append(
            RootCause(
                node_id=attrs.get("node_id", r),
                span_id=r,
                fault_type=fault_type(
                    attrs.get("node_id", ""), attrs.get("kind", ""), checks.get(r, ())
                ),
                confidence=round(max(0.0, min(1.0, confidence(r))), 4),
                evidence=evidence,
            )
        )
    for span_id in result.anomalous:
        if span_id in root_ids:
            continue
        path = path_of[span_id]
        result.propagated.append(
            PropagatedError(
                span_id=span_id,
                node_id=g.nodes[span_id].get("node_id", span_id),
                caused_by=list(dict.fromkeys(rep_of[r] for r in roots_of[span_id])),
                path=[rep_of.get(path[0], path[0]), *path[1:]],
            )
        )
    return result


def _tier(score: AnomalyScore) -> int:
    """Evidence strength: 0 deterministic (validators, failed step), 1 a judge's faulty,
    2 a judge's suspicious."""
    if score.judge == "stage-a":
        return 0
    return 1 if score.verdict == "faulty" else 2


def aggregate_scores(stored: Iterable[AnomalyScore]) -> dict[str, AnomalyScore]:
    """One score per span from stored Stage A / ensemble rows: the ensemble's aggregate if
    there is one, else the deterministic Stage A score, else the first single judge."""
    rank = {"ensemble": 0, "stage-a": 1}
    best: dict[str, AnomalyScore] = {}
    for s in stored:
        if s.judge.startswith("inherited:"):
            continue
        current = best.get(s.span_id)
        if current is None or rank.get(s.judge, 2) < rank.get(current.judge, 2):
            best[s.span_id] = s
    return best


def valid_path(g: nx.DiGraph, path: list[str]) -> bool:
    """Each step of a path follows a data edge, or goes from a call to its step, or from a
    step to a call inside a step that read from it (how a call is fed)."""
    for a, b in zip(path, path[1:], strict=False):
        if g.has_edge(a, b):
            continue
        parent = g.nodes[b].get("parent")
        if parent is not None and g.has_edge(a, parent):
            continue
        return False
    return True
