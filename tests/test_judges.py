"""P6-T1: versioned judge prompts; the judge's context; schema- and citation-valid scores."""

from __future__ import annotations

import itertools
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.gateway.gateway import Gateway
from agenttrace.judges import registry
from agenttrace.judges import sample as judge_sample
from agenttrace.judges.context import NotJudged, judge_context, summarize
from agenttrace.judges.node_judge import judge_span, render_prompt
from agenttrace.schemas.diagnosis import AnomalyScore
from agenttrace.schemas.run import Run
from agenttrace.workflow.support import build_support_workflow, make_resources

TICKET = {
    "ticket_id": "T-1",
    "customer_id": "C001",
    "subject": "Return a sweater",
    "body": "I want to return the wool sweater from order O1001 for a refund.",
}


def support_run(tmp_path: Path, run_id: str = "r1", scenario_id: str | None = "sc-001") -> Run:
    result = build_support_workflow().run(
        {"ticket": TICKET}, run_id=run_id, resources=make_resources(make_gateway(tmp_path))
    )
    return result.to_run(scenario_id=scenario_id, final_correct=True)


def judge_reply(prompt: str, cite: str | None = None) -> dict[str, Any]:
    span = re.search(r"^Span: (\S+)", prompt, re.M)
    assert span is not None
    return {
        "inputs_ok": 0.9,
        "inputs_reasoning": "The inputs agree with each other.",
        "output_ok_given_inputs": 0.8,
        "output_reasoning": "The output follows the contract.",
        "verdict": "ok",
        "evidence": [{"ref": cite or span.group(1), "claim": "matches its inputs"}],
    }


def judge_script(prompt: str) -> dict[str, Any]:
    if prompt.startswith("You are auditing"):
        return judge_reply(prompt)
    return default_script(prompt)


# ---------------------------------------------------------------- versioned prompts


def test_registered_prompts_are_unchanged() -> None:
    assert registry.check_versions() == []
    text, pid = registry.load_prompt("node_judge")
    assert pid == "node-judge-v1" and "(a) INPUTS" in text and "(b) OUTPUT" in text


def test_editing_a_released_prompt_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "prompts"
    shutil.copytree(registry.PROMPT_DIR, copy)
    monkeypatch.setattr(registry, "PROMPT_DIR", copy)
    monkeypatch.setattr(registry, "VERSIONS_PATH", copy / "VERSIONS.json")
    (copy / "node_judge_v1.md").write_text("an edited prompt {today}", encoding="utf-8")
    (copy / "node_judge_v2.md").write_text("a new draft", encoding="utf-8")
    assert registry.check_versions() == [
        "node_judge_v2.md is not registered",
        "node_judge_v1.md changed after release (sha mismatch)",
    ]
    with pytest.raises(ValueError, match="changed after release"):
        registry.load_prompt("node_judge", "v1")
    with pytest.raises(KeyError, match="not registered"):
        registry.load_prompt("node_judge", "v2")


# ---------------------------------------------------------------- what the judge sees


def test_step_context(tmp_path: Path) -> None:
    run = support_run(tmp_path)
    ctx = judge_context(run, "policy")
    assert ctx.kind == "agent" and "Decide approve, deny, or escalate" in ctx.description
    assert set(ctx.parents) == {"router", "retriever", "tool_agent"}  # input refs' producers
    assert set(ctx.parents["router"]) == {"category"}  # only what policy read from it
    assert ctx.citable[0] == "policy" and "KB-REF-001" in ctx.doc_ids
    tools = judge_context(run, "tool_agent")
    assert set(tools.calls) == {
        "tool_agent.order_lookup",
        "tool_agent.account_status",
        "tool_agent.refund_api",
    }
    assert set(tools.citable) >= set(tools.calls)


def test_call_context(tmp_path: Path) -> None:
    run = support_run(tmp_path)
    ctx = judge_context(run, "tool_agent.order_lookup")
    assert ctx.kind == "tool" and "Look up an order by ID" in ctx.description
    assert list(ctx.parents) == ["tool_agent"] and ctx.output["found"] is True
    search = judge_context(run, "retriever.search")
    assert search.kind == "retrieval" and search.citable == ["retriever.search", "retriever"]
    with pytest.raises(NotJudged, match="judge its step"):
        judge_context(run, "policy.llm")


def test_the_judge_never_sees_ground_truth(tmp_path: Path) -> None:
    run = support_run(tmp_path, scenario_id="sc-077")
    template, _ = registry.load_prompt("node_judge")
    for local in ("router", "policy", "writer", "tool_agent.order_lookup"):
        prompt = render_prompt(judge_context(run, local), template, "2026-01-20")
        assert "sc-077" not in prompt and "final_correct" not in prompt
        for leak in ("expected_final", "root_cause", "fault_code", "injector", "occurrence"):
            assert leak not in prompt


def test_parent_outputs_are_summarized() -> None:
    long = {"body": "x" * 1000, "items": list(range(20))}
    short = summarize(long)
    assert short["body"].endswith("[...]") and len(short["body"]) < 450
    assert short["items"][-1] == "[... 12 more]" and len(short["items"]) == 9


# ---------------------------------------------------------------- judging


def test_a_valid_judgement_becomes_an_anomaly_score(tmp_path: Path) -> None:
    run = support_run(tmp_path)
    gateway = make_gateway(tmp_path / "j", script=judge_script)
    res = judge_span(gateway, run, "policy", role="judge_fast")
    assert res.valid and res.attempts == 1
    score = res.score
    assert isinstance(score, AnomalyScore)
    assert score.span_id == "r1/policy" and score.judge == "judge_fast"
    assert score.prompt_version == "node-judge-v1" and score.verdict == "ok"
    assert score.evidence == ["policy: matches its inputs"]
    assert score.reasoning.startswith("(a) inputs:") and "(b) output:" in score.reasoning


def test_a_citation_outside_the_context_is_sent_back(tmp_path: Path) -> None:
    run = support_run(tmp_path)
    calls = itertools.count()

    def bad_then_good(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are auditing"):
            return judge_reply(prompt, cite="KB-XXX-999" if next(calls) == 0 else None)
        return default_script(prompt)

    res = judge_span(make_gateway(tmp_path / "j", script=bad_then_good), run, "policy")
    assert res.valid and res.attempts == 2
    assert "not in the context" in res.errors[0] and "KB-XXX-999" in res.errors[0]


def test_a_judge_that_never_answers_validly_gets_no_score(tmp_path: Path) -> None:
    run = support_run(tmp_path)

    def nonsense(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are auditing"):
            return {"verdict": "fine"}
        return default_script(prompt)

    res = judge_span(make_gateway(tmp_path / "j", script=nonsense), run, "writer")
    assert not res.valid and res.score is None and res.attempts == 3 and len(res.errors) == 3


def test_a_non_ok_verdict_needs_evidence() -> None:
    from agenttrace.judges.node_judge import JudgeReply

    with pytest.raises(ValueError):
        JudgeReply.model_validate(
            {
                "inputs_ok": 0.2,
                "inputs_reasoning": "r",
                "output_ok_given_inputs": 0.1,
                "output_reasoning": "r",
                "verdict": "faulty",
                "evidence": [],
            }
        )


# ---------------------------------------------------------------- the sample command


def test_the_draw_is_seeded_and_stratified(tmp_path: Path) -> None:
    runs = [support_run(tmp_path / str(i), run_id=f"sc-00{i}-default-s1") for i in range(4)]
    a = judge_sample.draw_sample(runs, 20)
    assert a == judge_sample.draw_sample(runs, 20)
    steps = Counter(local for _, local in a if "." not in local)
    assert {"router", "retriever", "tool_agent", "policy", "writer"} <= set(steps)
    assert any(local.startswith("tool_agent.") for _, local in a)
    assert any(local.startswith("retriever.search") for _, local in a)


def test_judge_sample_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    with TraceStore.open() as store:
        for i in range(3):
            store.save_run(support_run(tmp_path / str(i), run_id=f"sc-00{i}-default-s1"))
    monkeypatch.setattr(judge_sample, "REPORT_PATH", tmp_path / "judge_sample.json")
    monkeypatch.setattr(judge_sample, "ROOT", tmp_path)
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(
            lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}", script=judge_script)
        ),
    )
    assert judge_sample.main(["-n", "10"]) == 0
    out = capsys.readouterr().out
    assert "judge_fast (muse-spark-1.3): 10/10 schema-valid" in out
    assert "judge_nvidia (nvidia/nemotron-3-ultra-550b-a55b): 10/10 schema-valid" in out
    assert "judge_oss (qwen/qwen-2.5-72b-instruct): 10/10 schema-valid" in out and "PASS" in out
    report = json.loads((tmp_path / "judge_sample.json").read_text("utf-8"))
    assert len(report["sample"]) == 10 and report["summary"]["judge_fast"]["valid"] == 10
    assert report["results"]["judge_fast"][0]["score"]["prompt_version"] == "node-judge-v1"


def test_a_partial_rerun_keeps_the_other_judges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "judge_sample.json"
    monkeypatch.setattr(judge_sample, "REPORT_PATH", path)
    sample = [{"run_id": "r", "span": "policy"}]
    judge_sample.write_report(
        {
            "sample": sample,
            "summary": {"a": 1, "b": 2},
            "skipped": {},
            "results": {"a": [], "b": []},
        }
    )
    judge_sample.write_report(
        {"sample": sample, "summary": {"b": 3}, "skipped": {}, "results": {"b": ["new"]}}
    )
    doc = json.loads(path.read_text("utf-8"))
    assert doc["summary"] == {"a": 1, "b": 3} and doc["results"] == {"a": [], "b": ["new"]}
    # A different sample starts a fresh report.
    other = [{"run_id": "r", "span": "writer"}]
    judge_sample.write_report({"sample": other, "summary": {"b": 4}, "skipped": {}, "results": {}})
    assert json.loads(path.read_text("utf-8"))["summary"] == {"b": 4}


def test_judge_sample_reports_a_rejected_key_and_carries_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "s.db"))
    with TraceStore.open() as store:
        for i in range(3):
            store.save_run(support_run(tmp_path / str(i), run_id=f"sc-00{i}-default-s1"))
    monkeypatch.setattr(judge_sample, "REPORT_PATH", tmp_path / "judge_sample.json")
    monkeypatch.setattr(judge_sample, "ROOT", tmp_path)
    count = itertools.count()

    def gw(*a: Any, **k: Any) -> Any:
        g = make_gateway(tmp_path / f"g{next(count)}", script=judge_script)
        original = g.complete

        def complete(role: str, *args: Any, **kwargs: Any) -> Any:
            if role == "judge_oss":
                from agenttrace.gateway.types import ProviderRequestError

                raise ProviderRequestError("openrouter: HTTP 401: User not found.", status_code=401)
            return original(role, *args, **kwargs)

        g.complete = complete  # type: ignore[method-assign]
        return g

    monkeypatch.setattr(Gateway, "from_env", classmethod(lambda cls, *a, **k: gw()))
    assert judge_sample.main(["-n", "10", "--roles", "judge_fast", "judge_oss"]) == 0
    out = capsys.readouterr().out
    assert "judge_fast (muse-spark-1.3): 10/10 schema-valid" in out
    assert "judge_oss: skipped (provider rejected the key" in out and "401" in out
