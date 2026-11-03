"""P2-T4: export -> import yields an identical Run (native JSON and OTLP/JSON)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fake_llm import EXPL, TICKET, default_script, make_gateway
from pydantic import ValidationError
from test_store import traced_run

from agenttrace import cli
from agenttrace.schemas.run import Run
from agenttrace.store import TraceStore
from agenttrace.tracing.export import (
    TraceFormatError,
    _from_nanos,
    _nanos,
    export_runs,
    from_otel,
    import_runs,
    run_to_dict,
    runs_from_dict,
    to_otel,
)
from agenttrace.workflow.support import build_support_workflow, make_resources


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return traced_run(tmp_path, "r1")


def failed_run(tmp_path: Path) -> Run:
    def broken_policy(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the Policy Agent"):
            return {**EXPL}
        return default_script(prompt)

    gateway = make_gateway(tmp_path / "failed", script=broken_policy)
    result = build_support_workflow().run(
        {"ticket": TICKET}, run_id="bad", resources=make_resources(gateway)
    )
    return result.to_run()


# ---------------------------------------------------------------- the done-when


def test_native_export_import_is_identical(tmp_path: Path, run: Run) -> None:
    path = export_runs([run], tmp_path / "run.json")
    (back,) = import_runs(path)
    assert back == run
    assert back.model_dump_json() == run.model_dump_json()
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "agenttrace.run"


def test_otel_export_import_is_identical(tmp_path: Path, run: Run) -> None:
    path = export_runs([run], tmp_path / "otel.json", otel=True)
    (back,) = import_runs(path)
    assert back.model_dump_json() == run.model_dump_json()


def test_bundle_of_runs_round_trips(tmp_path: Path, run: Run) -> None:
    runs = [run, traced_run(tmp_path, "r2", "clean-02"), failed_run(tmp_path)]
    for otel in (False, True):
        back = import_runs(export_runs(runs, tmp_path / f"bundle-{otel}.json", otel=otel))
        assert [r.model_dump_json() for r in back] == [r.model_dump_json() for r in runs]


def test_store_to_file_to_another_store_is_identical(tmp_path: Path, run: Run) -> None:
    with TraceStore.open(tmp_path / "a.db") as a:
        a.save_run(run)
        exported = export_runs([a.get_run("r1")], tmp_path / "x.json")
    with TraceStore.open(tmp_path / "b.db") as b:
        for r in import_runs(exported):
            b.save_run(r)
        assert b.get_run("r1").model_dump_json() == run.model_dump_json()


# ---------------------------------------------------------------- OTLP structure


def test_otel_document_shape(run: Run) -> None:
    doc = to_otel([run])
    (resource,) = doc["resourceSpans"]
    (scope,) = resource["scopeSpans"]
    spans = scope["spans"]
    assert scope["scope"]["name"] == "agenttrace" and len(spans) == len(run.spans)
    assert {s["traceId"] for s in spans} == {spans[0]["traceId"]}
    assert re.fullmatch(r"[0-9a-f]{32}", spans[0]["traceId"])
    assert all(re.fullmatch(r"[0-9a-f]{16}", s["spanId"]) for s in spans)
    # Nesting is preserved through parentSpanId.
    by_name = {s["name"]: s for s in spans}
    assert by_name["policy.llm"]["parentSpanId"] == by_name["policy"]["spanId"]
    assert "parentSpanId" not in by_name["policy"]
    # Times are integer nanoseconds (as strings) with end >= start.
    for s in spans:
        assert int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"]) > 0


def test_otel_llm_spans_use_genai_conventions(run: Run) -> None:
    spans = to_otel([run])["resourceSpans"][0]["scopeSpans"][0]["spans"]
    llm = next(s for s in spans if s["name"] == "policy.llm")
    attrs = {a["key"]: a["value"] for a in llm["attributes"]}
    assert attrs["gen_ai.request.model"] == {"stringValue": "nvidia/nemotron-3.5-lightning-30b-a3b"}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "11"}
    assert attrs["gen_ai.usage.output_tokens"] == {"intValue": "7"}
    assert llm["kind"] == 3 and llm["status"] == {"code": 1}


def test_otel_marks_errors(tmp_path: Path) -> None:
    spans = to_otel([failed_run(tmp_path)])["resourceSpans"][0]["scopeSpans"][0]["spans"]
    policy = next(s for s in spans if s["name"] == "policy")
    assert policy["status"]["code"] == 2 and "AgentFailure" in policy["status"]["message"]


def test_nanosecond_conversion_is_exact() -> None:
    t = datetime(2026, 11, 2, 15, 47, 38, 6953, tzinfo=UTC)
    assert _nanos(t) == "1793634458006953000"
    assert _from_nanos(_nanos(t)) == t


# ---------------------------------------------------------------- traces from other systems


def foreign_doc() -> dict[str, Any]:
    """An OTLP trace as another framework (OpenInference-style) would emit it."""

    def attr(key: str, **value: Any) -> dict[str, Any]:
        return {"key": key, "value": value}

    def span(sid: str, name: str, start: int, end: int, attrs: list[Any], **extra: Any) -> Any:
        return {
            "traceId": "ab" * 16,
            "spanId": sid,
            "name": name,
            "startTimeUnixNano": str(start),
            "endTimeUnixNano": str(end),
            "attributes": attrs,
            **extra,
        }

    base = 1_793_634_458_000_000_000
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [attr("service.name", stringValue="other-app")]},
                "scopeSpans": [
                    {
                        "spans": [
                            # Listed out of order on purpose: import sorts by start time.
                            span(
                                "0000000000000002",
                                "planner.llm",
                                base + 1_000,
                                base + 5_000,
                                [
                                    attr("gen_ai.request.model", stringValue="some/model"),
                                    attr("gen_ai.usage.input_tokens", intValue="40"),
                                    attr("gen_ai.usage.output_tokens", intValue="9"),
                                    attr("input.value", stringValue='{"prompt": "hi"}'),
                                    attr("output.value", stringValue="plain text answer"),
                                ],
                                parentSpanId="0000000000000001",
                            ),
                            span(
                                "0000000000000001",
                                "planner",
                                base,
                                base + 9_000,
                                [attr("openinference.span.kind", stringValue="CHAIN")],
                            ),
                            span(
                                "0000000000000003",
                                "search",
                                base + 6_000,
                                base + 7_000,
                                [attr("openinference.span.kind", stringValue="RETRIEVER")],
                                parentSpanId="0000000000000001",
                                status={"code": 2, "message": "index offline"},
                            ),
                        ]
                    }
                ],
            }
        ]
    }


def test_foreign_otel_trace_is_mapped() -> None:
    (run,) = from_otel(foreign_doc())
    assert run.run_id == "ab" * 16 and run.workflow_version == "otel-import"
    assert [s.node_id for s in run.spans] == ["planner", "planner.llm", "search"]
    planner, llm, search = run.spans
    assert planner.kind == "agent" and planner.parent_span_id is None
    assert llm.kind == "llm" and llm.parent_span_id == planner.span_id
    assert (llm.model, llm.tokens_in, llm.tokens_out) == ("some/model", 40, 9)
    assert llm.input == {"prompt": "hi"} and llm.output == {"value": "plain text answer"}
    assert llm.agent == "planner"
    assert search.kind == "retrieval" and search.error == "index offline"
    assert (llm.ended_at - llm.started_at).total_seconds() == pytest.approx(4e-6)


def test_foreign_trace_survives_a_second_round_trip(tmp_path: Path) -> None:
    (run,) = from_otel(foreign_doc())
    (back,) = import_runs(export_runs([run], tmp_path / "again.json", otel=True))
    assert back.model_dump_json() == run.model_dump_json()


# ---------------------------------------------------------------- bad input


def test_bad_files_are_rejected(tmp_path: Path, run: Run) -> None:
    def write(name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    with pytest.raises(TraceFormatError, match="not JSON"):
        import_runs(write("a.json", "{nope"))
    with pytest.raises(TraceFormatError, match="JSON object"):
        import_runs(write("b.json", "[]"))
    with pytest.raises(TraceFormatError, match="unknown trace format"):
        runs_from_dict({"format": "zipkin", "version": 1})
    with pytest.raises(TraceFormatError, match="version 2"):
        runs_from_dict({**run_to_dict(run), "version": 2})
    with pytest.raises(TraceFormatError, match="no spans"):
        from_otel({"resourceSpans": []})
    tampered = run_to_dict(run)
    tampered["run"]["surprise"] = 1  # strict schema: unknown fields are refused
    with pytest.raises(ValidationError):
        runs_from_dict(tampered)


# ---------------------------------------------------------------- CLI


def test_cli_export_import_between_stores(
    tmp_path: Path, run: Run, capsys: pytest.CaptureFixture[str]
) -> None:
    src, dst = str(tmp_path / "src.db"), str(tmp_path / "dst.db")
    with TraceStore.open(src) as store:
        store.save_run(run)
        store.save_run(traced_run(tmp_path, "r2"))
    out = tmp_path / "out.json"

    assert cli.main(["export", "--scenario", "clean-01", "-o", str(out), "--db", src]) == 0
    assert "exported 2 run(s)" in capsys.readouterr().out
    assert cli.main(["import", str(out), "--db", dst]) == 0
    assert "imported r1" in capsys.readouterr().out
    with TraceStore.open(dst) as store:
        assert store.get_run("r1").model_dump_json() == run.model_dump_json()
        assert store.run_ids() == ["r1", "r2"]

    # Importing again clashes unless --replace.
    assert cli.main(["import", str(out), "--db", dst]) == 2
    assert "already stored" in capsys.readouterr().out
    assert cli.main(["import", str(out), "--db", dst, "--replace"]) == 0

    otel = tmp_path / "otel.json"
    assert cli.main(["export", "r1", "--otel", "-o", str(otel), "--db", src]) == 0
    assert "OTLP/JSON" in capsys.readouterr().out
    assert "resourceSpans" in json.loads(otel.read_text(encoding="utf-8"))


def test_cli_export_and_import_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "e.db")
    assert cli.main(["export", "-o", str(tmp_path / "o.json"), "--db", db]) == 2
    assert cli.main(["export", "missing", "-o", str(tmp_path / "o.json"), "--db", db]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["import", str(bad), "--db", db]) == 2
    assert "cannot import" in capsys.readouterr().out
