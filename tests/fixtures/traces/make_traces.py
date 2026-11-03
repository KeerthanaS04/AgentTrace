"""Generate the trace fixtures in this folder: one valid trace and corrupted variants.

    uv run python tests/fixtures/traces/make_traces.py

Each corrupted file is named for what is wrong with it:
- `<integrity code>.json` must fail `check_integrity` with that code;
- `schema_<what>.json` is so broken the schemas reject it before any check runs.
`tests/test_integrity.py` checks the files still match this script.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_ID = "fx"


def _hash(value: Any) -> str:
    # Same canonical form as agenttrace.gateway.cache.content_hash.
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _t(ms: int) -> str:
    return f"2026-11-03T10:00:00.{ms:03d}000Z"


def _span(span_id: str, kind: str, start: int, end: int, **fields: Any) -> dict[str, Any]:
    return {
        "span_id": f"{RUN_ID}/{span_id}",
        "run_id": RUN_ID,
        "node_id": span_id,
        "agent": span_id.split(".")[0],
        "kind": kind,
        "started_at": _t(start),
        "ended_at": _t(end),
        **fields,
    }


def valid_run() -> dict[str, Any]:
    """__input__ -> a (with a.llm inside) -> b (with b.tool inside)."""
    ticket = "where is my order?"
    return {
        "run_id": RUN_ID,
        "workflow_version": "fixture@1",
        "final_output": {"z": 2},
        "created_at": _t(0),
        "spans": [
            _span("__input__", "message", 0, 0, output={"ticket": ticket}),
            _span(
                "a",
                "agent",
                10,
                40,
                input={"ticket": ticket},
                input_refs=[
                    {
                        "span_id": f"{RUN_ID}/__input__",
                        "key": "ticket",
                        "content_hash": _hash(ticket),
                        "input_key": "ticket",
                    }
                ],
                output={"y": 1},
            ),
            _span(
                "a.llm",
                "llm",
                20,
                30,
                parent_span_id=f"{RUN_ID}/a",
                model="some/model",
                tokens_in=12,
                tokens_out=3,
                output={"text": '{"y": 1}'},
            ),
            _span(
                "b",
                "agent",
                50,
                80,
                input={"v": 1},
                input_refs=[
                    {
                        "span_id": f"{RUN_ID}/a",
                        "key": "y",
                        "content_hash": _hash(1),
                        "input_key": "v",
                    }
                ],
                output={"z": 2},
            ),
            _span("b.tool", "tool", 60, 70, parent_span_id=f"{RUN_ID}/b", output={"ok": True}),
        ],
    }


Corruption = Callable[[dict[str, Any]], None]


def _spans(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["node_id"]: s for s in run["spans"]}


def _ref(run: dict[str, Any], node: str) -> dict[str, Any]:
    ref: dict[str, Any] = _spans(run)[node]["input_refs"][0]
    return ref


def _ghost_parent(r: dict[str, Any]) -> None:
    _spans(r)["a.llm"]["parent_span_id"] = f"{RUN_ID}/ghost"


def _ghost_ref(r: dict[str, Any]) -> None:
    _ref(r, "b")["span_id"] = f"{RUN_ID}/ghost"


def _missing_key(r: dict[str, Any]) -> None:
    _ref(r, "b")["key"] = "w"


def _forged_hash(r: dict[str, Any]) -> None:
    _ref(r, "b")["content_hash"] = "sha256:" + "0" * 64


def _from_future(r: dict[str, Any]) -> None:
    _spans(r)["b"]["started_at"] = _t(35)  # a is still running until 40


def _child_outside(r: dict[str, Any]) -> None:
    _spans(r)["b.tool"]["ended_at"] = _t(90)  # b ended at 80


def _parent_loop(r: dict[str, Any]) -> None:
    _spans(r)["a"]["parent_span_id"] = f"{RUN_ID}/a.llm"  # a.llm's parent is a


def _flow_loop(r: dict[str, Any]) -> None:
    a = _spans(r)["a"]
    a["input_refs"].append(
        {"span_id": f"{RUN_ID}/b", "key": "z", "content_hash": _hash(2), "input_key": "z"}
    )


def _no_spans(r: dict[str, Any]) -> None:
    r["spans"] = []


def _duplicate_id(r: dict[str, Any]) -> None:
    _spans(r)["b.tool"]["span_id"] = f"{RUN_ID}/a"


def _end_before_start(r: dict[str, Any]) -> None:
    _spans(r)["b"]["ended_at"] = _t(45)


def _self_parent(r: dict[str, Any]) -> None:
    _spans(r)["b"]["parent_span_id"] = f"{RUN_ID}/b"


def _foreign_span(r: dict[str, Any]) -> None:
    _spans(r)["b"]["run_id"] = "someone-else"


def _unknown_field(r: dict[str, Any]) -> None:
    _spans(r)["a"]["surprise"] = 1


CORRUPTIONS: dict[str, Corruption] = {
    "orphan_parent": _ghost_parent,
    "orphan_ref": _ghost_ref,
    "ref_key_missing": _missing_key,
    "ref_hash_mismatch": _forged_hash,
    "ref_from_future": _from_future,
    "child_outside_parent": _child_outside,
    "parent_cycle": _parent_loop,
    "ref_cycle": _flow_loop,
    "empty": _no_spans,
    "schema_duplicate_span_id": _duplicate_id,
    "schema_end_before_start": _end_before_start,
    "schema_self_parent": _self_parent,
    "schema_foreign_span": _foreign_span,
    "schema_unknown_field": _unknown_field,
}


def documents() -> dict[str, dict[str, Any]]:
    """File name -> native export document."""

    def wrap(run: dict[str, Any]) -> dict[str, Any]:
        return {"format": "agenttrace.run", "version": 1, "run": run}

    docs = {"valid.json": wrap(valid_run())}
    for name, corrupt in CORRUPTIONS.items():
        run = copy.deepcopy(valid_run())
        corrupt(run)
        docs[f"{name}.json"] = wrap(run)
    return docs


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1) + "\n"


def main() -> None:
    for name, doc in documents().items():
        (HERE / name).write_text(render(doc), encoding="utf-8")
    print(f"wrote {len(documents())} fixtures to {HERE}")


if __name__ == "__main__":
    main()
