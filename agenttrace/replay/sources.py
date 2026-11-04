"""Patch sources (P5-T4): where a replacement output comes from.

    rule_patch(run, "tool_agent.order_lookup", resources)    # deterministic fix, or None
    model_patch(run, "policy", gateway, hint="...")           # a model's proposed output

Three sources, tagged on every patch, replay, and result as `patch_source`:

    oracle  the true value: the node's output in the clean run (upper bound). It needs
            ground truth, so it lives with the harness: `scenarios.oracle`.
    rule    a deterministic fix, computed now and handed over as the replacement output:
              tool call         re-call the tool with its recorded arguments against the
                                real tools; the result is validated against the tool's
                                output schema (a wrong value or malformed reply is replaced)
              search            re-run the search with its recorded query
              retriever step    re-run all its searches from its recorded queries
                                (the Retriever's own `select_docs`)
              tool_agent step   re-call every recorded tool call
            None for nodes whose output is a model's judgement (router, policy, writer,
            and model calls): no rule can recompute those.
    model   a model (role `patcher`) sees the node's contract, its inputs, and its
            suspect output, and proposes the output it should have produced, same keys.

Neither `rule` nor `model` sees ground truth (this package is outside the harness).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.patcher import Patch, PatchError, resolve_target, step_of
from agenttrace.schemas.base import Probability
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span
from agenttrace.workflow.agents.base import LLMReply
from agenttrace.workflow.agents.retriever import select_docs
from agenttrace.workflow.agents.tool_agent import ToolRequest, execute_call
from agenttrace.workflow.contracts import CONTRACTS
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import KnowledgeBase
from agenttrace.workflow.tools import ToolBox
from agenttrace.workflow.tools.tools import ToolInputError, tool_result_json

PATCHER_ROLE = "patcher"


class PatchSourceError(RuntimeError):
    """A source could not produce a patch (e.g. the model returned no valid output)."""


def _input_of(run: Run, name: str) -> Any:
    """A workflow input value as recorded on the `__input__` span."""
    span = next(s for s in run.spans if s.node_id == "__input__")
    return span.output[name]


def _fresh_toolbox(resources: Mapping[str, Any]) -> ToolBox:
    """A toolbox on its own copy of the fixture database, so a rule never mutates the
    one a replay will use."""
    box = resources["tools"]
    if not isinstance(box, ToolBox):
        raise TypeError("resources['tools'] must be a ToolBox")
    return ToolBox(box.db)


# ---------------------------------------------------------------- rule


def rule_patch(run: Run, target: str, resources: Mapping[str, Any]) -> Patch | None:
    """A deterministic replacement for `target`, or None if no rule applies to it."""
    span = resolve_target(run, target)
    note = ""
    output: dict[str, Any]
    if span.kind == "tool":
        try:
            result = _fresh_toolbox(resources).call(span.input["tool"], span.input["args"])
        except ToolInputError:
            return None  # the call itself was malformed: re-calling cannot fix it
        output = tool_result_json(result)
        note = f"re-called {span.input['tool']} with its recorded arguments (schema-validated)"
    elif span.kind == "retrieval":
        index = resources["kb_index"]
        assert isinstance(index, BM25Index)
        hits = index.search(span.input["query"], int(span.input.get("k", 3)))
        output = {"hits": [h.model_dump(mode="json") for h in hits]}
        note = "re-ran the search with its recorded query"
    elif span.kind == "agent" and span.node_id == "retriever":
        index, kb = resources["kb_index"], resources["kb"]
        assert isinstance(index, BM25Index) and isinstance(kb, KnowledgeBase)
        queries = list(span.output.get("queries") or span.input.get("queries") or [])
        _, docs = select_docs(index, kb, queries, _input_of(run, "ticket"), span.params)
        output = {"queries": queries, "docs": docs}
        note = f"re-ran the retriever's searches for its {len(queries)} recorded queries"
    elif span.kind == "agent" and span.node_id == "tool_agent":
        box = _fresh_toolbox(resources)
        calls = [
            execute_call(box, ToolRequest(tool=c["tool"], args=c["args"]))
            for c in span.output.get("tool_calls", [])
        ]
        output = {"tool_calls": calls}
        note = f"re-called the {len(calls)} recorded tool calls"
    else:
        return None
    return Patch(target=span.span_id, output=output, patch_source="rule", note=note)


# ---------------------------------------------------------------- model


PROMPT = """You are repairing one step of a customer-support workflow (the online retailer
Brightcart). The step's recorded output is suspected to be wrong. Using only the step's
inputs below, write the output the step should have produced.

Step: {node_id} ({kind})
What the step does: {description}
What its output should be: {expectations}

Inputs:
{inputs}

Recorded output (suspected wrong):
{output}
{hint}
Return `output` as a JSON object with exactly these keys: {keys}. Keep every value that
is already right; change only what is wrong."""


class ModelPatchReply(LLMReply):
    output: dict[str, Any] = Field(description="The corrected output, same keys.")
    rationale: str = Field(description="What was wrong and what you changed.")
    confidence: Probability


def _describe(span: Span) -> tuple[str, str]:
    contract = CONTRACTS.get(span.node_id)
    if contract is not None:
        return contract.description, contract.output_expectations
    return f"A {span.kind} call inside a workflow step.", "The result the call should return."


def model_patch(
    run: Run,
    target: str,
    gateway: Gateway,
    *,
    role: str = PATCHER_ROLE,
    hint: str | None = None,
) -> Patch:
    """A model's proposed replacement for `target`'s output (same keys as recorded)."""
    span = resolve_target(run, target)
    if not span.output:
        raise PatchError(f"{span.node_id} has no recorded output to repair")
    description, expectations = _describe(span)
    prompt = PROMPT.format(
        node_id=span.node_id,
        kind=span.kind,
        description=description,
        expectations=expectations,
        inputs=json.dumps(span.input, indent=1, sort_keys=True, default=str),
        output=json.dumps(span.output, indent=1, sort_keys=True, default=str),
        hint=f"\nWhat looks wrong: {hint}\n" if hint else "",
        keys=", ".join(sorted(span.output)),
    )
    step = step_of(run, span).node_id
    result = gateway.complete_structured(
        role, [{"role": "user", "content": prompt}], ModelPatchReply, tag=f"patch/{step}"
    )
    if result.value is None:
        raise PatchSourceError(
            f"{role} gave no valid patch for {span.node_id} after {result.attempts} tries"
        )
    reply = result.value
    if set(reply.output) != set(span.output):
        raise PatchSourceError(
            f"{role} returned keys {sorted(reply.output)}, expected {sorted(span.output)}"
        )
    return Patch(
        target=span.span_id,
        output=reply.output,
        patch_source="model",
        note=f"{role} ({reply.confidence:.2f}): {reply.rationale}",
    )
