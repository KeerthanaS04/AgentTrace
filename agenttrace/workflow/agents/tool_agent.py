"""Tool Agent: decide which tools to call, call them, and pass on the results.

Runs a short loop: the model sees the ticket, the tool specs, and the results so far,
and either requests more calls or says it is done. Tool input errors are recorded as
results (not raised), so the Policy Agent and later the debugger can see them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.workflow.agents.base import ExplainedReply, LLMReply, as_json, ask
from agenttrace.workflow.runtime import AgentOutput, StepContext
from agenttrace.workflow.tools import TOOLS, ToolBox, ToolInputError

MAX_ROUNDS = 3
MAX_CALLS = 5

PROMPT = """You are the Tool Agent in a customer-support system for the online retailer Brightcart.
Gather the account and order facts needed to resolve the ticket by calling tools.
- Look up the order the ticket is about (if any) and the customer's account status.
- For refund requests, once you know the order, call refund_api with mode "quote" for the
  amount the customer is asking about, to check whether it can be refunded. Never use mode "issue".
- Do not repeat a call that already has a result. Set done to true when you have what you need.

Available tools (JSON schemas):
{tools}

Ticket category: {category}
Order ID from the router: {order_id}
Ticket:
{ticket}

Tool results so far:
{results}"""


class ToolRequest(LLMReply):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolAgentReply(ExplainedReply):
    calls: list[ToolRequest] = Field(default_factory=list, description="Tool calls to make now.")
    done: bool = Field(description="True when no more calls are needed.")


class ToolAgent:
    name = "tool_agent"
    prompt_version = "tool-agent-v1"
    input_keys: tuple[str, ...] = ("ticket", "category", "order_id")
    output_keys: tuple[str, ...] = ("tool_calls",)

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        toolbox = ctx.resource("tools", ToolBox)
        specs = as_json([spec.schema_for_llm() for spec in TOOLS.values()])
        calls: list[dict[str, Any]] = []
        last: ToolAgentReply | None = None
        for _ in range(int(ctx.params.get("max_rounds", MAX_ROUNDS))):
            last = ask(
                ctx,
                PROMPT.format(
                    tools=specs,
                    category=inputs["category"],
                    order_id=inputs["order_id"],
                    ticket=as_json(inputs["ticket"]),
                    results=as_json(calls) if calls else "(none yet)",
                ),
                ToolAgentReply,
            )
            made = 0
            for req in last.calls:
                if len(calls) >= MAX_CALLS:
                    break
                if not _already_called(calls, req):  # also catches repeats within one batch
                    calls.append(_execute(toolbox, req))
                    made += 1
            if last.done or not made or len(calls) >= MAX_CALLS:
                break
        assert last is not None
        return AgentOutput(
            output={"tool_calls": calls},
            explanation=last.explanation(cited=[c["tool"] for c in calls]),
        )


def _already_called(calls: list[dict[str, Any]], req: ToolRequest) -> bool:
    return any(c["tool"] == req.tool and c["args"] == req.args for c in calls)


def _execute(toolbox: ToolBox, req: ToolRequest) -> dict[str, Any]:
    if req.tool == "refund_api" and req.args.get("mode") == "issue":
        return {
            "tool": req.tool,
            "args": req.args,
            "result": None,
            "error": "issue mode not allowed",
        }
    try:
        result = toolbox.call(req.tool, req.args)
    except ToolInputError as exc:
        return {"tool": req.tool, "args": req.args, "result": None, "error": str(exc)}
    return {
        "tool": req.tool,
        "args": req.args,
        "result": result.model_dump(mode="json"),
        "error": None,
    }
