"""Extract and validate JSON from model output."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} block.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("no JSON object found", text, 0)


def schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object only, no prose, matching this JSON Schema:\n"
        + json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )


def repair_instruction(error: str) -> str:
    return (
        "Your previous reply was not valid for the required schema. Error:\n"
        f"{error[:2000]}\n"
        "Reply again with only the corrected JSON object."
    )
