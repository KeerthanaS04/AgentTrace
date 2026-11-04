"""Record/replay engine (P5): reproduce runs, and re-run them with surgical changes."""

from agenttrace.replay.cache import Recorded, ReplayCache, fingerprint, node_key, span_key
from agenttrace.replay.executor import ExecutedReplay, PatchingReplay, execute_plan
from agenttrace.replay.patcher import Patch, PatchError, ReplayPlan, patch_fields, plan_patch
from agenttrace.replay.replay import exact_diffs, replay_run, run_inputs, run_model_params
from agenttrace.replay.sources import PatchSourceError, model_patch, rule_patch

__all__ = [
    "Patch",
    "PatchSourceError",
    "ExecutedReplay",
    "PatchingReplay",
    "execute_plan",
    "PatchError",
    "ReplayPlan",
    "patch_fields",
    "plan_patch",
    "Recorded",
    "ReplayCache",
    "exact_diffs",
    "fingerprint",
    "model_patch",
    "node_key",
    "replay_run",
    "rule_patch",
    "run_inputs",
    "run_model_params",
    "span_key",
]
