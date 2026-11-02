"""Per-agent model choice, loaded from config/agents.yaml.

A profile maps each workflow step to a model role (roles live in config/models.yaml),
so switching the models under test is a config edit, never a code change.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from agenttrace.gateway.config import GatewayConfig
from agenttrace.schemas.base import NonEmptyStr, StrictModel
from agenttrace.workflow.support import STEPS

DEFAULT_AGENTS_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"
DEFAULT_PROFILE = "default"


class AgentProfile(StrictModel):
    description: str = ""
    default_role: NonEmptyStr
    steps: dict[str, NonEmptyStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _known_steps(self) -> AgentProfile:
        unknown = set(self.steps) - set(STEPS)
        if unknown:
            raise ValueError(f"unknown steps {sorted(unknown)}; expected some of {list(STEPS)}")
        return self

    def roles(self) -> dict[str, str]:
        """Role for every step, with `default_role` filling the gaps."""
        return {step: self.steps.get(step, self.default_role) for step in STEPS}


class AgentsConfig(StrictModel):
    profiles: dict[str, AgentProfile] = Field(min_length=1)

    def profile(self, name: str) -> AgentProfile:
        if name not in self.profiles:
            raise KeyError(f"no profile {name!r} in agents config; have {sorted(self.profiles)}")
        return self.profiles[name]

    def check_roles(self, models: GatewayConfig) -> None:
        """Every role a profile uses must exist in the models config."""
        for name, prof in self.profiles.items():
            missing = sorted(set(prof.roles().values()) - set(models.roles))
            if missing:
                raise ValueError(f"profile {name!r} uses roles not in models config: {missing}")


def load_agents_config(path: Path | str = DEFAULT_AGENTS_PATH) -> AgentsConfig:
    return AgentsConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
