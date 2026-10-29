"""Load `config/models.yaml`: providers, per-model pricing, and role -> model bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from agenttrace.schemas.base import NonEmptyStr, NonNegativeFloat, StrictModel

# Placeholder for model IDs (or endpoints) not yet confirmed from a provider dashboard.
TODO_VERIFY = "TODO_VERIFY"
DEFAULT_CONFIG_PATH = Path("config/models.yaml")


class ModelPricing(StrictModel):
    """USD per million tokens."""

    input_per_m: NonNegativeFloat
    output_per_m: NonNegativeFloat


class ProviderConfig(StrictModel):
    base_url: NonEmptyStr
    api_key_env: NonEmptyStr
    # Whether the endpoint accepts `response_format={"type": "json_object"}`.
    supports_json_mode: bool = False
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    models: dict[str, ModelPricing] = Field(default_factory=dict)


class RoleConfig(StrictModel):
    provider: NonEmptyStr
    model: NonEmptyStr
    params: dict[str, Any] = Field(default_factory=dict)


class GatewayConfig(StrictModel):
    providers: dict[str, ProviderConfig]
    roles: dict[str, RoleConfig]

    @model_validator(mode="after")
    def _roles_reference_known_providers(self) -> GatewayConfig:
        for name, role in self.roles.items():
            if role.provider not in self.providers:
                raise ValueError(f"role {name!r} uses unknown provider {role.provider!r}")
        return self

    def pricing_for(self, provider: str, model: str) -> ModelPricing | None:
        return self.providers[provider].models.get(model)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> GatewayConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return GatewayConfig.model_validate(yaml.safe_load(fh))
