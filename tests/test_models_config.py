"""Checks on the real config/models.yaml (P0-T4)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from agenttrace.gateway import CompletionCache, Gateway, UnverifiedModelError, load_config
from agenttrace.gateway.config import REQUIRED_ROLES, TODO_VERIFY, GatewayConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


@pytest.fixture
def config() -> GatewayConfig:
    return load_config(CONFIG_PATH)


def test_config_loads_and_has_all_roles(config: GatewayConfig) -> None:
    assert config.missing_roles() == []
    assert set(REQUIRED_ROLES) <= set(config.roles)


def test_judges_and_agents_are_deterministic(config: GatewayConfig) -> None:
    for name, role in config.roles.items():
        assert role.params.get("temperature") == 0, f"{name} must run at temperature 0"


def test_muse_roles_use_meta(config: GatewayConfig) -> None:
    assert config.roles["judge_fast"].provider == "meta"
    assert config.roles["judge_deep"].provider == "meta"
    assert config.roles["judge_deep"].model == "muse-wonder-1.0"


def test_verified_meta_models_are_priced(config: GatewayConfig) -> None:
    # Meta credits are capped, so every confirmed Meta model must carry pricing.
    for name, role in config.roles.items():
        if role.provider == "meta" and role.model != TODO_VERIFY:
            assert config.pricing_for("meta", role.model), f"{name}: add pricing for {role.model}"


def test_judges_are_independent_providers(config: GatewayConfig) -> None:
    providers = {config.roles[r].provider for r in ("judge_fast", "judge_nvidia", "judge_oss")}
    assert len(providers) == 3


def _gateway(config: GatewayConfig, tmp_path: Path, seen: list[dict[str, object]]) -> Gateway:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), **json.loads(request.content)})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    return Gateway(
        config,
        cache=CompletionCache(tmp_path / "c.sqlite"),
        env={"MODEL_API_KEY": "k", "OPENROUTER_API_KEY": "k", "NVIDIA_API_KEY": "k"},
        transports={p: httpx.MockTransport(handler) for p in config.providers},
    )


def test_swapping_a_role_model_is_config_only(tmp_path: Path) -> None:
    """Done-when for P0-T4: change a role's model in YAML, nothing else, and calls follow."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["roles"]["agent_default"]["provider"] = "openrouter"
    raw["roles"]["agent_default"]["model"] = "some-org/model-a"
    path_a = tmp_path / "a.yaml"
    path_a.write_text(yaml.safe_dump(raw))

    raw["roles"]["agent_default"] = {
        "provider": "nvidia",
        "model": "some-org/model-b",
        "params": {"temperature": 0},
    }
    path_b = tmp_path / "b.yaml"
    path_b.write_text(yaml.safe_dump(raw))

    seen: list[dict[str, object]] = []
    for path in (path_a, path_b):
        gw = _gateway(load_config(path), tmp_path / path.stem, seen)
        gw.complete("agent_default", [{"role": "user", "content": "hi"}])

    assert seen[0]["model"] == "some-org/model-a"
    assert "openrouter.ai" in str(seen[0]["url"])
    assert seen[1]["model"] == "some-org/model-b"
    assert "integrate.api.nvidia.com" in str(seen[1]["url"])


def test_unverified_roles_are_refused(config: GatewayConfig, tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []
    gw = _gateway(config, tmp_path, seen)
    for name, role in config.roles.items():
        if role.model == TODO_VERIFY:
            with pytest.raises(UnverifiedModelError):
                gw.complete(name, [{"role": "user", "content": "hi"}])
    assert not seen


def test_local_base_url_env_override(config: GatewayConfig, tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []
    raw = config.model_dump()
    raw["roles"]["local_test"] = {"provider": "local", "model": "m", "params": {}}
    cfg = GatewayConfig.model_validate(raw)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url)})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    gw = Gateway(
        cfg,
        cache=CompletionCache(tmp_path / "c.sqlite"),
        env={"LOCAL_OPENAI_API_KEY": "x", "LOCAL_OPENAI_BASE_URL": "http://gpu-box:9000/v1"},
        transports={"local": httpx.MockTransport(handler)},
    )
    gw.complete("local_test", [{"role": "user", "content": "hi"}])
    assert seen[0]["url"] == "http://gpu-box:9000/v1/chat/completions"


def test_todo_verify_base_url_refused(config: GatewayConfig, tmp_path: Path) -> None:
    raw = config.model_dump()
    raw["providers"]["nvidia"]["base_url"] = TODO_VERIFY
    raw["roles"]["judge_nvidia"]["model"] = "some-model"
    seen: list[dict[str, object]] = []
    gw = _gateway(GatewayConfig.model_validate(raw), tmp_path, seen)
    with pytest.raises(UnverifiedModelError, match="base_url"):
        gw.complete("judge_nvidia", [{"role": "user", "content": "hi"}])
    assert not seen
