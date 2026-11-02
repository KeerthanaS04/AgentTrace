"""P1-T6: each agent's model comes from config/agents.yaml + config/models.yaml only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fake_llm import CONFIG_PATH, TICKET, make_gateway
from pydantic import ValidationError

from agenttrace.gateway import load_config
from agenttrace.workflow.agent_config import (
    DEFAULT_AGENTS_PATH,
    AgentProfile,
    AgentsConfig,
    load_agents_config,
)
from agenttrace.workflow.support import STEPS, build_support_workflow, make_resources

AGENT = {  # step -> agent name as it appears in the prompt
    "router": "Router",
    "retriever": "Retriever",
    "tool_agent": "Tool Agent",
    "policy": "Policy Agent",
    "writer": "Response Writer",
}


def models_called(
    tmp_path: Path, roles: dict[str, str], models_path: Path = CONFIG_PATH
) -> dict[str, set[str]]:
    """Run the workflow offline and return {step: model ids its requests named}."""
    calls: list[tuple[str, str]] = []
    gateway = make_gateway(tmp_path, calls=calls, config_path=models_path)
    result = build_support_workflow(roles).run(
        {"ticket": TICKET}, run_id="t", resources=make_resources(gateway)
    )
    assert result.status == "ok"
    by_agent: dict[str, set[str]] = {}
    for agent, model in calls:
        by_agent.setdefault(agent, set()).add(model)
    return {step: by_agent[name] for step, name in AGENT.items()}


# ---------------------------------------------------------------- the real config


def test_real_agents_config_is_valid() -> None:
    agents = load_agents_config()
    agents.check_roles(load_config(CONFIG_PATH))
    assert {"default", "ultra", "bolt", "mistral"} <= set(agents.profiles)


def test_profiles_cover_at_least_three_models() -> None:
    """Done-when for P1-T6: the workflow can run on >= 3 different models via config."""
    agents, models = load_agents_config(), load_config(CONFIG_PATH)
    whole_workflow = {
        name: {models.roles[r].model for r in prof.roles().values()}
        for name, prof in agents.profiles.items()
    }
    single_model = {next(iter(m)) for m in whole_workflow.values() if len(m) == 1}
    assert len(single_model) >= 3, whole_workflow


@pytest.mark.parametrize("profile", ["default", "ultra", "bolt", "mistral", "mixed"])
def test_each_profile_sends_every_step_to_its_configured_model(
    tmp_path: Path, profile: str
) -> None:
    roles = load_agents_config().profile(profile).roles()
    models = load_config(CONFIG_PATH)
    got = models_called(tmp_path, roles)
    for step in STEPS:
        assert got[step] == {models.roles[roles[step]].model}, step


def test_editing_yaml_alone_moves_one_agent(tmp_path: Path) -> None:
    """Change only config files; the policy step's requests follow, nothing else moves."""
    raw = yaml.safe_load(DEFAULT_AGENTS_PATH.read_text(encoding="utf-8"))
    raw["profiles"]["default"]["steps"] = {"policy": "agent_alt_2"}
    edited = tmp_path / "agents.yaml"
    edited.write_text(yaml.safe_dump(raw), encoding="utf-8")

    roles = load_agents_config(edited).profile("default").roles()
    models = load_config(CONFIG_PATH)
    got = models_called(tmp_path, roles)
    assert got["policy"] == {models.roles["agent_alt_2"].model}
    for step in ("router", "retriever", "tool_agent", "writer"):
        assert got[step] == {models.roles["agent_default"].model}, step


def test_editing_models_yaml_swaps_model_behind_a_role(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["roles"]["agent_alt_2"]["model"] = "some-org/brand-new-model"
    edited = tmp_path / "models.yaml"
    edited.write_text(yaml.safe_dump(raw), encoding="utf-8")

    roles = load_agents_config().profile("bolt").roles()
    got = models_called(tmp_path, roles, models_path=edited)
    assert all(m == {"some-org/brand-new-model"} for m in got.values())


# ---------------------------------------------------------------- validation


def test_default_role_fills_unlisted_steps() -> None:
    prof = AgentProfile(default_role="agent_alt_2", steps={"policy": "agent_default"})
    assert prof.roles() == {
        "router": "agent_alt_2",
        "retriever": "agent_alt_2",
        "tool_agent": "agent_alt_2",
        "policy": "agent_default",
        "writer": "agent_alt_2",
    }


def test_unknown_step_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown steps"):
        AgentProfile(default_role="agent_default", steps={"planner": "agent_default"})


def test_unknown_role_rejected_against_models_config() -> None:
    agents = AgentsConfig(profiles={"x": AgentProfile(default_role="agent_typo")})
    with pytest.raises(ValueError, match="agent_typo"):
        agents.check_roles(load_config(CONFIG_PATH))


def test_unknown_profile_names_the_choices() -> None:
    with pytest.raises(KeyError, match="default"):
        load_agents_config().profile("nope")
