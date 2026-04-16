"""Factory functions for creating environments and agents from config.

Usage:
    from src.registry import create_env, create_agent

    env = create_env(config)     # reads config["environment"]["type"]
    agent = create_agent(config)  # reads config["model"]["type"]

Environment type is determined by config["environment"]["type"]:
    "foraging" (default) -> ForagingEnv
    "odor_field"         -> OdorFieldEnv

Agent type is determined by config["model"]["type"]:
    "minimal_recurrent" (default) -> MinimalEnactiveAgent
"""

from __future__ import annotations

from typing import Dict

from src.interfaces import BaseAgent, BaseEnvironment


def create_env(config: Dict) -> BaseEnvironment:
    """Create an environment instance from config."""
    env_type = config.get("environment", {}).get("type", "foraging")

    if env_type == "foraging":
        from src.env import ForagingEnv
        return ForagingEnv(config)
    elif env_type == "odor_field":
        from src.envs.odor_field import OdorFieldEnv
        return OdorFieldEnv(config)
    else:
        raise ValueError(f"Unknown environment type: {env_type!r}")


def create_agent(config: Dict) -> BaseAgent:
    """Create an agent instance from config."""
    agent_type = config.get("model", {}).get("type", "minimal_recurrent")

    if agent_type == "minimal_recurrent":
        from src.agent import MinimalEnactiveAgent
        return MinimalEnactiveAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}")
