"""Run one minimal enactive simulation episode and save outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import yaml

from src.interfaces import BaseAgent, BaseEnvironment
from src.registry import create_agent, create_env
from src.eval import compute_metrics
from src.viz import plot_states, plot_trajectory


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(
    config: Dict,
    env: Optional[BaseEnvironment] = None,
    agent: Optional[BaseAgent] = None,
    save_outputs: bool = True,
) -> Dict:
    """Run a single episode and return metrics.

    Args:
        config: Full configuration dict.
        env: Pre-built environment (if None, created from config via registry).
        agent: Pre-built agent (if None, created from config via registry).
        save_outputs: If True, save metrics, log, and plots to output_dir.

    Returns:
        Dict with "metrics", "log", and "output_dir" keys.
    """
    if env is None:
        env = create_env(config)
    if agent is None:
        agent = create_agent(config)

    env_state = env.reset()
    agent.reset()
    obs = agent.sense(env_state)

    steps = int(config["simulation"].get("steps", 400))

    n_patches = len(env_state.patches)

    log = {
        "x": [],
        "y": [],
        "heading": [],
        "local_food": [],
        "local_risk": [],
        "in_patch": [],
        "patch_level": [],
        "h": [],
        "m": [],
        "action_turn": [],
        "action_speed": [],
        # Per-patch time series for animation
        "patch_levels": [],   # [[p0, p1, ...], ...]
        "odor_levels": [],    # [[p0, p1, ...], ...]
    }

    # Static layout info (stored once, not per-step)
    log["layout"] = {
        "world_size": env.get_layout().get("world_size", 20.0),
        "patch_centers": env.get_layout().get("patch_centers", []),
        "patch_radii": env.get_layout().get("patch_radii", []),
        "risk_center": env.get_layout().get("risk_center", [0, 0]),
        "risk_radius": env.get_layout().get("risk_radius", 0),
        "n_patches": n_patches,
    }
    # Ensure layout values are JSON-serializable
    rc = log["layout"]["risk_center"]
    if hasattr(rc, "tolist"):
        log["layout"]["risk_center"] = rc.tolist()

    for _ in range(steps):
        step = agent.step(obs)
        env_state, step_info = env.step(step.action)
        obs = agent.sense(env_state)

        log["x"].append(step_info.info["x"])
        log["y"].append(step_info.info["y"])
        log["heading"].append(float(env_state.heading))
        log["local_food"].append(float(step.observation[0]))
        log["local_risk"].append(float(step.observation[1]))
        log["in_patch"].append(step_info.info["in_patch"])
        log["patch_level"].append(step_info.info["patch_level"])
        log["h"].append(step.h.tolist())
        log["m"].append(step.m.tolist())
        log["action_turn"].append(float(step.action[0]))
        log["action_speed"].append(float(step.action[1]))
        log["patch_levels"].append([p.level for p in env_state.patches])
        log["odor_levels"].append([p.odor_level for p in env_state.patches])

    metrics = compute_metrics(log)

    output_dir = Path(config["simulation"].get("output_dir", "outputs/base"))

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(output_dir / "rollout_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f)

        plot_trajectory(log, env.get_layout(), str(output_dir / "trajectory.png"))
        plot_states(log, str(output_dir / "states.png"))

    return {"metrics": metrics, "log": log, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal enactive PoC simulation")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_episode(cfg)

    print("=== Simulation complete ===")
    print(f"Config: {args.config}")
    print(f"Output dir: {result['output_dir']}")
    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
