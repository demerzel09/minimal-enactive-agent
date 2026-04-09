"""Run one minimal enactive simulation episode and save outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

from src.agent import MinimalEnactiveAgent
from src.env import ForagingEnv
from src.eval import compute_metrics
from src.viz import plot_states, plot_trajectory


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(config: Dict) -> Dict:
    env = ForagingEnv(config)
    agent = MinimalEnactiveAgent(config)

    obs = env.reset()
    agent.reset()

    steps = int(config["simulation"].get("steps", 400))

    log = {
        "x": [],
        "y": [],
        "local_food": [],
        "local_risk": [],
        "in_patch": [],
        "patch_level": [],
        "reward": [],
        "h": [],
        "m": [],
        "action_turn": [],
        "action_speed": [],
    }

    for _ in range(steps):
        step = agent.step(obs)
        result = env.step(step.action)

        log["x"].append(result.info["x"])
        log["y"].append(result.info["y"])
        log["local_food"].append(result.info["local_food"])
        log["local_risk"].append(result.info["local_risk"])
        log["in_patch"].append(result.info["in_patch"])
        log["patch_level"].append(result.info["patch_level"])
        log["reward"].append(result.reward)
        log["h"].append(step.h.tolist())
        log["m"].append(step.m.tolist())
        log["action_turn"].append(float(step.action[0]))
        log["action_speed"].append(float(step.action[1]))

        obs = result.observation

    metrics = compute_metrics(log)

    output_dir = Path(config["simulation"].get("output_dir", "outputs/base"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "rollout_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f)

    plot_trajectory(log, env.get_layout(), str(output_dir / "trajectory.png"))
    plot_states(log, str(output_dir / "states.png"))

    return {"metrics": metrics, "output_dir": str(output_dir)}


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
