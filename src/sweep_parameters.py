"""Experiment 4: Parameter sensitivity analysis.

Sweep alpha_h, alpha_h/alpha_m ratio, and b_h to identify:
  1. Transition where h becomes too slow to matter or too fast (collapses to reaction)
  2. Effect of time-scale separation (alpha_h / alpha_m ratio)
  3. Threshold for b_h below which h dies at a fixed point
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.run_simulation import run_episode


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


METRICS_TO_TRACK = [
    "stay_leave_transitions",
    "mode_switch_count",
    "avg_local_food",
    "exploration_radius",
    "time_in_patch",
]

SEEDS = [7, 13, 42]


def sweep(base_cfg: Dict, param_setter, param_values, label: str) -> Dict:
    """Run sweep over one parameter axis."""
    results = {m: [] for m in METRICS_TO_TRACK}
    results["_stds"] = {m: [] for m in METRICS_TO_TRACK}
    results["_param_values"] = list(param_values)
    results["_label"] = label

    for val in param_values:
        seed_metrics = {m: [] for m in METRICS_TO_TRACK}
        for seed in SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            cfg["simulation"]["output_dir"] = f"outputs/sweep_tmp"
            param_setter(cfg, val)

            episode = run_episode(cfg)
            for m in METRICS_TO_TRACK:
                seed_metrics[m].append(episode["metrics"][m])

        for m in METRICS_TO_TRACK:
            results[m].append(float(np.mean(seed_metrics[m])))
            results["_stds"][m].append(float(np.std(seed_metrics[m])))

    return results


def plot_sweep(results: Dict, output_path: str, xlabel: str) -> None:
    vals = results["_param_values"]
    n_metrics = len(METRICS_TO_TRACK)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 3.5))
    fig.suptitle(results["_label"], fontsize=12)

    for i, m in enumerate(METRICS_TO_TRACK):
        ax = axes[i]
        means = results[m]
        stds = results["_stds"][m]
        ax.errorbar(vals, means, yerr=stds, marker="o", markersize=4, linewidth=1.5, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(m.replace("_", " "))
        ax.set_title(m.replace("_", "\n"), fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    base_cfg = load_config("configs/full.yaml")
    out_dir = Path("outputs/experiment4_parameter_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    # --- Sweep 1: alpha_h ---
    print("=== Sweep 1: alpha_h ===")
    alpha_h_values = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]

    def set_alpha_h(cfg, val):
        cfg["model"]["alpha_h"] = val

    r1 = sweep(base_cfg, set_alpha_h, alpha_h_values, "Sweep: alpha_h (alpha_m=0.40 fixed)")
    plot_sweep(r1, str(out_dir / "sweep_alpha_h.png"), "alpha_h")
    all_summaries["alpha_h"] = r1
    print("  done")

    # --- Sweep 2: alpha_h / alpha_m ratio ---
    print("=== Sweep 2: alpha_h / alpha_m ratio ===")
    # Fix alpha_h + alpha_m sum = 0.58 (current: 0.18 + 0.40), vary the ratio
    ratios = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5]
    # ratio = alpha_h / alpha_m. Keep alpha_m = 0.40, vary alpha_h.
    # But also test when alpha_h > alpha_m (ratio > 1)

    def set_ratio(cfg, ratio):
        alpha_m = 0.40
        alpha_h = alpha_m * ratio
        cfg["model"]["alpha_h"] = min(alpha_h, 0.95)
        cfg["model"]["alpha_m"] = alpha_m

    r2 = sweep(base_cfg, set_ratio, ratios, "Sweep: alpha_h/alpha_m ratio (alpha_m=0.40)")
    plot_sweep(r2, str(out_dir / "sweep_ratio.png"), "alpha_h / alpha_m")
    all_summaries["ratio"] = r2
    print("  done")

    # --- Sweep 3: b_h magnitude ---
    print("=== Sweep 3: b_h magnitude ===")
    # Scale both b_h components uniformly
    bh_scales = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    def set_bh(cfg, scale):
        # The handtuned b_h direction is [0.3, 0.2]; scale its magnitude
        cfg["model"]["b_h_scale"] = scale

    # We need to support b_h_scale in the agent. Instead, directly modify after init.
    # Simpler approach: override b_h in the config isn't directly supported,
    # so we modify the agent's _init_handtuned values proportionally.
    # For this sweep, we'll use a different approach: patch the agent after creation.

    # Actually, let's just modify b_h through a hook. The cleanest way is to
    # use the existing config and create a custom sweep.

    bh_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    from src.agent import MinimalEnactiveAgent
    from src.env import ForagingEnv
    from src.eval import compute_metrics
    from src.viz import plot_states, plot_trajectory

    def run_with_bh(config, bh_magnitude):
        """Run episode with overridden b_h magnitude."""
        env = ForagingEnv(config)
        agent = MinimalEnactiveAgent(config)
        # Override b_h: keep direction [0.6, 0.4] (normalized from [0.3, 0.2]), scale magnitude
        direction = np.array([0.6, 0.4])
        agent.b_h = direction * bh_magnitude

        env_state = env.reset()
        agent.reset()
        # Re-apply b_h override after reset (reset doesn't touch weights)
        agent.b_h = direction * bh_magnitude
        obs = agent.sense(env_state)

        steps = int(config["simulation"].get("steps", 400))
        log = {
            "x": [], "y": [], "local_food": [], "local_risk": [],
            "in_patch": [], "patch_level": [], "h": [], "m": [],
            "action_turn": [], "action_speed": [],
        }

        for _ in range(steps):
            step = agent.step(obs)
            env_state, step_info = env.step(step.action)
            obs = agent.sense(env_state)

            log["x"].append(step_info.info["x"])
            log["y"].append(step_info.info["y"])
            log["local_food"].append(float(step.observation[0]))
            log["local_risk"].append(float(step.observation[1]))
            log["in_patch"].append(step_info.info["in_patch"])
            log["patch_level"].append(step_info.info["patch_level"])
            log["h"].append(step.h.tolist())
            log["m"].append(step.m.tolist())
            log["action_turn"].append(float(step.action[0]))
            log["action_speed"].append(float(step.action[1]))

        return compute_metrics(log)

    r3_means = {m: [] for m in METRICS_TO_TRACK}
    r3_stds = {m: [] for m in METRICS_TO_TRACK}

    for bh_val in bh_values:
        seed_metrics = {m: [] for m in METRICS_TO_TRACK}
        for seed in SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            metrics = run_with_bh(cfg, bh_val)
            for m in METRICS_TO_TRACK:
                seed_metrics[m].append(metrics[m])

        for m in METRICS_TO_TRACK:
            r3_means[m].append(float(np.mean(seed_metrics[m])))
            r3_stds[m].append(float(np.std(seed_metrics[m])))

    r3 = {m: r3_means[m] for m in METRICS_TO_TRACK}
    r3["_stds"] = r3_stds
    r3["_param_values"] = bh_values
    r3["_label"] = "Sweep: b_h magnitude (direction=[0.6, 0.4])"
    plot_sweep(r3, str(out_dir / "sweep_bh.png"), "b_h magnitude")
    all_summaries["b_h"] = r3
    print("  done")

    # Save all results
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(to_serializable(all_summaries), f, indent=2)

    print(f"\nAll results saved to {out_dir}/")

    # Print key findings
    print("\n=== Key findings ===")

    # alpha_h: find peak of stay_leave_transitions
    slt = r1["stay_leave_transitions"]
    best_idx = int(np.argmax(slt))
    print(f"alpha_h: best stay_leave_transitions = {slt[best_idx]:.1f} at alpha_h = {alpha_h_values[best_idx]}")

    # ratio: find peak
    slt2 = r2["stay_leave_transitions"]
    best_idx2 = int(np.argmax(slt2))
    print(f"ratio: best stay_leave_transitions = {slt2[best_idx2]:.1f} at ratio = {ratios[best_idx2]}")

    # b_h: find threshold
    slt3 = r3["stay_leave_transitions"]
    for i, val in enumerate(bh_values):
        if slt3[i] > 1.5:
            print(f"b_h: behavior activates at b_h >= {val} (transitions = {slt3[i]:.1f})")
            break
    else:
        print(f"b_h: no clear activation threshold found")


if __name__ == "__main__":
    main()
