"""Run all challenge environments and produce a comparison summary."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.run_simulation import run_episode


CHALLENGE_CONFIGS = {
    "A_baseline": "configs/challenge/env_a_baseline.yaml",
    "B_distant": "configs/challenge/env_b_distant_patches.yaml",
    "C_fast_depl": "configs/challenge/env_c_fast_depletion.yaml",
    "D_risk_food": "configs/challenge/env_d_risk_near_food.yaml",
    "E_no_risk": "configs/challenge/env_e_no_risk.yaml",
    "F_small": "configs/challenge/env_f_small_world.yaml",
    "G_3patches": "configs/challenge/env_g_three_patches.yaml",
}

SEEDS = [7, 13, 42]

METRICS = [
    "stay_leave_transitions",
    "time_in_patch",
    "avg_local_food",
    "exploration_radius",
    "mode_switch_count",
]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_challenge_suite(override_model: Dict | None = None, label: str = "handtuned") -> Dict:
    """Run all challenge environments, optionally overriding model weights."""
    results = {}

    for env_name, config_path in CHALLENGE_CONFIGS.items():
        base_cfg = load_config(config_path)
        seed_metrics = {m: [] for m in METRICS}

        for seed in SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            cfg["simulation"]["output_dir"] = f"outputs/challenge/{label}/{env_name}/seed_{seed}"

            if override_model is not None:
                cfg["model"].update(override_model)

            episode = run_episode(cfg)
            for m in METRICS:
                seed_metrics[m].append(episode["metrics"][m])

        results[env_name] = {
            m: {"mean": float(np.mean(seed_metrics[m])), "std": float(np.std(seed_metrics[m]))}
            for m in METRICS
        }
        trans = results[env_name]["stay_leave_transitions"]["mean"]
        food = results[env_name]["avg_local_food"]["mean"]
        print(f"  {env_name}: transitions={trans:.1f}, food={food:.3f}")

    return results


def print_summary_table(results: Dict, label: str) -> str:
    lines = []
    lines.append(f"\n### {label}")
    lines.append("")

    header = f"| {'env':<15} |"
    sep = f"| {'-'*15} |"
    for m in METRICS:
        short = m.replace("stay_leave_", "").replace("avg_local_", "").replace("_", " ")
        header += f" {short:>14} |"
        sep += f" {'-'*14} |"
    lines.append(header)
    lines.append(sep)

    for env_name in CHALLENGE_CONFIGS:
        row = f"| {env_name:<15} |"
        for m in METRICS:
            mean = results[env_name][m]["mean"]
            std = results[env_name][m]["std"]
            row += f" {mean:>6.2f}+/-{std:>4.2f} |"
        lines.append(row)

    table = "\n".join(lines)
    print(table)
    return table


def plot_comparison(results: Dict, output_path: str, title: str) -> None:
    env_names = list(CHALLENGE_CONFIGS.keys())
    n_envs = len(env_names)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4))
    fig.suptitle(title, fontsize=12)

    x = np.arange(n_envs)
    for i, m in enumerate(METRICS):
        ax = axes[i]
        means = [results[e][m]["mean"] for e in env_names]
        stds = [results[e][m]["std"] for e in env_names]
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([e.split("_")[0] for e in env_names], fontsize=8)
        short = m.replace("stay_leave_", "").replace("avg_local_", "").replace("_", "\n")
        ax.set_title(short, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = Path("outputs/challenge")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Challenge Suite: Handtuned baseline ===")
    results = run_challenge_suite(label="handtuned")

    table = print_summary_table(results, "Handtuned v7 baseline")
    plot_comparison(results, str(out_dir / "handtuned_comparison.png"), "Handtuned v7: Challenge environments")

    with open(out_dir / "handtuned_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "handtuned_summary.md", "w") as f:
        f.write(table + "\n")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
