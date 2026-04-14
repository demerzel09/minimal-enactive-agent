"""Run multiple seeds for all experimental conditions and summarize."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np

from src.run_simulation import run_episode


CONDITIONS = {
    "full": "configs/full.yaml",
    "no_h": "configs/ablation_no_h.yaml",
    "no_m": "configs/ablation_no_m.yaml",
    "reduced_recurrence": "configs/ablation_reduced_recurrence.yaml",
}

SEEDS = [7, 13, 42, 99, 256]

METRICS_OF_INTEREST = [
    "time_in_patch",
    "stay_leave_transitions",
    "avg_patch_residence",
    "exploration_radius",
    "mode_switch_count",
    "mode_persistence",
    "avg_local_food",
    "history_dependence_proxy",
    "mean_step_distance",
]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_all() -> Dict[str, List[Dict[str, float]]]:
    results: Dict[str, List[Dict[str, float]]] = {}

    for cond_name, config_path in CONDITIONS.items():
        base_cfg = load_config(config_path)
        results[cond_name] = []

        for seed in SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            cfg["simulation"]["output_dir"] = f"outputs/multi_seed/{cond_name}/seed_{seed}"

            episode = run_episode(cfg)
            results[cond_name].append(episode["metrics"])
            print(f"  {cond_name} seed={seed} done")

    return results


def summarize(results: Dict[str, List[Dict[str, float]]]) -> str:
    lines = []
    lines.append("# Multi-seed comparison (5 seeds per condition)")
    lines.append("")
    lines.append(f"Seeds: {SEEDS}")
    lines.append("")

    # Header
    header = f"| {'metric':<30} |"
    sep = f"| {'-'*30} |"
    for cond in CONDITIONS:
        header += f" {cond:>22} |"
        sep += f" {'-'*22} |"
    lines.append(header)
    lines.append(sep)

    for metric in METRICS_OF_INTEREST:
        row = f"| {metric:<30} |"
        for cond in CONDITIONS:
            vals = [r[metric] for r in results[cond]]
            mean = np.mean(vals)
            std = np.std(vals)
            row += f" {mean:>8.3f} +/- {std:>6.3f} |"
        lines.append(row)

    return "\n".join(lines)


def main() -> None:
    print("=== Multi-seed experiment ===")
    results = run_all()

    summary = summarize(results)
    print("\n" + summary)

    out_dir = Path("outputs/multi_seed")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
