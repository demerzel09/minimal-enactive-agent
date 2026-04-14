"""Run per-environment GAs + larger universal GA, then produce 3-way comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.run_ga import (
    CHALLENGE_CONFIGS,
    EVAL_SEEDS,
    run_ga,
    decode_genome,
    handtuned_genome,
    run_challenge_suite_with_weights,
)


def run_per_env_gas(pop_size: int = 20, generations: int = 30) -> Dict[str, Dict]:
    """Run GA optimized for each individual environment."""
    per_env_results = {}
    for env_name in CHALLENGE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Per-env GA: {env_name}")
        print(f"{'='*60}")
        result = run_ga(
            pop_size=pop_size,
            generations=generations,
            seed=42,
            env_filter=env_name,
            verbose=True,
        )
        per_env_results[env_name] = result
    return per_env_results


def evaluate_on_all_envs(ga_result: Dict) -> Dict:
    """Evaluate a GA result's weights on all challenge environments."""
    decoded_weights = {k: np.array(v) for k, v in ga_result["decoded_weights"].items()}
    alpha_override = {
        "alpha_h": ga_result["decoded_alpha_h"],
        "alpha_m": ga_result["decoded_alpha_m"],
    }
    from src.run_ga import _run_genome_on_env, _load_config, decode_genome as _dec
    from src.agent import MinimalEnactiveAgent
    from src.env import ForagingEnv
    from src.eval import compute_metrics
    import copy

    METRICS = [
        "stay_leave_transitions", "time_in_patch", "avg_local_food",
        "exploration_radius", "mode_switch_count",
    ]

    results = {}
    for env_name, config_path in CHALLENGE_CONFIGS.items():
        with open(config_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        seed_metrics = {m: [] for m in METRICS}

        for seed in EVAL_SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = seed
            cfg["simulation"]["output_dir"] = f"outputs/ga_eval_comparison/{env_name}/seed_{seed}"
            cfg["model"]["init_mode"] = "handtuned"

            env = ForagingEnv(cfg)
            agent = MinimalEnactiveAgent(cfg)
            agent.alpha_h = alpha_override["alpha_h"]
            agent.alpha_m = alpha_override["alpha_m"]
            for wname, wval in decoded_weights.items():
                setattr(agent, wname, wval.copy())

            env_state = env.reset()
            agent.reset()
            obs = agent.sense(env_state)

            steps = int(cfg["simulation"].get("steps", 400))
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

            metrics = compute_metrics(log)
            for m in METRICS:
                seed_metrics[m].append(metrics[m])

        results[env_name] = {
            m: {"mean": float(np.mean(seed_metrics[m])), "std": float(np.std(seed_metrics[m]))}
            for m in METRICS
        }
    return results


def plot_three_way(ht: Dict, ga_uni: Dict, per_env_best: Dict, output_path: str) -> None:
    """3-way comparison: handtuned vs GA-universal vs per-env best."""
    envs = list(CHALLENGE_CONFIGS.keys())
    metrics = ["stay_leave_transitions", "avg_local_food"]
    titles = ["Patch Revisits (transitions)", "Food Acquisition (avg_local_food)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("3-way Comparison: Handtuned vs GA-universal vs Per-env GA ceiling", fontsize=13)

    x = np.arange(len(envs))
    width = 0.25

    for i, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        ht_means = [ht[e][m]["mean"] for e in envs]
        ht_stds = [ht[e][m]["std"] for e in envs]
        uni_means = [ga_uni[e][m]["mean"] for e in envs]
        uni_stds = [ga_uni[e][m]["std"] for e in envs]
        best_means = [per_env_best[e][m]["mean"] for e in envs]
        best_stds = [per_env_best[e][m]["std"] for e in envs]

        ax.bar(x - width, ht_means, width, yerr=ht_stds, capsize=2,
               alpha=0.7, label="Handtuned v7", color="steelblue")
        ax.bar(x, uni_means, width, yerr=uni_stds, capsize=2,
               alpha=0.7, label="GA-universal", color="coral")
        ax.bar(x + width, best_means, width, yerr=best_stds, capsize=2,
               alpha=0.7, label="GA per-env (ceiling)", color="mediumseagreen")

        ax.set_xticks(x)
        ax.set_xticklabels([e.split("_")[0] for e in envs], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = Path("outputs/ga_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load handtuned baseline
    with open("outputs/challenge/handtuned_results.json") as f:
        ht_results = json.load(f)

    # 2. Run larger universal GA
    print("\n" + "=" * 60)
    print("Universal GA (40 pop x 60 gen)")
    print("=" * 60)
    uni_result = run_ga(pop_size=40, generations=60, seed=7, env_filter=None)
    uni_metrics = evaluate_on_all_envs(uni_result)
    with open(out_dir / "universal_ga_metrics.json", "w") as f:
        json.dump(uni_metrics, f, indent=2)

    # 3. Run per-environment GAs
    per_env_ga_results = run_per_env_gas(pop_size=20, generations=30)

    # 4. Evaluate per-env best on their OWN environment
    per_env_best_metrics = {}
    for env_name, result in per_env_ga_results.items():
        env_metrics = evaluate_on_all_envs(result)
        per_env_best_metrics[env_name] = env_metrics[env_name]

    # Build per-env-best dict (each env uses its own specialist GA)
    per_env_combined = {}
    for env_name in CHALLENGE_CONFIGS:
        per_env_combined[env_name] = per_env_best_metrics[env_name]

    with open(out_dir / "per_env_best_metrics.json", "w") as f:
        json.dump(per_env_combined, f, indent=2)

    # 5. Print 3-way comparison table
    print("\n" + "=" * 70)
    print("3-WAY COMPARISON: Handtuned vs GA-universal vs Per-env ceiling")
    print("=" * 70)
    print(f"\n{'env':<16} {'HT food':>10} {'GA-uni food':>12} {'Per-env food':>13} {'gap%':>8}")
    print("-" * 62)
    for env in CHALLENGE_CONFIGS:
        ht_f = ht_results[env]["avg_local_food"]["mean"]
        uni_f = uni_metrics[env]["avg_local_food"]["mean"]
        best_f = per_env_combined[env]["avg_local_food"]["mean"]
        # gap = how much of the per-env ceiling does universal achieve?
        if best_f > 0.001:
            gap = uni_f / best_f * 100
        else:
            gap = 0.0
        print(f"{env:<16} {ht_f:>10.3f} {uni_f:>12.3f} {best_f:>13.3f} {gap:>7.0f}%")

    print(f"\n{'env':<16} {'HT trans':>10} {'GA-uni trans':>12} {'Per-env trans':>13}")
    print("-" * 55)
    for env in CHALLENGE_CONFIGS:
        ht_t = ht_results[env]["stay_leave_transitions"]["mean"]
        uni_t = uni_metrics[env]["stay_leave_transitions"]["mean"]
        best_t = per_env_combined[env]["stay_leave_transitions"]["mean"]
        print(f"{env:<16} {ht_t:>10.1f} {uni_t:>12.1f} {best_t:>13.1f}")

    # 6. Plot
    plot_three_way(ht_results, uni_metrics, per_env_combined,
                   str(out_dir / "three_way_comparison.png"))
    # Also save to docs/assets
    plot_three_way(ht_results, uni_metrics, per_env_combined,
                   "docs/assets/challenge_three_way_comparison.png")

    print(f"\nResults saved to {out_dir}/")
    print("Plot saved to docs/assets/challenge_three_way_comparison.png")


if __name__ == "__main__":
    main()
