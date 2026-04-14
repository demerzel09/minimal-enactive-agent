"""Experiment 3: Direct test of history dependence.

Construct two conditions with identical immediate observation but different
recent histories, then compare the agent's action at that moment.

Condition A (rich history): agent spends N steps inside the food patch.
  -> h should reflect low depletion pressure, low exploration drift.
Condition B (poor history): agent spends N steps far from the food patch.
  -> h should reflect high depletion pressure, high exploration drift.

After preconditioning, both agents are placed at the same position (patch edge)
and receive the same observation. We compare:
  1. The internal state h (should differ for Full, identical zero for no-h)
  2. The action output (should differ for Full, identical for no-h)
  3. The mode distribution m (should differ for Full)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.agent import MinimalEnactiveAgent
from src.env import ForagingEnv, EnvState


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_state_at(env: ForagingEnv, pos: np.ndarray) -> EnvState:
    """Create an EnvState snapshot at a specific position."""
    env.pos = pos.copy()
    return env._make_state()


def precondition(
    agent: MinimalEnactiveAgent,
    env: ForagingEnv,
    pos: np.ndarray,
    n_steps: int,
) -> None:
    """Run agent at a fixed position for n_steps to build up history in h."""
    env.pos = pos.copy()
    for _ in range(n_steps):
        env_state = env._make_state()
        obs = agent.sense(env_state)
        agent.step(obs)
        # Agent acts but we keep position fixed (preconditioning, not free movement)


def run_probe(
    agent: MinimalEnactiveAgent,
    env: ForagingEnv,
    probe_pos: np.ndarray,
) -> Dict:
    """Place agent at probe position and record one step's output."""
    env_state = make_env_state_at(env, probe_pos)
    obs = agent.sense(env_state)
    step = agent.step(obs)
    return {
        "obs": obs.tolist(),
        "h": step.h.tolist(),
        "m": step.m.tolist(),
        "action": step.action.tolist(),
    }


def run_history_test(
    config: Dict,
    precondition_steps: int = 60,
) -> Dict:
    """Run the full history dependence test for one config."""
    env = ForagingEnv(config)
    env.reset()

    patch_center = np.array(config["environment"]["patch_center"], dtype=float)
    patch_radius = float(config["environment"]["patch_radius"])

    # Positions for preconditioning
    pos_in_patch = patch_center.copy()  # center of food patch
    pos_outside = np.array([2.0, 2.0], dtype=float)  # far from patch

    # Probe position: patch edge (where food signal is moderate)
    probe_pos = patch_center + np.array([patch_radius * 0.9, 0.0])

    results = {}
    for condition, precond_pos in [("rich", pos_in_patch), ("poor", pos_outside)]:
        agent = MinimalEnactiveAgent(config)
        agent.reset()

        # Build up history
        precondition(agent, env, precond_pos, precondition_steps)

        # Record h state just before probe
        h_before_probe = agent.h.copy()

        # Probe at identical position
        probe = run_probe(agent, env, probe_pos)
        probe["h_before_probe"] = h_before_probe.tolist()
        results[condition] = probe

    return results


def run_experiment(
    n_seeds: int = 10,
    precondition_steps: int = 60,
) -> Dict:
    """Run history test across Full and no-h, multiple seeds."""
    configs = {
        "full": "configs/full.yaml",
        "no_h": "configs/ablation_no_h.yaml",
    }

    all_results = {}
    for cond_name, config_path in configs.items():
        base_cfg = load_config(config_path)
        seed_results = []

        for seed in range(n_seeds):
            cfg = base_cfg.copy()
            cfg["seed"] = seed
            result = run_history_test(cfg, precondition_steps)
            seed_results.append(result)

        all_results[cond_name] = seed_results

    return all_results


def analyze_and_plot(results: Dict, output_dir: str) -> Dict[str, float]:
    """Analyze results and create comparison plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}

    for model_name, seed_results in results.items():
        # Collect action differences across seeds
        action_diffs = []
        h_diffs = []
        m_diffs = []

        rich_actions = []
        poor_actions = []
        rich_h = []
        poor_h = []
        rich_m = []
        poor_m = []

        for sr in seed_results:
            a_rich = np.array(sr["rich"]["action"])
            a_poor = np.array(sr["poor"]["action"])
            action_diffs.append(np.linalg.norm(a_rich - a_poor))

            h_r = np.array(sr["rich"]["h_before_probe"])
            h_p = np.array(sr["poor"]["h_before_probe"])
            h_diffs.append(np.linalg.norm(h_r - h_p))

            m_r = np.array(sr["rich"]["m"])
            m_p = np.array(sr["poor"]["m"])
            m_diffs.append(np.linalg.norm(m_r - m_p))

            rich_actions.append(a_rich)
            poor_actions.append(a_poor)
            rich_h.append(h_r)
            poor_h.append(h_p)
            rich_m.append(m_r)
            poor_m.append(m_p)

        summary[f"{model_name}_action_diff_mean"] = float(np.mean(action_diffs))
        summary[f"{model_name}_action_diff_std"] = float(np.std(action_diffs))
        summary[f"{model_name}_h_diff_mean"] = float(np.mean(h_diffs))
        summary[f"{model_name}_m_diff_mean"] = float(np.mean(m_diffs))

        # Plot
        rich_actions = np.array(rich_actions)
        poor_actions = np.array(poor_actions)
        rich_h = np.array(rich_h)
        poor_h = np.array(poor_h)
        rich_m = np.array(rich_m)
        poor_m = np.array(poor_m)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"History dependence: {model_name}", fontsize=13)

        # h comparison
        ax = axes[0]
        n_seeds = len(seed_results)
        x = np.arange(rich_h.shape[1])
        width = 0.35
        ax.bar(x - width / 2, rich_h.mean(axis=0), width, yerr=rich_h.std(axis=0),
               label="rich (in patch)", color="tab:green", alpha=0.7)
        ax.bar(x + width / 2, poor_h.mean(axis=0), width, yerr=poor_h.std(axis=0),
               label="poor (outside)", color="tab:red", alpha=0.7)
        ax.set_xlabel("h dimension")
        ax.set_ylabel("h value")
        ax.set_title("Internal state h before probe")
        ax.set_xticks(x)
        ax.set_xticklabels([f"h[{i}]" for i in x])
        ax.legend(fontsize=8)

        # m comparison
        ax = axes[1]
        x = np.arange(rich_m.shape[1])
        ax.bar(x - width / 2, rich_m.mean(axis=0), width, yerr=rich_m.std(axis=0),
               label="rich", color="tab:green", alpha=0.7)
        ax.bar(x + width / 2, poor_m.mean(axis=0), width, yerr=poor_m.std(axis=0),
               label="poor", color="tab:red", alpha=0.7)
        ax.set_xlabel("mode")
        ax.set_ylabel("m value")
        ax.set_title("Mode distribution at probe")
        ax.set_xticks(x)
        ax.set_xticklabels(["exploit", "explore", "avoid"])
        ax.legend(fontsize=8)

        # action comparison
        ax = axes[2]
        labels = ["turn", "speed"]
        x = np.arange(2)
        ax.bar(x - width / 2, rich_actions.mean(axis=0), width, yerr=rich_actions.std(axis=0),
               label="rich", color="tab:green", alpha=0.7)
        ax.bar(x + width / 2, poor_actions.mean(axis=0), width, yerr=poor_actions.std(axis=0),
               label="poor", color="tab:red", alpha=0.7)
        ax.set_xlabel("action")
        ax.set_ylabel("value")
        ax.set_title("Action at probe")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out / f"history_dep_{model_name}.png", dpi=140)
        plt.close(fig)

    # Save summary
    with open(out / "history_dependence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    print("=== Experiment 3: History Dependence Direct Test ===")
    results = run_experiment(n_seeds=10, precondition_steps=60)

    summary = analyze_and_plot(results, "outputs/experiment3_history_dependence")

    print("\n--- Summary ---")
    print(f"Full model  action diff: {summary['full_action_diff_mean']:.4f} +/- {summary['full_action_diff_std']:.4f}")
    print(f"Full model  h diff:      {summary['full_h_diff_mean']:.4f}")
    print(f"Full model  m diff:      {summary['full_m_diff_mean']:.4f}")
    print(f"no-h model  action diff: {summary['no_h_action_diff_mean']:.4f} +/- {summary['no_h_action_diff_std']:.4f}")
    print(f"no-h model  h diff:      {summary['no_h_h_diff_mean']:.4f}")
    print(f"no-h model  m diff:      {summary['no_h_m_diff_mean']:.4f}")

    ratio = summary["full_action_diff_mean"] / max(summary["no_h_action_diff_mean"], 1e-10)
    print(f"\nAction diff ratio (full / no-h): {ratio:.2f}x")

    if summary["full_action_diff_mean"] > 0.05 and summary["no_h_action_diff_mean"] < 0.01:
        print("\n=> PASS: h creates history-dependent action; removing h eliminates it.")
    elif summary["full_action_diff_mean"] > summary["no_h_action_diff_mean"] * 2:
        print("\n=> PARTIAL: Full model shows stronger history dependence than no-h.")
    else:
        print("\n=> INCONCLUSIVE: Difference not clear enough.")


if __name__ == "__main__":
    main()
