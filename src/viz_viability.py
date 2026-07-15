"""Visualize the viability mechanism (experiment 12 follow-up, option 1).

Question: does the slow internal state h drive patch DEPARTURE *before* energy
drops (anticipatory), which is why GA-full keeps a comfortable viability margin
while the reactive GA-no_h skates the death boundary and eventually starves?

Runs the GA-best genome (from outputs/ga_viability/{full,no_h}/ga_result.json)
for one episode on a given config/seed, logging energy, h, mode, local_food and
patch occupancy, then plots full vs no_h and prints anticipation statistics
(energy and local_food at the moment of each patch exit).

Usage:
    uv run python -m src.viz_viability --config configs/odor_field_harsh.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.run_ga_viability import decode_genome, apply_genome_to_agent


def _run_logged_episode(genome: List[float], base_cfg: Dict, seed: int, use_h: bool) -> Dict:
    import copy
    from src.registry import create_env, create_agent

    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["model"]["init_mode"] = "handtuned"
    cfg["model"]["use_h"] = bool(use_h)
    cfg["simulation"]["output_dir"] = "outputs/viz_viability/tmp"

    env = create_env(cfg)
    agent = create_agent(cfg)
    apply_genome_to_agent(agent, decode_genome(np.asarray(genome, dtype=float)))

    env_state = env.reset()
    agent.reset()
    obs = agent.sense(env_state)

    steps = int(cfg["simulation"].get("steps", 2000))
    log = {"energy": [], "h0": [], "h1": [], "mode": [], "in_patch": [], "local_food": []}
    died_at = None
    for t in range(steps):
        step = agent.step(obs)
        env_state, info = env.step(step.action)
        obs = agent.sense(env_state)

        log["energy"].append(float(info.info["energy"]))
        log["h0"].append(float(step.h[0]))
        log["h1"].append(float(step.h[1]))
        log["mode"].append(int(np.argmax(step.m)))
        log["in_patch"].append(int(info.info["in_patch"]))
        log["local_food"].append(float(step.observation[0]))

        if info.info.get("alive", 1.0) == 0.0:
            died_at = t
            break
    log["died_at"] = died_at
    return log


def _exit_stats(log: Dict) -> Dict:
    """Energy and local_food at each patch-exit (in_patch 1 -> 0)."""
    inp = np.asarray(log["in_patch"], dtype=int)
    energy = np.asarray(log["energy"], dtype=float)
    food = np.asarray(log["local_food"], dtype=float)
    exits = np.where((inp[:-1] == 1) & (inp[1:] == 0))[0]  # index of last step inside
    return {
        "n_exits": int(len(exits)),
        "energy_at_exit_mean": float(np.mean(energy[exits])) if len(exits) else float("nan"),
        "food_at_exit_mean": float(np.mean(food[exits])) if len(exits) else float("nan"),
        "exit_idx": exits.tolist(),
    }


def visualize(config_path: str, seed: int, out_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    genomes = {}
    for cond in ["full", "no_h"]:
        rp = Path(f"outputs/ga_viability/{cond}/ga_result.json")
        genomes[cond] = json.load(open(rp, encoding="utf-8"))["best_genome"]

    full = _run_logged_episode(genomes["full"], base_cfg, seed, use_h=True)
    no_h = _run_logged_episode(genomes["no_h"], base_cfg, seed, use_h=False)

    fs = _exit_stats(full)
    ns = _exit_stats(no_h)
    print(f"=== Anticipation stats (seed {seed}) ===")
    print(f"FULL: exits={fs['n_exits']}, energy@exit={fs['energy_at_exit_mean']:.2f}, "
          f"food@exit={fs['food_at_exit_mean']:.3f}, died_at={full['died_at']}")
    print(f"NO_H: exits={ns['n_exits']}, energy@exit={ns['energy_at_exit_mean']:.2f}, "
          f"food@exit={ns['food_at_exit_mean']:.3f}, died_at={no_h['died_at']}")
    print("(higher energy@exit + higher food@exit for FULL => leaves patches EARLY/"
          "anticipatorily, before exhausting them, keeping a viability margin.)")

    # ---- plot ----
    C_FULL, C_NOH = "#1f77b4", "#d62728"
    C_H0, C_H1 = "#2ca02c", "#9467bd"
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    tf = np.arange(len(full["energy"]))
    tn = np.arange(len(no_h["energy"]))

    # 1) energy full vs no_h
    ax[0].plot(tf, full["energy"], color=C_FULL, lw=1.4, label="full (h)")
    ax[0].plot(tn, no_h["energy"], color=C_NOH, lw=1.4, ls="--", label="no_h")
    ax[0].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    if no_h["died_at"] is not None:
        ax[0].scatter([no_h["died_at"]], [0.0], color=C_NOH, marker="x", s=80, zorder=5, label="no_h death")
    ax[0].set_ylabel("energy\n(viability margin)")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Viability mechanism — harsh L3, seed {seed}: does h drive anticipatory departure?")

    # 2) full internal state h
    ax[1].plot(tf, full["h0"], color=C_H0, lw=1.2, label="h[0] depletion pressure")
    ax[1].plot(tf, full["h1"], color=C_H1, lw=1.2, label="h[1] exploration drift")
    ax[1].axhline(0.0, color="k", lw=0.6, alpha=0.3)
    ax[1].set_ylabel("full: h")
    ax[1].legend(loc="upper right", fontsize=8)

    # 3) full local_food + in_patch shading + exit marks
    ax[2].plot(tf, full["local_food"], color=C_FULL, lw=1.0, label="full local_food (odor dev.)")
    ax[2].fill_between(tf, 0, 1, where=np.asarray(full["in_patch"]) == 1,
                       color=C_FULL, alpha=0.10, transform=ax[2].get_xaxis_transform(),
                       label="full in-patch")
    for e in fs["exit_idx"]:
        ax[2].axvline(e, color=C_H1, lw=0.5, alpha=0.5)
    ax[2].set_ylabel("full: sensing")
    ax[2].legend(loc="upper right", fontsize=8)

    # 4) dominant mode: full vs no_h
    ax[3].plot(tf, full["mode"], color=C_FULL, lw=1.0, label="full mode")
    ax[3].plot(tn, no_h["mode"], color=C_NOH, lw=1.0, ls="--", alpha=0.7, label="no_h mode")
    ax[3].set_yticks([0, 1, 2])
    ax[3].set_yticklabels(["exploit", "explore", "avoid"])
    ax[3].set_ylabel("mode")
    ax[3].set_xlabel("timestep")
    ax[3].legend(loc="upper right", fontsize=8)

    for a in ax:
        a.grid(True, alpha=0.15)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/odor_field_harsh.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/viz_viability/mechanism.png")
    args = p.parse_args()
    visualize(args.config, args.seed, args.out)


if __name__ == "__main__":
    main()
