"""Viability-fitness GA — the T1 next step (redesign_viability_first.md §5 Phase 1, §6).

Experiment 10 (T1) showed that under stakes (energy/death), the HAND-TUNED full
model (with h) still collapses: it traps in one patch and dies, while no_h roams
and survives. Before concluding this is an ARCHITECTURE constraint, we must rule
out a PARAMETER problem (environment_first_roadmap.md §9): optimize ALL weights —
including the new interoceptive term W_he — under a VIABILITY fitness and ask
whether GA-full catches GA-no_h on survival.

  - GA-full catches GA-no_h  -> the hand-tuned W_he/weights were just bad
                                (parameter problem). Position advances.
  - GA-full does NOT catch it -> even with tuned weights, h can't escape the
                                trap under stakes (architecture constraint).

Key differences from run_ga.py (the foraging GA):
  - Genome INCLUDES W_he (interoceptive energy-deficit -> h).
  - Fitness is VIABILITY (survival_time, with mean energy margin as a shaper),
    NOT foraging score. Episodes terminate on death.
  - Same env for full and no_h; the only difference is use_h.

Fitness is the GA's (generational-scale) selection pressure, not an within-lifetime
reward for the agent — consistent with the reward-free core (the agent never sees it).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Reuse the generic genome operators (they act on flat arrays, layout-agnostic).
from src.run_ga import tournament_select, crossover, mutate


# ---------------------------------------------------------------------------
# Genome layout (extends run_ga.py's with W_he)
# ---------------------------------------------------------------------------

_SHAPES = {
    "W_hi": (2, 3),
    "W_hh": (2, 2),
    "W_hm": (2, 3),
    "W_he": (2,),          # NEW: interoceptive energy-deficit -> h
    "b_h": (2,),
    "W_uh": (3, 2),
    "W_uu": (3, 3),
    "W_ui": (3, 3),
    "b_u": (3,),
}

# Sign structure from handtuned (0 => fixed at 0). W_he is positive: low energy
# (high deficit) should raise depletion pressure / exploration drift.
_SIGNS = {
    "W_hi": np.array([[-1, +1, -1], [-1, -1, -1]]),
    "W_hh": np.array([[+1, +1], [+1, +1]]),
    "W_hm": np.array([[-1, +1, +1], [-1, +1, +0]]),
    "W_he": np.array([+1, +1]),
    "b_h": np.array([+1, +1]),
    "W_uh": np.array([[-1, -1], [+1, +1], [+1, -1]]),
    "W_uu": np.array([[+1, -1, -1], [-1, +1, -1], [-1, -1, +1]]),
    "W_ui": np.array([[+1, -1, +1], [-1, -1, -1], [-1, +1, +0]]),
    "b_u": np.array([-1, +0, -1]),
}

_ORDER = ["W_hi", "W_hh", "W_hm", "W_he", "b_h", "W_uh", "W_uu", "W_ui", "b_u"]

_HANDTUNED = {
    "W_hi": np.array([[-0.8, +0.3, -0.4], [-0.5, -0.2, -0.3]]),
    "W_hh": np.array([[+0.3, +0.15], [+0.2, +0.3]]),
    "W_hm": np.array([[-0.5, +0.1, +0.2], [-0.4, +0.3, +0.0]]),
    "W_he": np.array([+0.8, +0.3]),
    "b_h": np.array([+0.3, +0.2]),
    "W_uh": np.array([[-0.8, -0.5], [+0.6, +0.9], [+0.4, -0.2]]),
    "W_uu": np.array([[+0.5, -0.3, -0.2], [-0.3, +0.5, -0.2], [-0.2, -0.2, +0.4]]),
    "W_ui": np.array([[+0.7, -0.3, +0.4], [-0.5, -0.1, -0.4], [-0.1, +0.8, -0.0]]),
    "b_u": np.array([-0.1, +0.0, -0.2]),
}


def genome_length() -> int:
    # +2 for alpha_h, alpha_m
    return 2 + sum(int(np.prod(s)) for s in _SHAPES.values())


def handtuned_genome() -> np.ndarray:
    parts = [np.array([0.04]), np.array([0.40])]  # alpha_h, alpha_m
    for name in _ORDER:
        parts.append(np.abs(_HANDTUNED[name]).ravel())
    return np.concatenate(parts)


def decode_genome(genome: np.ndarray) -> Dict:
    idx = 0

    def _take(n: int) -> np.ndarray:
        nonlocal idx
        vals = genome[idx: idx + n]
        idx += n
        return vals

    alpha_h = float(np.clip(_take(1)[0], 0.005, 0.3))
    alpha_m = float(np.clip(_take(1)[0], 0.1, 0.8))

    weights = {}
    for name in _ORDER:
        shape = _SHAPES[name]
        n = int(np.prod(shape))
        magnitudes = np.clip(_take(n).reshape(shape), 0.0, 2.0)
        sign = _SIGNS[name]
        weights[name] = np.where(sign == 0, 0.0, sign * magnitudes)

    return {"alpha_h": alpha_h, "alpha_m": alpha_m, "weights": weights}


def apply_genome_to_agent(agent, decoded: Dict) -> None:
    agent.alpha_h = decoded["alpha_h"]
    agent.alpha_m = decoded["alpha_m"]
    for name, val in decoded["weights"].items():
        setattr(agent, name, val.copy())


# ---------------------------------------------------------------------------
# Viability fitness
# ---------------------------------------------------------------------------

# Fitness = survival_time (dominant) + MARGIN_WEIGHT * mean energy margin.
# survival_time gives gradient while the agent dies early; the margin term
# differentiates among full-survivors (how comfortably they self-maintain).
MARGIN_WEIGHT = 300.0


def _run_viability_episode(decoded: Dict, base_cfg: Dict, seed: int, use_h: bool) -> Dict:
    from src.registry import create_env, create_agent
    from src.eval import compute_metrics

    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["model"]["init_mode"] = "handtuned"
    cfg["model"]["use_h"] = bool(use_h)
    cfg["simulation"]["output_dir"] = f"outputs/ga_viability_eval/tmp_seed_{seed}"

    env = create_env(cfg)
    agent = create_agent(cfg)
    apply_genome_to_agent(agent, decoded)

    env_state = env.reset()
    agent.reset()
    obs = agent.sense(env_state)

    steps = int(cfg["simulation"].get("steps", 2000))
    log = {
        "x": [], "y": [], "local_food": [], "local_risk": [],
        "in_patch": [], "patch_level": [], "h": [], "m": [],
        "action_turn": [], "action_speed": [], "energy": [],
    }
    died = False
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
        log["energy"].append(float(step_info.info["energy"]))

        if step_info.info.get("alive", 1.0) == 0.0:
            died = True
            break

    log["died"] = died
    return compute_metrics(log)


def viability_fitness(genome: np.ndarray, base_cfg: Dict, seeds: List[int], use_h: bool) -> float:
    decoded = decode_genome(genome)
    scores = []
    for seed in seeds:
        try:
            m = _run_viability_episode(decoded, base_cfg, seed, use_h)
            scores.append(m["survival_time"] + MARGIN_WEIGHT * m["mean_viability_margin"])
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------

def init_population(pop_size: int, rng: np.random.Generator) -> np.ndarray:
    n = genome_length()
    base = handtuned_genome()
    population = np.zeros((pop_size, n))
    population[0] = base.copy()  # elitism seed = handtuned baseline
    for i in range(1, pop_size):
        noise = rng.normal(0.0, 0.15, size=n) * base
        population[i] = np.clip(base + noise, 0.0, 2.0)
    return population


def run_ga_viability(
    config_path: str,
    use_h: bool,
    pop_size: int = 24,
    generations: int = 25,
    elite_count: int = 3,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.12,
    seeds: Optional[List[int]] = None,
    ga_seed: int = 42,
    verbose: bool = True,
) -> Dict:
    seeds = seeds or [7, 13, 42]
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    rng = np.random.default_rng(ga_seed)
    label = "full" if use_h else "no_h"
    out_dir = Path(f"outputs/ga_viability/{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"GA-viability [{label}]: pop={pop_size} gen={generations} "
              f"genome_len={genome_length()} seeds={seeds} config={config_path}")

    population = init_population(pop_size, rng)
    history: List[Dict] = []
    best_fit = -np.inf
    best_genome = None

    for gen in range(generations):
        t0 = time.time()
        fitness = np.array([viability_fitness(g, base_cfg, seeds, use_h) for g in population])

        gbest = int(np.argmax(fitness))
        if fitness[gbest] > best_fit:
            best_fit = float(fitness[gbest])
            best_genome = population[gbest].copy()

        history.append({
            "generation": gen,
            "best_fitness": float(fitness[gbest]),
            "mean_fitness": float(np.mean(fitness)),
            "elapsed_sec": time.time() - t0,
        })
        if verbose:
            print(f"  [{label}] gen {gen:3d}: best={fitness[gbest]:8.1f} "
                  f"mean={np.mean(fitness):8.1f} ({time.time()-t0:.1f}s)")

        sorted_idx = np.argsort(fitness)[::-1]
        new_pop = np.zeros_like(population)
        for i in range(elite_count):
            new_pop[i] = population[sorted_idx[i]].copy()
        for i in range(elite_count, pop_size):
            p1 = tournament_select(fitness, rng)
            p2 = tournament_select(fitness, rng)
            child = crossover(population[p1], population[p2], rng)
            child = mutate(child, rng, rate=mutation_rate, scale=mutation_scale)
            new_pop[i] = child
        population = new_pop

    # Evaluate the best genome per-seed for a detailed viability report.
    decoded = decode_genome(best_genome)
    per_seed = []
    for seed in seeds:
        m = _run_viability_episode(decoded, base_cfg, seed, use_h)
        per_seed.append({
            "seed": seed,
            "survival_time": m["survival_time"],
            "died": m["died"],
            "mean_energy": m["mean_energy"],
            "min_energy": m["min_energy"],
            "unique_patches_visited": m.get("unique_patches_visited", float("nan")),
        })

    result = {
        "condition": label,
        "use_h": use_h,
        "config": config_path,
        "seeds": seeds,
        "best_fitness": best_fit,
        "best_genome": best_genome.tolist(),
        "decoded_alpha_h": decoded["alpha_h"],
        "decoded_alpha_m": decoded["alpha_m"],
        "decoded_weights": {k: v.tolist() for k, v in decoded["weights"].items()},
        "per_seed_best": per_seed,
        "history": history,
    }
    with open(out_dir / "ga_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        surv = np.mean([p["survival_time"] for p in per_seed])
        deaths = sum(p["died"] for p in per_seed)
        print(f"=== GA-viability [{label}] done: best_fit={best_fit:.1f}, "
              f"mean_survival={surv:.0f}, deaths={deaths}/{len(seeds)} ===")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Viability-fitness GA (T1 next step)")
    parser.add_argument("--config", type=str, default="configs/odor_field_viability.yaml")
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 42])
    parser.add_argument("--ga-seed", type=int, default=42)
    parser.add_argument("--condition", choices=["full", "no_h", "both"], default="both")
    args = parser.parse_args()

    results = {}
    if args.condition in ("full", "both"):
        results["full"] = run_ga_viability(
            args.config, use_h=True, pop_size=args.pop_size,
            generations=args.generations, seeds=args.seeds, ga_seed=args.ga_seed)
    if args.condition in ("no_h", "both"):
        results["no_h"] = run_ga_viability(
            args.config, use_h=False, pop_size=args.pop_size,
            generations=args.generations, seeds=args.seeds, ga_seed=args.ga_seed)

    if "full" in results and "no_h" in results:
        f = np.mean([p["survival_time"] for p in results["full"]["per_seed_best"]])
        n = np.mean([p["survival_time"] for p in results["no_h"]["per_seed_best"]])
        print("\n================= T1 GA VERDICT =================")
        print(f"GA-full  mean survival: {f:.0f}")
        print(f"GA-no_h  mean survival: {n:.0f}")
        if f >= 0.9 * n:
            print("=> GA-full CATCHES GA-no_h: parameter problem (hand-tuned weights were bad).")
        else:
            print("=> GA-full does NOT catch GA-no_h: architecture constraint under stakes.")
        Path("outputs/ga_viability").mkdir(parents=True, exist_ok=True)
        with open("outputs/ga_viability/verdict.json", "w", encoding="utf-8") as fh:
            json.dump({"ga_full_survival": float(f), "ga_no_h_survival": float(n)}, fh, indent=2)


if __name__ == "__main__":
    main()
