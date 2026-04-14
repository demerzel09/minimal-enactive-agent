"""Genetic algorithm to optimize weight magnitudes across all challenge environments.

Preserves the handtuned sign structure — GA only adjusts absolute values.
Fitness is a composite score across ALL challenge environments simultaneously.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from src.run_simulation import run_episode


# ---------------------------------------------------------------------------
# Challenge suite definition (same as run_challenge.py)
# ---------------------------------------------------------------------------

CHALLENGE_CONFIGS = {
    "A_baseline": "configs/challenge/env_a_baseline.yaml",
    "B_distant": "configs/challenge/env_b_distant_patches.yaml",
    "C_fast_depl": "configs/challenge/env_c_fast_depletion.yaml",
    "D_risk_food": "configs/challenge/env_d_risk_near_food.yaml",
    "E_no_risk": "configs/challenge/env_e_no_risk.yaml",
    "F_small": "configs/challenge/env_f_small_world.yaml",
    "G_3patches": "configs/challenge/env_g_three_patches.yaml",
}

EVAL_SEEDS = [7, 13, 42]

# ---------------------------------------------------------------------------
# Genome ←→ weight conversion
# ---------------------------------------------------------------------------

# Reference sign pattern from handtuned v7
_SIGN_W_hi = np.array([[-1, +1, -1], [-1, -1, -1]])
_SIGN_W_hh = np.array([[+1, +1], [+1, +1]])
_SIGN_W_hm = np.array([[-1, +1, +1], [-1, +1, +0]])  # 0 means fixed at 0
_SIGN_b_h = np.array([+1, +1])

_SIGN_W_uh = np.array([[-1, -1], [+1, +1], [+1, -1]])
_SIGN_W_uu = np.array([[+1, -1, -1], [-1, +1, -1], [-1, -1, +1]])
_SIGN_W_ui = np.array([[+1, -1, +1], [-1, -1, -1], [-1, +1, +0]])  # 0 fixed
_SIGN_b_u = np.array([-1, +0, -1])  # 0 fixed

# Also optimize alpha_h and alpha_m
# Genome layout: [alpha_h, alpha_m, |W_hi|..., |W_hh|..., |W_hm|..., |b_h|...,
#                 |W_uh|..., |W_uu|..., |W_ui|..., |b_u|...]

_SHAPES = {
    "alpha_h": (1,),
    "alpha_m": (1,),
    "W_hi": (2, 3),
    "W_hh": (2, 2),
    "W_hm": (2, 3),
    "b_h": (2,),
    "W_uh": (3, 2),
    "W_uu": (3, 3),
    "W_ui": (3, 3),
    "b_u": (3,),
}

_SIGNS = {
    "W_hi": _SIGN_W_hi,
    "W_hh": _SIGN_W_hh,
    "W_hm": _SIGN_W_hm,
    "b_h": _SIGN_b_h,
    "W_uh": _SIGN_W_uh,
    "W_uu": _SIGN_W_uu,
    "W_ui": _SIGN_W_ui,
    "b_u": _SIGN_b_u,
}


def genome_length() -> int:
    return sum(int(np.prod(s)) for s in _SHAPES.values())


def handtuned_genome() -> np.ndarray:
    """Encode the handtuned v7 weights as a genome (absolute values)."""
    parts = []
    # alpha_h, alpha_m
    parts.append(np.array([0.04]))
    parts.append(np.array([0.40]))
    # Weight matrices — store absolute values
    ht = _handtuned_weights()
    for name in ["W_hi", "W_hh", "W_hm", "b_h", "W_uh", "W_uu", "W_ui", "b_u"]:
        parts.append(np.abs(ht[name]).ravel())
    return np.concatenate(parts)


def _handtuned_weights() -> Dict[str, np.ndarray]:
    return {
        "W_hi": np.array([[-0.8, +0.3, -0.4], [-0.5, -0.2, -0.3]]),
        "W_hh": np.array([[+0.3, +0.15], [+0.2, +0.3]]),
        "W_hm": np.array([[-0.5, +0.1, +0.2], [-0.4, +0.3, +0.0]]),
        "b_h": np.array([+0.3, +0.2]),
        "W_uh": np.array([[-0.8, -0.5], [+0.6, +0.9], [+0.4, -0.2]]),
        "W_uu": np.array([[+0.5, -0.3, -0.2], [-0.3, +0.5, -0.2], [-0.2, -0.2, +0.4]]),
        "W_ui": np.array([[+0.7, -0.3, +0.4], [-0.5, -0.1, -0.4], [-0.1, +0.8, -0.0]]),
        "b_u": np.array([-0.1, +0.0, -0.2]),
    }


def decode_genome(genome: np.ndarray) -> Dict:
    """Convert genome back to model config overrides."""
    idx = 0

    def _take(n: int) -> np.ndarray:
        nonlocal idx
        vals = genome[idx : idx + n]
        idx += n
        return vals

    alpha_h = float(np.clip(_take(1)[0], 0.005, 0.3))
    alpha_m = float(np.clip(_take(1)[0], 0.1, 0.8))

    weights = {}
    for name in ["W_hi", "W_hh", "W_hm", "b_h", "W_uh", "W_uu", "W_ui", "b_u"]:
        shape = _SHAPES[name]
        n = int(np.prod(shape))
        magnitudes = np.clip(_take(n).reshape(shape), 0.0, 2.0)
        sign = _SIGNS[name]
        # Where sign == 0, value stays 0. Otherwise apply sign.
        weights[name] = np.where(sign == 0, 0.0, sign * magnitudes)

    return {
        "alpha_h": alpha_h,
        "alpha_m": alpha_m,
        "weights": weights,
    }


def apply_genome_to_agent(agent, decoded: Dict) -> None:
    """Override agent weights with decoded genome values."""
    agent.alpha_h = decoded["alpha_h"]
    agent.alpha_m = decoded["alpha_m"]
    w = decoded["weights"]
    agent.W_hi = w["W_hi"].copy()
    agent.W_hh = w["W_hh"].copy()
    agent.W_hm = w["W_hm"].copy()
    agent.b_h = w["b_h"].copy()
    agent.W_uh = w["W_uh"].copy()
    agent.W_uu = w["W_uu"].copy()
    agent.W_ui = w["W_ui"].copy()
    agent.b_u = w["b_u"].copy()


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_genome_on_env(decoded: Dict, config_path: str, seed: int, env_name: str) -> float:
    """Run a single decoded genome on one environment/seed. Returns fitness score."""
    from src.agent import MinimalEnactiveAgent
    from src.env import ForagingEnv
    from src.eval import compute_metrics

    base_cfg = _load_config(config_path)
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["simulation"]["output_dir"] = f"outputs/ga_eval/{env_name}/seed_{seed}"
    cfg["model"]["init_mode"] = "handtuned"

    env = ForagingEnv(cfg)
    agent = MinimalEnactiveAgent(cfg)
    apply_genome_to_agent(agent, decoded)

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

    score = (
        metrics["stay_leave_transitions"] * 0.3
        + metrics["avg_local_food"] * 100.0 * 0.3
        + metrics["time_in_patch"] * 50.0 * 0.2
        + metrics["mode_switch_count"] * 0.1
        + (1.0 - metrics.get("avg_local_risk", 0.0)) * 2.0 * 0.1
    )
    return score


def evaluate_fitness(genome: np.ndarray, verbose: bool = False,
                     env_filter: str | None = None) -> float:
    """Evaluate genome across challenge environments.

    Args:
        env_filter: If set, only evaluate on this single environment name.
                    If None, evaluate across all environments (universal).
    """
    decoded = decode_genome(genome)
    total_score = 0.0

    if env_filter is not None:
        target = {env_filter: CHALLENGE_CONFIGS[env_filter]}
    else:
        target = CHALLENGE_CONFIGS

    for env_name, config_path in target.items():
        env_scores = []
        for seed in EVAL_SEEDS:
            try:
                score = _run_genome_on_env(decoded, config_path, seed, env_name)
                env_scores.append(score)
            except Exception as e:
                if verbose:
                    print(f"  Error in {env_name}/seed_{seed}: {e}")
                env_scores.append(0.0)
        total_score += float(np.mean(env_scores))

    return total_score


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

def init_population(pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize population around the handtuned solution."""
    n = genome_length()
    base = handtuned_genome()
    population = np.zeros((pop_size, n))

    # First individual is the handtuned baseline (elitism seed)
    population[0] = base.copy()

    # Rest are noisy variants around the baseline
    for i in range(1, pop_size):
        noise = rng.normal(0.0, 0.15, size=n) * base  # proportional noise
        population[i] = np.clip(base + noise, 0.0, 2.0)

    return population


def tournament_select(fitness: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    """Tournament selection."""
    candidates = rng.choice(len(fitness), size=k, replace=False)
    return int(candidates[np.argmax(fitness[candidates])])


def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover."""
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child


def mutate(genome: np.ndarray, rng: np.random.Generator, rate: float = 0.15, scale: float = 0.1) -> np.ndarray:
    """Gaussian mutation with rate control."""
    g = genome.copy()
    mask = rng.random(len(g)) < rate
    g[mask] += rng.normal(0.0, scale, size=int(mask.sum())) * np.maximum(np.abs(g[mask]), 0.05)
    g = np.clip(g, 0.0, 2.0)
    return g


def run_ga(
    pop_size: int = 30,
    generations: int = 40,
    elite_count: int = 3,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
    env_filter: str | None = None,
) -> Dict:
    """Run GA optimization.

    Args:
        env_filter: If set, optimize for a single environment only.
    """
    rng = np.random.default_rng(seed)
    n = genome_length()

    label = env_filter or "universal"
    out_dir = Path(f"outputs/ga/{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        target = env_filter or "ALL environments"
        print(f"GA [{label}]: pop={pop_size}, gen={generations}, genome_len={n}, target={target}")

    population = init_population(pop_size, rng)
    history: List[Dict] = []
    best_ever_fitness = -np.inf
    best_ever_genome = None

    for gen in range(generations):
        t0 = time.time()

        # Evaluate fitness
        fitness = np.array([evaluate_fitness(g, env_filter=env_filter) for g in population])

        # Track stats
        gen_best = int(np.argmax(fitness))
        gen_stats = {
            "generation": gen,
            "best_fitness": float(fitness[gen_best]),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "elapsed_sec": time.time() - t0,
        }
        history.append(gen_stats)

        if fitness[gen_best] > best_ever_fitness:
            best_ever_fitness = fitness[gen_best]
            best_ever_genome = population[gen_best].copy()

        if verbose:
            print(
                f"  Gen {gen:3d}: best={fitness[gen_best]:.2f}, "
                f"mean={np.mean(fitness):.2f} +/- {np.std(fitness):.2f}, "
                f"elapsed={gen_stats['elapsed_sec']:.1f}s"
            )

        # Save checkpoint every 10 generations
        if gen % 10 == 0 or gen == generations - 1:
            _save_checkpoint(out_dir, gen, best_ever_genome, best_ever_fitness, history)

        # Selection & reproduction
        sorted_idx = np.argsort(fitness)[::-1]
        new_pop = np.zeros_like(population)

        # Elitism
        for i in range(elite_count):
            new_pop[i] = population[sorted_idx[i]].copy()

        # Fill rest via crossover + mutation
        for i in range(elite_count, pop_size):
            p1_idx = tournament_select(fitness, rng)
            p2_idx = tournament_select(fitness, rng)
            child = crossover(population[p1_idx], population[p2_idx], rng)
            child = mutate(child, rng, rate=mutation_rate, scale=mutation_scale)
            new_pop[i] = child

        population = new_pop

    # Final results
    decoded = decode_genome(best_ever_genome)
    result = {
        "env_filter": env_filter,
        "best_fitness": float(best_ever_fitness),
        "best_genome": best_ever_genome.tolist(),
        "decoded_alpha_h": decoded["alpha_h"],
        "decoded_alpha_m": decoded["alpha_m"],
        "decoded_weights": {k: v.tolist() for k, v in decoded["weights"].items()},
        "history": history,
        "handtuned_genome": handtuned_genome().tolist(),
    }

    with open(out_dir / "ga_result.json", "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n=== GA Complete ===")
        print(f"Best fitness: {best_ever_fitness:.2f}")
        print(f"Best alpha_h: {decoded['alpha_h']:.4f}")
        print(f"Best alpha_m: {decoded['alpha_m']:.4f}")
        _print_weight_comparison(decoded)

    return result


def _save_checkpoint(out_dir: Path, gen: int, genome: np.ndarray, fitness: float, history: List[Dict]) -> None:
    checkpoint = {
        "generation": gen,
        "best_fitness": float(fitness),
        "best_genome": genome.tolist(),
        "history": history,
    }
    with open(out_dir / f"checkpoint_gen{gen:03d}.json", "w") as f:
        json.dump(checkpoint, f, indent=2)


def _print_weight_comparison(decoded: Dict) -> None:
    """Print GA-optimized vs handtuned weights side by side."""
    ht = _handtuned_weights()
    print("\n--- Weight comparison (handtuned → GA) ---")
    for name in ["W_hi", "W_hh", "W_hm", "b_h", "W_uh", "W_uu", "W_ui", "b_u"]:
        ht_flat = ht[name].ravel()
        ga_flat = decoded["weights"][name].ravel()
        diffs = ga_flat - ht_flat
        max_diff = np.max(np.abs(diffs))
        print(f"  {name}: max_diff={max_diff:.3f}")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_ga_comparison(ga_result: Dict) -> Dict:
    """Run the challenge suite with GA-optimized weights and compare to handtuned."""
    from src.run_challenge import run_challenge_suite, print_summary_table, plot_comparison

    # Build model override from GA result
    decoded_weights = {k: np.array(v) for k, v in ga_result["decoded_weights"].items()}
    override = {
        "alpha_h": ga_result["decoded_alpha_h"],
        "alpha_m": ga_result["decoded_alpha_m"],
    }

    # Run GA-optimized suite
    print("\n=== Challenge Suite: GA-optimized ===")
    ga_results = run_challenge_suite_with_weights(decoded_weights, override, label="ga_optimized")

    ga_table = print_summary_table(ga_results, "GA-optimized")
    plot_comparison(
        ga_results,
        "outputs/challenge/ga_optimized_comparison.png",
        "GA-optimized: Challenge environments",
    )

    out_dir = Path("outputs/challenge")
    with open(out_dir / "ga_optimized_results.json", "w") as f:
        json.dump(ga_results, f, indent=2)

    return ga_results


def run_challenge_suite_with_weights(
    weights: Dict[str, np.ndarray],
    alpha_override: Dict,
    label: str = "ga_optimized",
) -> Dict:
    """Run challenge suite with explicit weight overrides (not just model config)."""
    from src.eval import compute_metrics
    from src.agent import MinimalEnactiveAgent
    from src.env import ForagingEnv

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
            cfg["simulation"]["output_dir"] = f"outputs/challenge/{label}/{env_name}/seed_{seed}"
            cfg["model"]["init_mode"] = "handtuned"

            env = ForagingEnv(cfg)
            agent = MinimalEnactiveAgent(cfg)

            # Override weights
            agent.alpha_h = alpha_override["alpha_h"]
            agent.alpha_m = alpha_override["alpha_m"]
            for wname, wval in weights.items():
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
        trans = results[env_name]["stay_leave_transitions"]["mean"]
        food = results[env_name]["avg_local_food"]["mean"]
        print(f"  {env_name}: transitions={trans:.1f}, food={food:.3f}")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run GA optimization for minimal enactive agent")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size")
    parser.add_argument("--generations", type=int, default=40, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Also run comparison after GA")
    args = parser.parse_args()

    result = run_ga(
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed,
    )

    if args.compare:
        run_ga_comparison(result)


if __name__ == "__main__":
    main()
