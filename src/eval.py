"""Simple qualitative metrics for the first PoC."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _positive_run_lengths(binary_series: np.ndarray) -> np.ndarray:
    """Lengths of contiguous runs where binary_series == 1."""
    lengths = []
    run = 0
    for v in binary_series.astype(int):
        if v == 1:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return np.array(lengths, dtype=int)


def compute_metrics(log: Dict) -> Dict[str, float]:
    in_patch = np.asarray(log["in_patch"], dtype=int)
    local_food = np.asarray(log["local_food"], dtype=float)
    local_risk = np.asarray(log["local_risk"], dtype=float)
    m = np.asarray(log["m"], dtype=float)
    x = np.asarray(log["x"], dtype=float)
    y = np.asarray(log["y"], dtype=float)

    # Stay/leave proxies
    enter_leave_events = np.sum(np.abs(np.diff(in_patch)))
    patch_run_lengths = _positive_run_lengths(in_patch)
    avg_patch_residence = float(np.mean(patch_run_lengths)) if len(patch_run_lengths) else 0.0

    # Explore/exploit proxy using displacement and dominant mode changes
    pos = np.column_stack([x, y])
    step_dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    exploration_radius = float(np.mean(np.linalg.norm(pos - pos.mean(axis=0), axis=1)))

    dominant_mode = np.argmax(m, axis=1)
    mode_switch_count = int(np.sum(np.diff(dominant_mode) != 0))
    mode_persistence = float(1.0 - mode_switch_count / max(1, len(dominant_mode) - 1))

    # History dependence proxy:
    # correlation between current leave tendency and trailing food trend.
    food_trend = np.convolve(local_food, np.ones(10) / 10.0, mode="same")
    leave_signal = 1 - in_patch
    if np.std(food_trend) > 1e-8 and np.std(leave_signal) > 1e-8:
        history_dependence = float(np.corrcoef(food_trend, leave_signal)[0, 1])
    else:
        history_dependence = 0.0

    metrics = {
        "total_steps": float(len(in_patch)),
        "avg_local_food": float(np.mean(local_food)),
        "avg_local_risk": float(np.mean(local_risk)),
        "time_in_patch": float(np.mean(in_patch)),
        "stay_leave_transitions": float(enter_leave_events),
        "avg_patch_residence": avg_patch_residence,
        "mean_step_distance": float(np.mean(step_dist)) if len(step_dist) else 0.0,
        "exploration_radius": exploration_radius,
        "mode_switch_count": float(mode_switch_count),
        "mode_persistence": mode_persistence,
        "history_dependence_proxy": history_dependence,
    }

    # Patch diversity metrics (require layout + patch_levels in log)
    if "layout" in log and "patch_levels" in log:
        layout = log["layout"]
        n_patches = layout.get("n_patches", 0)
        patch_centers = layout.get("patch_centers", [])
        patch_radii = layout.get("patch_radii", [])
        patch_levels = np.asarray(log["patch_levels"], dtype=float)  # [steps, n_patches]

        if n_patches > 0 and len(patch_centers) == n_patches:
            # Which patch is the agent inside at each step?
            patch_ids = []
            for t in range(len(x)):
                pid = -1
                for i in range(n_patches):
                    cx, cy = patch_centers[i]
                    r = patch_radii[i]
                    if (x[t] - cx) ** 2 + (y[t] - cy) ** 2 <= r ** 2:
                        pid = i
                        break
                patch_ids.append(pid)
            patch_ids = np.array(patch_ids, dtype=int)

            visited = set(patch_ids[patch_ids >= 0])
            metrics["unique_patches_visited"] = float(len(visited))
            metrics["unique_patch_fraction"] = float(len(visited) / n_patches)

            # Depleted revisit: visits where patch level < 20% of max
            in_any = patch_ids >= 0
            if in_any.sum() > 0:
                depleted_visits = 0
                total_visits = 0
                for t in range(len(patch_ids)):
                    pid = patch_ids[t]
                    if pid >= 0:
                        total_visits += 1
                        max_food_i = patch_levels[0, pid] if patch_levels[0, pid] > 0 else 0.5
                        if patch_levels[t, pid] < 0.2 * max_food_i:
                            depleted_visits += 1
                metrics["depleted_revisit_fraction"] = float(
                    depleted_visits / max(1, total_visits)
                )
            else:
                metrics["depleted_revisit_fraction"] = 0.0

            # Time to leave after depletion starts (per patch visit)
            leave_times = []
            for pid in visited:
                # Find contiguous runs inside this patch
                in_this = (patch_ids == pid).astype(int)
                runs = _positive_run_lengths(in_this)
                # For each run, check if patch was depleting
                idx = 0
                for run_len in runs:
                    while idx < len(in_this) and in_this[idx] == 0:
                        idx += 1
                    start = idx
                    end = min(start + run_len, len(patch_levels))
                    if end > start and end <= len(patch_levels):
                        levels_during = patch_levels[start:end, pid]
                        if len(levels_during) > 1 and levels_during[0] > levels_during[-1]:
                            leave_times.append(run_len)
                    idx = end
            metrics["avg_time_to_leave_depleted"] = float(
                np.mean(leave_times) if leave_times else 0.0
            )

    return metrics
