"""Minimal 2D closed-loop foraging environment for the first PoC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    info: Dict[str, float]


class ForagingEnv:
    """2D environment with one depleting food patch and sparse risk.

    The agent receives only local sensing:
    - local food
    - local risk
    - local food trend (delta from previous step)
    """

    def __init__(self, config: Dict):
        self.cfg = config
        env_cfg = config["environment"]

        self.world_size = float(env_cfg.get("world_size", 20.0))
        self.step_size = float(env_cfg.get("step_size", 0.7))
        self.turn_angle = float(env_cfg.get("turn_angle", 0.4))

        self.patch_center = np.array(env_cfg.get("patch_center", [10.0, 10.0]), dtype=float)
        self.patch_radius = float(env_cfg.get("patch_radius", 2.8))
        self.patch_max_food = float(env_cfg.get("patch_max_food", 1.0))
        self.patch_depletion_rate = float(env_cfg.get("patch_depletion_rate", 0.035))
        self.patch_regen_rate = float(env_cfg.get("patch_regen_rate", 0.004))

        self.risk_center = np.array(env_cfg.get("risk_center", [14.5, 6.0]), dtype=float)
        self.risk_radius = float(env_cfg.get("risk_radius", 1.2))
        self.risk_strength = float(env_cfg.get("risk_strength", 0.9))
        self.risk_noise_prob = float(env_cfg.get("risk_noise_prob", 0.03))
        self.risk_noise_strength = float(env_cfg.get("risk_noise_strength", 0.5))

        self.food_sensor_sigma = float(env_cfg.get("food_sensor_sigma", 2.0))
        self.risk_sensor_sigma = float(env_cfg.get("risk_sensor_sigma", 1.4))

        self.seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.pos = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.patch_level = self.patch_max_food
        self.prev_local_food = 0.0
        self.t = 0

    def reset(self) -> np.ndarray:
        env_cfg = self.cfg["environment"]
        self.pos = np.array(env_cfg.get("start_pos", [7.0, 10.0]), dtype=float)
        self.heading = float(env_cfg.get("start_heading", 0.0))
        self.patch_level = self.patch_max_food
        self.t = 0
        self.prev_local_food = self._local_food_signal()
        return self._observe()

    def step(self, action: np.ndarray) -> StepResult:
        """Action is [turn_delta, speed] in continuous space."""
        turn_delta = float(np.clip(action[0], -1.0, 1.0)) * self.turn_angle
        speed = float(np.clip(action[1], 0.0, 1.0)) * self.step_size

        self.heading += turn_delta
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.pos = self.pos + speed * direction
        self.pos = np.clip(self.pos, 0.0, self.world_size)

        # Closed-loop environmental changes after action.
        was_in_patch = self.in_patch(self.pos)
        if was_in_patch:
            self.patch_level = max(0.0, self.patch_level - self.patch_depletion_rate)
        else:
            self.patch_level = min(self.patch_max_food, self.patch_level + self.patch_regen_rate)

        local_food = self._local_food_signal()
        local_risk = self._local_risk_signal()

        reward = local_food - 0.3 * local_risk
        info = {
            "local_food": local_food,
            "local_risk": local_risk,
            "in_patch": float(was_in_patch),
            "patch_level": self.patch_level,
            "x": self.pos[0],
            "y": self.pos[1],
        }

        self.t += 1
        obs = self._observe()
        return StepResult(observation=obs, reward=reward, info=info)

    def _observe(self) -> np.ndarray:
        local_food = self._local_food_signal()
        local_risk = self._local_risk_signal()
        food_delta = local_food - self.prev_local_food
        self.prev_local_food = local_food
        return np.array([local_food, local_risk, food_delta], dtype=float)

    def _local_food_signal(self) -> float:
        d = np.linalg.norm(self.pos - self.patch_center)
        spatial = np.exp(-(d**2) / (2 * self.food_sensor_sigma**2))
        return float(np.clip(self.patch_level * spatial, 0.0, 1.0))

    def _local_risk_signal(self) -> float:
        d = np.linalg.norm(self.pos - self.risk_center)
        base = self.risk_strength * np.exp(-(d**2) / (2 * self.risk_sensor_sigma**2))
        sparse_spike = self.risk_noise_strength if self.rng.random() < self.risk_noise_prob else 0.0
        return float(np.clip(base + sparse_spike, 0.0, 1.0))

    def in_patch(self, pos: np.ndarray) -> bool:
        return np.linalg.norm(pos - self.patch_center) <= self.patch_radius

    def get_layout(self) -> Dict[str, Tuple[float, float]]:
        return {
            "world_size": self.world_size,
            "patch_center": self.patch_center,
            "patch_radius": self.patch_radius,
            "risk_center": self.risk_center,
            "risk_radius": self.risk_radius,
        }
