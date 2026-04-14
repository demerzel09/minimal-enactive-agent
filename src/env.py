"""Minimal 2D closed-loop foraging environment for the first PoC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EnvState:
    """Raw environment state exposed to the agent's sensing apparatus."""
    pos: np.ndarray
    heading: float
    patch_center: np.ndarray
    patch_level: float
    risk_center: np.ndarray
    risk_strength: float
    risk_spike: float
    in_patch: bool
    t: int


@dataclass
class StepInfo:
    """Logging information returned after each environment step."""
    info: Dict[str, float]


class ForagingEnv:
    """2D environment with one depleting food patch and sparse risk.

    The environment manages physics (movement, depletion) and exposes raw state.
    Observation is computed by the agent's own sensing apparatus, not by the environment.
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

        self.seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.pos = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.patch_level = self.patch_max_food
        self.t = 0

    def reset(self) -> EnvState:
        env_cfg = self.cfg["environment"]
        self.pos = np.array(env_cfg.get("start_pos", [7.0, 10.0]), dtype=float)
        self.heading = float(env_cfg.get("start_heading", 0.0))
        self.patch_level = self.patch_max_food
        self.t = 0
        return self._make_state()

    def step(self, action: np.ndarray) -> Tuple[EnvState, StepInfo]:
        """Apply action, update physics, return new state."""
        turn_delta = float(np.clip(action[0], -1.0, 1.0)) * self.turn_angle
        speed = float(np.clip(action[1], 0.0, 1.0)) * self.step_size

        self.heading += turn_delta
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.pos = self.pos + speed * direction
        self.pos = np.clip(self.pos, 0.0, self.world_size)

        # Closed-loop environmental changes after action.
        is_in_patch = self.in_patch(self.pos)
        if is_in_patch:
            self.patch_level = max(0.0, self.patch_level - self.patch_depletion_rate)
        else:
            self.patch_level = min(self.patch_max_food, self.patch_level + self.patch_regen_rate)

        self.t += 1

        state = self._make_state()
        info = StepInfo(info={
            "in_patch": float(is_in_patch),
            "patch_level": self.patch_level,
            "x": self.pos[0],
            "y": self.pos[1],
        })
        return state, info

    def _make_state(self) -> EnvState:
        risk_spike = self.risk_noise_strength if self.rng.random() < self.risk_noise_prob else 0.0
        return EnvState(
            pos=self.pos.copy(),
            heading=self.heading,
            patch_center=self.patch_center,
            patch_level=self.patch_level,
            risk_center=self.risk_center,
            risk_strength=self.risk_strength,
            risk_spike=risk_spike,
            in_patch=self.in_patch(self.pos),
            t=self.t,
        )

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
