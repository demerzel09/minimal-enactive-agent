"""Minimal 2D closed-loop foraging environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PatchState:
    """State of a single food patch."""
    center: np.ndarray
    radius: float
    level: float
    max_food: float


@dataclass
class EnvState:
    """Raw environment state exposed to the agent's sensing apparatus."""
    pos: np.ndarray
    heading: float
    patches: List[PatchState]
    risk_center: np.ndarray
    risk_strength: float
    risk_spike: float
    in_any_patch: bool
    t: int


@dataclass
class StepInfo:
    """Logging information returned after each environment step."""
    info: Dict[str, float]


class ForagingEnv:
    """2D environment with depleting food patches and sparse risk.

    Supports one or multiple food patches. The environment manages physics
    (movement, depletion) and exposes raw state. Observation is computed
    by the agent's own sensing apparatus, not by the environment.

    Config formats (backward compatible):
      Single patch (legacy):
        environment:
          patch_center: [10, 10]
          patch_radius: 2.8
          ...

      Multiple patches:
        environment:
          patches:
            - center: [7, 10]
              radius: 2.5
            - center: [16, 12]
              radius: 2.0
          patch_max_food: 1.0       # shared defaults
          patch_depletion_rate: 0.035
          patch_regen_rate: 0.004
    """

    def __init__(self, config: Dict):
        self.cfg = config
        env_cfg = config["environment"]

        self.world_size = float(env_cfg.get("world_size", 20.0))
        self.step_size = float(env_cfg.get("step_size", 0.7))
        self.turn_angle = float(env_cfg.get("turn_angle", 0.4))

        # Parse patches (backward compatible with single-patch config)
        self.default_max_food = float(env_cfg.get("patch_max_food", 1.0))
        self.default_depletion = float(env_cfg.get("patch_depletion_rate", 0.035))
        self.default_regen = float(env_cfg.get("patch_regen_rate", 0.004))
        self.default_radius = float(env_cfg.get("patch_radius", 2.8))

        if "patches" in env_cfg:
            self.patch_configs = env_cfg["patches"]
        else:
            self.patch_configs = [{
                "center": env_cfg.get("patch_center", [10.0, 10.0]),
                "radius": env_cfg.get("patch_radius", 2.8),
            }]

        self.n_patches = len(self.patch_configs)
        self.patch_centers = [np.array(p["center"], dtype=float) for p in self.patch_configs]
        self.patch_radii = [float(p.get("radius", self.default_radius)) for p in self.patch_configs]
        self.patch_max_foods = [float(p.get("max_food", self.default_max_food)) for p in self.patch_configs]
        self.patch_depletions = [float(p.get("depletion_rate", self.default_depletion)) for p in self.patch_configs]
        self.patch_regens = [float(p.get("regen_rate", self.default_regen)) for p in self.patch_configs]
        self.patch_levels = [mf for mf in self.patch_max_foods]

        # Risk
        self.risk_center = np.array(env_cfg.get("risk_center", [14.5, 6.0]), dtype=float)
        self.risk_radius = float(env_cfg.get("risk_radius", 1.2))
        self.risk_strength = float(env_cfg.get("risk_strength", 0.9))
        self.risk_noise_prob = float(env_cfg.get("risk_noise_prob", 0.03))
        self.risk_noise_strength = float(env_cfg.get("risk_noise_strength", 0.5))

        self.seed = int(config.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.pos = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.t = 0

    def reset(self) -> EnvState:
        env_cfg = self.cfg["environment"]
        self.pos = np.array(env_cfg.get("start_pos", [7.0, 10.0]), dtype=float)
        self.heading = float(env_cfg.get("start_heading", 0.0))
        self.patch_levels = [mf for mf in self.patch_max_foods]
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
        in_any = False
        for i in range(self.n_patches):
            if self._in_patch_i(self.pos, i):
                self.patch_levels[i] = max(0.0, self.patch_levels[i] - self.patch_depletions[i])
                in_any = True
            else:
                self.patch_levels[i] = min(self.patch_max_foods[i], self.patch_levels[i] + self.patch_regens[i])

        self.t += 1

        state = self._make_state()
        info = StepInfo(info={
            "in_patch": float(in_any),
            "patch_level": self.patch_levels[0],  # primary patch for backward compat
            "x": self.pos[0],
            "y": self.pos[1],
        })
        # Add per-patch levels for multi-patch logging
        for i in range(self.n_patches):
            info.info[f"patch_{i}_level"] = self.patch_levels[i]
        return state, info

    def _make_state(self) -> EnvState:
        risk_spike = self.risk_noise_strength if self.rng.random() < self.risk_noise_prob else 0.0
        patches = [
            PatchState(
                center=self.patch_centers[i].copy(),
                radius=self.patch_radii[i],
                level=self.patch_levels[i],
                max_food=self.patch_max_foods[i],
            )
            for i in range(self.n_patches)
        ]
        return EnvState(
            pos=self.pos.copy(),
            heading=self.heading,
            patches=patches,
            risk_center=self.risk_center,
            risk_strength=self.risk_strength,
            risk_spike=risk_spike,
            in_any_patch=any(self._in_patch_i(self.pos, i) for i in range(self.n_patches)),
            t=self.t,
        )

    def _in_patch_i(self, pos: np.ndarray, i: int) -> bool:
        return np.linalg.norm(pos - self.patch_centers[i]) <= self.patch_radii[i]

    def in_patch(self, pos: np.ndarray) -> bool:
        return any(self._in_patch_i(pos, i) for i in range(self.n_patches))

    def get_layout(self) -> Dict:
        layout = {
            "world_size": self.world_size,
            "risk_center": self.risk_center,
            "risk_radius": self.risk_radius,
            "n_patches": self.n_patches,
        }
        # Backward compat: single patch uses old keys
        if self.n_patches == 1:
            layout["patch_center"] = self.patch_centers[0]
            layout["patch_radius"] = self.patch_radii[0]
        # Always include list form
        layout["patch_centers"] = [c.tolist() for c in self.patch_centers]
        layout["patch_radii"] = self.patch_radii
        return layout
