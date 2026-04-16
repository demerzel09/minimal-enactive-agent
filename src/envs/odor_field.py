"""Odor field environment: many small patches with lingering scent.

Extends ForagingEnv with odor dynamics — depleted patches retain
residual odor that decays slowly, creating a noisy but informative
gradient field. The agent senses this odor field rather than food
directly, so gradient-following alone is unreliable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.env import ForagingEnv
from src.interfaces import EnvState, PatchState, StepInfo


class OdorFieldEnv(ForagingEnv):
    """ForagingEnv extended with odor dynamics.

    Each patch maintains an odor_level that tracks food_level but decays
    more slowly after depletion. This creates residual "ghost" odors at
    depleted patches, making the sensory landscape unreliable.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        env_cfg = config["environment"]
        self.odor_decay_rate = float(env_cfg.get("odor_decay_rate", 0.005))
        self.odor_levels = [mf for mf in self.patch_max_foods]

    def reset(self) -> EnvState:
        state = super().reset()
        self.odor_levels = [mf for mf in self.patch_max_foods]
        # Re-create state with odor info
        return self._make_state()

    def step(self, action: np.ndarray) -> Tuple[EnvState, StepInfo]:
        state, info = super().step(action)

        # Update odor levels: tracks food but decays more slowly
        for i in range(self.n_patches):
            if self.patch_levels[i] >= self.odor_levels[i]:
                self.odor_levels[i] = self.patch_levels[i]
            else:
                self.odor_levels[i] = max(
                    self.patch_levels[i],
                    self.odor_levels[i] - self.odor_decay_rate,
                )

        # Re-create state with odor info
        return self._make_state(), info

    def _make_state(self) -> EnvState:
        risk_spike = self.risk_noise_strength if self.rng.random() < self.risk_noise_prob else 0.0
        patches = [
            PatchState(
                center=self.patch_centers[i].copy(),
                radius=self.patch_radii[i],
                level=self.patch_levels[i],
                max_food=self.patch_max_foods[i],
                odor_level=self.odor_levels[i],
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
