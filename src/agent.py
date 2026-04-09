"""Minimal recurrent agent with slow internal state h and behavior mode m."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AgentStep:
    action: np.ndarray
    h: np.ndarray
    m: np.ndarray


class MinimalEnactiveAgent:
    """Small inspectable recurrent rate-based policy.

    Full model:
      h_{t+1} = (1-a_h) h_t + a_h tanh(W_hh h_t + W_hm m_t + W_hi i_t + b_h)
      u_{t+1} = (1-a_m) u_t + a_m tanh(W_uh h_t + W_uu u_t + W_ui i_t + b_u)
      m_t = softmax(u_t)
      a_t = W_a m_t + b_a

    Ablations:
      - no_h: freezes h to zero and removes its downstream effect.
      - no_m: bypasses mode variable and maps [obs, h] to action directly.
    """

    def __init__(self, config: Dict):
        self.cfg = config
        model_cfg = config["model"]
        self.obs_dim = int(model_cfg.get("obs_dim", 3))
        self.h_dim = int(model_cfg.get("h_dim", 2))
        self.m_dim = int(model_cfg.get("m_dim", 3))
        self.action_dim = 2

        self.alpha_h = float(model_cfg.get("alpha_h", 0.12))
        self.alpha_m = float(model_cfg.get("alpha_m", 0.35))

        self.use_h = bool(model_cfg.get("use_h", True))
        self.use_m = bool(model_cfg.get("use_m", True))

        seed = int(config.get("seed", 0)) + 17
        self.rng = np.random.default_rng(seed)

        scale = float(model_cfg.get("init_scale", 0.6))
        self.W_hh = self.rng.normal(0.0, scale, size=(self.h_dim, self.h_dim))
        self.W_hm = self.rng.normal(0.0, scale, size=(self.h_dim, self.m_dim))
        self.W_hi = self.rng.normal(0.0, scale, size=(self.h_dim, self.obs_dim))
        self.b_h = self.rng.normal(0.0, 0.1, size=(self.h_dim,))

        self.W_uh = self.rng.normal(0.0, scale, size=(self.m_dim, self.h_dim))
        self.W_uu = self.rng.normal(0.0, scale, size=(self.m_dim, self.m_dim))
        self.W_ui = self.rng.normal(0.0, scale, size=(self.m_dim, self.obs_dim))
        self.b_u = self.rng.normal(0.0, 0.1, size=(self.m_dim,))

        # Action decoder from mode to [turn, speed].
        self.W_a = np.array([
            [0.0, 0.8, -0.8],   # turn: exploit ~0, explore left/right swings
            [0.25, 1.0, 0.85],  # speed: exploit slower, explore/avoid faster
        ])
        self.b_a = np.array([0.0, 0.2])

        # Direct mapping for no-m ablation.
        self.W_direct = self.rng.normal(0.0, scale, size=(self.action_dim, self.obs_dim + self.h_dim))
        self.b_direct = self.rng.normal(0.0, 0.1, size=(self.action_dim,))

        self.h = np.zeros(self.h_dim, dtype=float)
        self.u = np.zeros(self.m_dim, dtype=float)
        self.m = np.ones(self.m_dim, dtype=float) / self.m_dim

    def reset(self) -> None:
        self.h.fill(0.0)
        self.u.fill(0.0)
        self.m = np.ones(self.m_dim, dtype=float) / self.m_dim

    def step(self, obs: np.ndarray) -> AgentStep:
        obs = np.asarray(obs, dtype=float)

        if self.use_h:
            h_tilde = np.tanh(self.W_hh @ self.h + self.W_hm @ self.m + self.W_hi @ obs + self.b_h)
            self.h = (1.0 - self.alpha_h) * self.h + self.alpha_h * h_tilde
        else:
            self.h.fill(0.0)

        if self.use_m:
            h_for_m = self.h if self.use_h else np.zeros_like(self.h)
            u_tilde = np.tanh(self.W_uh @ h_for_m + self.W_uu @ self.u + self.W_ui @ obs + self.b_u)
            self.u = (1.0 - self.alpha_m) * self.u + self.alpha_m * u_tilde
            self.m = self._softmax(self.u)
            action = self.W_a @ self.m + self.b_a
        else:
            self.m = np.ones(self.m_dim, dtype=float) / self.m_dim
            direct_input = np.concatenate([obs, self.h])
            action = self.W_direct @ direct_input + self.b_direct

        # Clamp into environment action ranges: turn in [-1,1], speed in [0,1]
        action[0] = float(np.clip(np.tanh(action[0]), -1.0, 1.0))
        action[1] = float(np.clip(0.5 * (np.tanh(action[1]) + 1.0), 0.0, 1.0))

        return AgentStep(action=action, h=self.h.copy(), m=self.m.copy())

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.sum(e)
