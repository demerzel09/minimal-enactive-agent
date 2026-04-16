"""Minimal recurrent agent with slow internal state h and behavior mode m."""

from __future__ import annotations

from typing import Dict

import numpy as np

# Canonical data structures and ABC live in interfaces.py.
# Re-export AgentStep so existing ``from src.agent import AgentStep`` works.
from src.interfaces import (  # noqa: F401  — re-exports
    BaseAgent,
    AgentStep,
    EnvState,
)


class MinimalEnactiveAgent(BaseAgent):
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
        self.recurrence_scale = float(model_cfg.get("recurrence_scale", 1.0))

        # Sensor parameters — part of the agent's body, not the environment.
        # Preferred location: config["model"]["sensors"]
        # Fallback: config["environment"] (legacy format)
        sensors_cfg = model_cfg.get("sensors", {})
        env_cfg = config.get("environment", {})

        def _sensor(name: str, default: float) -> float:
            return float(sensors_cfg.get(name, env_cfg.get(name, default)))

        self.food_sensor_sigma = _sensor("food_sensor_sigma", 2.0)
        self.odor_sensor_sigma = _sensor("odor_sensor_sigma", 0.0)
        self.odor_noise_strength = _sensor("odor_noise_strength", 0.0)
        self.risk_sensor_sigma = _sensor("risk_sensor_sigma", 1.4)
        self.prev_local_food = 0.0

        seed = int(config.get("seed", 0)) + 17
        self.rng = np.random.default_rng(seed)

        init_mode = str(model_cfg.get("init_mode", "random"))
        if init_mode == "handtuned":
            self._init_handtuned()
        else:
            self._init_random(model_cfg)

        # Ablation C: scale down recurrent self-connections.
        if self.recurrence_scale != 1.0:
            self.W_hh = self.W_hh * self.recurrence_scale
            self.W_uu = self.W_uu * self.recurrence_scale

        # Action decoder from mode to [turn, speed].
        self.W_a = np.array([
            [0.0, 0.8, -0.8],   # turn: exploit straight, explore/avoid swing
            [0.25, 1.0, 0.85],  # speed: exploit moderate, explore/avoid fast
        ])
        self.b_a = np.array([0.0, 0.2])

        # Direct mapping for no-m ablation.
        self.W_direct = self.rng.normal(0.0, 0.5, size=(self.action_dim, self.obs_dim + self.h_dim))
        self.b_direct = self.rng.normal(0.0, 0.1, size=(self.action_dim,))

        self.h = np.zeros(self.h_dim, dtype=float)
        self.u = np.zeros(self.m_dim, dtype=float)
        self.m = np.ones(self.m_dim, dtype=float) / self.m_dim

    def _init_random(self, model_cfg: Dict) -> None:
        scale = float(model_cfg.get("init_scale", 0.6))
        self.W_hh = self.rng.normal(0.0, scale, size=(self.h_dim, self.h_dim))
        self.W_hm = self.rng.normal(0.0, scale, size=(self.h_dim, self.m_dim))
        self.W_hi = self.rng.normal(0.0, scale, size=(self.h_dim, self.obs_dim))
        self.b_h = self.rng.normal(0.0, 0.1, size=(self.h_dim,))

        self.W_uh = self.rng.normal(0.0, scale, size=(self.m_dim, self.h_dim))
        self.W_uu = self.rng.normal(0.0, scale, size=(self.m_dim, self.m_dim))
        self.W_ui = self.rng.normal(0.0, scale, size=(self.m_dim, self.obs_dim))
        self.b_u = self.rng.normal(0.0, 0.1, size=(self.m_dim,))

    def _init_handtuned(self) -> None:
        """Hand-designed weights with interpretable causal structure.

        State interpretation:
          h[0] = depletion pressure (rises when food is scarce)
          h[1] = exploration drift (rises when outside patch)
        Mode interpretation:
          m[0] = exploit (stay in patch)
          m[1] = explore (leave and move)
          m[2] = avoid  (flee from risk)
        Input: i = [local_food, local_risk, food_delta]
        """
        # --- h dynamics ---
        # W_hi: input -> h (food actively suppresses depletion/exploration)
        self.W_hi = np.array([
            [-0.8, +0.3, -0.4],  # food strongly suppresses depletion; risk raises it
            [-0.5, -0.2, -0.3],  # food suppresses exploration drift
        ])
        # W_hh: h self-persistence (weakened to avoid fixed-point trapping)
        self.W_hh = np.array([
            [+0.3, +0.15],  # mild persistence, weak cross-talk
            [+0.2, +0.3],   # depletion mildly drives exploration
        ])
        # W_hm: mode -> h feedback
        self.W_hm = np.array([
            [-0.5, +0.1, +0.2],  # exploiting→depletion↓, explore/avoid→depletion↑
            [-0.4, +0.3, +0.0],  # exploiting→explore_drift↓, exploring→explore_drift↑
        ])
        # Positive bias: depletion and exploration RISE by default (when no food)
        self.b_h = np.array([+0.3, +0.2])

        # --- mode dynamics ---
        # W_uh: h -> mode (stronger influence so h actually drives switching)
        self.W_uh = np.array([
            [-0.8, -0.5],  # depletion→exploit↓, explore_drift→exploit↓
            [+0.6, +0.9],  # depletion→explore↑, explore_drift→explore↑
            [+0.4, -0.2],  # depletion→avoid↑, explore_drift→avoid↓
        ])
        # W_uu: mode self-persistence and mutual inhibition
        self.W_uu = np.array([
            [+0.5, -0.3, -0.2],  # exploit persists, competes with explore/avoid
            [-0.3, +0.5, -0.2],  # explore persists, competes with exploit/avoid
            [-0.2, -0.2, +0.4],  # avoid persists weakly
        ])
        # W_ui: input -> mode direct
        self.W_ui = np.array([
            [+0.7, -0.3, +0.4],  # food→exploit↑, risk→exploit↓, food_rising→exploit↑
            [-0.5, -0.1, -0.4],  # food→explore↓, food_falling→explore↑
            [-0.1, +0.8, -0.0],  # risk→avoid↑
        ])
        self.b_u = np.array([-0.1, +0.0, -0.2])

    def sense(self, env_state: EnvState) -> np.ndarray:
        """Compute observation from environment state through the agent's body.

        This is the agent's active sensing — observation is an act, not a gift.
        Sensor parameters (sigma) belong to the agent, not the environment.

        Two sensing modes:
          - Direct food sensing (odor_sensor_sigma == 0): detects food level via
            food_sensor_sigma. Sharp falloff, no residual after depletion.
          - Odor-based sensing (odor_sensor_sigma > 0): detects odor field via
            odor_sensor_sigma (wider range). Depleted patches still smell.
            Temporal noise simulates turbulent plume.
        """
        use_odor = self.odor_sensor_sigma > 0
        local_food = 0.0
        for patch in env_state.patches:
            d = np.linalg.norm(env_state.pos - patch.center)
            if use_odor:
                sigma = self.odor_sensor_sigma
                source = patch.odor_level
            else:
                sigma = self.food_sensor_sigma
                source = patch.level
            spatial = np.exp(-(d ** 2) / (2 * sigma ** 2))
            local_food += source * spatial

        # Add temporal noise to odor sensing (turbulent plume)
        if use_odor and self.odor_noise_strength > 0:
            local_food += self.rng.normal(0.0, self.odor_noise_strength)

        local_food = float(np.clip(local_food, 0.0, 1.0))

        d_risk = np.linalg.norm(env_state.pos - env_state.risk_center)
        risk_base = env_state.risk_strength * np.exp(-(d_risk ** 2) / (2 * self.risk_sensor_sigma ** 2))
        local_risk = float(np.clip(risk_base + env_state.risk_spike, 0.0, 1.0))

        food_delta = local_food - self.prev_local_food
        self.prev_local_food = local_food

        return np.array([local_food, local_risk, food_delta], dtype=float)

    def reset(self) -> None:
        self.h.fill(0.0)
        self.u.fill(0.0)
        self.m = np.ones(self.m_dim, dtype=float) / self.m_dim
        self.prev_local_food = 0.0

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

        return AgentStep(observation=obs.copy(), action=action, h=self.h.copy(), m=self.m.copy())

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.sum(e)
