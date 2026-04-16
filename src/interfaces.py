"""Abstract base classes for environment and agent.

These interfaces define the contract between environment and agent
in the closed sensorimotor loop. All concrete implementations must
conform to these interfaces to be usable with the registry and
run_episode infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class PatchState:
    """State of a single food patch."""
    center: np.ndarray
    radius: float
    level: float
    max_food: float
    odor_level: float = 0.0


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


@dataclass
class AgentStep:
    """Output of a single agent step."""
    observation: np.ndarray
    action: np.ndarray
    h: np.ndarray
    m: np.ndarray


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class BaseEnvironment(ABC):
    """Interface for all environments in the closed sensorimotor loop.

    An environment manages the physical world state and responds to
    agent actions. It does NOT compute observations — that is the
    agent's responsibility via sense().
    """

    @abstractmethod
    def reset(self) -> EnvState:
        """Reset the environment to its initial state."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[EnvState, StepInfo]:
        """Apply an action, update physics, return new state and info."""
        ...

    @abstractmethod
    def get_layout(self) -> Dict:
        """Return static geometry for visualization."""
        ...


class BaseAgent(ABC):
    """Interface for all agents in the closed sensorimotor loop.

    An agent actively senses the environment state through its body,
    maintains internal state, and produces actions. Observation is
    the agent's act, not a gift from the environment.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        ...

    @abstractmethod
    def sense(self, env_state: EnvState) -> np.ndarray:
        """Compute observation from raw environment state through the agent's body."""
        ...

    @abstractmethod
    def step(self, obs: np.ndarray) -> AgentStep:
        """Given observation, update internal state and produce action."""
        ...
