from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from medica.core.constants import MG_DIMENSIONS
from medica.core.state import OrganismState


@dataclass
class DynamicsModel:
    coupling: np.ndarray
    decay: np.ndarray
    noise_scale: float = 0.0

    def __post_init__(self):
        self.coupling = np.asarray(self.coupling, dtype=float)
        self.decay = np.asarray(self.decay, dtype=float)
        n = len(self.decay)
        if self.coupling.shape != (n, n):
            raise ValueError("coupling matrix must be NxN")

    @classmethod
    def default(cls, noise_scale: float = 0.002) -> "DynamicsModel":
        n = len(MG_DIMENSIONS)
        coupling = np.zeros((n, n))
        for i in range(n - 1):
            coupling[i, i + 1] = 0.04
            coupling[i + 1, i] = 0.02
        return cls(coupling=coupling, decay=np.full(n, 0.05), noise_scale=noise_scale)

    def derivative(self, x: np.ndarray, intervention: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = self.coupling @ np.tanh(x) - self.decay * x
        if intervention is not None:
            dx = dx + np.asarray(intervention, dtype=float)
        return dx


class MedicaSimulator:
    def __init__(self, model: DynamicsModel, rng_seed: Optional[int] = None):
        self.model = model
        self.rng = np.random.default_rng(rng_seed)

    def step(self, state: OrganismState, dt: float = 0.01, intervention: Optional[np.ndarray] = None) -> OrganismState:
        dx = self.model.derivative(state.values, intervention)
        noise = self.rng.normal(0.0, self.model.noise_scale, size=state.values.shape)
        return OrganismState(
            values=state.values + dx * dt + noise * np.sqrt(dt),
            timestamp=state.timestamp + dt,
            provenance="simulation",
        )

    def run(self, initial_state: OrganismState, steps: int, dt: float = 0.01, intervention: Optional[np.ndarray] = None):
        trajectory = [initial_state.copy()]
        current = initial_state.copy()
        for _ in range(steps):
            current = self.step(current, dt, intervention)
            trajectory.append(current)
        return trajectory
