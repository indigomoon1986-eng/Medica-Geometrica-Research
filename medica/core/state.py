from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np

from medica.core.constants import MG_DIMENSIONS


@dataclass
class OrganismState:
    values: np.ndarray
    timestamp: float = field(default_factory=time.time)
    provenance: str = "unknown"
    uncertainty: Optional[np.ndarray] = None
    source_record: Optional[dict] = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != (len(MG_DIMENSIONS),):
            raise ValueError(f"state must have {len(MG_DIMENSIONS)} dimensions")
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=float)
            if self.uncertainty.shape != self.values.shape:
                raise ValueError("uncertainty must match state shape")

    @property
    def n(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict:
        return {
            "values": {k: float(v) for k, v in zip(MG_DIMENSIONS, self.values)},
            "timestamp": self.timestamp,
            "provenance": self.provenance,
        }

    def copy(self) -> "OrganismState":
        return OrganismState(
            values=self.values.copy(),
            timestamp=self.timestamp,
            provenance=self.provenance,
            uncertainty=None if self.uncertainty is None else self.uncertainty.copy(),
            source_record=None if self.source_record is None else dict(self.source_record),
        )


@dataclass
class GeometricReference:
    center: np.ndarray
    metric: np.ndarray
    label: str = "reference"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.metric = np.asarray(self.metric, dtype=float)
        n = len(self.center)
        if self.metric.shape != (n, n):
            raise ValueError("metric must be NxN")
        self.metric = (self.metric + self.metric.T) / 2.0

    @classmethod
    def identity(cls, label: str = "research_baseline") -> "GeometricReference":
        n = len(MG_DIMENSIONS)
        return cls(center=np.zeros(n), metric=np.eye(n), label=label)
