from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from medica.core.geometry import GeometricEngine
from medica.core.state import GeometricReference, OrganismState


@dataclass
class CounterfactualResult:
    label: str
    trajectory: list
    final_distortion: float
    final_coherence: float


@dataclass
class OODAResult:
    state: OrganismState
    orient: dict
    results: List[CounterfactualResult]
    receipts: List[dict]
    reference_label: str


class OODALoop:
    def __init__(self, platform, steps: int = 500, dt: float = 0.01):
        self.platform = platform
        self.steps = steps
        self.dt = dt
        self.history: List[OODAResult] = []
        self.receipts: List[dict] = []
        self.reference_history: List[np.ndarray] = []

    def observe(self, values, provenance: str = "observation", source_record: Optional[dict] = None) -> OrganismState:
        return self.platform.witness(values, provenance=provenance, source_record=source_record)

    def orient(self, state: OrganismState) -> dict:
        return GeometricEngine.relate(state, self.platform.reference)

    def decide(self, state: OrganismState, hypotheses: Dict[str, np.ndarray]) -> List[CounterfactualResult]:
        return self.platform.test(state, hypotheses, steps=self.steps, dt=self.dt)

    def act(self, result: CounterfactualResult) -> dict:
        receipt = self.platform.become(result)
        self.receipts.append(receipt)
        return receipt

    def update_reference(self, state: OrganismState, alpha: float = 0.1) -> None:
        new_center = (1.0 - alpha) * self.platform.reference.center + alpha * state.values
        self.platform.reference = GeometricReference(
            center=new_center,
            metric=self.platform.reference.metric,
            label=f"adaptive_ref_t{state.timestamp:.2f}",
        )
        self.reference_history.append(new_center.copy())

    def cycle(self, values, hypotheses, provenance="observation", source_record=None, update_ref=False, alpha=0.1):
        state = self.observe(values, provenance, source_record)
        orient = self.orient(state)
        results = self.decide(state, hypotheses)
        receipts = [self.act(r) for r in results]
        if update_ref:
            self.update_reference(state, alpha)
        out = OODAResult(state, orient, results, receipts, self.platform.reference.label)
        self.history.append(out)
        return out

    def run_stream(self, data_stream, hypotheses, update_ref=False, alpha=0.1):
        outputs = []
        for item in data_stream:
            if isinstance(item, dict):
                values = item.get("values")
                provenance = item.get("provenance", "observation")
                source_record = item.get("source_record")
            else:
                values, provenance = item
                source_record = None
            outputs.append(self.cycle(values, hypotheses, provenance, source_record, update_ref, alpha))
        return outputs
