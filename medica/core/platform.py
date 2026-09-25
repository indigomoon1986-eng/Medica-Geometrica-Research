from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from medica.core.constants import MG_DIMENSIONS, RESEARCH_DISCLAIMER
from medica.core.dynamics import DynamicsModel, MedicaSimulator
from medica.core.geometry import GeometricEngine
from medica.core.ooda import CounterfactualResult, OODALoop
from medica.core.state import GeometricReference, OrganismState
from medica.ingest.mappers import ObservationBundle, to_state_vector
from medica.provenance.receipts import create_receipt


class CounterfactualEngine:
    def __init__(self, simulator: MedicaSimulator, reference: GeometricReference):
        self.simulator = simulator
        self.reference = reference

    def compare(self, initial_state, scenarios, steps=1000, dt=0.01):
        results = []
        for label, perturbation in scenarios.items():
            trajectory = self.simulator.run(initial_state, steps, dt, perturbation)
            final = trajectory[-1]
            results.append(
                CounterfactualResult(
                    label=label,
                    trajectory=trajectory,
                    final_distortion=GeometricEngine.distortion(final, self.reference),
                    final_coherence=GeometricEngine.coherence(final, self.reference),
                )
            )
        return results


class MonteCarloEngine:
    def __init__(self, simulator: MedicaSimulator):
        self.simulator = simulator

    def run(self, initial_state, runs, steps, dt, intervention=None):
        return np.asarray([
            self.simulator.run(initial_state, steps, dt, intervention)[-1].values
            for _ in range(runs)
        ])


class MedicaGeometricaPlatform:
    def __init__(self, model=None, reference=None, seed=None):
        self.model = model or DynamicsModel.default()
        self.reference = reference or GeometricReference.identity()
        self.simulator = MedicaSimulator(self.model, seed)
        self.counterfactuals = CounterfactualEngine(self.simulator, self.reference)
        self.monte_carlo = MonteCarloEngine(self.simulator)
        self.ooda = OODALoop(self)
        self.disclaimer = RESEARCH_DISCLAIMER

    def witness(self, values, provenance="observation", source_record=None):
        return OrganismState(values=np.asarray(values, dtype=float), provenance=provenance, source_record=source_record)

    def witness_observation(self, bundle: ObservationBundle):
        values, uncertainty = to_state_vector(bundle)
        return OrganismState(values=values, provenance=bundle.provenance, uncertainty=uncertainty, source_record=bundle.as_dict())

    def relate(self, state):
        self.counterfactuals.reference = self.reference
        return GeometricEngine.relate(state, self.reference)

    def test(self, state, hypotheses, steps=500, dt=0.01):
        self.counterfactuals.reference = self.reference
        return self.counterfactuals.compare(state, hypotheses, steps, dt)

    def become(self, result):
        return create_receipt({
            "scenario": result.label,
            "coherence": result.final_coherence,
            "distortion": result.final_distortion,
            "status": "SIMULATED_RESEARCH_RESULT",
            "disclaimer": RESEARCH_DISCLAIMER,
            "dimensions": list(MG_DIMENSIONS),
        })

    def default_hypotheses(self):
        n = len(MG_DIMENSIONS)
        return {
            "baseline": np.zeros(n),
            "inflammatory_nudge": np.array([0.0, 0.0, 0.02] + [0.0] * (n - 3)),
            "structural_nudge": np.array([0.0] * 4 + [0.02] + [0.0] * (n - 5)),
        }
