from __future__ import annotations

import numpy as np

from medica.core.state import GeometricReference, OrganismState


class GeometricEngine:
    @staticmethod
    def distance(state: OrganismState, reference: GeometricReference) -> float:
        delta = state.values - reference.center
        d2 = float(delta.T @ reference.metric @ delta)
        return float(np.sqrt(max(d2, 0.0)))

    @staticmethod
    def distortion(state: OrganismState, reference: GeometricReference) -> float:
        delta = state.values - reference.center
        return float(delta.T @ reference.metric @ delta)

    @staticmethod
    def coherence(state: OrganismState, reference: GeometricReference) -> float:
        d = GeometricEngine.distance(state, reference)
        return 1.0 / (1.0 + d)

    @staticmethod
    def relate(state: OrganismState, reference: GeometricReference) -> dict:
        return {
            "distance": GeometricEngine.distance(state, reference),
            "distortion": GeometricEngine.distortion(state, reference),
            "coherence": GeometricEngine.coherence(state, reference),
            "reference": reference.label,
        }
