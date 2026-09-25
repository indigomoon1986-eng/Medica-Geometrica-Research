from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from medica.core.constants import MG_DIMENSIONS


@dataclass
class LesionGeometry:
    longest_mm: Optional[float] = None
    shortest_mm: Optional[float] = None
    border_irregularity_1_to_5: Optional[float] = None
    surface_area_mm2: Optional[float] = None
    color_variation_1_to_5: Optional[float] = None
    elevation_mm: Optional[float] = None


@dataclass
class LabPanel:
    wbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    crp: Optional[float] = None
    esr: Optional[float] = None
    tsh: Optional[float] = None
    creatinine: Optional[float] = None
    glucose: Optional[float] = None


@dataclass
class Vitals:
    heart_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    temperature_c: Optional[float] = None
    pain_0_to_10: Optional[float] = None


@dataclass
class ContextRecord:
    days_since_onset: Optional[float] = None
    chemical_exposure: Optional[bool] = None
    family_skin_cancer: Optional[bool] = None
    notes: str = ""


@dataclass
class ObservationBundle:
    lesion: LesionGeometry = field(default_factory=LesionGeometry)
    labs: LabPanel = field(default_factory=LabPanel)
    vitals: Vitals = field(default_factory=Vitals)
    context: ContextRecord = field(default_factory=ContextRecord)
    provenance: str = "observation"

    def as_dict(self) -> dict:
        return {
            "lesion": self.lesion.__dict__,
            "labs": self.labs.__dict__,
            "vitals": self.vitals.__dict__,
            "context": self.context.__dict__,
            "provenance": self.provenance,
        }


def _norm(value: Optional[float], center: float, scale: float) -> float:
    if value is None:
        return 0.0
    return float((value - center) / scale)


def to_state_vector(bundle: ObservationBundle):
    lesion, labs, vitals, ctx = bundle.lesion, bundle.labs, bundle.vitals, bundle.context
    values = np.zeros(len(MG_DIMENSIONS))
    uncertainty = np.ones(len(MG_DIMENSIONS))
    if labs.hemoglobin is not None:
        values[0] = _norm(labs.hemoglobin, 14.0, 2.0); uncertainty[0] = 0.2
    if ctx.days_since_onset is not None:
        values[1] = -_norm(ctx.days_since_onset, 30.0, 30.0); uncertainty[1] = 0.3
    if labs.crp is not None:
        values[2] = _norm(labs.crp, 3.0, 10.0); uncertainty[2] = 0.2
    elif labs.esr is not None:
        values[2] = _norm(labs.esr, 15.0, 20.0); uncertainty[2] = 0.3
    if labs.glucose is not None:
        values[3] = _norm(labs.glucose, 95.0, 25.0); uncertainty[3] = 0.25
    if lesion.elevation_mm is not None:
        values[4] = _norm(lesion.elevation_mm, 1.0, 2.0); uncertainty[4] = 0.2
    if lesion.longest_mm is not None:
        irreg = lesion.border_irregularity_1_to_5 or 1.0
        values[5] = _norm(lesion.longest_mm * irreg, 10.0, 15.0); uncertainty[5] = 0.15
    pain = vitals.pain_0_to_10 or 0.0
    irreg = lesion.border_irregularity_1_to_5 or 0.0
    values[6] = -_norm(pain + irreg, 5.0, 5.0); uncertainty[6] = 0.4
    if vitals.heart_rate is not None:
        values[7] = _norm(vitals.heart_rate, 72.0, 15.0); uncertainty[7] = 0.2
    if labs.wbc is not None:
        values[8] = _norm(labs.wbc, 7.0, 3.0); uncertainty[8] = 0.2
    uncertainty[9] = 0.8
    d = 0.0
    if ctx.chemical_exposure:
        d += 0.4
    if ctx.family_skin_cancer:
        d += 0.2
    values[10] = d
    uncertainty[10] = 0.35 if ctx.chemical_exposure is not None else 0.8
    return values, uncertainty
