"""Research pharmacopeia catalog.

This is a label library for simulation hypotheses.
It is not a formulary, not dosing advice, and not a prescription engine.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CatalogEntry:
    code: str
    name: str
    research_class: str
    notes: str


PHARMACOPEIA: List[CatalogEntry] = [
    CatalogEntry("RX-NONE", "no_perturbation", "control", "Baseline simulation branch."),
    CatalogEntry("RX-OBS", "observation_only", "control", "Watchful waiting as a modeled branch."),
    CatalogEntry("RX-TOP-AB", "topical_antimicrobial_label", "topical", "Label only. No dose."),
    CatalogEntry("RX-TOP-ST", "topical_steroid_label", "topical", "Label only. No dose."),
    CatalogEntry("RX-WOUND", "wound_care_protocol_label", "procedure", "Dressing / wound-care branch label."),
    CatalogEntry("RX-BX", "biopsy_completed", "procedure", "Marks that histopathology entered the record."),
]


def list_catalog() -> List[dict]:
    return [e.__dict__ for e in PHARMACOPEIA]


def get_entry(code: str) -> CatalogEntry:
    for entry in PHARMACOPEIA:
        if entry.code == code:
            return entry
    raise KeyError(code)
