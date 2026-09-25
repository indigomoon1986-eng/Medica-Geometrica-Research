from medica.core.platform import MedicaGeometricaPlatform
from medica.ingest.mappers import ContextRecord, LabPanel, LesionGeometry, ObservationBundle, Vitals, to_state_vector


def test_witness_relate_shape():
    mg = MedicaGeometricaPlatform(seed=1)
    state = mg.witness([0.1] * 11, provenance="unit")
    rel = mg.relate(state)
    assert set(rel) >= {"distance", "distortion", "coherence"}
    assert rel["distance"] >= 0


def test_ooda_updates_history():
    mg = MedicaGeometricaPlatform(seed=1)
    out = mg.ooda.cycle([0.2] * 11, mg.default_hypotheses(), update_ref=True)
    assert len(mg.ooda.history) == 1
    assert out.receipts[0]["algorithm"] == "sha256"
    assert len(mg.ooda.reference_history) == 1


def test_observation_mapper():
    bundle = ObservationBundle(
        lesion=LesionGeometry(12.0, 8.0, 3.0, 70.0, 2.0, 1.0),
        labs=LabPanel(7.0, 14.0, 5.0, 18.0, 2.0, 1.0, 95.0),
        vitals=Vitals(72.0, 120.0, 36.8, 2.0),
        context=ContextRecord(40.0, True, False, "research"),
    )
    values, uncertainty = to_state_vector(bundle)
    assert values.shape == (11,)
    assert uncertainty.shape == (11,)
