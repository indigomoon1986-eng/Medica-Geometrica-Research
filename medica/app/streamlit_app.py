from medica.core.constants import DIMENSION_LABELS, MG_DIMENSIONS, RESEARCH_DISCLAIMER
from medica.core.platform import MedicaGeometricaPlatform
from medica.ingest.mappers import ContextRecord, LabPanel, LesionGeometry, ObservationBundle, Vitals
from medica.pharmacopeia.library import list_catalog

import streamlit as st

st.set_page_config(page_title="Medica Geometrica Research", layout="wide")
st.title("Medica Geometrica Research")
st.warning(RESEARCH_DISCLAIMER)

if "platform" not in st.session_state:
    st.session_state.platform = MedicaGeometricaPlatform(seed=42)
platform = st.session_state.platform

left, right = st.columns(2)
with left:
    st.subheader("Lesion geometry")
    longest = st.number_input("Longest diameter (mm)", 0.0, 100.0, 12.0)
    shortest = st.number_input("Shortest diameter (mm)", 0.0, 100.0, 8.0)
    border = st.slider("Border irregularity (1-5)", 1, 5, 3)
    area = st.number_input("Surface area (mm2)", 0.0, 5000.0, 75.0)
    color_var = st.slider("Color variation (1-5)", 1, 5, 2)
    elevation = st.number_input("Elevation / depth (mm)", 0.0, 20.0, 1.5)
    st.subheader("Labs")
    wbc = st.number_input("WBC", 0.0, 30.0, 7.2)
    hb = st.number_input("Hemoglobin", 0.0, 20.0, 14.1)
    crp = st.number_input("CRP", 0.0, 200.0, 4.0)
    esr = st.number_input("ESR", 0.0, 150.0, 18.0)
    tsh = st.number_input("TSH", 0.0, 20.0, 2.1)
with right:
    st.subheader("Vitals / context")
    hr = st.number_input("Heart rate", 30.0, 180.0, 76.0)
    sbp = st.number_input("Systolic BP", 70.0, 220.0, 122.0)
    temp = st.number_input("Temperature C", 34.0, 42.0, 36.8)
    pain = st.slider("Pain 0-10", 0, 10, 2)
    days = st.number_input("Days since onset", 0.0, 3650.0, 45.0)
    chem = st.checkbox("Possible chemical / acid exposure")
    fam = st.checkbox("Family history of skin cancer")
    update_ref = st.checkbox("Adapt reference after this observation", value=True)
    run = st.button("Run OODA cycle")

if run:
    bundle = ObservationBundle(
        lesion=LesionGeometry(longest, shortest, float(border), area, float(color_var), elevation),
        labs=LabPanel(wbc, hb, crp, esr, tsh, None, None),
        vitals=Vitals(hr, sbp, temp, float(pain)),
        context=ContextRecord(days, chem, fam, ""),
        provenance="research_console",
    )
    state = platform.witness_observation(bundle)
    out = platform.ooda.cycle(state.values, platform.default_hypotheses(), provenance=bundle.provenance, source_record=bundle.as_dict(), update_ref=update_ref, alpha=0.12)
    st.subheader("ORIENT")
    st.json(out.orient)
    st.subheader("State")
    st.json({k: float(v) for k, v in zip(MG_DIMENSIONS, out.state.values)})
    st.subheader("DECIDE / ACT")
    for r, receipt in zip(out.results, out.receipts):
        st.write(f"{r.label}: coherence={r.final_coherence:.4f} distortion={r.final_distortion:.4f}")
        st.caption(f"receipt {receipt['digest']}")

with st.expander("Dimension labels"):
    st.json(DIMENSION_LABELS)
with st.expander("Pharmacopeia catalog (labels only)"):
    st.json(list_catalog())
