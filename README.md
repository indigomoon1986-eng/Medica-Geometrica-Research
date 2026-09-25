# Medica Geometrica Research

Research simulation software for modeling organism state as geometry.

**Not a diagnostic or treatment system.**
**Does not authorize medical action.**
**A human clinician or researcher remains the only decision-maker.**

Canonical loop:

```
WITNESS -> RELATE -> TEST -> BECOME
```

wired as a recursive OODA cycle:

```
OBSERVE -> ORIENT -> DECIDE -> ACT -> update reference -> next observation
```

## Proposition

Health is not merely a number. Health can be modeled as a changing geometry.

That proposition is a **research hypothesis**, not a clinical claim.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[app]"
```

## CLI

```bash
PYTHONPATH=. python -m medica.app.cli demo
PYTHONPATH=. python -m medica.app.cli ooda --json examples/lesion_stream.json
streamlit run medica/app/streamlit_app.py
```

See `docs/RESEARCH_BOUNDARY.md`.
