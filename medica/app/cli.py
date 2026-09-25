from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medica.core.constants import RESEARCH_DISCLAIMER
from medica.core.platform import MedicaGeometricaPlatform


def _print_cycle(i, out):
    print(f"cycle {i} provenance={out.state.provenance}")
    print(f"  orient distance={out.orient['distance']:.4f} distortion={out.orient['distortion']:.4f} coherence={out.orient['coherence']:.4f}")
    for r in out.results:
        print(f"  decide {r.label} coherence={r.final_coherence:.4f}")
    print(f"  act receipt={out.receipts[0]['digest'][:16]}...")


def cmd_demo(_args):
    print(RESEARCH_DISCLAIMER)
    platform = MedicaGeometricaPlatform(seed=42)
    n = 11
    base = np.array([0.25, -0.10, 0.30, 0.05, -0.20, 0.10, -0.15, 0.08, 0.12, -0.05, 0.20])
    rng = np.random.default_rng(7)
    stream = []
    for t in range(6):
        drift = np.zeros(n); drift[5] = 0.02 * t
        stream.append((base + drift + rng.normal(0, 0.02, n), f"synthetic_t{t}"))
    outs = platform.ooda.run_stream(stream, platform.default_hypotheses(), update_ref=True, alpha=0.15)
    for i, out in enumerate(outs):
        _print_cycle(i, out)


def cmd_ooda(args):
    print(RESEARCH_DISCLAIMER)
    payload = json.loads(Path(args.json).read_text())
    platform = MedicaGeometricaPlatform(seed=payload.get("seed", 42))
    stream = [(np.array(item["values"], dtype=float), item.get("provenance", "observation")) for item in payload["stream"]]
    hypotheses = platform.default_hypotheses()
    outs = platform.ooda.run_stream(stream, hypotheses, update_ref=payload.get("update_ref", True), alpha=payload.get("alpha", 0.1))
    for i, out in enumerate(outs):
        _print_cycle(i, out)


def cmd_app(_args):
    print("Launch with: streamlit run medica/app/streamlit_app.py")


def main():
    parser = argparse.ArgumentParser(description="Medica Geometrica research CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    p_ooda = sub.add_parser("ooda")
    p_ooda.add_argument("--json", required=True)
    p_ooda.set_defaults(func=cmd_ooda)
    sub.add_parser("app").set_defaults(func=cmd_app)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
