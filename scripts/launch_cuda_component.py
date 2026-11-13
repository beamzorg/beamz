#!/usr/bin/env python3
"""Run a benchmark with an immutable CUDA component without replacing the install."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    component = args.component.resolve()
    if hashlib.sha256(component.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("Candidate CUDA component hash differs")
    if "beamz" in sys.modules or "beamz._cuda" in sys.modules:
        raise RuntimeError("Candidate selection must precede BeamZ imports")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location("beamz._cuda", component)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["beamz._cuda"] = module
    import beamz

    beamz._cuda = module
    sys.argv = [str(args.script), *args.arguments]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
