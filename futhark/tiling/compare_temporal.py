#!/usr/bin/env python3
"""Compare Futhark's temporal schedule with plain stepping, bit for bit.

Runs each case with BEAMZ_FUTHARK_TEMPORAL=0 and =2 and compares every state
leaf (fields, CPML memory, DFT accumulators). Run from the repository root with
XLA_PYTHON_CLIENT_PREALLOCATE=false.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests" / "hardware"), str(ROOT / "scripts")]

import beamz as bz  # noqa: E402
from beamz.simulation.execute import build_scan, initial_program_state  # noqa: E402
from test_futhark_backend import MIXED, PML, _simulation  # noqa: E402
from beamz.devices.boundaries import PEC  # noqa: E402


def run(simulation, steps, temporal):
    os.environ["BEAMZ_FUTHARK_TEMPORAL"] = str(temporal)
    program = simulation.compile(num_steps=steps, backend="futhark")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    return jax.block_until_ready(build_scan(program, donate_state=False)(state, program.coefficients))


def compare(name, simulation, steps):
    a = run(simulation, steps, 0)
    b = run(simulation, steps, 2)
    leaves_a, tree = jax.tree_util.tree_flatten_with_path(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    bad = []
    for (path, x), y in zip(leaves_a, leaves_b):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            bad.append(f"{jax.tree_util.keystr(path)} shape {x.shape} vs {y.shape}")
            continue
        if x.dtype.kind == "f":
            diff = x.view(np.uint32 if x.dtype == np.float32 else np.uint64) != y.view(
                np.uint32 if y.dtype == np.float32 else np.uint64
            )
        else:
            diff = x != y
        if diff.any():
            scale = float(np.abs(x).max()) if x.dtype.kind == "f" else 1
            err = float(np.abs(x - y).max()) if x.dtype.kind == "f" else 0
            bad.append(f"{jax.tree_util.keystr(path)}: {int(diff.sum())}/{diff.size} differ, max {err:.3g} (scale {scale:.3g})")
    print(f"{name} steps={steps}: {'IDENTICAL' if not bad else 'DIFFERENT'}", flush=True)
    for line in bad[:12]:
        print("   ", line)
    return not bad


def realistic(shape, steps, source="mode"):
    from benchmark_cuda_realistic import build_simulation

    spec = SimpleNamespace(shape=shape, steps=steps, pml=12, monitors=2, frequencies=3,
                           material="binary", source=source, monitor_type="field")
    return build_simulation(spec)


def main():
    ok = True
    cases = sys.argv[1:] or ["pec", "mixed", "pml", "realistic"]
    for steps in (90, 45):
        if "pec" in cases:
            ok &= compare("pec-lossy", _simulation([PEC(edges="all")], monitors=False, lossy=True, steps=steps), steps)
        if "mixed" in cases:
            ok &= compare("pec-cpml-dft", _simulation(list(MIXED), monitors=True, lossy=False, steps=steps), steps)
        if "pml" in cases:
            ok &= compare("cpml-lossy-dft", _simulation([PML], monitors=True, lossy=True, steps=steps), steps)
    if "realistic" in cases:
        ok &= compare("realistic 40x56x72", realistic((40, 56, 72), 64), 64)
    # Large enough for core tiles; a single Gaussian source keeps plain
    # stepping deterministic (overlapping mode-source entries are not).
    if "gaussian" in cases:
        for steps in (64, 33):
            ok &= compare("gaussian 64x96x224", realistic((64, 96, 224), steps, "gaussian"), steps)
        # One step: no pass, only the final plain step on the tiled state (a
        # simulation needs at least two time entries).
        ok &= compare("gaussian 64x96x224", realistic((64, 96, 224), 2, "gaussian"), 1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
