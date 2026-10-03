"""Export the 6-PPW PSR electric material arrays without advancing FDTD.

Run this script from each source checkout with its own installed native raster:
    JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        .venv/bin/python /path/to/audit_psr_material.py /tmp/materials.npz

The working directory selects the checkout to audit. Record its commit, native
raster binary hash, and environment alongside the output.
"""

import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

from scripts.investigate_passive_soi import build_experiment
from tests.differential.passive_soi.experiments import ExperimentOptions

sim, *_ = build_experiment(
    "polarization_splitter_rotator", 6, ExperimentOptions(exact_center=True)
)
program = sim.compile(num_steps=1, backend="jax")
np.savez_compressed(
    sys.argv[1],
    **{k: np.asarray(getattr(program.grid, k)) for k in ["eps_ex", "eps_ey", "eps_ez"]},
)
print("Saved material audit", sys.argv[1])
