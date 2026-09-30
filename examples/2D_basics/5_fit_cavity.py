"""Uniform-grid FIT cavity with a native electric-current source.

Run: JAX_PLATFORMS=cuda uv run examples/2D_basics/5_fit_cavity.py
"""

import numpy as np

from beamz import FITCurrentSource, FITSimulation, UniformFITMesh, um

mesh = UniformFITMesh(shape=(32, 48), spacing=0.1 * um, polarization="tm")
signal = np.exp(-(((np.arange(240) - 25) / 8) ** 2))
source = FITCurrentSource("Ez", position=(2.4 * um, 1.6 * um), signal=signal)
sim = FITSimulation(mesh, permittivity=2.25, sources=[source])
results = sim.run(240, record_fields=["Ez"])
dataset = results.to_xarray()
print(dataset)
print("Final physical energy per unit length:", float(sim.energy()), "J/m")
