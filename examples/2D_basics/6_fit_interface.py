"""Lossless planar dielectric FIT cavity on CUDA JAX.

Run: JAX_PLATFORMS=cuda uv run examples/2D_basics/6_fit_interface.py
"""

import jax
import jax.numpy as jnp

from beamz import FITSimulation, PlanarDielectricInterface, UniformFITMesh, um

mesh = UniformFITMesh((32, 48), 0.1 * um, polarization="te")
interface = PlanarDielectricInterface((0.8660254, 0.5), 2.4 * um, 2.0, 12.0)
sim = FITSimulation(mesh, interface=interface)
y, x = mesh.coordinates("Ey")
initial = jnp.sin(jnp.pi * jnp.asarray(x)[None, :] / (48 * mesh.spacing))
initial = jnp.broadcast_to(initial, (len(y), len(x)))
sim.set_fields(Ey=initial)
energy = float(sim.energy(conserved=True))
sim.run(128)
print("JAX backend:", jax.default_backend(), "devices:", jax.devices())
print(
    "Relative conserved-energy drift:", float(sim.energy(conserved=True)) / energy - 1
)
print("Maximum constitutive residual:", float(sim.state.max_solve_residual))
