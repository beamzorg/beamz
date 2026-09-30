# Experimental FIT backend

`FITSimulation` is an independent uniform-grid finite integration engine next
to BEAMZ's FDTD `Simulation`. It supports 2D TE/TM in the xy plane and 3D,
full-domain PEC walls, positive scalar/cell-grid permittivity and permeability,
electric conductivity, and impressed electric-current sources.

```python
import numpy as np
from beamz import FITCurrentSource, FITSimulation, UniformFITMesh, um

mesh = UniformFITMesh((32, 48), 0.1 * um, polarization="tm")
signal = np.exp(-(((np.arange(240) - 25) / 8) ** 2))
source = FITCurrentSource("Ez", (2.4 * um, 1.6 * um), signal)
sim = FITSimulation(mesh, permittivity=2.25, sources=[source])
results = sim.run(240, record_fields=["Ez"])
dataset = results.to_xarray()
```

Mesh shape counts cells in `(y, x)` or `(z, y, x)` order. Source positions use
`(x, y)` or `(x, y, z)` in metres. Source signals are current densities in A/m²,
sampled at the electric update's half time step and zero after the waveform ends.
The source occupies one dual face and is not an electrical port with fixed
integrated current under refinement. In 2D, quantities use a one-metre extrusion;
reported energy is energy per unit length, numerically J/m.

`run(n)` advances **additional** steps. `step()` advances one step. Field history
is opt-in and stores every step for the specified names; no full field trajectory
is stored by default. `set_fields` accepts native E/H arrays, enforces PEC and
retains unspecified fields. H corresponds to `(n - 1/2) dt` when E is at `n dt`.

The xarray adapter preserves separate native coordinates for each component and
separate E/H sample times. It does not interpolate onto a common grid. Integrated
voltages and magnetic fluxes are in `sim.state`; native E/H are in `sim.fields`.
`sim.energy(conserved=True)` evaluates the source-free, lossless leapfrog
quadratic, which differs from the instantaneous physical staggered energy.

`FITSimulation.from_design(design, resolution=...)` reuses current scalar
rasterization for explicitly dimensioned, origin-zero designs with integer cell
counts. This adapter uses a provisional diagonal material average. To retain
interface geometry, supply an explicit interface to the constructor instead.

## Dielectric interfaces

```python
from beamz import PlanarDielectricInterface

mesh = UniformFITMesh((32, 48), 0.1 * um, polarization="te")
interface = PlanarDielectricInterface(
    normal=(0.8660254, 0.5),
    offset=2.4 * um,
    permittivity_minus=2.0,
    permittivity_plus=12.0,
)
sim = FITSimulation(mesh, interface=interface)
```

The plane divides two isotropic dielectrics in SI coordinates. Host-side exact
square/cube clipping supplies fractions and normals. `FITInterfaceMaterial`
also accepts prepared cell fractions and normals for future geometry adapters.
The effective dielectric tensor uses harmonic averaging normal to the interface
and arithmetic averaging tangentially. A symmetric positive cell-corner assembly
couples staggered electric samples.

Interface mode evolves displacement flux `state.d` and obtains electric voltages
by a matrix-free, Jacobi-preconditioned JAX conjugate-gradient solve each step.
There are no host callbacks in stepping or general sparse matrix allocations.
`electric_operator` defines the actual constitutive law; `m_epsilon` contains
only its scalar reference. PEC constrains the free principal system. The default
relative solve tolerance is `5e-7` in float32 and `1e-11` in float64, with at most
256 iterations; constructor options `interface_solve_tolerance` and
`interface_solve_maxiter` override these. `step`/`run` check the accumulated
maximum residual and raise if it exceeds twenty times the requested tolerance.

The initial interface implementation is lossless: nonzero conductivity and a
separate `permittivity` override are rejected. Curved geometry, conformal PEC,
PML and shared photonic ports remain future work. See the
[interface validation](../../../docs/fit_interface_validation.md) for measured
accuracy and GPU costs; improved accuracy does not imply faster stepping.

The initial backend does not yet support PML, shared FDTD sources/monitors,
modal S-parameters, nonuniform meshes, dispersive media, or material/shape
optimization. `Simulation(backend="fit")` is a future API proposal; use
`FITSimulation` directly today. Float64 requires enabling JAX x64 before
constructing `FITSimulation(..., precision="float64")`.

See the [development plan](../../../docs/fit_backend_plan.md) and
[GPU baseline record](../../../docs/fit_baseline_gpu.json).

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m pytest tests/test_fit.py tests/test_fit_interfaces.py
.venv/bin/python scripts/benchmark_fit_interfaces.py --platform cuda
```
