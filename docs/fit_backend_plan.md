# FIT backend development plan

Add a Finite Integration Technique (FIT) backend alongside BEAMZ's existing
FDTD engine. The first measurable objective is better accuracy at photonic
material interfaces, followed by graded meshes and conformal geometry. A later
electronics extension targets full-wave interconnects, transmission lines,
packages, and microwave structures.

The existing FDTD engine remains the default and a numerical reference. A FIT
formulation on a uniform orthogonal grid with matching material approximation
and leapfrog stepping recovers Yee FDTD. Accuracy improvements must therefore
be attributed to geometry and constitutive discretization, rather than the
FIT name. Compare current BEAMZ rasterization with both improved FDTD interface
treatment and the proposed FIT formulation.

## Scope

| Area | Initial scope | Later extension |
| --- | --- | --- |
| Solver | Explicit time-domain FIT, 2D TE/TM and 3D | Frequency-domain and implicit solves |
| Mesh | Uniform, then graded orthogonal Cartesian coordinates | Subgridding, curvilinear and unstructured meshes |
| Interfaces | Geometry-aware dielectric constitutive operators | General anisotropic and dispersive interfaces |
| Geometry | Interfaces represented within Cartesian cells | Conformal PEC cut cells and general meshes |
| Materials | Positive nondispersive scalar epsilon/mu and conductivity | Debye, Lorentz, Drude and surface impedance |
| Optimization | Material coefficients on fixed meshes | Shape derivatives and mesh-coordinate optimization |
| Electronics | PEC structures and voltage/current ports | Lumped elements and circuit coupling |

The first nonuniform mesh uses independent coordinate arrays along each axis.
Refining an axis extends refinement across the domain; this is not localized
subgridding. Smaller cells still constrain the explicit timestep. Memory
savings do not necessarily imply runtime savings.

BEAMZ already exposes permeability, conductivity and PEC. Electronics support
requires additional material physics, electrical ports, conductor treatment,
and validation, rather than a different set of Maxwell equations.

## Architecture

Separate three responsibilities:

1. Mesh topology: oriented edges/faces and curl, divergence and gradient
   incidence operators. These contain connectivity and orientation, not lengths
   or material coefficients.
2. Geometry and constitutive laws: primal/dual lengths, areas and volumes;
   material fractions and interface normals; electric and magnetic operators.
3. Execution: integrated state, time stepping, boundaries, source application,
   monitoring, results and differentiation.

Electric voltages `e` live on primal edges; magnetic fluxes `b` on primal faces.
Electric displacement fluxes are `d = M_epsilon e`; magnetic dual-edge voltages
are `h = M_reluctivity b`. With consistent orientations:

```text
b[n+1/2] = b[n-1/2] - dt C e[n]
d[n+1]   = d[n] + dt (C.T h[n+1/2] - j[n+1/2])
```

On orthogonal grids the initial constitutive operators are diagonal. Apply
incidence operators with JAX array stencils rather than assembled general sparse
matrices. Perform meshing, clipping and geometry intersections outside the
compiled step loop. Keep static topology separate from differentiable runtime
material coefficients in the later optimization implementation.

The eventual shared API should select a backend before meshing and allocation:

```python
sim = Simulation(
    design=design,
    backend="fit",
    mesh_spec=mesh_spec,
    sources=sources,
    monitors=monitors,
    boundaries=boundaries,
)
```

This is a future API proposal, not the milestone 1 interface. Initially expose
`FITSimulation` independently so the core can be validated before altering
the existing engine's source and monitor machinery.

Integration touches `simulation/core.py`, `design/meshing.py`, Yee coordinates,
source/monitor compilers, PML construction, modal projection and
`data/xarray.py`. Several currently assume scalar `resolution`. Existing source
and monitor specifications may be reused, but their compiled numerical
representations must understand native coordinates and integration weights.

## Milestones and acceptance gates

### Milestone 0 Establish the baseline

Choose and record the starting source revision without overwriting existing
dependency changes. Record installed versions, device, precision, test selection,
dependency hashes and baseline failures. Use the existing marked fast CI gate
and small repeatable PEC workloads in 2D/3D. Separate first execution, which
includes compilation, from warm execution and synchronize device work before
timing. Performance records are descriptive, not CI thresholds.

Initial record: source revision `65b1adaf432edb67156b4dd98406483d020fdc24`
(Release v0.3.1). At the start, local `main` was 374 commits behind the locally
recorded `origin/main`; no fetch, branch change or upstream integration was
performed. `pyproject.toml` and `uv.lock` already contained CUDA JAX dependency
changes and are preserved. The existing environment had JAX 0.9.0 and no pytest;
pytest and pytest-cov were installed into `.venv` without editing dependencies.

The pre-FIT fast gate passed 177 tests with 383 deselected. Initial validation
used CPU because the sandbox could not access CUDA. Outside the sandbox this
machine exposes an NVIDIA RTX 5070 Laptop GPU; the repeated CUDA baseline also
passed all 177 existing tests. Environment and timing observations are in the
historical [CPU record](fit_baseline.json) and [GPU record](fit_baseline_gpu.json).
The full integration/characterization suite and coverage remain outside this
initial baseline.

Reproduce with the installed environment:

```bash
.venv/bin/python scripts/record_fit_baseline.py --platform cuda
```

### Milestone 1 Uniform grid FIT

Implement complete native Yee cell complexes, metric-free incidence operators,
diagonal constitutive laws, explicit stepping, full-domain PEC walls and simple
electric-current sources. Support 2D TE/TM and 3D. Preserve integrated SI units
internally and expose native E/H values and coordinates in results.

Require matching Yee results under matching assumptions, curl/divergence and
gradient/curl identities, exact operator transpose pairing, stable source-free
evolution and appropriate energy/charge behavior. Validate a cavity against its
discrete dispersion relation and continuum frequency. Distinguish the physical
staggered energy from the conserved leapfrog quadratic.

Initial implementation is in `beamz/simulation/fit/`, with public exports,
`tests/test_fit.py`, and `examples/2D_basics/5_fit_cavity.py`. It supports scalar
or cell-grid positive epsilon/mu, electric conductivity, optional field history,
and xarray export. `from_design` reuses scalar rasterization for explicitly
dimensioned origin-zero geometry; domain lengths must be integer cell multiples.
Cell-to-native arithmetic averaging is a provisional diagonal approximation,
not an interface accuracy improvement.

Known reference limitation: the inspected FDTD H-update coefficient functions
use `MU_0` without relative permeability in their lossless source coefficient.
Cross-engine equivalence consequently targets `mu_r = 1`; FIT's non-unit
permeability is validated independently against the cavity dispersion relation.
The FDTD behavior is not changed as part of this work.

The initial backend uses its own impressed-current specification and has no
shared FDTD source/monitor/PML compilation, modal S-parameters, or automatic
backend selection. Field-initialization differentiation is checked as a JAX
compatibility smoke test; material/shape optimization belongs to milestone 6.

Initial verification on CPU: all 29 FIT acceptance tests passed. The final
marked fast CI gate passed 206 tests with 383 deselected, including the 177
existing tests. Ruff lint, formatting checks for changed code and `git diff
--check` passed; the example executed successfully. A separate heterogeneous
3D float64 check measured relative conserved-energy drift of approximately
`5e-15` after 100 steps and exercised the 3D Design/xarray adapter. These results
validate the initial uniform-grid backend, not the later interface/conformal
accuracy objectives.

### Milestone 2 Dielectric interface accuracy

Retain intersections, fractions and normals. Compare diagonal averaging with
interface-oriented coupled constitutive laws. Tangential E and normal D obey
different continuity conditions; scalar averaging is generally insufficient.
Check symmetry, positivity, consistency and stability of the assembled operator,
including coupling between staggered samples. A local effective tensor alone
does not prove global stability.

Require convergence improvement for translated and rotated dielectric interfaces
against analytical Fresnel/slab solutions and converged references. Compare
current antialiasing and improved FDTD smoothing at similar cost. If constitutive
application requires a solve, benchmark it immediately and define convergence
and preconditioning requirements before adopting it.

Initial planar implementation is available through `PlanarDielectricInterface`
or prepared `FITInterfaceMaterial` geometry. Exact host-side cell clipping
supplies fractions/normals. Harmonic normal and arithmetic tangential averaging
form a positive tensor; paired cell-corner gather/scatter assembly gives a
symmetric positive global permittivity on free PEC degrees. The displacement
update uses a matrix-free Jacobi-preconditioned JAX CG solve for electric
voltages, with checked residuals. Nonzero conductivity is currently rejected
in coupled mode. The diagonal backend remains available for comparisons.

The initial CUDA convergence study covers three resolutions, three interface
angles and two translations per angle, using an independent interface-fitted
FEM reference checked at two finer resolutions. At 28 cells per axis, mean
error over eighteen cavity frequencies was 0.1549% for the tensor operator,
0.3427% for scalar fractions and 0.2732% for staircasing. Small warm GPU
workloads cost approximately eight times more with the coupled solve.
See [validation methods and results](fit_interface_validation.md) and the
[raw convergence/timing report](fit_interface_convergence.json).

This completes the initial interface prototype, not every milestone 2 gate.
Open gates include Fresnel/slab validation, comparison with newer upstream
FDTD smoothing, larger/high-contrast workloads and improved preconditioning.
Do not infer universal accuracy order or curved-interface support from this
planar study. Milestone 3 integration remains separate.

The newer upstream CUDA engine combines JAX orchestration with native CUDA
kernels through JAX FFI; this older checkout does not contain that extension.
The FIT prototype uses CUDA-enabled JAX stencils and matrix-free CG, avoiding
generic sparse matrices. Native CUDA availability does not resolve generic
JAX sparse performance concerns automatically.

CUDA verification: all 54 FIT core/interface tests and all 177 existing baseline
tests passed. The interface example ran on `CudaDevice(id=0)` and measured
approximately `8.3e-8` relative conserved-energy drift over 128 steps. Ruff
lint/format checks and whitespace validation passed. These checks do not replace
the remaining acceptance gates above.

Next work is the [performance exploration](fit_performance_exploration.md).
The measured slowdown makes equivalent-accuracy cost an acceptance gate before
milestone 3 integration; retain this prototype for reproducible comparisons.

### Milestone 3 Complete photonics integration

Adapt CPML, mode injection, flux/DFT accumulation, modal projection and common
results. Preserve E/H temporal staggering, port orientation, native spatial
reference planes and source power normalization. Require straight-waveguide
power balance, phase, low spurious reflection and cross-engine agreement.

### Milestone 4 Nonuniform Cartesian meshes

Introduce explicit primal/dual coordinate arrays, graded spacing and stability
selection. Update all physical sampling and quadrature, including source
placement, PML profiles and modal solver integration. Verify the mode solver's
nonuniform-grid capabilities before selecting an adapter or replacement.

Require convergence on graded grids, low artificial reflection at transitions,
correct boundary absorption and port normalization. Keep localized refinement,
local time stepping and arbitrary mesh topology deferred.

### Milestone 5 Conformal photonics

Extend interface-aware operators to curved and sloped dielectric boundaries,
extruded polygons and sidewalls. Validate bends, rings and crossings. Compare
error against wall time and memory, not just cell count. Prioritize effective
index, complex S-parameters, bend transmission and resonance frequency before
high-Q loss extraction.

Conformal PEC cut cells are a later decision: tiny effective cells can impose
severe explicit timestep restrictions. Select stabilization/cell treatment based
on a published formulation and verify its stability rather than clipping metric
values heuristically.

### Milestone 6 Differentiable FIT

Differentiate continuous material coefficients on fixed topology first. Validate
gradients by finite differences; implement checkpointing or a discrete adjoint
for long runs. Demonstrate a small optimization and record peak memory. For
linear solves, use an implicit derivative only with validated primal/adjoint
convergence. Polygon clipping, interface crossings and remeshing introduce
nonsmooth changes; shape differentiation needs a separate formulation.

### Milestone 7 Electronics

Add physically normalized voltage/current ports and transmission-line extraction;
then conductor treatments and passive Debye/Lorentz/Drude models. Validate
transmission-line impedance/propagation, cavity modes, S-parameters and conductor
loss before adding lumped-element or circuit coupling. Distinguish dielectric
conformal interfaces from conductor surface/cut-cell models.

## Validation policy

Use analytical cases, discrete invariants and convergence studies independently.
Sweep several resolutions, interface angles and subcell translations. Preserve
staircasing as an explicit comparison choice; do not silently switch geometry
approximations. Report mesh, constitutive law, boundary model, precision and
timestep with each result.

Do not evaluate interface accuracy solely through pointwise fields at a
discontinuity or corner. Use physical observables and converged reference
solutions. Target improved error versus total cost while preserving stability
and existing FDTD behavior; do not claim universal accuracy order from a few
successful geometries.

## References

- [Clemens and Weiland, Discrete Electromagnetism with the Finite Integration Technique](https://www.jpier.org/issues/volume.html?paper=00080103)
- [Farjadpour et al., Improving accuracy by sub-pixel smoothing in FDTD](https://arodriguez.princeton.edu/publications/improving-accuracy-sub-pixel-smoothing-finite-difference-time-domain)
- [Cylindrical and Nonconformal Material Interfaces in the Finite Integration Technique](https://d-nb.info/1156179939/34)
- [Conformal FDTD methods and timestep restrictions](https://www.sciencedirect.com/science/article/pii/S0021999107000642)
- [JAX matrix-free linear solves with implicit gradients](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)
