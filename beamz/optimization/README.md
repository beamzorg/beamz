# BeamZ optimization

Use `beamz.optimization` for topology inverse design with BeamZ simulations,
metre units, and JAX objectives. The same workflow supports full autodiff and
spectral adjoint gradients.

## End-to-end FDTD autodiff

`Simulation.differentiable()` builds a reusable, pure solver whose fields and raw
monitor accumulators can be differentiated with `jax.grad`. Construct it once
outside JAX transformations; material values are runtime inputs and do not change
the compiled-program cache key.

```python
import jax
import jax.numpy as jnp

solver = simulation.differentiable(backend="cuda", checkpoint_interval=32)


def loss(density):
    epsilon = 1.0 + 11.0 * jax.nn.sigmoid(density)
    state = solver(epsilon)
    return -jnp.sum(solver.monitor_flux(state, "output"))


value, gradient = jax.value_and_grad(loss)(density)
```

JAX density maps and pointwise projections can replace `sigmoid` in this
example; apply the existing 2D filters slice by slice for a 3D design. This path
differentiates the discrete electromagnetic updates and the
objective together; it does not need `compute_overlap_gradient` or a manually
constructed adjoint source. A complete runnable example is
[`autodiff_cuda.py`](../../examples/optimization/autodiff_cuda.py).
For a step-by-step waveguide example with gradient checks and before/after field
plots, open [`cuda_autodiff_waveguide_bend.ipynb`](../../examples/notebooks/cuda_autodiff_waveguide_bend.ipynb).

- `backend="jax"` supports the portable 2D TM/TE and 3D solvers.
- `backend="cuda"` uses native streamed forward and reverse-mode Yee/CPML
  kernels for the existing single-GPU, float32 3D physics. Both lossy materials
  and mixed absorbing/PEC faces are supported. The CUDA component must be
  rebuilt for ABI 28.
- `checkpoint_interval=None` selects approximately sqrt(T) steps per chunk;
  a positive integer controls the memory/recomputation tradeoff. Zero retains
  the trajectory for small reference problems. Chunk recomputation works with
  conductive materials and CPML; no time-reversible assumption is needed.
- `solver(epsilon, conductivity=sigma, state=initial_state,
  source_amplitudes=amplitudes)` accepts differentiable material values, initial
  fields/CPML memories, and one real amplitude per original source. Omitted
  arguments preserve compiled materials and use a fresh initial state.
- Uniform-grid cell permittivity is sampled differentiably onto the Yee
  supports, including the transpose of averaging and clipped edge sampling.
  Directly rasterized or nonuniform Yee materials should be supplied as an
  `(epsilon_x, epsilon_y, epsilon_z)` tuple on the compiled Ex/Ey/Ez supports.
  Conductivity accepts cell arrays, scalars, or corresponding Yee arrays. Cell
  conductivity describes physical electric loss and uses the ordinary solver's
  absorber-combination rule. Only the artificial sponge contribution receives
  matched magnetic loss; CPML damping stays in its auxiliary state. A Yee tuple
  directly replaces total electric losses (including any sponge contribution)
  and preserves compiled magnetic losses.
- `solver.monitor_fields(state, name)` returns complex DFT fields on canonical
  axes in `(Ex, Ey, Ez, Hx, Hy, Hz)` order, normalized by the actual window weight
  with H phasors aligned to E time. `solver.monitor_flux` returns signed planar
  flux per frequency and includes the compiled integration weights. These
  helpers preserve the monitor's configured native/physical units; source
  normalization belongs in the objective.

For eligible slab-source/vector-DFT plans, checkpoint chunks use native forward
graphs. Permittivity and initial-state gradients with fixed source amplitudes and
decay coefficients use a fused native reverse step with aliased field cotangents,
material accumulation, and two CPML cotangent banks. Conductivity, source-amplitude,
and clock derivatives retain the general native phase VJP path. Both paths
differentiate every discrete timestep and the monitor objective.

For guided-wave devices, optimize mode power rather than total flux, which can
include radiation. Freeze a projector from an ordinary unnormalized run with a
3D `ModeMonitor`; it uses the same mode basis as detached result analysis:

```python
from beamz.optimization.modes import build_mode_projector

initial_result = simulation.run(backend="cuda")  # normalize_source=None
project = build_mode_projector(initial_result, "output")


def loss(epsilon):
    state = solver(epsilon)
    amplitudes = project(solver.monitor_fields(state, "output"))
    # Shape: (frequency, direction, mode). Directions 0/1 mean +axis/-axis.
    return -(jnp.abs(amplitudes[0, 1, 0]) ** 2) / fixed_reference_mode_power
```

Keep the projector's port materials, geometry, frequencies, and mesh fixed. Its
mode solve and normalization are setup operations; its linear field projection
is differentiable. Gradient agreement verifies the discrete objective, while
binary layouts, longer time windows, and finer meshes require forward checks.

The differentiable path retains fixed geometry, grid/time spacing, boundaries,
source profiles and their launch normalization, and monitor definitions. Source
mode solving and rasterization are setup operations, and source profiles are not
recomputed when a design changes. Keep design regions away from source supports
when their normalization depends on local material values. `Simulation.run()`
continues to provide ordinary decoded analysis results; use the differentiable
state and helpers within a gradient objective.

CUDA provides first-order reverse mode; forward-mode JVPs and higher derivatives
of native FFI calls are not implemented. BF16 absorber state, sharded execution,
and coupled full-tensor constitutive updates are outside the native autodiff
path. For JAX tensor optimization or independent magnetic losses, use
`solver.run_coefficients(state, coefficients, source_amplitudes=None)` with
dynamic `UpdateCoefficients`. Checkpointing stores full chunk-boundary states, so
large field-recorder histories still consume their configured storage.

## Recommended workflow

Given an ordinary `simulation` with a fixed mode or Gaussian-beam source and monitors:

```python
import jax.numpy as jnp
import beamz.optimization as opt

region = opt.TopologyDesignRegion(
    center=(2.5e-6, 2.5e-6, 1.1e-6),
    size=(3e-6, 3e-6, 200e-9),
    pixel_size=simulation.resolution,
    eps_bounds=(1, 4),
    transformations=[opt.FilterProject(radius=150e-9, beta=5)],
)
design = opt.InverseDesign(
    simulation=simulation,
    design_region=region,
    output_monitor_names=["input", "output"],
    gradient_backend="adjoint",  # or "autodiff"
)


def transmission(data):
    incoming = opt.utils.get_amps(data, "input", direction="+", mode_index=0)
    outgoing = opt.utils.get_amps(data, "output", direction="+", mode_index=0)
    return jnp.mean(jnp.abs(outgoing / incoming) ** 2)


optimizer = opt.AdamOptimizer(
    design=design,
    learning_rate=0.1,
)
result = optimizer.run(transmission, steps=10, checkpoint="results/inverse-design.npz")
result.plot_optimization()
final_data = result.simulation_data()
```

By default, parameters use `(y, x)` order and the pattern is extruded through
finite z thickness. Set `parameterization="volumetric"` for independent `(z, y, x)`
voxels. `FilterProject` and `ErosionDilationPenalty` work in both dimensions;
the volumetric conic filter uses a spherical kernel and FFT convolution.

The density grid's `pixel_size` can differ from the FDTD `resolution`. Transform
and project on the density grid, then linearly sample physical density at FDTD
cell centers. The derivative includes the exact interpolation transpose.
Region bounds must align to the FDTD grid and sizes must be integer multiples
of `pixel_size`. Fixed-density masks constrain the density grid; interpolation
can blend their interfaces on a different FDTD grid.

A simulation may use one fixed `ModeSource` or 3D `GaussianBeamSource`. Its
source spectrum normalizes the modal amplitudes; `ModePower` divides by incident
source power to give coupling efficiency. Keep density away from source and
mode-monitor planes (two cells, or a larger margin for a beam's `max_shift`).
Source and mode-monitor planes must also stay outside the absorber along their
normal axis; measuring a mode inside PML understates its coupling efficiency.
For independent polarization experiments, combine objective functions over shared
parameters with `jax.value_and_grad`; see the
[polarization-insensitive grating-coupler notebook](../../examples/notebooks/polarization_insensitive_grating_coupler.ipynb).

`InverseDesign.export_design` thresholds a volumetric density into one extruded
polygon layer per z voxel. It supports `contour` or `pixel_exact` footprints,
including fixed density masks. Large volumes can create many polygons; re-simulate
the export to assess thresholding and rasterization effects.

The design must align to an unsharded uniform grid,
use scalar lossless materials, and stay outside the absorber and fixed mode
planes. The spectral adjoint additionally requires decayed fields and full-run
rectangular DFT monitors and nonperiodic boundaries. Use autodiff for periodic
boundaries; nonzero Bloch wavevectors do not support mode sources or monitors.

`FilterProject` and `ErosionDilationPenalty` provide differentiable smoothing,
projection, and soft geometric penalties. They do not guarantee minimum feature
sizes, connectivity, or a binary final design. Verify the exported geometry with
an independent simulation and resolution checks.

Use `OptimizationSchedule` with `LinearSchedule` or `StepSchedule` for projection,
penalty, and objective-weight continuation. A schedule has its own fixed horizon;
`steps` always means the target total number of updates:

```python
result = optimizer.run(transmission, steps=20, resume="results/inverse-design.npz")
result.final_params
result.history[-1].post_process_val
simulation = result.to_simulation()
```

By default, retain scalar history and the latest parameters, gradient, and optimizer
state. Set `store_full_results=True` to retain parameter/gradient snapshots.
`checkpoint_every=5` saves every five updates and at the final update; callbacks
receive `callback(result)`. Both workflows share the same result and checkpoint
format. The two checkpoint formats previously introduced on this branch are
rejected explicitly; rerun the examples to generate current checkpoints.
Numerical fingerprint 6 also rejects earlier histories: the current 3D mode solver
searches additional eigenmode candidates before applying TE/TM preference,
including for a one-mode monitor request. Earlier one-candidate requests could
select the same physical mode for both polarizations. Frequency-aligned reference
normalization corrections are also included in this fingerprint.

See the [mode-converter notebook](../../examples/notebooks/topology_mode_converter.ipynb)
and [scheduled WDM notebook](../../examples/notebooks/topology_wdm_4channel.ipynb)
for complete inverse-design examples.

## Lower-level control

`TopologyProblem` binds a `TopologySpec` mask to a simulation and a composable
modal objective. Its existing API remains available:

```python
problem = opt.TopologyProblem(
    simulation,
    topology,  # TopologySpec containing the design mask and optimizer settings
    opt.ModePower("output", reference_monitor="input"),
    gradient_backend="autodiff",
)
result = problem.run(steps=20, checkpoint="results/topology.npz")
geometry = problem.export_design(result)
```

For resident multi-GPU checkpointed gradients, pass `sharding=dict(axis="x",
num_devices=2, backend="gpu")` to `TopologyProblem` with `gradient_backend="autodiff"`.
Density and optimizer moments are partitioned; result snapshots are copied to
host memory. Its public `value_and_grad` preserves NumPy gradient returns.
The [distributed support matrix](../simulation/README.md#differentiable-execution)
lists physics and observation restrictions. `InverseDesign` and spectral-adjoint
distribution are separate follow-up interfaces.

`ModePower` and `SoftMinModePower` combine using arithmetic, for example
`transmission - 0.2 * reflection`. `autodiff.py`, `projections.py`, and
`polygonize.py` contain the density transforms, SSP projection, and geometry export
helpers for custom loops.

## Export topology to GDS

Both workflows and their results accept the same geometry controls:

```python
from beamz import Material
from beamz.design import export_gds

pixels = result.export_design(mode="pixel_exact")
contours = result.export_design(mode="contour")  # default
refined = result.export_design(mode="contour", sample_spacing=10e-9)

export_gds(
    refined,
    "device.gds",
    precision=1e-9,  # metres; 1 nm database grid
    layer_map={Material(4.0): (1, 0)},  # select your core material/layer
)
```

- `pixel_exact` unions the full footprints of density cells strictly above the
  threshold. It preserves the thresholded mask, including its staircase edges.
- `contour` reconstructs boundaries by linear interpolation. Edge samples extend
  to the exact region boundary, so contour closure cannot inset connecting ports.
- `sample_spacing` sets the maximum spacing of optional bilinear sampling before
  contour extraction, in metres. It is only valid for contour mode and refines
  the representation without recovering missing physical information. It does
  not change the FDTD mesh or smooth the density with an additional filter.

All density arrays use `(y, x)` order. For `density_to_polygons` and
`density_to_shapely_geometry`, `x0, y0` are **cell-edge origins**, while density
values are at cell centers. `dx, dy` are the physical cell sizes. Cropped arrays
need their true edge origin; retain surrounding samples when external material
context matters. `region_mask` clips a full density field to an exact union of
cells, with the exterior values supplying interpolation context.

To hold a waveguide connection during optimization, pass disjoint boolean
`fixed_solid` and `fixed_void` arrays to `TopologyDesignRegion` or `TopologySpec`.
Region arrays have `params_shape`; spec arrays have `region_mask.shape` and must
lie inside that region. Fixed cells are applied before filtering and after
projection, have zero gradients with respect to their parameters, and retain
exact footprints during geometry export. They do not guarantee connectivity
elsewhere in the design. Masks are copied and participate in checkpoint identity.

`export_gds` preserves holes and uses a private database at the requested
precision, without changing gdsfactory's global grid. `layer_map` selects only
specified materials; omitting it retains automatic material-to-layer assignment.
Explicitly select the core to exclude region-clearing and background polygons.

Run the [export showcase](../../examples/optimization/topology_gds_export.py) to
optimize a small 2D mode converter, compare the previous export with all three
current variants, and reimport and simulate the actual GDS files on two meshes:

```bash
JAX_PLATFORMS=cpu uv run --extra gds python examples/optimization/topology_gds_export.py
```

The script writes GDS masks, boundary zooms, an interpolation comparison, and
measured transmission to the ignored `results/gds-export/` directory. Revalidate
geometry changes and GDS quantization separately from mesh convergence; smoother
contours alone do not establish better performance.
