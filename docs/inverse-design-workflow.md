# Inverse-design workflow

`beamz.optimization` follows Tidy3D's single-simulation inverse-design workflow while using **BeamZ objects, metre units, `(y, x)` arrays, and JAX**. Both gradient backends use the same objective and optimizer interface.

Inverse-design regions, transforms, optimizers, and results are part of BeamZ's
core optimization module. Import them directly from `beamz.optimization`.
Both optimization workflows use one result and checkpoint format. The previous
branch-only checkpoint formats are rejected explicitly; the notebooks generate
fresh checkpoints when rerun.

The [1-to-3 splitter notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/tidy3d_inverse_design_splitter.ipynb) ports the single-simulation part of Flexcompute's [inverse-design plugin example](https://www.flexcompute.com/tidy3d/examples/notebooks/InverseDesign/). It records geometry, grid, solver, and normalization differences. It is a workflow port at coarser resolution, not a cross-solver numerical reproduction.

## Define a region, objective, and optimizer

Given an ordinary BeamZ `simulation` with fixed mode sources and monitors:

```python
import jax
import jax.numpy as jnp
import beamz.optimization as bi

region = bi.TopologyDesignRegion(
    size=(2e-6, 1.6e-6, float("inf")),
    center=(3e-6, 2e-6, 0),
    eps_bounds=(1, 4),
    pixel_size=simulation.resolution,
    transformations=[bi.FilterProject(radius=200e-9, beta=2)],
    initialization_spec=bi.UniformInitializationSpec(value=0.5),
)
design = bi.InverseDesign(
    simulation=simulation,
    design_region=region,
    output_monitor_names=["input", "output"],
    gradient_backend="adjoint",  # or "autodiff"
)

def post_process_fn(data):
    output = data["output"].amps.sel(direction="+", mode_index=0).values
    incident = data["input"].amps.sel(direction="+", mode_index=0).values
    return jnp.mean(jnp.abs(output / incident) ** 2)

optimizer = bi.AdamOptimizer(
    design=design, learning_rate=0.04,
)
result = optimizer.run(post_process_fn, steps=20, checkpoint="results/inverse-design.npz")
result.plot_optimization()
params = result.final_params
final_simulation = result.to_simulation()
final_data = result.simulation_data()
```

For your own optimization loop, use the same objective:

```python
objective_fn = design.make_objective_fn(post_process_fn)
value, gradient = jax.value_and_grad(objective_fn)(region.initial_parameters)
```

You can also differentiate a function that calls `design.to_simulation_data(params)` directly. `design.to_simulation(params)` returns an ordinary BeamZ simulation for inspection and independent verification; differentiating arbitrary changes to that simulation's Python geometry is not supported.

For field visualization, add a `FieldMonitor` to a copy of that ordinary simulation and run it independently. In a 2D xy simulation, a z-normal monitor with positive x/y extents samples the full plane at material-cell centers, interpolating the staggered Yee components. The splitter notebook uses `Simulation.plot()` for layout, `Simulation.plot_eps()` for initial/final material grids, and `SimulationResults.plot_field()` for electric-field squared magnitude and amplitude. Two-dimensional `Simulation.plot()` uses the same material colors, shaded PML, source arrows, and monitor markers as 3D geometry layouts. Both display micrometre coordinates, including axis limits and manually added annotations; simulation inputs remain in metres. Native layout geometry does not compile the FDTD grid, while material-grid simulations show their actual raster. The only direct Matplotlib array image follows the reference's random-parameter illustration. These are forward diagnostics; the inverse-design callback interface still exposes modal data only.

## Vector 3D with an extruded planar design

A finite positive `TopologyDesignRegion.size[2]` defines the extrusion thickness
when the base simulation is 3D. Parameters remain `(y, x)`; the same transformed
pattern is broadcast across the region's z cells. All three electric and three
magnetic components propagate in the 3D solver. Both gradient backends use the
same region, modal callback, optimizer, and checkpoint interface.

```python
region = bi.TopologyDesignRegion(
    center=(2.5e-6, 2.5e-6, 1.1e-6),
    size=(3e-6, 3e-6, 200e-9),
    pixel_size=50e-9,
    eps_bounds=(1, 4),
    transformations=[bi.FilterProject(radius=150e-9, beta=5)],
)
design = bi.InverseDesign(simulation=simulation_3d, design_region=region,
                          gradient_backend="adjoint")
# The ordinary simulation includes the full 3D continuous material grid.
sim = design.initial_simulation
sim.plot_eps(z=1.1e-6)
sim.plot_eps(x=2.5e-6)
# After optimization, export ordinary polygons with their extrusion thickness.
geometry = design.export_design(result.params[-1], threshold=0.5)
verified = simulation_3d.updated_copy(design=geometry, resolution=40e-9).run()
```

The [3D bend notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/tidy3d_topology_bend_3d.ipynb)
recreates Flexcompute's [topology waveguide bend](https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd18TopologyBend/).
It includes a coarse full-vector gradient comparison, 50 nm optimization, native
layout/field plots, restart validation, a spectrum, duration verification, and
independent polygon rerasterization. Its mesh, pulse, iteration count, and
projection schedule differ from the reference and are stated in the notebook.

In the saved execution, ten updates improve source-normalized target-mode power
from **0.004988 to 0.734932**; the 800 fs verification agrees to six decimal
places. The coarse adjoint/autodiff gradient-vector difference is **0.000251%**.
The same binary polygon export gives **0.714983 at 50 nm** and **0.745034 at
40 nm**. These meshes resolve the 200 nm thickness with four and five cells,
respectively; the example does not establish mesh convergence.

Three-dimensional modal callbacks use the coupled forward/backward overlap
system from ordinary BeamZ analysis, including its public physical `+`/`−` direction labels. Source symmetry is checked at the fixed source cross-section, so distant topology updates do not change the launch when the ordinary simulation is rebuilt. The material pullback interpolates cell
permittivities onto each electric Yee support and sums the sensitivity through
the extrusion. Fixed mode planes may be normal to x, y, or z. The same uniform,
unsharded, scalar lossless material restrictions apply. The source and its mode
basis remain fixed and outside the design region.

Full autodiff retains checkpointed vector fields and CPML states; its memory cost
can be substantial. The spectral backend stores design-region DFTs for Ex, Ey,
and Ez and checks forward and adjoint decay. Its recorded DFT count is independent
of timestep count, not of design volume or number of objective frequencies.

General volumetric filter/penalty workflows and thickness/sidewall derivatives
are not part of this extruded-pattern interface. Changing thickness changes the
fixed geometry; it is not a differentiable parameter in this milestone.

## Data and objective semantics

`data[name].amps` supports `.values`/`.data`, `.shape`, `.dims`, `.coords`, and exact `.sel(direction=..., f=..., mode_index=...)`. A scalar selection removes that dimension; a list preserves it and its requested order. Dimensions are `(direction, f, mode_index)`. This is a trace-preserving modal subset, **not the full xarray API**. Selection does not interpolate frequencies. Use `jax.numpy` on `.values` inside objectives; converting traced values to ordinary NumPy, Python `float`, or `.item()` breaks differentiation.

Built-in `ModePower`, `SoftMinModePower`, and weighted combinations are callable
with the same data and can be passed directly to `optimizer.run`. For example,
`optimizer.run(bi.ModePower("output", reference_monitor="input"), steps=20)`
optimizes normalized transmission. Built-in objectives without a reference
normalize by nominal source power; a custom callback chooses its own normalization.

Complex amplitudes follow ordinary BeamZ modal analysis. Divide by measured incoming amplitudes when a transmission ratio is intended. Power and phase/interference objectives are supported if their final result is a real scalar. `bi.utils.get_amps(data, monitor_name, **selectors)` and `bi.utils.sum_abs_squared(array)` provide shorthand. Include all monitors read by the callback in `output_monitor_names`; the default selects all mode monitors.

`make_objective_fn(..., maximize=True)` returns `post_process - penalty`; `maximize=False` returns `-post_process - penalty`. Adam ascends this internal score. Recorded `history[i].objective_before` is `post_process - penalty` for maximization and `post_process + penalty` for minimization. Histories describe the state **before** each update; `params[-1]` is the final updated state. Evaluate `simulation_data()` to inspect that final state.

## Regions and transformations

Region coordinates use the simulation's public frame, including centered domains. Bounds must align to the uniform grid, and `pixel_size` must equal `simulation.resolution`. The rectangular mask and Yee-material mapping are handled internally. Existing exclusions around CPML, domain edges, sources, and mode planes still apply. Parameters use `(y, x)` order instead of Tidy3D's three-dimensional parameter layout.

`transformations` are applied in order with `(density, pixel_size)`. `FilterProject` rounds its conic radius up to cells, uses reflection padding by default, then applies tanh projection and clips numerical roundoff to `[0, 1]`. `penalties` receive transformed density and pixel size and return a weighted real scalar. `ErosionDilationPenalty(length_scale=..., weight=...)` measures the RMS difference between filtered closing and opening. Its norm has tiny smoothing near zero for finite derivatives. These are soft controls, not hard feature-size guarantees.

The reference splitter mirrors its illustrative noise about physical y and then initializes optimization uniformly; it does not enforce symmetry during updates. The BeamZ notebook matches this and also mirrors its gradient-check perturbation. With BeamZ `(y, x)` arrays, that reflection is `np.flip(params, axis=0)`.

Uniform, seeded random, and custom-array initialization are available. Specifications support `.updated_copy(...)`. An omitted random seed is resolved once when the specification is created; use an explicit seed to reproduce it in another process. A seed reproduces random initialization; `CustomInitializationSpec(params=...)` copies the array. Custom transforms must preserve shape and return finite density in `[0, 1]`.

## Numerical symmetry

For uniform-grid 2D TM, symmetric geometry and input parameters should produce symmetric Ez fields and symmetric gradients for a reflection-invariant objective. No mode, field, gradient, or optimizer update needs to be projected onto a symmetric subspace. The splitter notebook checks this independently of its mirrored random initialization.

The investigation found four placement issues:

- Topology cells were sampled from one side of staggered Yee locations. The material map now averages adjacent cells and blends partially covered supports with the original fixed Yee raster. Ordinary material simulations use the same map.
- The TM source taper omitted the final transverse node of its cell interval, shifting its center by half a cell. The launch includes both bounding nodes.
- The uniform TM monitor basis used cell-centered material against node-sampled tangential fields. Its material profile is now interpolated onto the transverse nodes before cropping.
- Adjoint DFT storage used the previous nearest-cell mask and missed interpolation boundary samples. It now includes every affected Yee sample, and the material VJP returns their contributions to design cells.

In the recorded 1.668 ps uniform-design audit, relative mirror mismatch fell from **22.3493% to 0.0000049%** for complex Ez and from **70.5046% to 0.0002969%** for the objective gradient. The standalone fundamental mode on the centered symmetric waveguide was already symmetric to roughly 1e-13 relative error. The main errors occurred after mode solving. The notebook includes reproducible checks and native BeamZ plots; `results/topology_examples/tidy3d_splitter_symmetry.json` records that 50 nm run. The current notebook uses 25 nm and writes separate artifacts under `results/topology_examples/tidy3d_splitter_25nm/`. Its optimization now uses 2.502 ps because an updated design failed the adjoint decay guard at the reference duration.

The fresh run also exposed residual excitation from the internal adjoint pulse: extending time alone did not pass the decay check. A Gaussian envelope with width three periods and an eight-width leading margin reduced the perturbed-design adjoint terminal/peak ratio from 1.19e-4 to 2.27e-5 at 2.502 ps, while relative gradient-vector difference from autodiff stayed at about 0.1122%. The measured pulse DFT normalizes its source strength; the default 1e-4 decay guard remains unchanged. The 25 nm rerun additionally differences this pulse in time to suppress DC excitation. It also revealed accumulated float32 clock drift in the forward solver: observation time now comes from the integer step count, anchored to the initial or continued state. This aligns monitor DFT phases with the spectral adjoint and prevents drift over long runs. Both changes retain the unchanged decay guard. Independent full-autodiff and finite-difference checks remain required.

After all eleven unconstrained updates in the archived 50 nm notebook, parameter mirror mismatch is **0.000729%**, complex-field mismatch is **0.001181%**, and outer modal powers differ by **0.000475%** relative to their mean. All twelve code cells in that 50 nm run executed successfully and all seven saved figures use the intended plotting utilities. The regression suite passes 69 tests; Ruff, Pyright, and the strict documentation build also pass.

This does not establish exact symmetry for all mode problems. An additional deliberately off-center aperture test showed finite mode-domain boundary error: reflected modal electric profiles differed by about 7.3e-4 with six padding cells, decreasing to 3.3e-7 with twelve. Include enough of the evanescent tails and check aperture convergence. The placement corrections above target uniform-grid 2D TM; no corresponding 3D or nonuniform-grid claim is made. Arithmetic topology interpolation preserves reflection and positive material bounds but is not a mesh-convergence guarantee.

The current **25 nm** run has 26,880 parameters, complete-gradient disagreement of **0.0020%**, and a post-processing score of **0.306 → 0.949** after 11 updates. Its final field mirror mismatch is **0.000813%**. This is a fresh optimization at finer resolution, not a convergence comparison of one fixed design. Saved outputs live under `results/topology_examples/tidy3d_splitter_25nm/`.

## Continue and inspect a run

```python
# Reach 30 total updates, preserving the optimizer state and schedule position.
result = optimizer.run(post_process_fn, steps=30, resume=result)

# A checkpoint path can be used directly instead of loading it separately.
result = optimizer.run(
    post_process_fn, steps=40, resume="results/inverse-design.npz",
    checkpoint="results/inverse-design.npz", checkpoint_every=5,
)

params = result.final_params
final_simulation = result.to_simulation()
final_data = result.simulation_data()
```

`result.history` is a tuple of `OptimizationStep` records. Each record identifies
its one-based step, projection beta, score before the update, post-processing
value, penalty, gradient norm, parameter change, and elapsed time. `objective`
is the score after the update when the arbitrary-mask workflow explicitly
computes it; otherwise it is `None`. `final_params` includes the last update.

The default keeps scalar records and the latest parameters, gradient, and
optimizer state. `store_full_results=True` retains parameter/gradient snapshots;
`to_simulation(index)` and `simulation_data(index)` inspect retained states.
After enabling snapshots on a resumed compact run, snapshots form a contiguous
suffix of the run; their settings retain the absolute schedule position.
Only the latest optimizer state is stored. Callbacks receive `callback(result)`.
Checkpoints are saved before callbacks at the configured interval and always
at the final update.

Checkpoints contain numeric arrays and finite JSON metadata, never executable
objects. Their fingerprint covers the numerical mapping revision, simulation,
region, backend, optimizer settings, schedule, and objective code/captured
values. Snapshot storage does not change the physical problem identity.
For callable objects other than built-in modal objectives, supply `objective_id`;
changing such an objective requires changing its identifier. Changed backends,
learning rates, objectives, or schedules require a fresh run, optionally with
`CustomInitializationSpec(params=old_result.final_params)`.

Both workflows save schema 2. The previous two branch-only schema-1 formats are
incompatible with this API and are rejected; rerun the examples rather than
reusing their earlier histories. Exact continuation assumes an unchanged
numerical environment.

## Current compatibility boundary

This is a supported subset of the **inverse-design workflow**, not a drop-in `tidy3d` package. It keeps BeamZ objects and metre units. It supports one simulation, 2D or extruded 3D scalar lossless topology, differentiable modal callbacks, transformations/penalties, Adam, and result continuation. `InverseDesign` defaults to `adjoint`; the lower-level `TopologyProblem` still defaults to `autodiff`.

The [backend documentation](inverse-design.md#two-gradient-backends) describes adjoint decay requirements, monitor restrictions, and host orchestration. No general volumetric filter workflow, shape, lossy/dispersive material, field/flux/far-field objective, `InverseDesignMulti`, expression-metric, or general traced `Simulation.run` support is implied. Additional backend capabilities will be exposed through this interface.

## Scheduled optimization

Use fixed-horizon schedules for projection sharpness, a multiplier on the design
penalties, and keyword arguments to the modal objective:

```python
from beamz.optimization import (
    AdamOptimizer, LinearSchedule, OptimizationSchedule, StepSchedule,
)

schedule = OptimizationSchedule(
    beta=LinearSchedule(start=1, stop=50, num_steps=50),
    penalty_weight=1.0,
    objective_kwargs={"leak_weight": StepSchedule(before=0, after=1, switch_step=17)},
)
# post_process_fn(data, leak_weight) returns the scalar modal objective.
optimizer = AdamOptimizer(design, learning_rate=0.1, schedule=schedule)
partial = optimizer.run(post_process_fn, steps=25)
result = optimizer.run(post_process_fn, steps=50, resume=partial)
```

Steps are zero-based. Linear schedules clamp at their final value after their own
`num_steps` horizon. Shortening the optimizer run to pause it does not change that
horizon, and extending the run does not restart either schedule. Adam moments are
preserved through the projection ramp and objective-weight switch.

`beta` overrides the region's `FilterProject` transformations. Set
`TopologyDesignRegion(..., penalty_input="parameters")` when penalties should act
on the raw parameters, as in the low-level Flexcompute WDM example; the default
`"density"` continues to penalize the transformed density.

`result.schedule_history` records the settings used before each update.
`result.settings(index)` gives the settings for a saved parameter state.
`result.to_simulation(index)`, `simulation_data(index)`, and `export_design()` use
the corresponding beta, including the last applied beta for the final updated
parameters. Calling `design.to_simulation(params)` directly uses the region's
configured beta; supply `beta=...` to override it explicitly.

Checkpoints include the schedule identity and each update’s projection beta.
Changing an endpoint, switch step, keyword weight, or penalty-input convention
rejects continuation. Both gradient backends use this same interface.

## Broadband adjoint source grouping

Set `InverseDesign(..., adjoint_source_grouping="auto")` to try a smaller
broadband spatial basis. The default `"frequency"` retains one solve per active
frequency. Full autodiff does not use source grouping.

The optional strategy factors the actual complex source vectors across
frequencies. Groups with any frequency farther than 10% from their mean retain independent
frequency solves so that pulse-spectrum normalization remains well conditioned. Every normalized nonzero source must be reconstructed with relative
2-norm error at most 1e-7; otherwise the backend retains more basis vectors or
falls back to independent-frequency solves. Each basis solve measures its pulse
DFT at every required frequency, and its frequency-dependent complex weights are
applied to the final field overlap. Mode profiles are not assumed to be constant
across the band. This follows Tidy3D's broadband grouping principle, but is a
BeamZ spatial-basis implementation rather than Tidy3D's port-grouping algorithm.

Diagnostics expose `source_grouping`, `adjoint_solves`,
`source_reconstruction_error` when a basis is used, and `adjoint_timesteps`.
Adjoints may finish before the configured maximum duration only after the pulse
has ended and the full dual-state terminal/peak ratio has stayed below
`min(1e-7, decay_tolerance * 1e-3)` for 512 consecutive steps. The normal forward
and adjoint decay guards still apply. This stopping rule is exclusive to the
spectral adjoint; full autodiff differentiates the full configured duration.

Fewer solves do not necessarily mean lower runtime: each broadband basis solve
accumulates more DFTs. The [four-channel WDM notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/tidy3d_wdm_4channel.ipynb)
checks gradients, uses coarse-grid timings to select a backend, and reports its
cold and warmed cost on the training grid.
