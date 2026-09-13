# Inverse-design workflow

`beamz.plugins.invdes` follows Tidy3D's single-simulation inverse-design workflow while using **BeamZ objects, metre units, `(y, x)` arrays, and JAX**. Both gradient backends use the same objective and optimizer interface.

The [1-to-3 splitter notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/tidy3d_inverse_design_splitter.ipynb) ports the single-simulation part of Flexcompute's [inverse-design plugin example](https://www.flexcompute.com/tidy3d/examples/notebooks/InverseDesign/). It records geometry, grid, solver, and normalization differences. It is a workflow port at coarser resolution, not a cross-solver numerical reproduction.

## Define a region, objective, and optimizer

Given an ordinary BeamZ `simulation` with fixed mode sources and monitors:

```python
import jax
import jax.numpy as jnp
import beamz.plugins.invdes as bi

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
    design=design, learning_rate=0.04, num_steps=20,
    results_cache_fname="results/inverse-design.npz",
)
result = optimizer.run(post_process_fn=post_process_fn)
result.plot_optimization()
params = result.get_last("params")
final_simulation = result.sim_last
final_data = result.sim_data_last()
```

For your own optimization loop, use the same objective:

```python
objective_fn = design.make_objective_fn(post_process_fn)
value, gradient = jax.value_and_grad(objective_fn)(region.initial_parameters)
```

You can also differentiate a function that calls `design.to_simulation_data(params)` directly. `design.to_simulation(params)` returns an ordinary BeamZ simulation for inspection and independent verification; differentiating arbitrary changes to that simulation's Python geometry is not supported.

For field visualization, add a `FieldMonitor` to a copy of that ordinary simulation and run it independently. In a 2D xy simulation, a z-normal monitor with positive x/y extents samples the full plane at material-cell centers, interpolating the staggered Yee components. The splitter notebook uses `Simulation.plot()` for layout, `Simulation.plot_eps()` for initial/final material grids, and `SimulationResults.plot_field()` for electric-field squared magnitude and amplitude. Two-dimensional `Simulation.plot()` uses the same material colors, shaded PML, source arrows, and monitor markers as 3D geometry layouts. Both display micrometre coordinates, including axis limits and manually added annotations; simulation inputs remain in metres. Native layout geometry does not compile the FDTD grid, while material-grid simulations show their actual raster. The only direct Matplotlib array image follows the reference's random-parameter illustration. These are forward diagnostics; the inverse-design callback interface still exposes modal data only.

## Data and objective semantics

`data[name].amps` supports `.values`/`.data`, `.shape`, `.dims`, `.coords`, and exact `.sel(direction=..., f=..., mode_index=...)`. A scalar selection removes that dimension; a list preserves it and its requested order. Dimensions are `(direction, f, mode_index)`. This is a trace-preserving modal subset, **not the full xarray API**. Selection does not interpolate frequencies. Use `jax.numpy` on `.values` inside objectives; converting traced values to ordinary NumPy, Python `float`, or `.item()` breaks differentiation.

Complex amplitudes follow ordinary BeamZ modal analysis. Divide by measured incoming amplitudes when a transmission ratio is intended. Power and phase/interference objectives are supported if their final result is a real scalar. `bi.utils.get_amps(data, monitor_name, **selectors)` and `bi.utils.sum_abs_squared(array)` provide shorthand. Include all monitors read by the callback in `output_monitor_names`; the default selects all mode monitors.

`make_objective_fn(..., maximize=True)` returns `post_process - penalty`; `maximize=False` returns `-post_process - penalty`. Adam ascends this internal score. Reported `objective_fn_val` is `post_process - penalty` for maximization and `post_process + penalty` for minimization. Histories describe the state **before** each update; `params[-1]` is the final updated state. Evaluate `sim_data_last()` to inspect that final state.

## Regions and transformations

Region coordinates use the simulation's public frame, including centered domains. Bounds must align to the uniform grid, and `pixel_size` must equal `simulation.resolution`. The rectangular mask and Yee-material mapping are handled internally. Existing exclusions around CPML, domain edges, sources, and mode planes still apply. Parameters use `(y, x)` order instead of Tidy3D's three-dimensional parameter layout.

`transformations` are applied in order with `(density, pixel_size)`. `FilterProject` rounds its conic radius up to cells, uses reflection padding by default, then applies tanh projection and clips numerical roundoff to `[0, 1]`. `penalties` receive transformed density and pixel size and return a weighted real scalar. `ErosionDilationPenalty(length_scale=..., weight=...)` measures the RMS difference between filtered closing and opening. Its norm has tiny smoothing near zero for finite derivatives. These are soft controls, not hard feature-size guarantees.

The reference splitter mirrors its illustrative noise about physical y and then initializes optimization uniformly; it does not enforce symmetry during updates. The BeamZ notebook matches this and also mirrors its gradient-check perturbation. With BeamZ `(y, x)` arrays, that reflection is `np.flip(params, axis=0)`.

Uniform, seeded random, and custom-array initialization are available. Specifications support `.updated_copy(...)`. A seed reproduces random initialization; `CustomInitializationSpec(params=...)` copies the array. Custom transforms must preserve shape and return finite density in `[0, 1]`.

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
optimizer = optimizer.updated_copy(num_steps=30)
result = optimizer.continue_run(result, post_process_fn=post_process_fn)

# Restore the last checkpoint and finish to num_steps:
result = optimizer.complete_run_from_history(post_process_fn=post_process_fn)
# Explicit num_steps here means additional updates:
result = optimizer.continue_run_from_history(
    num_steps=5, post_process_fn=post_process_fn
)
```

`result.history`, `result.keys`, `get(key, index)`, and `get_last(key)` expose parameters, gradients, optimizer states, objective values, post-processing values, and penalties. `store_full_results=False` retains the latest vector state and all scalar history. `get_sim(index)` and `get_sim_data(index)` evaluate saved parameters. Callbacks receive `(result, step_index=..., aux_data=...)`, with post-processing and penalty values in `aux_data`.

Checkpoints store arrays and JSON, not executable design objects. Restore with `optimizer.load_result(path, post_process_fn)` or the continuation methods. The fingerprint checks the numerical mapping revision, simulation, region, backend, optimizer settings, and function code/captured values. Histories from before the symmetry and timestep-clock corrections are incompatible with the current objective; the earlier splitter history is preserved separately as `tidy3d_splitter_workflow_before_symmetry_fix.npz`. For callable objects, supply `objective_id`; changing the objective requires changing that identifier. Exact continuation assumes an unchanged numerical environment. Changes to backend, learning rate, or objective start a new history using `CustomInitializationSpec(params=old_result.get_last("params"))`.

## Current compatibility boundary

This is a supported subset of the **inverse-design workflow**, not a drop-in `tidy3d` package. It keeps BeamZ objects and metre units. It supports one simulation, 2D scalar lossless topology, differentiable modal callbacks, transformations/penalties, Adam, and result continuation. `InverseDesign` defaults to `adjoint`; the lower-level `TopologyProblem` still defaults to `autodiff`.

The [backend documentation](inverse-design.md#two-gradient-backends) describes adjoint decay requirements, monitor restrictions, and host orchestration. No differentiable 3D, shape, lossy/dispersive material, field/flux/far-field objective, `InverseDesignMulti`, expression-metric, or general traced `Simulation.run` support is implied. Additional backend capabilities will be exposed through this interface.
