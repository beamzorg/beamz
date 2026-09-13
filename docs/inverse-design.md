# Topology optimization

For a Tidy3D-style region → callback → optimizer workflow, start with [`beamz.plugins.invdes`](inverse-design-workflow.md). It keeps BeamZ objects and metre units and supports both gradient backends. The lower-level `TopologyProblem` API below remains available.

BeamZ can optimize a density region through its full 2D JAX FDTD solver. Define the simulation and region, choose a modal objective, and call the optimizer. Gradients include the density transform, material-to-Yee sampling, update coefficients, CPML, spectral monitors, and modal projection.

Start with the executed [mode-converter notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_mode_converter.ipynb). The [broadband demultiplexer notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_broadband_demultiplexer.ipynb) extends this to two output ports and two wavelength bands. The [gradient-checking notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_gradient_checks.ipynb) demonstrates the numerical checks and checkpoint memory tradeoff. These notebooks also live in `examples/notebooks/` in this checkout.

For a direct port of an existing reference problem, use [Meep's filtered broadband bend recreated in BeamZ](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/meep_filtered_waveguide_bend.ipynb). Its first part preserves the reference geometry, indices, ten wavelengths, nodal mapping, and 72-evaluation MMA schedule. The same notebook then refines that baseline against dilated, nominal, and eroded threshold scenarios, with an equal-gradient-budget nominal control. Install the optional optimizer with `uv sync --extra dev --extra mma`. The notebook records solver differences, gradient checks, binary export verification, and threshold sensitivity; it does not claim cross-solver numerical equivalence or guaranteed minimum feature size.

## Basic API

Given a `Simulation` with fixed `ModeSource` and `ModeMonitor` planes, create a boolean mask matching its cell material grid in `(y, x)` order. Coordinates in `simulation.design` and compiled material grids use the solver's local frame; use that design when constructing `TopologySpec`.

```python
from beamz.optimization import ModePower, TopologyProblem, TopologySpec

region = TopologySpec(
    design=simulation.design,
    region_mask=mask,
    resolution=simulation.resolution,
    eps_min=1.0,
    eps_max=4.0,
    filter_radius=200e-9,
    learning_rate=0.04,
    beta_schedule=(1, 16),
)
problem = TopologyProblem(
    simulation,
    region,
    ModePower("output", mode_index=1, reference_monitor="input"),
    checkpoint_interval=32,
)
result = problem.run(40, initial_density=density, checkpoint="result.npz")
physical_density = problem.physical_density(result)
exported_design = problem.export_design(result, threshold=0.5)
verified = simulation.updated_copy(design=exported_design).run(backend="jax")
```

`ModePower` maximizes power in one mode. With `reference_monitor="input"`, it divides by that monitor's power at the same frequency, defaulting to the **positive-direction fundamental mode**. Use `reference_mode_index` and `reference_direction` for a different incident mode/direction. Without a reference monitor, it divides by nominal source power. These normalizations can differ on a finite grid. Frequency matching uses a relative tolerance of `1e-10` and never interpolates a missing spectral sample.

`value(density, beta=...)` evaluates the objective. `value_and_grad(density, beta=...)` returns the value and a NumPy gradient with respect to the latent density. Values outside the design mask have zero derivative. `material_simulation` constructs an ordinary simulation with the identical continuous-density Yee map, useful for comparing the optimization objective with ordinary modal analysis.

The runner re-evaluates each updated design before recording its objective. History entries include beta, objective before/after the update, gradient norm, maximum update, and elapsed time. The beta schedule changes the material projection, so the history need not be monotonic. A callback receives the immutable result after each update.

## Two gradient backends

Choose the derivative implementation independently of the design, objective, and optimizer:

```python
autodiff = TopologyProblem(simulation, region, objective,
                          gradient_backend="autodiff", checkpoint_interval=32)
adjoint = TopologyProblem(simulation, region, objective,
                         gradient_backend="adjoint", adjoint_decay_tolerance=1e-4)
value, gradient = adjoint.value_and_grad(density, beta=2)
print(adjoint.gradient_diagnostics)
```

| Backend | Simulation derivative | Field storage | Appropriate use |
| --- | --- | --- | --- |
| `autodiff` (default) | JAX reverse mode through the complete finite-duration FDTD computation | Checkpointed timestep states and recomputation | Reference gradients, transient or insufficiently decayed runs |
| `adjoint` | Custom simulation VJP, pulsed adjoint FDTD, and spectral field overlap | Forward electric DFTs on design Yee supports, plus the current adjoint state and its design DFTs | Converged pulsed, frequency-domain 2D problems |

The adjoint path follows Tidy3D's **forward preparation → objective VJP → adjoint sources/runs → material-gradient post-processing** architecture. The open-source reference is pinned in the [roadmap](inverse-design-roadmap.md). BeamZ uses JAX for the objective and density/material maps. Its custom simulation VJP replaces differentiation through the time loop with independent adjoint runs. The adjoint timestep is the transpose of BeamZ's linear, source-free Maxwell/CPML update, including Yee staggering and boundary masks; JAX generates this *single-step* transpose. It does not backpropagate through either full time loop. Complex source amplitudes are propagated as two real quadratures. The internal adjoint pulse is the discrete temporal difference of a modulated Gaussian with a three-period width and an eight-width leading margin. Differencing suppresses DC excitation of stationary dual states; normalization by its measured DFT retains the desired frequency-domain source strength.

This is an architectural counterpart, not a copy of Tidy3D's boundary-integral derivatives or proprietary solver. The first version supports scalar material/topology derivatives only. It groups all objective contributions at the same frequency into one source pattern, including reference-monitor sensitivities. There is at most one adjoint run per active frequency, independent of the number of density parameters; zero source patterns are skipped. Tidy3D additionally groups compatible spatial patterns across frequencies, which BeamZ does not yet implement.

The forward and adjoint pulses must decay. Every gradient call checks their terminal/peak field norms and raises an actionable error if the ratio exceeds `adjoint_decay_tolerance`; it never silently switches backends. Diagnostics include both decay ratios, the actual adjoint-run count, and the number of stored forward design DFT values. The adjoint norm includes scaled CPML dual-state variables. The decay threshold is **not a guaranteed gradient-error bound**: compare longer run durations and use autodiff/finite differences for new problem families. The spectral formula omits terminal-state terms and is not the exact derivative of a short, truncated run.

Adjoint monitors must sample every timestep with a rectangular window over the full run. Both forward and adjoint use the simulation's configured timestep and step count. The forward observation clock is derived from the integer step count and the run origin, rather than repeated float32 time additions; this avoids phase drift on long fine-grid runs and preserves explicit continuation clocks. The driver performs host-side orchestration and checks; individual FDTD runs are JIT compiled. It is not a general externally jittable or higher-order differentiable simulation API. The stored design DFTs do not grow with the number of timesteps, but source waveforms, compiled executables, and other allocations remain; this is not a measured peak-memory claim. `checkpoint_interval` controls the autodiff backend only. Checkpoints distinguish the adjoint backend, decay tolerance, and numerical mapping revision. Histories created before the symmetry corrections must start a new run because their numerical objective has changed.

The executed [gradient-checking notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_gradient_checks.ipynb) compares both backends, finite differences, and run-duration convergence.

## External parameter mappings

Set `filter_radius=0` and `projection_type="identity"` in `TopologySpec` to supply an already-mapped physical density. This is useful for reference-specific nodal grids, symmetry operations, or external optimizers. The density passed to the problem still has the full simulation-grid shape and values in `[0, 1]`. The design mask preserves fixed geometry outside it.

```python
import jax
import jax.numpy as jnp

# mapping maps your parameters to a full-grid physical density, using JAX.
physical_density, pullback = jax.vjp(mapping, jnp.asarray(parameters))
value, density_gradient = problem.value_and_grad(physical_density)
parameter_gradient = pullback(jnp.asarray(density_gradient))[0]
```

Design-cell permittivity is averaged onto staggered Yee supports, with partial coverage blended against the original fixed raster. The adjoint stores every affected support, including interpolation boundaries. This preserves reflections through the material map; see the [symmetry investigation](inverse-design-workflow.md#numerical-symmetry).

The identity projection bypasses projection only; a nonzero filter radius still applies the selected filter. External optimizer state and mapping parameters must be saved by that workflow, as demonstrated in the Meep notebook.

## Worst-frequency and fabrication-scenario objectives

`SoftMinModePower` takes the same monitor, frequency, and reference arguments as `ModePower`, with an additional positive `temperature` (default `0.03`). It replaces the mean with `-temperature * logsumexp(-power / temperature)`, a smooth lower bound on the weakest selected transmission. For N frequencies its gap below the exact minimum is at most `temperature * log(N)`. The score can be negative and is not itself a physical transmission. `problem.spectra` still returns the unreduced powers.

```python
from beamz.optimization import SoftMinModePower

objective = SoftMinModePower(
    "output", reference_monitor="input", temperature=0.03
)
problem = TopologyProblem(simulation, region, objective)
```

Weighted combinations preserve each term's reducer. Checkpoint identity distinguishes the mean, smooth minimum, and temperature. To optimize fabrication scenarios, evaluate the same problem on separately mapped densities and combine their scalar scores with another smooth minimum at the same temperature. The nested reduction equals a smooth minimum over every scenario/frequency pair. The notebook differentiates this combination, checks it against finite differences, and verifies the resulting binary geometries independently.

The current scenario model changes density thresholds inside the design region. It does not calibrate those thresholds to physical etch offsets, alter fixed leads, or enforce minimum feature sizes.

## Broadband and multiport objectives

`ModePower(..., frequencies=[f1, f2, ...])` averages modal power at those selected monitor frequencies. Omitting `frequencies` selects every frequency on the target monitor. All values are in Hz, and each selected frequency must exist on the target and reference monitors. The reference monitor may contain additional frequencies or list them in a different order. Power is normalized **per frequency before averaging**.

Combine terms with ordinary arithmetic. The optimizer maximizes the resulting score, so negative weights penalize unwanted powers:

```python
short_to_upper = ModePower("upper", frequencies=short_band, reference_monitor="input")
long_to_lower = ModePower("lower", frequencies=long_band, reference_monitor="input")
short_leakage = ModePower("lower", frequencies=short_band, reference_monitor="input")
long_leakage = ModePower("upper", frequencies=long_band, reference_monitor="input")
reflection = ModePower("input", direction="-", reference_monitor="input")

objective = (0.5 * (short_to_upper + long_to_lower)
             - 0.2 * (short_leakage + long_leakage)
             - 0.1 * reflection)
problem = TopologyProblem(simulation, region, objective)
result = problem.run(100, initial_density=density)
spectra = problem.spectra(result.state.density, beta=result.beta)
```

Arithmetic constructs an immutable `WeightedObjective`; it can also be constructed explicitly with `terms=` and `weights=`. Weights are finite real scalars and are not normalized automatically. The aggregate uses one forward simulation, followed by one reverse pass for `autodiff` or up to one adjoint FDTD run per active frequency for `adjoint`, including derivatives of reference amplitudes. Multiple output ports do not imply multiple input excitations: there is still exactly one fixed `ModeSource`.

`problem.spectra` performs one forward solve and returns a tuple of **unweighted per-frequency physical powers/ratios** in `objective.terms` order. Each array follows that term's requested frequency order, or its target monitor's order by default. The weighted scalar is `sum(weight * spectrum.mean() ...)`. Use the spectra to distinguish desired transmission, unwanted-port power, and reflection from the overall optimization score.

The source pulse must cover the requested band. `ModeSpec(num_freqs=...)` controls broadband source-profile reconstruction independently of monitor sampling. Keep the source, mode planes, mesh, and timestep fixed while changing density. Dense spectral sweeps and independent geometry verification remain necessary; a few training frequencies do not prove performance everywhere between them.

## Checkpoint and resume

```python
partial = problem.run(40, initial_density=density, stop_after=20,
                      checkpoint="result.npz")
result = problem.run(40, resume=problem.load("result.npz"),
                     checkpoint="result.npz")
```

`stop_after` is an absolute completed-step limit; `num_steps` defines the entire beta schedule and must remain unchanged on resume. Checkpoints atomically save density, optimizer state, history, schedule, and problem identity in an NPZ archive without pickle. Loading rejects a different simulation, region configuration, or objective, including changed spectral selections, weights, or reference modes. Checkpoints created with the original default `ModePower` specification retain their problem identity. Exact continuation is tested in the same environment; bitwise reproducibility across hardware or JAX versions is not guaranteed.

## Supported scope

- Uniform, unsharded 2D xy grids, with both TM and TE full-solver gradient checks.
- Design-rasterized scalar, lossless, nondispersive `Material` objects, relative permeability one, and a fixed timestep satisfying the design's CFL bound.
- One fixed `ModeSource`; multi-frequency line `ModeMonitor` or `FieldMonitor` acquisitions normal to x or y. Objectives are `ModePower` or weighted combinations of modal-power terms.
- Existing conic/morphology density filters, Heaviside, SSP, or identity projection, and Adam/SGD updates through `TopologySpec`. The examples exercise conic filtering and Heaviside projection; full-solver tests also check SSP and identity gradients. External optimizers can consume `value_and_grad`.
- Fixed geometry outside the design mask. The mask must avoid absorbers, the domain boundary, and source/modal-monitor planes with a two-cell sampling margin. This is a numerical exclusion margin, not proof that a modal plane is far enough from evanescent fields.
- JAX execution with either checkpointed discrete reverse mode or pulsed spectral adjoint FDTD. For autodiff, the default chunk length is 32; `None` keeps ordinary reverse-mode history for small reference comparisons. Adjoint-specific restrictions are listed above. Native CUDA FFI backends do not yet have an optimization derivative.

Three-dimensional, lossy/dispersive/anisotropic, shape, and multi-source optimization remain outside the supported API. The high-level workflow supports custom real scalar objectives over complex modal amplitudes. Fabrication variants can already be combined by an external differentiable mapping and scalar aggregation, as demonstrated in the bend notebook; there is no dedicated multi-simulation problem class yet. The [Tidy3D review and roadmap](inverse-design-roadmap.md) maps those later extensions.

## Interpreting results

A correct derivative is the derivative of the chosen finite-time discrete simulation. It does not establish grid convergence, source decay, modal-extraction accuracy, or global optimality. Coarse-grid modal estimates can exceed unity; the API does not silently clamp them. Use measured incoming power, place ports in sufficiently long uniform leads, and verify mesh, aperture, and run-time sensitivity.

The saved example reports the continuous-density objective separately from the thresholded polygons. Exported polygons are independently rasterized and simulated at finer resolutions and longer run times. Filtering and projection encourage smooth/binary geometry but do not guarantee minimum feature size, connectivity, or fabrication robustness.
