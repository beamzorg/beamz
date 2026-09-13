# BeamZ optimization

Use `beamz.optimization` for topology inverse design with BeamZ simulations,
metre units, and JAX objectives. The same workflow supports full autodiff and
spectral adjoint gradients.

## Recommended workflow

Given an ordinary `simulation` with fixed mode sources and monitors:

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

Parameters use `(y, x)` order. In 3D, the planar pattern is extruded through the
region's finite z thickness. The design must align to an unsharded uniform grid,
use scalar lossless materials, and stay outside the absorber and fixed mode
planes. The spectral adjoint additionally requires decayed fields and full-run
rectangular DFT monitors.

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

See the [workflow guide](../../docs/inverse-design-workflow.md) for schedules,
restart semantics, native plotting, export, and executed notebook examples.

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

`ModePower` and `SoftMinModePower` combine using arithmetic, for example
`transmission - 0.2 * reflection`. The [backend guide](../../docs/inverse-design.md)
describes this workflow and gradient validation. `autodiff.py`, `projections.py`,
and `polygonize.py` contain the density transforms, SSP projection, and geometry
export helpers for custom loops.
