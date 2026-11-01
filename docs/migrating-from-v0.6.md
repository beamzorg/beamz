# Migrating from BeamZ v0.6

The `dev/refactor_3110` branch simplifies the package and deliberately changes
some public APIs. This guide describes changes from `main` before that branch
is merged. It supplements the [v0.5 migration guide](migrating-to-v0.5.md).

## Geometry

`Box`, `Sphere`, `Rectangle`, `Circle`, `Ring`, `CircularBend`, `Polygon`, and
`Taper` are factory functions returning the shared, immutable `Shape` class.
Their usual construction syntax remains available. Replace subclass checks
such as `isinstance(shape, Box)` with a `Shape` check and, when necessary,
`shape.shape_kind`:

```python
from beamz.design import Box, Material, Shape

shape = Box(
    center=(0, 0, 0), size=(1e-6, 2e-6, 3e-6),
    material=Material(permittivity=4),
)
assert isinstance(shape, Shape)
assert shape.shape_kind == "box"
changed = shape.with_material(Material(permittivity=2))
```

`Solid` was an intermediate development name; use `Shape` in imports and type
annotations. Use `updated_copy(...)` and `with_material(...)` for changes.

## Inverse design

| Previous API | Current API |
| --- | --- |
| `AdamOptimizer` | `Optimizer(..., method="adam")`; `"sgd"` is also supported |
| `TopologySpec` / `TopologyProblem` | `MaskedDesignRegion` / `InverseDesign` for an existing density mask |
| A rectangular design region | `TopologyDesignRegion` with physical center, size, and pixel size |
| `TopologyState` / `TopologyResult` | `InverseDesignResult` returned by `Optimizer.run(...)` |

For an existing `simulation` with an output `ModeMonitor` named `output`, and a
boolean `mask` selecting its design cells:

```python
from beamz.optimization import (
    InverseDesign, MaskedDesignRegion, ModePower, Optimizer,
)

region = MaskedDesignRegion(
    region_mask=mask, pixel_size=simulation.resolution,
    eps_bounds=(1, 4), filter_radius=0.15e-6,
)
design = InverseDesign(
    simulation, region, ModePower("output"), gradient_backend="autodiff",
)
optimizer = Optimizer(design, learning_rate=0.04, method="adam")
result = optimizer.run(steps=20, checkpoint="optimization.npz")
```

The optimizer maximizes the objective by default; set `maximize=False` for a
loss. `steps` is the target total update count, including any resumed updates.
For an interrupted scheduled run, use
`optimizer.run(steps=20, stop_after=5, checkpoint="optimization.npz")`, then
`optimizer.run(steps=20, resume="optimization.npz")`. Preserve the schedule's
total horizon when resuming.

Checkpoints contain numerical arrays and validated metadata, without pickle.
Optimization checkpoints use schema 2; older schemas require a new run. Resume
with the same design, objective, optimizer settings, and schedule. For a custom
objective whose identity cannot be inferred reliably, provide a stable
`objective_id` on `Optimizer` and change it when the objective changes.
`store_full_results=True` retains parameter and gradient snapshots; the default
keeps scalar history and the latest state.

## Low-level mode solver

Normal source and monitor workflows continue to use `ModeSpec`. Direct
`solve_grid(...)` callers must now construct `Materials` once, rather than pass
material components and grid edges as separate solver arguments:

```python
from beamz.devices.modes import Materials, solve_grid

materials = Materials.from_components(
    x_edges=x_edges, y_edges=y_edges, eps_xx=eps_xx,
    # Optional tensor components and normal_axis also belong here.
)
modes = solve_grid(material_grid=materials, wavelength=1.55, num_modes=2)
```

This low-level API uses **micrometres** for edges, wavelength, normal coordinate,
and bend radius, and hertz for frequencies. Ordinary BeamZ geometry uses metres.
Tensor arrays have shape `(len(x_edges) - 1, len(y_edges) - 1)`; omitted diagonal
permittivity components inherit `eps_xx`, and omitted permeability is identity.

## Plotting and result storage

Use the canonical plotting arguments:

| Removed argument | Replacement |
| --- | --- |
| `field_name` | `field` |
| `field_monitor_name` | `monitor_name` on `results.plot_field(...)` |
| `f` | `frequency` on result field plots |
| Ignored `monitor` argument to `plot_dft_field` | Remove it; the first argument is `AnalysisData`, which owns monitor geometry |
| `source_normalize` on `plot_dft_field` | Supply `source_normalization` to normalize, or leave it `None` for unscaled fields |
| Viewer `mode` or `open_browser` options | Remove them; `view3d(...)` returns static Matplotlib cross sections |

For example, use
`results.plot_field("output", field="Ez", frequency=2e14, val="abs^2")`.
Plane plots retain the coordinates used during acquisition, including on shifted
and nonuniform grids. Detached results do not require a live simulation.

New simulation result archives use `beamz.simulation.results/2` and retain plane
coordinates. The current loader also reads version 1 and restores missing
coordinates. Older BeamZ readers that only support version 1 cannot read new
version-2 archives. Simulation result archives and optimization checkpoints are
separate formats.
