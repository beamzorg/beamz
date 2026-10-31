# Migrating to BeamZ v0.5

BeamZ v0.5 is a breaking, pre-1.0 architecture release. Simulations and public
devices are immutable specifications; execution state and completed results have
separate ownership. Removed compatibility modules are not preserved as aliases.

## API mapping

| v0.4 API | v0.5 API | Migration notes |
| --- | --- | --- |
| `beamz.simulation.core.Simulation` | `beamz.Simulation` or `beamz.simulation.Simulation` | The legacy `core` module was removed. Configuration is immutable. |
| `Simulation.run_compiled()` and `compile_simulation()` | `Simulation.run()` | All execution now uses the compiled engine. `Simulation.compile()` remains available for advanced inspection or prewarming. |
| `Simulation.run_compiled_until_decay()` | Repeated `Simulation.advance()` calls | There is no built-in decay-loop replacement. Evaluate the stopping criterion from each detached result and pass its state into the next chunk. |
| `RunState`, `EngineState`, and live simulation fields | `SimulationState` | `step()` returns state. `advance()` returns `SimulationRun(results, state)` for continuation and branching. |
| Mutable simulation snapshots and monitor side channels | `SimulationResults` and `MonitorResults` | Read acquisitions from `results.monitor(name)` or `results.monitors[name]`; a completed result is detached from the simulation. |
| `CompiledSimulation`, `CompiledRunConfig`, and `MonitorState` | No public replacement | Compiled plans, executable caches, and monitor accumulators are private implementation details. |
| `RegularGrid` | `GridSpec`, `Design.rasterize()`, and `MaterialGrid` | Use `GridSpec` for simulation grid configuration and rasterize only when direct material arrays are required. |
| `Medium` | `Material` | Materials are immutable value specifications. |
| Public `Structure` construction | `Box`, `Rectangle`, `Circle`, `Ring`, `CircularBend`, `Polygon`, `Taper`, or `Sphere` | Use concrete immutable geometry and `updated_copy(...)` or `with_material(...)` for changes. |
| Generic `Monitor` | `FieldMonitor`, `FieldRecorder`, `FluxMonitor`, or `ModeMonitor` | Configure the concrete acquisition required by the analysis. |
| `Boundary`, `BoundarySpec`, and `AbsorbingLayer` | `PEC`, `PML`, and `Absorber` | Boundaries are canonical immutable device specifications; use `Absorber` for the former sponge-style absorbing layer. |
| `PortSpec` | `Port` | A single canonical port creates matching source and monitor specifications. |
| `ModeSolver` and public `solve_modes()` workflows | `ModeSpec` on `ModeSource` or `ModeMonitor` | Mode solving is planned internally. Read modal results with `results.mode(name)` or `beamz.analysis.mode_data(...)`. |
| `beamz.data` xarray wrappers | `SimulationResults.to_xarray()` or `beamz.analysis.to_xarray(...)` | Labeled data is created lazily from detached results. Generic `colocate_dataset`, `field_intensity`, and `poynting_vector` helpers were removed without one-to-one replacements. |
| `beamz.visual` plotting functions and browser scene | `design.plot()`, `simulation.plot()`, `simulation.view3d()`, and `results.plot_field()` | Plotting now uses static Matplotlib-backed analysis adapters; the interactive browser viewer was removed. |
| `TopologyManager` | `InverseDesign`, design-region configuration, and `Optimizer` | Immutable optimization configuration is separate from evolving density and optimizer state in `InverseDesignResult`. |
| Top-level `transform_density`, `compute_overlap_gradient`, and `create_optimization_mask` | `beamz.optimization.autodiff.transform_density` and `beamz.optimization.topology` helpers | Advanced optimization functions are no longer part of the top-level `beamz` namespace. |
| Top-level `ShardingConfig` | `beamz.simulation.model.ShardingConfig` | Sharding remains an advanced execution option passed to `compile()`, `advance()`, or `run()`. |
| `beamz.devices.sources.signals.ramped_cosine` | `beamz.ramped_cosine` or `beamz.devices.sources.time.ramped_cosine` | Source-time definitions now live together in the source-time module. |

## Execution lifecycle

Use `run()` for a complete, detached result. Use `advance()` only when you need
continuation state for checkpointing or branching.

```python
simulation = simulation.updated_copy(sources=new_sources)
results = simulation.run()

# Use this form only when continuation state is required.
first = simulation.advance(num_steps=100)
second = simulation.advance(state=first.state, num_steps=100)
```

## Removed modules

The legacy `beamz.simulation.core`, `beamz.simulation.compiled`,
`beamz.simulation.fields`, `beamz.simulation.ops`, `beamz.simulation.specs`,
and `beamz.simulation.yee` modules were removed. `beamz.data` and `beamz.visual`
were also removed in favor of detached result adapters and `beamz.analysis`.
