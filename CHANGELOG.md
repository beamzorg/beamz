# Changelog

## v0.5.1 - 2026-09-12

### Added

- Added an optional native CUDA execution backend for supported 3D simulations,
  with fused Yee updates, graph replay, and source, CPML, and DFT monitor
  acceleration. CUDA sources are included in the package for optional builds.
- Added GDSFactory component preparation and S-parameter workflows, with
  executable documentation and distribution-level smoke coverage.
- Added and refreshed MMI, ring resonator, and grating coupler notebooks, including
  rendered outputs and v0.5 API migration examples.

### Changed

- Improved simulation progress reporting and performance statistics, separating
  compilation time from execution throughput.
- Default PML thickness now resolves to 12 grid cells per selected face when no
  physical thickness is specified.
- Improved layout cross-sections with smooth geometry and removed seams between
  overlapping structures. Face-on sources and monitors render as rectangles.
- Made field plot colorbars aspect-aware and separated horizontal colorbars from
  axis labels.

### Fixed

- Fixed mode refinement on rectilinear Yee grids.
- Preserved 2D mode-source profiles and corrected mode-source launch power
  calibration, centered 3D source symmetry, and reverse source placement.
- Corrected 3D GDS placement and hardened packaged GDSFactory workflow validation.

## v0.5.0 - 2026-08-19

This is a breaking, pre-1.0 architecture release. BeamZ now separates immutable
simulation configuration, compiled execution, evolving runtime state, and detached
analysis results. Public devices are immutable specifications, compiled plans and
caches are private implementation details, and completed results no longer retain or
mutate a live simulation.

Existing v0.4 code may require import and lifecycle changes. The most common
migrations are listed below; removed compatibility modules are not preserved as
aliases.

### Added

- Added lossless full-tensor Farjadpour permittivity updates for 2D TE and 3D
  FDTD, with inverse diagonals at E supports and stable average-multiply-average
  cross coupling at shared grid nodes, while retaining the explicit diagonal
  update policy.
- Integrated the complete MicroMode finite-difference eigensolver into
  `beamz.devices.modes`, including the latest guarded Yee-grid refinement and
  validation work from `beamzorg/micromode@80c57d8`.
- Added a public native mode-solver package for rasterized solves, modal sweeps,
  overlap analysis, and optional HDF5 result persistence through
  `beamz[mode-io]`.
- Added immutable `SimulationState`, `SimulationRun`, and detached
  `SimulationResults` ownership contracts.
- Added canonical immutable source, monitor, boundary, port, material, geometry, and
  topology specifications with functional `updated_copy(...)` operations.
- Added result-owned source normalization, modal projection, S-parameter extraction,
  labeled-data adapters, plotting, and field-video support through `beamz.analysis`.
- Added explicit compilation, continuation, sharding, cache, packaging, public-API,
  architecture, and physics regression coverage.

### Changed

- Farjadpour smoothing is now the default: standalone rasterization retains full
  tensors and simulations use the diagonal policy unless full coupling is
  requested explicitly.
- Mode sources, monitors, and ports now share `ModeSpec` and `ModeData` from
  `beamz.devices.modes`; the former source import paths remain compatibility
  re-exports.
- Removed the external `micromode` runtime dependency. Mode solving and source
  compilation now use the BeamZ-owned implementation directly.
- `Simulation.run()` is now the normal complete execution path and returns detached
  `SimulationResults`. Use `Simulation.advance()` for continuation or checkpointing,
  and `Simulation.step()` only for single-timestep debugging.
- Simulation configuration changes now return a new value through
  `Simulation.updated_copy(...)`; simulations and device specifications no longer own
  mutable runtime fields or monitor buffers.
- Grid materialization is deferred until compilation or execution. `GridSpec`
  describes the requested grid, while `Design.rasterize()` returns material data.
- The Python `RegularGrid` rasterizer and callback-based `CustomMaterial` API
  were removed. Rust rasterization is the only geometry path for `Design`,
  simulations, imported GDS, STL, Gmsh, meshio, and raw-mesh scenes. Pass
  pre-sampled spatial coefficients directly as `MaterialGrid`.
- GDS support is now optional and loaded lazily. Install `beamz[gds]` before calling
  the GDS import or export functions.
- Plotting and labeled-data conversion are loaded lazily from `beamz.analysis` so the
  numerical engine does not depend on presentation or xarray state.

### Breaking API migration

See the [v0.5 migration guide](docs/migrating-to-v0.5.md) for the complete API
mapping and lifecycle guidance.

### Removed

- Removed the legacy `beamz.simulation.core`, `beamz.simulation.compiled`,
  `beamz.simulation.fields`, `beamz.simulation.ops`, `beamz.simulation.specs`, and
  `beamz.simulation.yee` public modules.
- Removed the legacy `beamz.data` and `beamz.visual` packages in favor of detached
  result adapters and `beamz.analysis`.
- Removed compatibility aliases and mutable compiler/device hooks that duplicated the
  canonical immutable APIs described above.

## v0.4.3 - 2026-06-26

### Changed
- Updated cosine crossing and modal source example notebooks.

### Fixed
- Fixed mode-source flux calculation and expanded related mode-source and tidy API coverage.

## v0.4.2 - 2026-06-22

### Changed
- Improved matplotlib field plotting with permittivity overlays for field-frame and DFT field views.
- Refined material overlay styling so real fields use darker structure overlays and power fields use lighter overlays.

### Fixed
- Expanded visualization coverage for real-field and power-field overlay behavior.

## v0.4.1 - 2026-06-22

### Changed
- Improved matplotlib field plotting aliases, marker orientation handling, and DFT field component extraction.
- Clarified README terminology around sub-pixel averaging.

### Fixed
- Preserved `SimulationResults.plot_field` behavior while forwarding monitor and field aliases through keyword arguments.
- Added clearer errors for missing DFT field components and expanded visualization coverage.

## v0.4.0 - 2026-06-20

### Added
- Added TFSF support for mode sources.
- Added CPML waveguide benchmarking documentation and scripts.
- Added expanded tests for mode sources, PML/CPML behavior, animation, visualization, and curl kernels.

### Changed
- Improved CPML behavior to reduce waveguide reflections.
- Updated examples and notebooks for modal source and monitor workflows.
- Refined license labeling and documentation around recommended and experimental examples.

## v0.3.2 - 2026-06-13

### Changed
- Reduced compiled-engine memory use and simulation compile overhead for larger 3D runs.
- Simplified boundary and compiled simulation internals while preserving the public API.
- Streamlined examples, benchmark scripts, and development tooling to reduce repository size.
- Updated project licensing metadata to Apache-2.0.

### Fixed
- Improved 3D permittivity handling and memory estimation in simulation setup.
- Fixed mode profile data handling for visualization workflows.
- Adjusted tests and CI coverage around compiled-engine, boundary, and 3D constitutive behavior.

## v0.3.1 - 2026-05-31

### Added
- Added a compact demo example for the demux workflow.

### Changed
- Reduced memory load in boundary and compiled simulation paths.

### Fixed
- Fixed the demux example.

## v0.3.0 - 2026-05-26

### Added
- Added modal port workflows with `Port` and `ModeMonitor` support.
- Added xarray-backed result accessors and plotting conveniences for simulation data.
- Added matplotlib visualization helpers for snapshots, fields, layouts, mode fields, and styled DFT views.
- Added UBC PDK support and improved gdsfactory component handling.
- Added broader 2D/3D physics, monitor, source, and engine-equivalence test coverage.

### Changed
- Replaced the external mode-solver dependency path with `micromode`.
- Improved 3D mode-source normalization, source quadrature handling, Yee phase-plane calibration, and monitor DFT/modal projection behavior.
- Refactored CPML/PML handling, including sponge-style absorbing layers and compatibility through `AbsorbingLayer`.
- Improved material sampling on Yee components, full-PEC 3D sampling, and source scattering/S-parameter calculations.
- Promoted the Design material API and added simulation-domain/depth convenience aliases.
- Updated development tooling around `uv`, `ruff`, `vulture`, and Makefile audit commands.

### Fixed
- Fixed 2D and 3D mode-source handedness, power normalization, and source-plane phase-referenced test expectations.
- Fixed cropped Yee profile interpolation and modal monitor projection alignment.
- Fixed 3D CPML compiled-engine behavior and related monitor/source reconstruction cases.
- Restored the dipole example and reverted an incompatible material-handling refactor.
