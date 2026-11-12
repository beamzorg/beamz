# Changelog

## Unreleased

- Added `AnisotropicMedium`, `TensorPole` and `GyrotropicPole`. They provide
  full symmetric permittivity and permeability tensors, tensor-valued
  Lorentz/Drude resonances (oriented poles via `rotated()`), and
  non-reciprocal gyrotropic resonances such as magnetized plasmas. Hyperbolic
  media take their negative branch from a pole. These features run on JAX,
  CPU or GPU.
- FDTD now honours non-unit permeability, including off-diagonal tensors, which
  rasterization previously rejected.
- Full-tensor permittivity accepts conductive and sponge-PML loss on JAX. The
  update applies loss semi-implicitly through `(eps + sigma dt / 2 eps0)^-1`.
  `bz.Absorber` therefore now works with every tensor medium and absorbs the
  backward waves that a PML amplifies in hyperbolic media. BeamZ warns when
  such a medium overlaps a PML.
- Periodic and Bloch seams average tensor cross terms with the seam phase. This
  also covers the full-tensor permittivity kernel, and rectilinear full-tensor
  grids may now use periodic or Bloch boundaries on JAX.
- Mode sources and mode monitors inside media with tensor poles are solved at
  each frequency with the poles folded in. Mode ports outside tensor media are
  no longer rejected.
- Added `AnisotropicMedium.fold_fast_poles(max_resonance)`. It replaces
  ultraviolet poles with a static shift plus a slower, passive pole that
  matches the low-frequency response, which relaxes the pole timestep limit.
- Mode sources and mode monitors may now sit in magnetic media. 3D ports
  reproduce diagonal and full symmetric permeability tensors with the
  propagation's own discretization, which puts diagonals on the H sites and
  cross terms at cell centres. 2D ports support diagonal permeability; 2D TM
  ports overlapping in-plane (xy) magnetic coupling are still rejected. Magnetic
  tensor poles fold into `mu(f)` for both sources and monitors, and the folding
  is kept in result archives. 3D mode-source launches now map their incident H
  increment through `mu^-1`, so the launched power is correct in magnetic media.
  Ports overlapping gyrotropic media are still rejected.
- `Simulation.differentiable()` and `TopologyProblem` differentiate through
  tensor media that lie outside the design region.
- `TopologySpec(material_min=..., material_max=...)` accepts anisotropic and
  pole-dispersive design materials. Density blends the high-frequency tensors
  and fills each cell with the endpoints' poles.
- `TopologySpec(materials=(m0, m1, m2, ...))` lets a design region choose
  among K >= 2 materials, for example air, SiO2, Si and a tilted LiNbO3
  crystal. Each may be a `Material`, a static tensor or a pole-dispersive
  `AnisotropicMedium`. The density gains `K - 1` nested channels, each
  filtered and projected like a two-material density, so binarization picks
  one material per cell. High-frequency tensors blend linearly, and every
  material's poles fill the cell with that material's fraction. Gradients use
  the autodiff backend, and `export_design` paints each material.
  `material_min`/`material_max` remain the K = 2 shorthand. Its physics is
  unchanged, but `export_design` now writes the actual design materials
  rather than scalar stand-ins with their mean permittivity.
- Added general complex-conjugate pole-residue (CCPR) tensor poles, so fitted
  pole-residue models can be anisotropic and oriented. They add an `E'`
  coupling tensor `B`: `chi = (A - i omega B) / (omega0^2 - omega^2 - i gamma
  omega)`. Build them with `TensorPole.ccpr` or `TensorPole.critical_point`.
  `AnisotropicMedium.from_pole_residue(x, y, z)` converts scalar `PoleResidue`
  fits along principal axes, and `rotated()` orients them, `B` included. The
  implicit `B` term solves with `eps_inf + sum dt/(1 + gamma dt/2) B` and runs
  on JAX and streamed CUDA. Mode ports fold the `B` term. The timestep check
  for such media is a von Neumann analysis of the scheme.
- The timestep check for Lorentz, Drude and gyrotropic tensor poles is now the
  same von Neumann analysis of the implemented scheme. The analytic bound
  `S^2 + (Omega dt / 2)^2 < 1` it replaces accepted timesteps that diverge for
  some multi-pole anisotropic media; for one tilted crystal the true limit is
  11% lower. The analysis is cached and independent of grid size.
  `beamz.simulation.anisotropic.max_stable_timestep` takes an optional 2D
  `polarization`.
- Tensor permeability and tensor-pole media run on single-device
  `backend="cuda_streamed"`. New native kernels apply their corrections; the
  CUDA component is now version 0.31.0 / ABI 31, so rebuild it after updating.
  Tensor-media executables compile without XLA command buffers, which crashed
  on replay for some of these programs.
- `AnisotropicMedium` accepts full 3x3 tensors, given as nine row-major values
  or a nested matrix. Permittivity and permeability may be non-symmetric
  (non-Hermitian, so BeamZ warns). Electric `conductivity` and the new
  `magnetic_conductivity` take full tensors with a passive symmetric part; an
  antisymmetric part is lossless and non-reciprocal. `rotated()` now rotates
  conductivity instead of rejecting it.
- Added `AnisotropicMedium.from_complex_permittivity()`, which mirrors fdtdx.
  A complex `eps` or `mu` (scalar, three values, or 3x3) at a reference
  frequency or wavelength becomes real tensors plus
  `sigma = omega0 eps0 Im(eps)`. A Hermitian `eps` therefore becomes an
  antisymmetric conductivity. Added `complex_permittivity()` and
  `complex_permeability()`, which include conduction.
- These terms are applied by a box-local, semi-implicit full-tensor update on
  JAX and on `backend="cuda_streamed"`. Inside the box, the E kernel runs in
  vacuum. The native tensor-media inverse kernel gains antisymmetric cross
  terms. Mode ports must avoid these media.
- Added `anisotropic_library` with sapphire, lithium niobate, hBN, and
  alpha-MoO3 crystal models.
- Added `examples/anisotropic/`: asymmetric refraction in tilted hyperbolic
  calcite, permittivity and permeability iso-frequency contours, and a one-way
  surface magnetoplasmon. Added the "Anisotropic, Hyperbolic, and Gyrotropic
  Media" guide.

## v0.6.2 - 2026-11-04

- Corrected native Yee material and field sampling for 3D mode ports. Solver
  results now carry component-specific coordinates; previews colocate fields
  before launch preparation. Symmetric dielectric ports no longer require forced
  mirror projection to remove sampling-induced mode asymmetry.
- Kept automatic-grid snapping points and mesh overrides in the same physical
  positions when copying centered simulations, including source attachment.
- Updated the cosine-crossing example with 20-cell-per-wavelength automatic
  meshing and PML padding outside the physical region.
- Corrected the 1×4 MMI example to excite the fundamental TM-like slot mode,
  with mode-identity checks and separate Ex, Ey, and Ez previews of mode 0.
  The notebook now uses automatic rectilinear meshing at 16 points per wavelength,
  five source frequency profiles, and 401 monitor wavelengths. Added convergence
  checks with a 20-minute execution budget and explicit provisional-spectrum labels.
- Refreshed executed modal-source, microring, and MMI notebook results.
- Added `AutoTermination(max_wall_time=...)` to stop at a chunk boundary while
  retaining acquired measurements and reporting `wall_time_limit`. Updated the
  microring add-drop notebook to 15 cells per wavelength with a 600-second budget
  and convergence-based early stopping.

### Migration notes

- Code consuming raw 3D `solve_grid()` fields must use each component's own
  coordinates rather than assume colocated samples. Port previews already
  interpolate the native Yee fields onto common cell centres.
- Recheck selected mode identities and numerical results after upgrading:
  corrected 3D sampling can change them, and symmetric dielectric ports no longer
  need forced mirror projection to correct sampling-induced asymmetry.

**Full Changelog**: https://github.com/beamzorg/beamz/compare/v0.6.1...v0.6.2

## v0.6.1 - 2026-11-03

- 2D mode preview, launch and measurement now share aperture bounds, explicit
  padding, target indices and optional confinement. Fixed x-directed previews
  and transverse Yee sampling; removed full-domain launch solves and tapering.

- Mode previews now show solver fields by default, retaining eigensolver sampling
  and normalization. Use `ModeSource.solve_modes(..., field_representation="launch")`
  to inspect prepared 3D launch fields. The preview choice does not change injection.

- 3D mode sources, previews, and monitors now solve on their mesh-snapped aperture
  by default, without automatic transverse padding or tapering. Additional solve
  cells require `ModeSpec.solver_padding_cells`; nondecaying raw fields produce
  edge diagnostics and a warning. Aperture-confinement selection is opt-in.

### Additional fixes and maintenance

- Shared multimode candidate selection to prevent duplicate preview modes and
  repeated eigensolves; improved Yee refinement and rectilinear-grid metrics.
- Fixed S-parameter interpolation when source and destination frequency grids
  have different lengths, and simplified complex interpolation and residuals.
- Consolidated shared ring and circular-bend geometry operations.
- Updated locked numerical and development dependencies, including JAX/JAXlib
  0.10.2, and patched msgpack and Tornado security updates.
- Restricted release-workflow token permissions and refreshed documentation,
  the mode-source notebook, and regression coverage.

### Migration notes

- Mode solves no longer add automatic transverse padding or apply aperture
  tapering. Existing simulations may select different modes or produce different
  numerical results. Enlarge the source/monitor aperture or explicitly set
  `ModeSpec.solver_padding_cells` and check aperture convergence. Use matching
  mode specifications for source launch and modal measurement.
- Default previews retain raw solver sampling and normalization. Request
  `field_representation="launch"` for prepared 3D launch fields.
- The optional CUDA component remains version 0.30.0 / ABI 30.

**Full Changelog**: https://github.com/beamzorg/beamz/compare/v0.6.0...v0.6.1

## v0.6.0 - 2026-10-25

### Added

- Added end-to-end differentiable FDTD with portable JAX and accelerated native
  CUDA reverse-mode execution, checkpointing, and differentiable modal objectives.
- Added 2D and volumetric 3D topology inverse design with autodiff and spectral
  adjoint gradients, broadband objectives, schedules, and fabrication-aware exports.
- Added dispersive materials, a bundled optical material library, Bloch-periodic
  boundaries, and executed metasurface and CMOS RGB sensor examples.
- Added consistent full-tensor dielectric ports on uniform and rectilinear grids,
  native CUDA full-tensor updates, and compact JAX/CUDA material storage.
- Added experimental CUDA multistep scheduling and opt-in region rasterization
  for Gmsh-derived scenes.
- Added passive SOI differential benchmarks and expanded inverse-design,
  grating-coupler, and interface-smoothing notebooks and validation reports.

### Changed

- BeamZ now requires Python 3.11–3.14. Python 3.10 users must upgrade their
  environment before installing v0.6.0.
- The optional CUDA component is now version 0.30.0 / ABI 30. Rebuild the
  extension when upgrading from an earlier ABI.
- Reduced CPML and material storage overhead, reused modal solves, and improved
  sharded CUDA storage selection and arithmetic consistency.
- Updated dependencies for security fixes and added locked Python and Rust
  dependency audits in CI.

### Fixed

- Corrected 2D mode launch and extraction in all four directions, normal Yee
  sampling in 3D modal projection, tensor-axis mapping, and symmetric mode seeds.
- Made forced mode mirror symmetry opt-in and stabilized CPML coefficients,
  including timestep-independent default alpha and compact 2D profiles.
- Fixed electrical conductivity binding in differentiable simulations.

- Fixed topology contours shrinking away from design-region boundaries (#331).
  Region clearing now uses exact cell footprints and fixed geometry supplies
  interpolation context in `TopologyProblem`.
- Added `mode="pixel_exact"` and optional bilinear contour `sample_spacing` to both
  optimization export workflows, and fixed solid/void masks for port cells.
- Added explicit GDS coordinate precision and material/layer selection, preserving
  holes without intermediate quantization through a global layout grid.
- Added a runnable mode-converter export showcase with GDS round-trip geometry and
  transmission comparisons on 100 nm and 50 nm meshes.

### Execution limits

- Native CUDA full-tensor updates require canonical, unpadded float32 streamed
  storage. Full-tensor gradients use JAX; CUDA multistep scheduling has a narrower
  supported physics and storage envelope. See [CUDA documentation](cuda/README.md)
  for backend requirements and measured performance limits.

**Full Changelog**: https://github.com/beamzorg/beamz/compare/v0.5.3...v0.6.0

## v0.5.3 - 2026-10-04

### What's Changed

- docs: add Zenodo DOI badge and release citation by @QuentinWach in [#305](https://github.com/beamzorg/beamz/pull/305)
- build(deps): bump tornado from 6.5.8 to 6.5.9 by @dependabot[bot] in [#299](https://github.com/beamzorg/beamz/pull/299)
- build(deps): bump gitpython from 3.1.59 to 3.1.62 by @dependabot[bot] in [#300](https://github.com/beamzorg/beamz/pull/300)
- build(deps): bump urllib3 from 2.7.0 to 2.8.0 by @dependabot[bot] in [#301](https://github.com/beamzorg/beamz/pull/301)
- build(deps): bump parry3d-f64 from 0.30.2 to 0.31.1 by @dependabot[bot] in [#291](https://github.com/beamzorg/beamz/pull/291)
- fix: preserve simulation materials in detached 3D modal analysis by @QuentinWach in [#310](https://github.com/beamzorg/beamz/pull/310)

**Full Changelog**: https://github.com/beamzorg/beamz/compare/v0.5.2...v0.5.3

## v0.5.2 - 2026-10-01

### Added

- Added multi-GPU CUDA execution and improved streamed CUDA/JAX scaling,
  continuation, and shard-local material preparation for large 3D simulations.
- Added optional constant-radius bent-waveguide eigenmode evaluation through
  `solve_grid(..., bend_radius=..., bend_axis=...)`.
- Added software citation metadata and preparation for Zenodo release archiving.

### Changed

- Improved CUDA storage layout selection and CPML execution, and reduced
  full-field communication in sparse JAX source and monitor operations.
- Removed the experimental `cuda_hopper` backend; use `cuda_streamed` or JAX.
  Rebuild the optional CUDA extension for ABI 21 / component version 0.21.0.
- Updated dependencies, validation tooling, benchmarks, and documentation.

### Fixed

- Fixed package imports on Windows by making Unix process-memory tracing optional.
- Improved material-coefficient equivalence across mesh, GDS, and native Design
  imports, including mesh-region ownership, clipping, adaptive sampling, and
  boundary classification.
- Hardened CUDA graph execution, modal DFT continuation, and distributed CPML
  ownership and halo exchange.

### Validation limits

- GPU performance depends on hardware and geometry. Large-capacity benchmark
  completion does not establish optical convergence or full-size backend parity.
- Some CUDA CPML auxiliary-state and refined raw-DFT comparisons remain outside
  tolerance; see the linked validation reports and benchmark evidence in the
  release notes. Bent-mode radiation loss requires mesh/domain/PML convergence.

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
