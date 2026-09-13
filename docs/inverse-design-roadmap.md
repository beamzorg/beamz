# Tidy3D inverse design and a BeamZ implementation roadmap

Research snapshot: 2026-09-21. The Tidy3D mapping is a source-and-documentation review. The BeamZ baseline analysis below refers to the pinned pre-implementation checkout; the topology milestone and selectable gradient backends are now implemented in this working tree, with status and validation recorded below.

**Current milestone: scheduled four-channel WDM.** `beamz.optimization` now provides `LinearSchedule`, `StepSchedule`, and `OptimizationSchedule` for projection beta, penalty scaling, and objective keyword weights. Schedule horizons are independent of pause/resume boundaries; saved histories, simulation reconstruction, and polygon export retain the applied beta. Parameter-space penalties reproduce the reference WDM's regularizer placement. The optional spectral source-basis strategy preserves frequency-dependent modal profiles, checks every reconstructed source, and falls back to independent frequencies for full-rank or wide-band sources. The default remains frequency grouping. See the [scheduled workflow](inverse-design-workflow.md#scheduled-optimization) and `examples/notebooks/tidy3d_wdm_4channel.ipynb`, based on Flexcompute's [four-channel WDM](https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd9WDM/).

The notebook's 75 nm, 20-frequency check gives complete-gradient differences from full autodiff of **0.00261%** for independent-frequency adjoints and **0.00276%** for the broadband basis. The latter uses **12 solves instead of 20** and takes **65 s versus 128 s** in this execution; its maximum normalized source reconstruction error is **8.1e-8**. Full autodiff is faster on this CPU (**19 s** on the coarse grid, **27 s** warmed on the 50 nm training grid), so the notebook selects it for training. These timings are example-specific, not universal performance claims. Tidy3D groups by eligible ports/frequencies; BeamZ's verified spatial basis is a different implementation of broadband reuse.

The executed notebook retains all **50 reference updates**, the switch at update 18, and a checkpoint restart at update 25. All eight code cells completed and all six figures were visually inspected. The continuous training-grid model shows four separated passbands. After binary polygon export, desired-channel transmission at 50 nm is **64.5%, 70.4%, 82.6%, and 24.8%**; at 30 nm it is **41.4%, 45.2%, 42.5%, and 15.1%**. Doubling the finer-grid duration from 6 to 12 ps changes those fractions by less than 2e-7. The mesh/export sensitivity is substantial, so these results demonstrate the workflow rather than a mesh-converged device. The reference uses a 15 nm design mesh; this example trains at 50 nm. Validation covers **82 distinct targeted tests**, including a vector 3D broadband-basis comparison; Ruff, Pyright, and the strict documentation build pass.

**Earlier milestone: extruded topology in vector 3D.** The same `InverseDesign` workflow now accepts a finite-thickness design region in a 3D simulation. Its `(y, x)` parameters are broadcast through z, with planar filters/penalties, Adam, saved results, and ordinary extruded-polygon export. Both full-timestep autodiff and the spectral Maxwell adjoint include all six vector field components. Three-dimensional modal objectives use the ordinary coupled overlap system and public physical direction labels. See the [3D workflow](inverse-design-workflow.md#vector-3d-with-an-extruded-planar-design) and `examples/notebooks/tidy3d_topology_bend_3d.ipynb`, based on Flexcompute's [3D waveguide bend](https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd18TopologyBend/).

This extension also corrects two 3D source issues exposed during validation: compact source crops must retain the complete Yee boundary supports, and source-profile symmetry must depend on the fixed source cross-section rather than distant device geometry. Otherwise ordinary re-simulation could change the injected source when the topology changed. Gradient tests cover x/y/z mode planes, both TE/TM cases, reverse propagation, physical-direction modal parity, restart, and export. General volumetric transforms, shape derivatives, lossy/dispersive material gradients, and independent multi-excitation optimization remain missing.

The executed 3D bend uses a **50 nm** uniform mesh (44 × 100 × 100 cells), 3,600 planar parameters, and ten Adam updates. Source-normalized target-mode power improves from **0.004988 to 0.734932**; doubling duration from 400 to 800 fs changes it by less than 0.000001. On the 100 nm gradient fixture, adjoint and full-autodiff vectors differ by **0.000251%**, with an independent directional finite-difference check. Binary polygon export gives **0.714983 at 50 nm** and **0.745034 at 40 nm**, so mesh convergence remains unproven. All ten code cells executed and all seven figures were visually inspected. **103 targeted regression tests pass**; Ruff, Pyright, and the strict documentation build also pass. The notebook states its differences from the Tidy3D reference and makes no numerical-parity or fabrication-readiness claim.

The following implementation notes record the earlier 2D milestones and their historical validation. Their original scope statements are snapshots; current support is described above and in the workflow guide.

**Earlier milestone: a shared inverse-design workflow.** `beamz.optimization` now provides `TopologyDesignRegion`, initialization specifications, `FilterProject`, `ErosionDilationPenalty`, `InverseDesign`, `AdamOptimizer`, and saved results with continuation. It follows the single-simulation Tidy3D workflow while deliberately retaining BeamZ simulation objects, metre units, `(y, x)` parameters, and JAX. Real scalar callbacks can use complex modal amplitudes through `data[name].amps.sel(...).values`, including power and phase objectives. Both gradient backends use this same interface. See the [workflow guide](inverse-design-workflow.md).

The executed `tidy3d_inverse_design_splitter.ipynb` ports the single-simulation portion of Flexcompute's [inverse-design plugin notebook](https://www.flexcompute.com/tidy3d/examples/notebooks/InverseDesign/), with explicit grid, pulse, solver, and normalization differences. It now uses a **25 nm** grid (320 × 248 cells; 26,880 design parameters), refined from 50 nm. The complete adjoint and autodiff gradient vectors differ by **0.0020%** on its symmetric perturbed initial design. After ten Adam updates and one saved-checkpoint continuation, the weighted post-processing score improves from **0.306 to 0.949**; this score is not a transmission fraction. Summed output power divided by measured incoming power is **0.942**. These are fresh 2.502 ps BeamZ optimization results, not a fixed-design mesh-convergence study, numerical equivalence to Tidy3D, or fabrication qualification. The previous 50 nm outputs are preserved separately.

Initial workflow validation: **77 targeted regression tests passed**, including the new workflow tests; the ten new workflow/region tests also passed after the final numerical-range and checkpoint-identity fixes. The notebook includes native BeamZ device/field plots, optimization history, and numeric modal powers. This API still supports one simulation and 2D scalar lossless topology with modal data; 3D, shape derivatives, field/flux objectives, and `InverseDesignMulti` remain future work.

**Symmetry investigation.** The centered standalone fundamental mode was symmetric; the main errors came from one-sided topology sampling, a missing TM source-window node, a half-cell shift in the TM monitor material basis, and incomplete adjoint boundary DFT coverage. All four are corrected. In the archived 50 nm notebook, initial complex-field mirror mismatch is **0.0000051%** and initial objective-gradient mismatch is **0.0002842%**, without projecting modes, fields, gradients, or Adam updates onto a symmetric subspace. The internal adjoint pulse was also smoothed to reduce residual off-band excitation while retaining the 1e-4 decay guard. **69 targeted tests pass**, including reflected fields/gradients/ports, ordinary solver parity, finite differences, both polarizations/backends, broadband objectives, and restart. See the linked investigation for the before/after audit and finite-aperture limitations.

**Fine-grid validation and consistent layouts.** Two-dimensional `Simulation.plot()` now shares the 3D material colors, PML hatching, source arrows, monitor markers, and micrometre display coordinates. The 25 nm run exposed float32 timestep-clock drift affecting DFT phase: observation time now comes from the integer step count and an anchored run origin, including continuation. The internal adjoint pulse also suppresses DC excitation through a discrete temporal difference, retaining its measured-DFT normalization and the 1e-4 decay guard. The new numerical revision rejects incompatible old optimizer histories. After 11 unconstrained updates, final field mirror mismatch is **0.000813%** and outer modal-power mismatch is **0.000372%**. The current targeted coverage includes 138 passing checks across plotting, execution/continuation, long-run clocks, modal analysis, and inverse-design gradients. All twelve notebook code cells executed; all seven figures were visually inspected.

**Two gradient backends.** `TopologyProblem(..., gradient_backend="autodiff")` retains the existing checkpointed JAX derivative of the full finite-duration FDTD computation. `gradient_backend="adjoint"` uses a custom simulation VJP: prepare forward design-region electric DFTs, differentiate the objective into monitor sensitivities, construct frequency-grouped adjoint sources, run pulsed adjoint FDTD, overlap spectral fields, and pull material gradients back through the Yee/density map. This follows the division in Tidy3D's [simulation primitive][autograd], [forward preparation][forward], and [adjoint setup/post-processing][backward]. No Tidy3D code or cloud execution is required at runtime.

The new path transposes BeamZ's single linear Maxwell/CPML timestep, including its boundary masks and Yee staggering. It does not autodifferentiate either complete time loop. This discrete spectral formulation differs from Tidy3D's geometry boundary-integral rules; shape derivatives are not implemented. The initial scope is the same scalar, lossless, 2D TM/TE topology API, with full-run rectangular DFT monitors. Each active frequency requires at most one adjoint run; compatible spatial-pattern grouping across frequencies remains missing. Both methods use the existing optimizer, objective composition, filters/projections, histories, and export workflow. Checkpoint identities prevent accidental continuation with a different derivative backend or adjoint decay tolerance.

Both forward and adjoint pulse decay are checked, with explicit errors for unconverged runs. The spectral gradient omits terminal-state terms, so the decay ratio is not a gradient-error guarantee and duration convergence remains necessary. The saved `topology_gradient_checks.ipynb` outputs, recorded before the symmetry corrections, include both backends: relative gradient-vector differences from autodiff are **0.0529% (TM)** and **0.0979% (TE)** on the stated 100 nm fixture. Extending the TM run from 320 to 480 fs changes its adjoint gradient by approximately **0.00003%**, while retaining 320 complex forward design DFT samples at either duration. These are correctness and stored-data checks, not peak-memory or speed benchmarks. Broadband weighted/soft-min objectives and measured-input sensitivities are also tested for both polarizations. These older saved measurements describe the previous numerical revision; the current corrected TM fixture stores 357 samples (17 × 21 nodes for a 16 × 20 cell patch), and the current regression suite rechecks TM/TE gradients and duration convergence. All results are BeamZ-only numerical checks.

The development priority is reusable solver/optimization features demonstrated by published reference notebooks. Further fabrication refinement of the existing bend is deferred. Extruded 3D support is now implemented. Next are independent multi-excitation objectives, efficient broadband source grouping, and shape derivatives; each needs its own numerical validation and reference notebook.

Validation for this addition: **168 targeted tests passed**, including the earlier topology/modal/Yee checks and the new backend gradient, orthogonal-port, decay, zero-source, and restart tests. The updated notebook executed top-to-bottom and all three saved figures were visually inspected. Ruff, Pyright on the changed optimization/runtime modules, and the strict documentation build passed. Validation was local on the JAX CPU backend; no GPU performance or Tidy3D cross-solver equivalence is claimed.

Tidy3D checkout: `c7f41d17ac5de2f7a87e811bbfa1d051f6f9c762` on `develop`, reporting version 2.12.0. I also fetched release `v2.12.0` (`4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9`); `git diff v2.12.0 HEAD -- tidy3d tests` was empty. Source links below are pinned to that release commit. BeamZ checkout: `37256510066a143764f0afd413d1cc76afd06dd5`, version 0.5.1.

**Main finding.** Flexcompute's statement is substantially correct about the inverse-design client: the repository contains automatic-differentiation integration, adjoint preparation, local gradient reconstruction, geometry/material derivative rules, optimization utilities, and advanced workflows. It does not contain the production cloud FDTD solver. `local_gradient=True` selects local gradient post-processing; its forward and adjoint simulations still run through cloud jobs. BeamZ can use this as an architectural reference, but needs its own solver integration and numerical validation. See the [local gradient strategy][strategy] and [execution engine][engine].

**What is implemented in Tidy3D**

“Implemented” here means concrete source exists, with documentation and/or relevant tests inspected. “Recipe” means a demonstrated composition of lower-level features, rather than a dedicated high-level feature class. I did not run paid simulations or execute Tidy3D's numerical test suite.

| Capability | What the client implements | Boundary of the claim |
| --- | --- | --- |
| Differentiable simulations | Autograd primitives and custom vector-Jacobian products for simulation execution; traced input extraction, forward/adjoint orchestration, and gradient return. Standard `td` objects work with `web.run`; asynchronous execution is supported. | The derivative crosses an external solver call; it is not backpropagation through an open FDTD time-stepping implementation. [Source][autograd] |
| Density/topology optimization | Spatial permittivity/conductivity derivatives through `CustomMedium`; design-grid interpolation backpropagation; high-level `TopologyDesignRegion` with material bounds, initialization, transformations, penalties, and uniform-axis controls. | Supports volumetric parameter grids as well as extruded designs. The high-level region's default is uniform in z. [Media][medium], [region][region] |
| Primitive shape optimization | Derivatives for box center/size, sphere center/radius, cylinder center/radius/length/sidewall angle, and polygon-slab vertices/slab bounds/sidewall angle. | These are explicit boundary derivative implementations, not automatic differentiation through arbitrary CAD rasterization. [Geometry source][geometry], [polygon slabs][polyslab] |
| Composite and mesh shapes | Derivatives through geometry groups and Boolean clipping operations; triangular surface-mesh vertex derivatives. | Topological changes, invalid meshes, and nonsmooth Boolean events still need careful handling. [Geometry source][geometry], [mesh source][mesh] |
| Lossy and anisotropic materials | Permittivity and conductivity derivatives; nested diagonal anisotropic and custom anisotropic media. | Do not infer support for every full-tensor medium or every material parameter from forward-solver support. [Source][medium] |
| Dispersive materials | Pole-residue, Sellmeier, Lorentz, Drude, and Debye parameter derivatives, including spatially varying counterparts. | Requires a solver that supports the corresponding constitutive models. [Source][medium], [dispersive tests][disp-tests] |
| Source optimization | Custom current/field datasets and supported center coordinates; Gaussian and astigmatic Gaussian beam positions, angles, polarization, and waist parameters. | A whitelist of supported source parameters; not arbitrary source-time or mode-eigensolver differentiation. [Source implementations][sources], [source tests][source-tests] |
| Modal and field objectives | Mode amplitudes, Gaussian overlap amplitudes, diffraction amplitudes, spectral E/H fields, supported permittivity outputs, and derived intensity/Poynting/flux quantities. | Complex amplitudes permit power, phase-sensitive, overlap, and interference objectives with a real scalar loss. [Monitor derivative implementations][monitor-data] |
| Flux objectives | `FluxMonitor(enable_adjoint=True)` retains hidden surface-field information for differentiation. | Default flux monitors do not opt in; storing additional frequencies has a memory cost. [Flux bridge][flux] |
| Far-field objectives | Differentiable local `FieldProjector` post-processing of near fields. | Direct differentiation of server-side projected monitor results is not the supported path. [Documentation][ad-docs], [projection implementation][projection] |
| Broadband objectives | Builds adjoint sources from objective derivatives and groups them by frequency or spatial source pattern, choosing the smaller group count. | The standard grouping uses `min(active frequencies, spatial-port groups)` per forward problem; “one extra solve” is not universal. A spatial-port group is an internal source grouping, not simply the number of named waveguides. [Implementation][sim-data] |
| Multiple simulations and S-matrices | `InverseDesignMulti`, batch differentiation, and supported optical/modal/terminal component-modeler objectives. | Multiple incident excitations generally require multiple forward problems. Weighted or smooth worst-case objectives can combine them. [Design wrapper][design], [modelers][smatrix] |
| Fabrication-oriented transforms | Conic, circular, and Gaussian filters; tanh and ramp projections; differentiable morphology; mirror/rotation/diagonal symmetry; grayscale diagnostics. | Useful manufacturability controls, not a foundry DRC guarantee. [Utility implementation][invdes-utils] |
| Subpixel-smoothed projection | A Hammond-style smoothed projection that retains interface sensitivity at very sharp projection strengths, including infinite beta. | This implementation requires a 2D array and assumes an appropriate uniform design grid. It is distinct from the solver's interface/subpixel material treatment. [Source][projections] |
| Fabrication penalties | Erosion/dilation penalties and curvature penalties, including Bézier-based shape helpers. | Soft penalties influence the objective; they do not constitute a general hard-constraint optimizer. [Source][penalties] |
| Level-set design | A documented parameterized level-set workflow with differentiable gap and curvature penalties. | A notebook recipe built on Autograd and custom media; no general `LevelSetDesignRegion` in the inspected high-level plugin. [Example][levelset] |
| Fabrication robustness | Multi-simulation objectives over geometric/material variants; a fabrication-aware WDM example integrates PreFab. | Variant aggregation is a supported recipe. PreFab's process prediction is an external integration, not all implemented inside Tidy3D. [Multi-objective example][robust], [PreFab example][prefab] |
| Optimization lifecycle | `InverseDesign`, `AdamOptimizer`, histories, callbacks, initialization policies, serialization, and continuation from saved results. Lower-level Adam helpers also exist. | The high-level `DesignRegionType` is topology-focused; shape capabilities extend beyond that wrapper. [Optimizer][optimizer], [result][result], [region][region] |
| Topology-to-shape handoff | `PolySlabSet` extracts and tracks solid/hole contours, boundary masks, smoothing, vertex updates, and self-intersection repair; separate curvature utilities. | A useful foundation for topology followed by shape refinement, not a universal differentiable topology-changing CAD pipeline. [Source][polyslab-set] |
| User-defined derivative rules | `custom_vjp`, `NumericalStructureConfig`, and derivative helpers allow user-created structures and custom gradients from forward/adjoint data. | The user supplies the derivative rule. The hook does not discover correct derivatives for arbitrary geometry builders. [Types][custom-types], [run integration][autograd] |
| Framework integration | Native Autograd support and a `to_torch` bridge for PyTorch workflows. | The old JAX adjoint plugin was removed in 2.10; this is not a reason for BeamZ to replace its existing JAX stack. [PyTorch source][pytorch], [migration docs][ad-docs] |
| Gradient-free design exploration | Grid and Monte Carlo sampling; wrappers for Bayesian optimization, genetic algorithms, and particle swarm; float, integer, and categorical parameters. | Several algorithms rely on optional third-party packages. They complement adjoint optimization for discrete or low-dimensional searches. [Source][design-methods], [docs][design-docs] |

The examples include mode converters, splitters, wavelength multiplexers, gratings, metasurfaces, and level-set Y branches. These demonstrate how to compose the capabilities above; they are not separate solver primitives.

**How the adjoint implementation is divided**

```mermaid
flowchart LR
    P[Design parameters] --> T[Geometry or density transforms]
    T --> S[Simulation inputs]
    S --> F[Forward FDTD solve]
    F --> M[Monitor data]
    M --> J[Real scalar objective]
    J --> V[Objective derivatives]
    V --> A[Adjoint source construction]
    A --> B[Adjoint FDTD solves]
    F --> G[Field overlap and boundary integration]
    B --> G
    G --> R[Material and geometry derivatives]
    R --> D[Gradients of design parameters]
```

The most useful reading order is:

1. [autograd.py][autograd]: primitive boundary, input validation, tracing, custom hooks, and VJP registration.
2. [forward.py][forward] and [strategy.py][strategy]: required gradient monitors, forward data retention, and local versus remote execution.
3. [monitor_data.py][monitor-data] and [sim_data.py][sim-data]: convert objective sensitivities into physical adjoint sources and group the required solves.
4. [backward.py][backward] and [derivative_utils.py][derivative-utils]: frequency alignment, normalization, bounded field data, memory chunking, and derivative context.
5. [medium.py][medium], [geometry][geometry], [polyslab.py][polyslab], and [mesh.py][mesh]: actual parameter derivative mathematics. Shape gradients use tangential electric and normal displacement fields; PEC interfaces additionally require magnetic-field information.
6. [invdes][invdes] and [autograd/invdes][invdes-utils]: parameterization, fabrication utilities, and the user-facing optimizer workflow.

The client also contains local parallel-adjoint execution: eligible canonical adjoint bases can launch alongside the forward run, then be combined once objective derivatives are known. Eligibility currently covers mode amplitudes, diffraction amplitudes, and point field samples; unsupported outputs fall back to the sequential path. This can exchange extra simulations/storage for lower latency. It is an advanced scheduling optimization, not a prerequisite for BeamZ's first implementation. [Source][parallel]

**Limits that affect a parity plan**

- Traced runs need supported frequency-domain data. A time-domain-only objective is not covered by this spectral adjoint route. Setup rejects unsupported traced paths rather than silently accepting every simulation setting. Point-cloud and thin-lens-overlap monitor cases have explicit rejection paths. [Validation][autograd]
- Geometry/material support is field-specific. A forward model's existence does not prove it has an adjoint derivative. Do not promise EME, coupled heat/charge, arbitrary nonlinear or time-varying adjoints, or higher-order solver derivatives based on this review.
- The default traced-structure budget is 500; geometry grouping helps. Gradient data can be large because fields and permittivities are retained over traced regions and frequencies. [Configuration][config], [backward processing][backward]
- Symmetry reduction and source grouping interact: the code warns that separately grouped adjoint ports may violate forward symmetry. BeamZ needs explicit symmetry tests rather than assuming every reduction is safe. [Source][sim-data]
- Some older examples describe restrictions that the current implementation has relaxed. Use pinned implementation and current validation rules as the authority for support decisions.
- Tests cover plumbing, transformations, and numerical comparisons with finite differences, including shapes, sources, dispersive media, and custom hooks. Their presence is evidence of validation infrastructure, not proof that every combination passes today. [Numerical tests][numerical-tests]

**BeamZ baseline before this implementation**

| Existing component | Evidence in this checkout | Consequence |
| --- | --- | --- |
| Density configuration and optimizer state | [topology.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/optimization/topology.py): `TopologySpec`, `TopologyState`, Adam/SGD, beta continuation, material interpolation, and masks. | Extend these concepts; avoid a second competing topology API. |
| Differentiable density transforms | [autodiff.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/optimization/autodiff.py): conic filtering, smooth morphology, fixed-structure padding, and JAX VJPs. | A substantial part of the topology front end already exists. The implementations are oriented toward 2D arrays. |
| Subpixel-smoothed projection | [projections.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/optimization/projections.py): 2D Hammond SSP1, including infinite beta. | This advanced item already has a BeamZ counterpart; verify conventions rather than reimplementing it. |
| Contour conversion | [polygonize.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/optimization/polygonize.py): thresholded contours with nested holes, Shapely geometry, and BeamZ polygons. | Reuse for final geometry verification and a later topology-to-shape stage. |
| Overlap and disk storage utilities | [topology.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/optimization/topology.py), [adjoint_memmap.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/optimization/adjoint_memmap.py). | Useful building blocks; they do not define an objective-specific, solver-consistent adjoint by themselves. |
| Explicit solver state and JAX stepping | [model.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/simulation/model.py), [execute.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/simulation/execute.py), [kernels.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/simulation/kernels.py). | Strong foundation for a discrete adjoint reference, including CPML state. |
| Spectral monitors and modal/S-parameter analysis | [observe.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/simulation/observe.py), [mode_projection.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/analysis/mode_projection.py), [sparameters.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/analysis/sparameters.py). | Reuse numerical conventions; existing NumPy-based result/analysis paths need a differentiable boundary or explicit VJP. |
| Material and backend infrastructure | [materials.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/design/materials.py), [CUDA runtime](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/simulation/cuda/runtime.py). | Tensor-valued forward materials and CUDA execution exist, but do not establish their optimization-gradient support. |

The main missing connection is a supported `parameters -> simulation -> objective -> gradient` path. `TopologySpec.apply_gradient` currently expects a supplied permittivity gradient. `compute_overlap_gradient` sums time-reversed field products; it does not construct objective sources or establish the coefficient, source, normalization, and time-staggering derivatives for a particular loss. Its tests verify overlap arithmetic, while the finite-difference tests in [test_optimization.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/tests/integration/test_optimization.py) verify density transforms. I did not find an end-to-end electromagnetic objective-gradient check in this checkout.

There is a useful partial runtime separation already: the compiled scan accepts `(state, coeffs)`. However, public execution obtains coefficients from `program.coefficients`, the compiled-program key includes material values, and material-to-coefficient compilation is not exposed as an optimization path. The baseline checkpointing TODO identified the same next step (now implemented in [execute.py](https://github.com/beamzorg/beamz/blob/37256510066a143764f0afd413d1cc76afd06dd5/beamz/simulation/execute.py)). We should build on the existing coefficient argument, not claim all solver coefficients are hard-coded inside the scan.

I found no general solver custom-VJP integration or registered CUDA differentiation rule in the inspected BeamZ files. I also found no Lorentz/Drude/Debye auxiliary-state implementation in the current material/runtime path. Dispersive inverse design therefore includes forward-solver work, whereas basic topology optimization does not require it.

**Recommended BeamZ architecture**

Keep JAX and Optax. Separate three contracts:

1. **Design mapping:** design parameters become runtime materials on a fixed mesh, with a differentiable map or explicit pullback. Later, shape parameters can use boundary derivative rules.
2. **Simulation derivative:** runtime materials and fixed simulation configuration produce objective-relevant monitor arrays. A backend implements their VJP using a discrete or spectral adjoint.
3. **Optimization problem:** a scalar objective combines outputs, fabrication penalties, and simulation variants; the runner owns schedules, histories, callbacks, and restart state.

Use ordinary JAX differentiation through small simulations as a correctness reference after exposing the missing runtime material path. Then add a memory-efficient backend. For Tidy3D-like frequency-domain photonic objectives, the intended scalable route is a spectral adjoint storing selected fields over design regions. Checkpointed discrete reverse-mode is an alternative with closer agreement to the finite-time discrete solver and broader potential objective support. Both can fit behind the same API; benchmark accuracy, memory, and runtime before selecting a default.

The spectral and discrete derivatives need not agree exactly at finite mesh resolution and finite run time. The validation must separate a wrong implementation from discretization, interpolation, and incomplete-decay errors. Do not equate the two adjoint formulations without that study.

For the first supported problem, freeze the mesh, timestep, source waveform, port cross-sections, and run length. Make material values dynamic. Keep the design region away from sources/ports so mode profiles remain fixed. Later support for moving ports or changing port materials must account for mode and normalization derivatives. Keep executable-cache identity separate from result-cache identity: reusing an executable across densities must never reuse old simulation results.

Tentative implementation areas, not new APIs already available:

| Responsibility | Likely BeamZ location |
| --- | --- |
| Runtime material/coefficient binding and reusable compiled plans | `beamz/simulation/compile.py`, `model.py`, `execute.py` |
| Differentiable simulation primitive and backend VJP | New `beamz/optimization/adjoint.py`, with runtime support in `simulation/` |
| Differentiable modal/field reductions and adjoint source rules | New `beamz/optimization/objectives.py`, sharing conventions with `analysis/` and `devices/` |
| Problem definition, result history, restart, and callbacks | New `beamz/optimization/problem.py` and `result.py`; reuse `TopologySpec`/`TopologyState` |
| Fabrication penalties and robust variants | New `beamz/optimization/penalties.py`; extend existing transforms |
| Shape pullbacks and parameterizations | Later `beamz/optimization/shape.py`; integrate deliberately with the Rust rasterizer |

**Proposed implementation sequence**

1. **Establish a solver-gradient reference.** Expose runtime material arrays and differentiable coefficient construction on a fixed 2D mesh, reuse the compiled scan, and implement a differentiable spectral field/modal reduction. Start with one polarization, nondispersive dielectrics, a fixed source, and fixed output modes. Check random directional derivatives against centered finite differences across a step-size sweep. Cover CPML, complex amplitudes, source normalization, and the exact material-to-Yee mapping. Acceptance: correct sign and magnitude in full electromagnetic objectives, plus a small step that improves the intended objective. A gradient-norm or nonzero-gradient test alone is insufficient.

2. **Implement a scalable adjoint backend.** Add design-region spectral field recording, objective-to-source VJPs, correct source/DFT normalization, forward/adjoint grid alignment, and material-gradient accumulation. Compare with the small discrete reference and finite differences; measure convergence with mesh and run time. In parallel with this design decision, evaluate checkpointed discrete reverse-mode as the alternative backend. Add an explicit differentiation strategy for CUDA before advertising CUDA optimization support; its FFI calls do not automatically inherit JAX differentiation. Acceptance: measured memory scaling, stable gradient agreement, and reusable executables across changing density values.

3. **Ship the first complete user workflow.** Connect existing filters/projections and optimizer state to a problem runner, with objective histories, continuation schedules, callbacks, checkpoints, and restart. Use a 2D mode converter or splitter as the first example. Add clear errors for unsupported features. Acceptance: a reproducible optimization improves the physical objective; a resumed run matches uninterrupted execution within declared tolerances; the thresholded/exported geometry is independently re-simulated. Save actual optical performance, not just the penalized training loss.

4. **Add practical photonic robustness and 3D.** First extend the validated material adjoint to vector 3D with extruded planar design parameters, then volumetric parameters. Add broadband/multiport objectives, batches over independent excitations, smooth worst-case aggregation, eroded/nominal/dilated variants, symmetry handling, and explicit length-scale/curvature penalties. Acceptance: gradient checks across polarizations/frequencies/variants, backend parity, and final-grid verification of the binary designs. Generalize filters to 3D deliberately; keep SSP's 2D restriction explicit until a separate implementation is validated.

5. **Add shape and level-set workflows.** Start with boxes and polygon vertices, then splines, extrusion thickness and sidewall angle, cylinders/spheres, Boolean combinations, and triangle meshes. Reuse contour conversion for topology-to-shape refinement, adding stable vertex identities and self-intersection handling. Choose between boundary-integral pullbacks and a differentiable rasterization route per geometry, and validate interface motion under mesh refinement. Add a parameterized level-set mapping as a separate design representation. Acceptance: directional shape derivatives converge and refined shapes retain performance after export/re-rasterization.

6. **Close advanced parity gaps.** Implement dispersive forward models and their gradients, diagonal anisotropic optimization, supported source parameters, local far-field objectives, custom derivative hooks, optional PyTorch integration, and adjoint scheduling optimizations. Add gradient-free search adapters for small/discrete parameter spaces when they have a concrete use case. Each feature needs its own capability matrix and numerical fixtures; “all inverse design” should not be a single completion flag.

The user selected **reliable topology optimization with a simple API** as the first milestone. Steps 1–3 deliver that milestone. Advanced geometry, dispersive materials, and full 3D parity remain later extensions.

**Selected first milestone: reliable topology optimization with a simple API**

The user-facing outcome is: supply a base simulation, a topology region, and an objective; run optimization; inspect, resume, and export the result. Users should not need to construct adjoint sources, retain forward fields, calculate overlaps, or manually pass permittivity gradients.

Proposed initial scope, using engineering defaults rather than additional user decisions:

- Fixed-grid 2D simulations, one initially validated polarization, nondispersive dielectric materials, one incident excitation, and a single target frequency.
- One density design region using existing `TopologySpec`/`TopologyState`, conic filtering, tanh or SSP projection, fixed connections, and Adam with beta continuation.
- A built-in normalized modal-transmission objective, followed by a small set of composable differentiable reductions. Keep ports and sources outside the changing design region.
- A problem runner with sensible defaults, reproducible initialization, progress/history, checkpoints, resume, and final density-to-polygon export.
- A JAX reference path first. Select the memory-efficient gradient implementation from measured evidence; native CUDA optimization is supported only after its gradient path is validated.
- A compact 2D waveguide mode converter as the proposed acceptance example. Confirm the particular modes and dimensions while building the fixture.

The simple API should expose the simulation, region, objective, and run settings. Reuse the existing topology configuration rather than duplicating filter and optimizer settings in a competing configuration object. Return final parameters, physical density, geometry, objective history, and enough state to resume. Exact public names remain provisional until a minimal end-to-end example exercises the design.

The milestone is complete when all of these acceptance conditions hold:

| Requirement | Evidence required |
| --- | --- |
| Correct electromagnetic gradients | Directional finite-difference comparisons over several perturbation sizes and deterministic cases; include the density transform, material mapping, CPML, monitor reduction, and normalization. Set numerical tolerances from precision/convergence studies and record them in the tests. |
| Useful optimization | The mode-converter example improves normalized target-mode transmission over its initial design under a fixed evaluation setup. Track physical performance separately from penalties. |
| Reliable restart | Interrupted and resumed runs reproduce uninterrupted parameters and objective history within declared backend tolerances. |
| Practical execution | Changing densities reuses the executable without reusing stale results. Record peak memory and iteration time for the acceptance example. |
| Trustworthy output geometry | Threshold, export, re-rasterize, and re-simulate the final geometry; report its performance and mesh/run-time convergence separately from the continuous-density result. |
| Simple usage | The example runs from a problem definition and a runner call, with no user-authored adjoint or gradient-storage code. Unsupported configurations fail with specific explanations. |

The first reviewable change should expose differentiable runtime material binding and add a small full-solver directional-gradient fixture. Subsequent changes add the practical gradient backend, connect `TopologySpec` to the runner, and complete restart/export verification. The user-visible milestone includes the runner and verified example; the gradient prototype alone does not complete it.

Hardware/memory targets remain to be measured before committing to the production adjoint backend. No calendar estimate is assigned until the prototype establishes the actual solver work. The 2D scope and mode-converter example are proposed implementation defaults, not additional preferences explicitly selected by the user.

Implementation recommendation: write BeamZ-native JAX code against these contracts and the cited mathematics. Tidy3D's repository carries LGPL-2.1 and BeamZ's carries Apache-2.0; any decision to incorporate upstream source should be handled explicitly as a dependency/provenance decision. This research adds no Tidy3D implementation code to BeamZ.

**First milestone implementation status**

The working tree now adds `TopologyProblem`, `ModePower`, and `TopologyResult`, reusing `TopologySpec`. This closes the previously missing end-to-end solver-gradient connection for uniform 2D scalar lossless dielectric problems. See the [usage guide](inverse-design.md).

- The material binder supplies dynamic Yee coefficients to the existing JAX timestep. A chunk-checkpointed discrete adjoint reuses the exact forward transitions, including CPML and spectral monitors. Ordinary reverse mode remains a small-problem reference. A spectral adjoint and native CUDA differentiation remain future work.
- Modal objectives share the ordinary analysis mode basis and phase conventions. An optional fixed input monitor normalizes by measured forward fundamental-mode power, including its derivative. Nominal-source normalization remains available, with its finite-grid limitations stated explicitly.
- The runner supports continuation, history, callbacks, atomic non-pickle checkpoints, restart with problem/schedule validation, continuous-material re-simulation, and thresholded polygon export.
- Seventeen new tests pass: full-solver finite differences for TM/TE with and without incoming-mode normalization, an SSP full-solver gradient check, chunk/ordinary adjoint parity, optimizer improvement, exact same-environment resume, export/fixed-port preservation, and rejection of unsupported or invalid configurations. Eighty-five related optimization, Yee-grid, and boundary-update tests also pass.
- The executed [mode-converter notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_mode_converter.ipynb) optimizes at 50 nm and independently re-solves the exported geometry down to 25 nm, extending the run from 160 to 320 fs. The executed [gradient-check notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_gradient_checks.ipynb) records perturbation sweeps, ordinary-modal parity, checkpoint memory estimates, and exact restart. Their saved outputs are the evidence for numerical values and timings.

This delivers a bounded first topology workflow, not full Tidy3D parity. The 100 nm exploratory optimizer exposed modal/discretization errors above 100% transmission; no objective values are clamped. The tutorial therefore uses a finer design grid, measured incoming power, independent binary export, and separate resolution/run-time checks. Further port-distance, aperture, fabrication, and spectral studies remain important before treating a design as production-ready. Filter radius is not a guaranteed minimum feature size.

**Broadband and multiport milestone**

The same 2D solver now supports frequency-selected `ModePower` terms and immutable weighted combinations. A single broadband excitation can optimize desired transmission across several output ports while penalizing reflection and crosstalk. `problem.spectra` exposes each term's unweighted spectrum. Mode bases, source normalization, and electric/magnetic phase corrections are evaluated at each selected frequency; reference spectra match by frequency, not array position.

Seventeen additional tests cover composition, invalid selections, combined full-solver gradients for TM/TE, comparison with ordinary modal analysis, ragged DFT offsets, different reference frequency counts/order, explicit reference direction, optimization, restart, and checkpoint identity. All 119 selected regression and validation tests pass. The first milestone's default single-frequency API and checkpoint identity are retained.

The executed [broadband-demultiplexer notebook](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/topology_broadband_demultiplexer.ipynb) adapts Flexcompute's [WDM example](https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd9WDM/), which cites [Cheung et al., OFC 2024](https://doi.org/10.1364/OFC.2024.W1A.6). It uses two bands near 1300/1550 nm, three training wavelengths per band, and independent intermediate-wavelength, mesh, and run-time checks. Its material index, dimensions, channel count, spectral spacing, and objective aggregation differ from the reference; it is a workflow adaptation, not a reproduction of the published silicon device or a numerical Tidy3D benchmark.

At the six training wavelengths, independent verification of the exported binary design on a 20 nm mesh gives 90.34–93.43% desired-port transmission, at most 0.802% unwanted-port transmission, and at least 20.52 dB port isolation. The notebook also retains a 36-point spectrum and mesh/run-time comparisons. These are simulated results for this 2D example.

Multi-source/S-matrix optimization, fabrication-variant batches and penalties, 3D gradients, and native CUDA adjoints remain later milestones. The public API still accepts scalar lossless materials on a fixed uniform 2D xy grid. Supporting multiple spectral/port terms does not remove those restrictions.

**Direct reference recreation: Meep's filtered broadband bend**

The executed [Meep bend recreation](https://github.com/beamzorg/beamz/blob/main/examples/notebooks/meep_filtered_waveguide_bend.ipynb) ports [the original notebook at commit cedb7aef](https://github.com/NanoComp/meep/blob/cedb7aef80dc2c4ac2c9c981b916a2ce50fcfb01/python/examples/adjoint_optimization/03-Filtered_Waveguide_Bend.ipynb). It retains the 3.4/1.44 material indices, 0.5 µm guides, 2.5 µm square design region, 1.5–1.6 µm wavelength grid, 51×51 nodal parameters, conic/tanh/symmetry mapping, and six 12-evaluation MMA stages. Solver, material interpolation, waveform, and stopping differences are explicit in the notebook. This is a reference-problem recreation, not a numerical cross-solver equivalence result.

The implementation adds `projection_type="identity"` for external parameter mappings and the optional `mma` dependency extra. Recreating the upward output also exposed and fixed a 2D modal-projection direction error: y-normal line modes inherited a legacy magnetic sign, so the positive port basis could carry negative physical flux. Projection now orients that basis by signed Poynting flux. Tests cover both axes and polarizations, plus full-solver bend derivatives and ordinary-analysis agreement.

The saved run increases the continuous mean transmission from 1.04% to 98.50%. The binary export gives 84.23–92.65% transmission (89.77% mean) on a 12.5 nm / 900 fs verification grid. Changing from 25 to 12.5 nm shifts some wavelengths by up to 3.35 percentage points, so mesh sensitivity remains visible. At 25 nm, the dilated/nominal/eroded threshold scenarios have worst-wavelength transmissions of 56.0% / 80.9% / 14.3%. This provides a concrete baseline for robust optimization. As in the reference, filtering is not an enforced minimum-feature constraint, and the nominal objective does not optimize fabrication corners.

**Fabrication-robust extension within the same notebook**

Part II of that notebook now refines the nominal result against all three threshold scenarios and ten wavelengths. `SoftMinModePower` supplies a smooth lower bound on the weakest frequency; applying the same reduction across scenario scores yields a smooth minimum over all 30 pairs. Its gradient includes the scenario mapping, FDTD response, and measured incident-power normalization. The notebook checks the composed derivative against finite differences.

The robust refinement uses 72 MMA evaluations, each with three FDTD gradient evaluations. A nominal-only control starts from the same baseline and uses 216 single-scenario MMA evaluations, matching the additional gradient budget. Both use the same six projection stages and equal forward-only stage diagnostics. Binary exports are then checked at additional wavelengths and intermediate thresholds, with finer-grid checks for the baseline and robust designs. This is robustness to a sampled density-threshold model inside the design region; physical etch calibration, fixed-lead variations, and explicit minimum-feature constraints remain future work.

In the executed extension, the 25 nm / 900 fs verification covers five thresholds and 29 wavelengths. Worst binary transmission is 13.44% for the original baseline, 0.29% for the equal-budget nominal control, and 89.33% for the robust refinement; nominal mean transmissions are 89.54%, 95.44%, and 93.27%, respectively. At 12.5 nm, over the three optimized thresholds and ten training wavelengths, the baseline/robust minima are 13.23% / 92.01%. The robust design's largest corresponding 25-to-12.5 nm spectral change is 2.28 percentage points. All 145 selected regression and gradient tests pass, including new worst-frequency objective and checkpoint checks.

[autograd]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/autograd.py
[strategy]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/strategy.py
[engine]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/engine.py
[forward]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/forward.py
[backward]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/backward.py
[parallel]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/parallel_adjoint.py
[custom-types]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/types.py
[flux]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/web/api/autograd/flux_monitor.py
[medium]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/medium.py
[geometry]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/geometry
[polyslab]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/geometry/polyslab.py
[mesh]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/geometry/mesh.py
[derivative-utils]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/autograd/derivative_utils.py
[monitor-data]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/data/monitor_data.py
[sim-data]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/data/sim_data.py
[sources]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/source
[projection]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/components/field_projection
[config]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/config/sections.py
[invdes]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/invdes
[invdes-utils]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/autograd/invdes
[region]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/invdes/region.py
[design]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/invdes/design.py
[optimizer]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/invdes/optimizer.py
[result]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/invdes/result.py
[polyslab-set]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/invdes/polyslab_set.py
[projections]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/autograd/invdes/projections.py
[penalties]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/autograd/invdes/penalties.py
[smatrix]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/smatrix
[pytorch]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/pytorch/wrapper.py
[design-methods]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tidy3d/plugins/design/method.py
[disp-tests]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tests/test_components/autograd/test_autograd_dispersive_vjps.py
[source-tests]: https://github.com/flexcompute/tidy3d/blob/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tests/test_components/autograd/test_autograd_sources.py
[numerical-tests]: https://github.com/flexcompute/tidy3d/tree/4cacc2755d5858e0de5a8066ff017f1b9e3ad9f9/tests/test_components/autograd/numerical
[ad-docs]: https://docs.flexcompute.com/projects/tidy3d/en/latest/api/plugins/autograd.html
[design-docs]: https://docs.flexcompute.com/projects/tidy3d/en/latest/api/plugins/design.html
[levelset]: https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd10YBranchLevelSet/
[robust]: https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd4MultiObjective/
[prefab]: https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd23FabricationAwareInvdes/
