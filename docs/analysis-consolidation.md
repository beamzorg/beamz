# Analysis consolidation and optimization migration

The analysis-only simplification keeps existing public and private call signatures.
`AnalysisData` still owns detached acquisition inputs. `AxisSlice` and `FieldView`
remain the geometry and scalar-display foundations.

`analysis.mode_projection.build_modal_projection_plan(projections)` now builds the
coupled forward/backward tangential measurement once. The returned
`ModalProjectionPlan` exposes coefficient rows and projection with conditioning,
component reconstruction residuals, complex reconstruction scales and signed
reconstructed flux, including coherent cross terms. Both legacy
`_modal_coefficient_rows_3d` and `_project_modal_coefficients_3d_group` use this
assembly. Well-conditioned systems retain direct solves; poorly conditioned
systems retain pseudoinverses. Acquired components must already be temporally
and spatially phase aligned and colocated in the plan's component order.
The existing S-parameter boundary still performs those operations. Grouped
colocation now reconstructs monitor coordinates once for all components.

DFT images, retained-material overlays and slice videos share physical monitor
aperture edges. Uniform images retain their image renderer; nonuniform static
plots retain explicit mesh edges. Videos use the common FieldView image primitive
and preserve their existing image rendering, labels and color limits. Geometry
and raster cross-sections share axes, device overlays, titles and limits while
retaining distinct geometry/raster providers, PML styles and layout behavior.
Analytic geometry setup views still avoid compilation.

## Later coordinated work

- Optimization can bind the new coupled plan's coefficient rows after composing
  its differentiable acquisition colocation and temporal/spatial phase operators.
  Preserve the current `build_mode_projector` callable and ModePower binding
  contracts. This pass deliberately leaves optimization source unchanged.
- Mode-basis solving, source launch, preview and analysis still assemble different
  adapters around device solvers. A shared basis needs cross-package ownership,
  directional conventions, retained constitutive data and solver diagnostics.
- Recorded frames and DFT components still coexist under AnalysisData's field
  mapping. Replacing rank/frequency inference needs an explicit acquisition-kind
  contract agreed with results decoding; do not guess it from matching lengths.
- Geometry cross-sections and permittivity rasters remain separate supported
  products. Their providers should not be collapsed into raster-only rendering.

No kernel or GPU execution code changed. CPU numerical evidence does not establish
GPU parity; the available NVIDIA device could not be opened by `nvidia-smi`.

## Measurement and validation

At baseline `433d59f5946f250b43e38487191609666e830b01`, physical Python lines
were 52,291 package-wide and 5,479 in analysis. This pass leaves 52,230
and 5,418, respectively: **61 net lines removed**, including the new
plan. Counts include blanks, comments and docstrings in every `beamz/**/*.py`;
tests and this document are excluded. This is substantially below the static
review's planning range; previous coordinate/projection consolidations are not
counted again.

Validation covers the analysis contract, plotting/video, modal S-parameters,
modal power/reference/objectives, retained tensor materials, solve caching,
normal sampling, differentiable projection, and straight-waveguide physical
S-parameters with sponge/CPML at 10/12 points per wavelength. New independent
Lorentz-overlap comparisons cover x/y/z normals, nonuniform quadrature and
ill-conditioned coherent mode bases; a batched colocation test verifies complex
interpolation and single geometry reconstruction across components.
