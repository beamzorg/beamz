# Differential cases

Solver-neutral JSON specifications live in `cases/`. They describe physical
inputs and observable outputs without exposing any solver's internal field
layout. `case_schema.py` validates them before an adapter is allowed to run.

Oracle priority is explicit in every case:

1. analytical solution;
2. mathematical invariant;
3. independent-solver consensus;
4. historical BeamZ regression data.

External adapters should return only the named observables (for example
effective index, reflected power, or an S-parameter). Optional Meep and FDTDX
jobs can therefore be added without turning either solver into the sole truth.

## Passive-SOI device benchmarks

The `cases/passive_soi_*.json` manifests implement Liu and Poon's
Lumerical/Tidy3D comparison (arXiv:2506.16665) using the closest BeamZ-supported
equivalents. They contain the geometry, provenance, simulation protocol,
published reference values, and remaining solver-specific limitations. Device
code lives in `passive_soi/`; layouts are generated on demand and checked
against normalized fingerprints of the paper's GDS artifacts.

Run the crossing sweep with:

```console
uv sync --extra test --extra gds
uv run pytest tests/differential/test_crossing.py \
  -m hardware --validation-report=validation-results-crossing.json
```

Run one published directional-coupler comparison with:

```console
BEAMZ_VALIDATION_ARTIFACT_DIR=validation-artifacts \
uv run pytest tests/differential/test_directional_coupler.py \
  -m hardware -k 6ppw \
  --validation-report=validation-results-directional-coupler-6ppw.json
```

Each device manifest defines its resolution sweep and broadband comparison
settings; add `-k` to select one resolution. Setting
`BEAMZ_VALIDATION_ARTIFACT_DIR` retains plots, raw monitor data, S-parameters,
and run metadata. Set `BEAMZ_EXECUTION_BACKEND` to `jax` or `cuda_streamed` to
select a backend; reports record the backend that actually ran.

Each broadband run monitors all requested wavelengths at once.

Power validation uses the complete published resolution sweep rather than the
reference value at the BeamZ run's resolution. Each manifest identifies the
region the paper describes as converged and pools the Lumerical and Tidy3D
values there to estimate one nominal result. For each BeamZ resolution, the
tolerance is the largest deviation from that nominal among the two published
solver values at the same PPW, with plot-digitization precision as a floor.
Reports retain both the converged samples and the same-PPW samples used for the
run's tolerance.

## Rectilinear-grid references

`rectilinear_grid_references.json` contains frozen x/y grid-boundary coordinates
for the solver-neutral cases declared in `rectilinear_grid_cases.py`. The cases
cover a homogeneous domain, high-index and coupled rectangles, a ring/bus
coupler, an explicit mesh override, snapping points, and an enforced coarse
override. Keeping the fixture immutable makes changes to mesh density and
grading explicit in review while leaving the normal test suite independent of
external packages.

## 2x2 MMI (initial lowest-resolution case)

The MMI uses the paper's `mmi2x2_with_sbend` layout, pinned to the physical
fingerprint of its reference GDS at the same source revision as the crossing
and directional coupler. The MMI and directional coupler share the four-port
adapter in `passive_soi/four_port.py`.

Only 6 cells per wavelength and a 20 nm source bandwidth are enabled for the
MMI so far. Run its geometry checks and lowest-resolution comparison with:

```console
uv run pytest tests/differential/test_mmi2x2.py \
  --validation-report=validation-results-mmi2x2-6ppw.json
```

The comparison uses the explicitly reported 1550 nm cross-power values from
[Section 3.3 of the paper](https://arxiv.org/html/2506.16665v3): 0.376 for
Lumerical and 0.358 for Tidy3D, with the suite's existing cross-solver tolerance.
It also checks the passive-device output-power bound. Passing at this coarse
resolution does not establish mesh convergence.

## Mode converter and polarization splitter rotator

Both conversion cases currently enable only **6 cells per wavelength** with a
**20 nm source bandwidth**. Run their geometry checks and simulations with:

```console
uv run pytest tests/differential/test_mode_conversion.py \
  --validation-report=validation-results-mode-conversion-6ppw.json
```

Use `-k mode_converter` or `-k polarization_splitter_rotator` to select one
case. The mode converter launches TE0 and measures TE1 conversion plus TE0
crosstalk at the wide output. The splitter rotator launches TM0 and measures
TE0 conversion plus TM0 crosstalk at the upper output. Its nitride top cladding
starts at the silicon substrate plane, matching the reference pipeline.
Both channels share each output's DFT monitor, with distinct modal projections.
The physical sources and mode monitors solve five candidate modes, matching the
reference pipeline's `mode_num=5` setting.
The power bound sums the measured output modes; it is not a measurement of
all guided and radiated power.

The silicon polygons in `passive_soi/layouts/` are extracted from the pinned
reference GDS files. Modern GDSFactory changes the mode-converter layout and
cannot reproduce the reference splitter rotator's 405 nm port width on its
2 nm port grid. These fixtures preserve the original geometry without a
runtime download or a dependency on the older GDSFactory release. Manifests
record the source revision, GDS checksum, fixture checksum, physical-union
fingerprint, and original YAML ports. Layer-1 silicon alone determines the
physical bounds; annotation layers do not enlarge the simulation domain.

The comparison ranges use values explicitly stated in Sections 3.4 and 3.5 of
[the paper](https://arxiv.org/html/2506.16665v3). Both devices are strongly
resolution-dependent at 6 ppw, so these tests do not establish convergence.
The current BeamZ results are strict expected failures against those published
ranges: the mode converter measures 0.200 TE1 power and the splitter rotator
measures 0.871 TE0 power at 1550 nm. Pytest will fail with an unexpected pass
when either comparison starts agreeing, requiring its marker and documentation
to be updated.

Offline analysis of the retained 6 ppw DFT fields rules out output-mode
misidentification as the source of these differences. At 1550 nm, the mode
converter's target-plane flux is 0.216 and its unnormalized TE1 projection is
0.217; the splitter rotator's values are 0.860 and 0.856 for TE0. The output
projection residual is about 7% in both cases and its condition number is 1.
The corresponding source-plane residuals are 5.1% and 2.7%. The discrepancy is
therefore already present in the coarse-grid propagated fields.

The splitter-rotator reference selects raw solver mode index 2 for its nominal
TM0 source. BeamZ instead orders candidates by polarization and selects the
highest-effective-index TM mode. At 6 ppw BeamZ finds TE0, TM0, and TE1 at
effective indices 2.652, 2.266, and 1.976, respectively. This avoids treating a
resolution-dependent raw mode index as physical mode identity and may explain
why BeamZ produces the paper's converged-like splitter behavior at the lowest
resolution. This is an inference from the source code and computed modes, not a
confirmed diagnosis of either reference solver.

The mode converter has an additional broadband normalization limitation. Its
measured incident modal power ranges from 0.856 to 1.119 across the 20 nm band,
and the selected output-power ratio reaches 1.123 at one edge. At 1550 nm its
target-plane flux and TE1 projection agree, so aperture projection does not
explain its low conversion. Retained artifacts now include per-port modal
powers, effective indices, projection residuals, condition numbers, and raw
flux comparisons for future runs.

Two controlled 6 ppw mode-converter reruns further isolate the discrepancy.
Replacing the graded absorber with CPML changes TE1 conversion only from 0.2003
to 0.2011. Enabling full Farjadpour interface averaging under CPML changes it
to 0.2129. The latter still shows a 1.136 peak selected-output ratio at a band
edge. Boundary reflection and diagonal-only interface averaging therefore do
not account for the published-power gap or broadband normalization error.

Cross-section and retained-field analysis instead identifies coarse-grid phase
matching as the leading explanation. On the exact 6 ppw raster, isolated
wide-guide TE1 and narrow-guide TE0 effective indices are 2.4926 and 2.5062.
The ordinary coupled-section solve returns 2.5204 and 2.4865, which would imply
a 22.8 um transfer length. Applying the same rectilinear Yee refinement used by
the source changes the coupled pair to 2.4822 and 2.4374, for a 17.3 um
transfer length. Independent spatial system identification of the retained
Ex/Ey fields gives 2.4742 and 2.4251, or 15.8 um. This agreement shows that the
time-domain field follows the refined Yee phase advance rather than the
ordinary cross-section eigenvalues.

Three internal mode planes at 27, 36, and 45 um confirm this interpretation in
a separate 6 ppw time-domain run. The two coupled-supermode powers remain
approximately constant along the straight section: 0.856/0.125 at the start,
0.841/0.144 at the middle, and 0.832/0.138 at the end. Their unwrapped relative
phase advances from 0.640 to 2.429 to 4.166 rad. The 3.526 rad advance over
18 um corresponds to an effective-index splitting of 0.0483 and a 16.0 um
transfer length, independently matching the 15.8 um full-field estimate. The
4.0--4.7% projection residual and unit condition number make fit instability
an unlikely explanation. The coupling section therefore preserves modal power
while the unexpectedly rapid relative phase advance changes its interference
at the output.

The refined modes are also strongly detuned: their electric-field localization
is approximately 12% and 85% in the wide guide. A two-mode estimate therefore
limits ideal transfer to about 46% before bend, taper, and power-normalization
effects. Static refined-mode checks are nonmonotonic across 6, 8, and 10 ppw,
with estimated transfer lengths of 17.3, 18.7, and 17.6 um. This is consistent
with the paper's report that the converter is unusually sensitive to mesh size
and to shifted spectral peaks and valleys. A mesh-placement sweep is the next
diagnostic; simply changing the absorber or increasing the mode count is not
expected to resolve the mismatch.

## Single-bus ring resonator

Issue #104's maintainer audit adds a sixth device beyond the five cases reported
in the paper: the supplementary repository's default `ring_single`. The case
pins its GDS at the same source revision and runs the repository's lowest
setting of 6 cells per wavelength over 1540--1560 nm with 0.2 nm sampling.

```console
uv run pytest tests/differential/test_ring_resonator.py \
  --validation-report=validation-results-ring-6ppw.json
```

The supplementary repository does not commit Lumerical or Tidy3D ring results,
so this case does not invent a cross-solver target. It records the complex TE0
through and reflection spectra, resonance wavelengths, median free spectral
range, loaded Q, runtime, grid size, and terminal field-decay ratio. The
field-decay ratio must reach the repository's `1e-5` auto-shutoff threshold
before any passivity or resonance metric can be accepted. The hardware test is
a strict expected failure at the pinned runtime, so an unexpectedly converged
run also requires review and removal of that marker before it can count as a
passing validation.

The 6 ppw CUDA run contains 3,615,840 cells and identifies the dominant TE0
resonances at 1541.78, 1549.14, and 1556.77 nm. Their median spacing is 7.49 nm;
the deepest dip has a sampled loaded Q of 740 and the normalized through-port
extinction is 7.71 dB. The 24,199-step run reached its 3.20 ps time limit with a
remaining field-decay ratio of 0.121. These resonance values are retained as
lowest-setting characterization only; the ring is not accepted as a converged
validation result.
