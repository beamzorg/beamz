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

The default MMI hardware case uses 6 cells per wavelength and a 20 nm source
bandwidth; the experiment builder additionally supports 10/15/20/25 PPW. Run its geometry checks and lowest-resolution comparison with:

```console
uv run pytest tests/differential/test_mmi2x2.py \
  --validation-report=validation-results-mmi2x2-6ppw.json
```

The comparison records same-resolution reference agreement and checks the
passive-device output-power bound across the sampled band.

## Mode converter and polarization splitter rotator

Both conversion hardware cases default to **6 cells per wavelength** with a
**20 nm source bandwidth**; the experiment builders support 10/15/20/25 PPW. Run their geometry checks and simulations with:

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

Conversion comparisons record same-resolution agreement and only evaluate
converged-reference agreement at eligible resolutions. A selected-output bound
checks for normalization or passivity problems across the wavelength band.

## Single-bus ring resonator

The paper's sixth device is the supplementary repository's default
`ring_single`. The case pins its GDS at the same source revision and runs the
lowest published setting of 6 cells per wavelength over 1540--1560 nm with
0.2 nm monitor sampling.

```console
uv run pytest tests/differential/test_ring_resonator.py \
  --validation-report=validation-results-ring-6ppw.json
```

The benchmark uses the upstream ring script's default Lumerical silicon runtime
of `30 * domain_x * 2 / c` (6.40 ps for this domain). The upstream Tidy3D helper
uses half that duration. Section 3.6 and Figure 26 publish Lumerical and Tidy3D
ring results at 6, 10, 15, 20, and 25 cells per wavelength. Following the
paper's analysis script, the benchmark cubically interpolates the 0.2 nm
samples to approximately 0.02 nm, measures the first complete resonance in
wavelength order at half depth, and computes `Q = wavelength / FWHM`. At 6 PPW
the paper reports a 0.84 nm Lumerical FWHM and Q of 1839.4; Tidy3D's FWHM lies
in its reported 0.88--0.90 nm range and its series starts at Q 1756.5. The
paper concludes that the ring requires 20 PPW for mesh convergence.

The case records the complex TE0 through and reflection spectra, resonance
wavelengths, median free spectral range, lowest-resonance FWHM and Q, runtime,
grid size, and terminal field-decay ratio. The
field-decay ratio must reach the repository's `1e-5` auto-shutoff threshold
before any passivity or resonance metric can be accepted. The hardware test is
a strict expected failure at the pinned runtime, so an unexpectedly converged
run also requires review and removal of that marker before it can count as a
passing validation.

## Investigation controls and interpretation (PR #245)

The four new device builders accept `ExperimentOptions` for explicit duration,
monitor offset/aperture, absorber thickness, source-profile count, frequency
sampling, and smoothing experiments. Defaults preserve the pinned setup;
6/10/15/20/25 PPW are now supported for controlled refinement runs. Enabling a
resolution is not evidence that it has been validated.

The MMI and conversion hardware tests now check physical output bounds and
record same-PPW agreement separately. Below the manifest's convergence region,
`reference_agreement` is null: a passing coarse-grid validity check is not a
passing paper-reproduction claim. Eligible resolutions compare against the
range of the published converged samples plus digitization uncertainty. A
single eligible point still does not establish BeamZ mesh convergence.
The older crossing/coupler comparisons retain their historical rule.

Expected failures are restricted to the known output-power or field-decay
assertion; unrelated exceptions are ordinary failures. Artifact files additionally
retain complex incoming/outgoing modal amplitudes and all realized grid edges.
AI-assisted implementation and investigation: OpenAI Codex.
