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
