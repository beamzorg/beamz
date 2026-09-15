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
