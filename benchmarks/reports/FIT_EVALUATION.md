# Finite Integration Technique (FIT) backend: evaluation summary

Recorded 2026-09-30 to 2026-10-13 on the `cdaunt/beamz` fork, branch
`feat/fit-interface-prototype`, which has an experimental `beamz/simulation/fit/` prototype.
The goal was **better accuracy at the same wall time as Yee FDTD, or the same accuracy
faster**. Equal per-cell throughput was not required.

## Finding

**For photonics, a separate FIT backend is not justified. For RF/microwave (microstrip and
similar), FIT is worth a gated follow-up stage**; see below.

**Photonics.** In every matched test, FIT's
accuracy gain came from **material sampling or mesh alignment**, and a Yee update given the
same sampling or metrics reproduces it. The coupled (tensor) FIT operator improves interface
accuracy but costs about 8× per step. At equal accuracy it was 50–70× slower than polarized
FDTD in 3D.

**Where FIT-style ideas do help** (and transfer to Yee):
- **Exact interface fractions.** In a 2D cavity, accuracy at a 1% target came 5–6.5× faster.
- **Layer-aligned or graded rectilinear meshes for planar structures** (e.g. microstrip).
  The same accuracy comes with far fewer cells.
- **Full-tensor treatment of curved walls.** See
  [CURVED_INTERFACE_SMOOTHING.md](CURVED_INTERFACE_SMOOTHING.md); the native
  `farjadpour_full` option does this without a new backend.

## Measurements

All comparisons hold geometry, materials, physical duration and observable fixed. Errors are
measured against independent references: fitted FEM (Netgen/NGSolve) or analytic transfer
matrices. Times are warm and exclude preparation and compilation.

**2D TM dielectric cavity, six interface geometries** (JAX, CUDA). Fastest grid reaching a
1% worst mode error, summed over the six cases:

| Method | Warm total | Speed-up vs current FDTD |
|---|---:|---:|
| Current FDTD (native sampling) | 1.193 s | 1.00× |
| FDTD with exact dual-node fractions | 0.241 s | 4.95× |
| FIT, staircase | 0.209 s | 5.72× |
| FIT, exact primal fractions | 0.184 s | 6.49× |

At 20 cells/axis, FDTD and FIT with the *same* dual fractions both give 0.678% error. The
gain is the material sampling, not the FIT time step.

**3D dielectric cavity, six planar interfaces** (float64). Fastest grid summed over the cases:

| Error target | Polarized diagonal FDTD | Polarized full FDTD | Coupled FIT |
|---|---:|---:|---:|
| 1% | 0.448 s | 0.677 s | 31.8 s |
| 0.5% | 0.869 s | 1.158 s | 53.9 s |
| 0.2% | 6.07 s | 9.45 s | 307.6 s |

At 62.5 nm the worst-case errors are similar: 0.178% diagonal, 0.140% full, 0.168% coupled
FIT. Coupled FIT's global constitutive inversion per step dominates its cost. On coarse
high-contrast oblique pilots it was more accurate (e.g. 0.55% vs 1.62% at 0.25 µm), but that
did not translate into a speed advantage.

**Microstrip quasi-TEM, layered cross-section** (CPU, against a fitted vector FEM reference
with h-drift of 0.012%):

| Method | Cells | All-mode error | Warm CPU |
|---|---:|---:|---:|
| Yee uniform, native sampling, 0.1 µm | 7,200 | 0.27% | 0.088 s |
| FIT x/z graded, 0.1 µm | 1,598 | 0.65% | 0.037 s |
| Yee x/z graded, same metric law | 1,598 | 0.65% | 0.029 s |
| FIT layer-aligned, 0.035 µm | 60,030 | 0.12% | 9.89 s |
| Yee layer-aligned, same metric law | 60,030 | 0.12% | 8.95 s |

An independently coded matched nonuniform Yee update reproduces FIT's operator and eigenmode
evolution. It gives the same error and storage, and is slightly faster on CPU. Graded meshing reaches a 1% target about 3× faster than
the uniform raster. At 0.2%, only the layer-aligned meshes qualify. This is the "good in some
cases" regime: **planar, layered, electronics-style structures**. The benefit comes from the
mesh, not the FIT formulation.

## Recommendation for photonics

- Don't add a separate FIT engine for optical work. Keep the prototype as a reference and test oracle.
- Bring the transferable pieces into Yee FDTD:
  1. full-tensor interface smoothing for curved photonics, with its memory fixes;
  2. exact interface fractions where native sampling is approximate.
- Revisit coupled FIT or published explicit tensor updates only with an open-boundary 3D
  implementation that beats `farjadpour_full` at equal accuracy. The 2D eight-triplet update
  (Werner et al., arXiv:1212.4857) was 7.57× faster at equal accuracy on a 2D ring cavity,
  but has no 3D open-boundary version yet.

## Follow-up stage: FIT for RF and microstrip

The photonics tests could not show FIT's main RF strengths. On a Cartesian grid, FIT and Yee
are algebraically equivalent, which is why matched Yee controls reproduced every result so
far. FIT is different in what it makes natural:

- **Circuit quantities are the unknowns.** FIT stores edge voltages (∫E·dl) and face fluxes,
  so voltage/current ports, lumped R/L/C elements, impedance extraction and SPICE-style
  coupling are integrals over existing degrees of freedom, not post-processing.
- **Metrics live in the operator.** Graded, layer-aligned rectilinear meshes, which suit thin
  substrates, metal thickness and small gaps, use exact primal/dual lengths and areas. The
  prototype already has this (`RectilinearFITMesh`, `insert_features`, `LayeredDielectric`,
  grid-aligned PEC boxes).
- **Conformal metal.** Commercial RF FIT codes rely on conformal or partially filled PEC cells
  (e.g. the perfect boundary approximation) for oblique and curved traces, bond wires and
  vias. Yee needs the same, but FIT's metric form is where it is usually formulated.

**What is already validated in the prototype:**
- quasi-TEM microstrip Z0 and ε_eff on layer-aligned/graded meshes: 0.14% Z0 error at
  17.5 nm, against an FEM reference with drift below 0.02%;
- layered TE/TM cavity frequencies;
- grid-aligned PEC solids;
- energy/divergence conservation.

**Stage plan.** Every step needs a matched Yee control (same mesh, metrics and dt). FIT is
credited only where it is more accurate at equal cost or adds a capability Yee lacks.

| Step | Deliverable | Acceptance gate |
|---|---|---|
| 1 | Voltage/current (lumped) ports on edge integrals; broadband S11/S21 for a microstrip line with CPML | Z0 and S-parameters within an agreed tolerance of FEM/analytic and a commercial reference (e.g. CST); port reflection below a set level |
| 2 | Graded-mesh production path: CPML, sources, monitors and mode/port solver on nonuniform grids | Grading-transition reflection and convergence sweeps; wall time to target S21 error vs uniform Yee |
| 3 | Conductor loss: finite-conductivity / surface-impedance metal, thin sheets | Attenuation constant vs a reference over frequency |
| 4 | Conformal PEC for oblique or curved traces, with a published stable cut-cell scheme | Accuracy vs staircase PEC at equal wall time; no timestep collapse from small cells |
| 5 | Lumped R/L/C elements and dispersive substrates (Debye/Lorentz) | Filter or resonator response vs reference; passivity |
| 6 | Decision | Keep FIT as the RF engine only if steps 1–4 beat matched Yee on accuracy per wall time, or Yee cannot express the capability cleanly. Otherwise move the pieces into Yee, as for photonics. |

Suggested benchmark suite: a straight 50 Ω microstrip (S-parameters, Z0, loss), a microstrip
step/gap discontinuity, an oblique or curved microstrip bend, a via transition, and a coupled-line
or stub filter.

## Limitations

- **Benchmark adapters, not production code.** These are closed cavities, periodic/PEC microstrip and JAX prototypes, not the production port/PML pipeline or the native CUDA kernels.
- **CPU timings don't carry over.** CPU stencil timings do not predict GPU throughput.
- **Unmeasured areas.** Dispersion, loss, broadband S-parameters, ports and peak memory were
  not measured for FIT. These are exactly the RF stage's gates.
- **No RF verdict yet.** The microstrip result is a periodic/PEC quasi-TEM cross-section, not
  a driven RF simulation. It shows the mesh benefit, not FIT's RF value.
- **Screening, not certified bounds.** Accuracy targets use the references' observed refinement drift as a screening margin; they are not certified error bounds.

## Sources and reproduction

Detailed reports, raw JSON, notebooks and scripts are **not part of this repository**. They are
working files under `docs/` and `scripts/` on the `cdaunt/beamz` fork, branch
`feat/fit-interface-prototype`, and will move to an examples repository. The FIT prototype is
`beamz/simulation/fit/` on that branch. The detailed reports are:
- `fit_performance_exploration.md` (2D cavity, tracker);
- `fit_3d_benchmark_results.md` (3D cavity);
- `fit_microstrip_fdtd_fit_results.md` and `fit_layered_validation.md` (microstrip);
- `explicit_anisotropic_results.md` and `open_waveguide_results.md` (published explicit updates).

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_matched_cavity.py --platform cuda
.venv/bin/python scripts/benchmark_matched_cavity_3d.py --reference-dir docs/fit_3d_references --timing
.venv/bin/python scripts/benchmark_microstrip_fdtd_fit.py --reference-dir docs/fit_microstrip_mode_references
.venv/bin/python scripts/fit_benchmark_tracker.py
```

Each source report lists its exact options and reference-generation steps.
