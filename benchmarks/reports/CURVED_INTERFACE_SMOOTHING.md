# Curved interfaces: full-tensor smoothing on coarser meshes, and its memory

Measured 2026-10-12/13 on a local RTX 5070 Laptop GPU (8 GB; 6.87 GB usable at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`), driver 580.173.02, Python 3.11.15, JAX 0.9.0,
JAX x64 enabled (native raster/material arrays stay float32). Code: BeamZ v0.5.2, pinned at `8ab968ac`, with the two setup-memory
patches below. "Dev today" means `main` at `80ef0182` (2026-10-13), also the base of
`feat/cuda-full-autodiff`. Its relevant code paths match the pinned revision, but they were
not re-measured there.

## Finding

On curved walls, `smoothing="farjadpour_full"` reaches a given accuracy on a **2× coarser
mesh** than the default diagonal ("polarized") averaging. In the 3D bend below this is
**13.2× faster** and needs **5.8× less GPU memory** at comparable accuracy. A setup-only fix, which stores uniform
CPML constants as scalars, produces the measured memory reductions for every method. It also
lets a fine full-tensor run fit on an 8 GB GPU.

Diagonal averaging remains the right default for grid-aligned walls. It is exact there:
straight-guide controls agree with full tensor to roundoff. For coarse targets (≳1.2 pp
here) it is also the fastest option.

## Why: what each method assigns to a cut cell

At a flat wall the field component along the wall sees the arithmetic mean of ε, and the
component across it sees the harmonic mean. The exact cell tensor is therefore
`ε = ε_harm·n nᵀ + ε_arith·(I − n nᵀ)`, aligned with the wall normal **n**
(Farjadpour et al., Opt. Lett. 31, 2972 (2006); Kottke et al., PRE 77, 036611 (2008)).

The study notebook (see Reproduction) illustrates this: a permittivity ellipse
per method in a cut cell, plus the samples each Ex update reads.

| | `volume` | `farjadpour_diagonal` (default) | `farjadpour_full` |
|---|---|---|---|
| Cut-cell value | arithmetic mean, all directions | diagonal of the exact tensor | full exact tensor |
| Exact for a flat wall at | no orientation | grid-aligned walls only | any orientation |
| Ex update reads | Dx at the same point | Dx at the same point | Dx plus neighbouring Dy, Dz |
| Measured cost per cell | 1× | 1× | ~1.45× time, ~1.5× memory |

A bend sweeps the wall normal through every in-plane angle. Diagonal averaging drops the
ε_xy terms there, so it needs a finer mesh to converge.

## Workload

- **Structure:** a 0.5 × 2 µm SiN core (n = 2.0) in SiO2 (n = 1.44), with an R = 20 µm, 90° bend at λ = 1.55 µm.
- **Domain:** 31 × 34 × 10 µm with CPML, run for 800 fs.
- **Source and observable:** a Gaussian mode-0 source. The observable is mode-0 power 6 µm after the bend, normalised by the measured incident modal power and jointly projected onto six bound modes.
- **Grids:** uniform, at 125, 100 and 62.5 nm.
- **Straight controls:** straight-guide controls were run for each method and grid.
- **Timing:** synchronized GPU execution, excluding setup and JIT. Rows with four timed runs report the median of three warm repeats, with a spread of at most 0.15 s.

The **reference** is the 62.5 nm full-tensor run of the same solver. Its screening margin is
**0.74 pp**: twice the refinement change, plus incident-mixture and straight-control
indicators. Because the reference uses full tensor, it favours full tensor. Volume (99.3 →
97.7%) and diagonal (96.5 → 96.1%) both move monotonically toward it as the mesh is refined.
There is no independent 3D scattering reference. pp = percentage points of transmission.

## Measurements

| Smoothing | Grid | Cells | Error vs reference | GPU time | Peak GPU, dev today | Peak GPU, with fixes |
|---|---:|---:|---:|---:|---:|---:|
| volume | 125 nm | 5.4 M | 4.00 pp | 9.1 s | 0.84 GB | 0.56 GB |
| volume | 100 nm | 10.5 M | 3.54 pp | 27.8 s | 1.60 GB | 1.10 GB |
| volume | 62.5 nm | 43.2 M | 2.37 pp | 174.3 s | 6.15 GB | 4.30 GB |
| diagonal (default) | 125 nm | 5.4 M | 1.15 pp | 9.0 s | 0.85 GB | 0.54 GB |
| diagonal (default) | 100 nm | 10.5 M | 1.12 pp | 27.9 s | 1.65 GB | 1.07 GB |
| diagonal (default) | 62.5 nm | 43.2 M | 0.76 pp | 174.2 s | 6.15 GB | 4.30 GB |
| full tensor | 125 nm | 5.4 M | 0.32 pp | 13.2 s | 1.75 GB | 0.74 GB |
| full tensor | 100 nm | 10.5 M | 0.09 pp | 39.6 s | not measured | 1.59 GB |
| full tensor | 62.5 nm | 43.2 M | reference | 251.1 s | **out of memory** | 6.38 GB |

| Need | Dev today | With the fixes | Gain |
|---|---|---|---|
| ~1.2 pp, defaults | diagonal 125 nm: 9.0 s, 0.85 GB | same run: 9.0 s, 0.54 GB | 1.6× less memory |
| ≤ 0.8 pp, default smoothing | diagonal 62.5 nm: 174 s, 6.15 GB | full 125 nm: 13.2 s, 0.74 GB | 13.2× faster, 8.3× less memory |
| ≤ 0.8 pp, already using full | full 125 nm: 13.2 s, 1.75 GB | full 125 nm: 13.2 s, 0.74 GB | 2.4× less memory |
| Fine full-tensor run, 8 GB GPU | out of memory | 251 s, 6.38 GB | now possible |

Like-for-like, with the fixes applied to both methods, full at 125 nm against diagonal at
62.5 nm is 13.2× faster with 5.8× less memory. The time gain comes from choosing full tensor,
which dev already offers. The fixes contribute the memory reduction.

## The memory fixes

Both fixes change setup only, not physics:
- volume and diagonal give bit-identical guided powers;
- full-tensor transmission moves by ~1e-6 (float32 operation order);
- bend run times match the earlier runs to ≤ 0.1 s (one single-run straight control varied by 1.9 s).

For full tensor, the timing comparison is against runs with fix 1 only, not fully unpatched runs.

1. **Uniform CPML constants** (`beamz/devices/_boundary_compile.py`). This produces the measured savings.
   Spatially uniform conductivity/permeability, and the matching `sig_*`/`mu_*` Yee arrays, are
   stored as exact 0-d scalars for 3D CPML grids. The patch applies cleanly to dev. At a coarse
   500 nm check grid, CPML arrays, profiles and masks are bit-identical after broadcast, and
   stored boundary material bytes fall from 12.6 to 4.2 MB.
2. **Inverse-tensor upload** (`beamz/simulation/compile.py`, full tensor only). Each electric
   support uploaded all six inverse-tensor components and then kept one (three at nodes).
   Selecting them on the host first cuts that upload payload 4×. On its own it did **not**
   lower the measured peak: full@125 measured 1.75 GB unpatched and 1.79 GB with this fix
   alone. Its contribution to the 62.5 nm success is unmeasured, because the CPML fix has not
   been run without it. It needs a small rebase onto dev.

Without either fix, the 62.5 nm full-tensor bend fails during setup on a 998 MiB allocation.
With fix 2 alone it still fails, in boundary lowering on a 166 MiB allocation. With both
fixes it peaks at 6.38 of 6.87 GB. A 50 nm reference would need about 12.5 GB.

## Limitations

- **Single structure and observable.** This is one bend, measured on mode-0 power. Absolute phase, radiation accounting and other geometries are not measured.
- **Same-solver reference.** Within its 0.74 pp margin, full@125 (95.00%) and diagonal@62.5 (96.08%) are not separable. "At least as accurate" is the defensible claim.
- **Laptop GPU and pinned revision.** Memory is the JAX peak in use, measured on pinned v0.5.2; it needs re-measuring on dev before merging.
- **Fix 2 is unproven.** It needs its own measurement before it is included.
- **Not a new sampling scheme.** The 2D published eight-triplet update (Werner et al., arXiv:1212.4857) showed a 7.57× equal-accuracy speedup on a ring cavity, but it has no open-boundary 3D implementation, so it is not part of this comparison.

## Reproduction

The benchmark scripts, raw per-case JSON and notebook are **not part of this repository**. They
are on the `cdaunt/beamz` fork, branch `feat/fit-interface-prototype`, as working files, and
will move to an examples repository. The notebook is `docs/explicit_anisotropic_experiment.ipynb`;
the runner is `scripts/run_sin_gpu_study.py`. They need a pinned upstream checkout with its
native extension built:

```bash
.venv/bin/python scripts/run_sin_gpu_study.py --beamz-root <pinned-beamz-checkout> \
  --output-dir <results-dir> --report-name <name> \
  --compact-inverse-coefficients --compact-boundary-constants --compact-boundary-scope all \
  --extra-repeats bend:farjadpour_full:125 bend:farjadpour_diagonal:62.5 --fresh
```

These numbers will be re-measured on current `main` before the memory fix is proposed for merging.
