# PR #245: origin/main refresh on the local RTX 3090

Merged `origin/main` at `1ccaa587` into PR #245, starting from `24a8c1ee`.
All benchmark FDTD runs in this campaign used the **local NVIDIA RTX 3090**, float32
fields, the pinned reference geometry, and unchanged acceptance thresholds.
No remote GPU was used. Complete paper reproduction and mesh/time convergence
remain unresolved.

## What changed

- Reconciled the PR's absolute monitor clock with main's streamed timing support.
  Ordinary continuation chunks share one integer-indexed time origin; an explicitly
  offset continuation state retains its supplied time. CPU and native CUDA tests
  check accumulated complex DFTs across chunks, not only the returned clock.
- Extended main's material-preparation reuse to dense, single-device simulations.
  A short final chunk reuses the existing material, coefficient, metric, and
  boundary banks. Source/monitor plans are rebuilt; changed physics invalidates
  reuse. Tests cover complete-state parity, source/monitor changes, and changed
  materials, timestep, and boundaries.

For the matched **15-PPW MMI**, the complete retained `monitor_data.npz` hash is
identical before and after reuse. Peak JAX allocation fell from **6.32 to 4.79 GiB
(24.2%)**. Stepping time was essentially unchanged, **172.23 vs 172.19 s**;
end-to-end elapsed time was **347.30 vs 259.76 s**. Host peak RSS did **not**
improve (**10.01 vs 10.39 GiB**). These are one sample per revision, not a repeated
performance gate; JAX allocation statistics exclude native CUDA allocations.
See [the paired record](preparation-reuse-comparison.json).

The automatic-termination loop now builds normalized results once after its last
chunk. Every check still inspects the same raw DFT vectors and field energy;
source-off, nonfinite, growth, decay, and monitor-stability criteria are unchanged.
Tests cover field-only and monitor-based convergence, both with and without
performance timing. The paired 6.4-ps ring has an identical complete raw-monitor
hash and the same
stopping step/reason. End-to-end time fell from **121.58 to 84.06 s (30.9%)**,
while stepping stayed **36.90 vs 36.98 s**. This one-sample comparison includes
both dense reuse and deferred result construction; it does not isolate their
individual speedups. Peak-energy/decay reductions differ by about `1e-7` relative,
consistent with float32 rounding. See [the paired record](deferred-result-comparison.json).

## Device results

Target-mode power is sampled exactly at 1550 nm, with the pinned 20-nm bandwidth.
The three non-ring devices retain six frequency samples (the protocol's five
plus exact center). Decay uses the unchanged `1e-5` gate; selected-output power
uses the unchanged `1.02` bound at those retained samples. These gates are not a
complete all-mode energy balance or proof of agreement with another solver.

| Device | PPW | Target-mode power | Terminal energy ratio | Stepping time (s) | Peak host RSS (GiB) |
|---|---:|---:|---:|---:|---:|
| MMI | 6 | 37.4274% | 2.14e-09 | 5.67 | 2.86 |
| MMI | 10 | 45.9778% | 9.74e-10 | 34.00 | 4.78 |
| MMI | 15 | 48.0324% | 6.28e-10 | 172.19 | 10.39 |
| MMI | 20 | 48.5343% | 5.16e-10 | 519.45 | 18.61 |
| Converter | 6 | 21.1471% | 2.5e-07 | 35.65 | 4.87 |
| Converter | 10 | 28.9626% | 4.07e-08 | 208.75 | 11.94 |
| Converter | 15 | 45.9416% | 1.19e-08 | 1107.65 | 25.05 |
| PSR | 6 | 85.2446% | 1.12e-08 | 32.87 | 4.29 |
| PSR | 10 | 89.1934% | 9.46e-09 | 234.95 | 9.58 |
| PSR | 15 | 91.2069% | 8.46e-09 | 1044.34 | 23.45 |

All completed MMI, converter, and PSR runs pass those decay and selected-output
gates. MMI and converter at the previously completed meshes reproduce the prior
corrected center powers to the reported precision. The new **20-PPW MMI yields
48.5343%**, inside both the same-PPW 48.4–48.6% range and the declared converged
47.8–49.0% reference envelope. The 15→20 PPW change is **0.502 percentage points**,
smaller than the 10→15 PPW change of 2.055 points. This supports the reference
comparison but does not establish a full mesh-convergence study.

The new **15-PPW PSR yields 91.2069% conversion**, up 2.013 percentage points
from 10 PPW, and lies within the same-PPW 90.1–94.0% reference range. It remains
below the 20-PPW minimum for the stricter 94.6–95.2% converged reference gate.
Its completed peak host RSS was **23.45 GiB**; no 20-PPW PSR run was attempted.

The final **15-PPW converter yields 45.9416% conversion**, inside both the
same-PPW 44.3–49.2% range and the declared 42.9–51.4% converged reference envelope.
Its 10→15 PPW change is still **16.979 percentage points**, so this one eligible
reference match is not a mesh-convergence demonstration. It completed with
**25.05 GiB peak host RSS** after the second cache-relocation pass. No 20-PPW
converter run was attempted.

The [pinned manifests](../../cases/) and [previous comparison](../pr245-controls/README.md)
retain the published Lumerical/Tidy3D series. Equal nominal PPW is not an
identical realized mesh. No tolerance or reference geometry was tuned to pass.

### PSR change after main

At 6 PPW, current transmission is 85.2446%, versus the previous corrected
85.5344%; at 10 PPW it is 89.1934%, versus 89.2234%. Reprojecting the old 6-PPW
fields with the current analysis gives 85.5400%, so extraction alone explains
little of the shift. Grid edges and sampled source waveforms are identical.

A direct material audit finds 2,592 changed Ex-permittivity samples and 2,441
changed Ey samples (less than 0.08% of each component), with maximum absolute
changes of 0.6984 and 2.1756; Ez differences are roundoff-sized. Thus the old and
new runs do not have identical discrete constitutive arrays. This is consistent
with main's rasterization corrections, but is not a full solver-isolated causal
attribution of the power shift. The audit reused the previous checkout's native
raster binary and records its hash; it did not independently rebuild that old
binary. See [material audit](psr-material-audit.json),
[sparse cell differences](psr-material-differences.npz), and
[old-field reprojection](psr-reprojection.json).

## Ring duration and matched bus

Both duration caps use the same pinned 6-PPW reference geometry and retain 102
frequency samples. The pinned short case remains an expected failure; its decay
criterion has not been weakened.

| Cap (ps) | Terminal energy ratio | Max selected output | Max input reflection | FWHM (nm) | Loaded Q | Stepping time (s) |
|---|---:|---:|---:|---:|---:|---:|
| 6.4 | 0.0511 | 1.27324 | 7.15% | 0.94094 | 1638.5 | 36.98 |
| 51.2 | 6.44e-07 | 0.99649 | 71.09% | 0.48048 | 3208.7 | 295.37 |

The matched 6.4-ps empty bus has maximum transmission error **0.2325%**, maximum
input reflection **0.0454%**, and terminal energy ratio **2.54e-09**. It passes the
existing 1% transmission control. `ring-bus-matching.json` checks all grid edges,
evaluation frequencies, and sampled source arrays against the short ring run.

The optimized 51.2-ps run completed in **370.39 s** end to end, including
**295.37 s** of GPU stepping. The superseded unoptimized attempt had not produced
a result after 2249 s; that interrupted run is not a completed timing baseline.

The long-duration linewidth/Q remain provisional. Published first-resonance
values are about 0.84–0.90 nm and Q 1756.5–1839.4, so the long BeamZ run does not
reproduce them. Strong resonant reflection remains unresolved. Rebuilding the
three broadband profile drives for a different FFT record changes the drive;
these separately rebuilt duration caps are not a fixed-source convergence
experiment. The existing 0.2-nm frequency spacing and extraction limitations also
remain. Passing decay at one endpoint does not establish duration convergence.

## Capacity attempts

The table retains every interrupted attempt. The memory guard stopped processes
when available host RAM fell below 1.5 GiB, preserving the interactive desktop.
The original long ring was superseded by the optimized run after 37.5 minutes;
its interruption is not an out-of-memory failure. These attempts produced
**no benchmark result**. The sampled RSS values are lower bounds on the attempted processes'
peaks, not measurements of the RAM needed to finish. The host has 30 GiB total
and other applications remained running. The first MMI attempt overlapped CPU
validation; the retry ran without concurrent validation.

| Attempt | Cells | Maximum polled RSS (GiB) | Stop reason |
|---|---:|---:|---|
| mmi-20 | 35,250,600 | 13.12 | host_memory_guard_1.5_GiB_available |
| converter-15 | 45,813,600 | 14.67 | host_memory_guard_1.5_GiB_available |
| psr-15 | 47,545,344 | 14.39 | host_memory_guard_1.5_GiB_available |
| mmi-20-reused | 35,250,600 | 14.45 | host_memory_guard_1.5_GiB_available |
| ring-51p2 | 3,615,840 | 4.20 | superseded_by_deferred_result_construction |
| converter-15-headroom | 45,813,600 | 21.95 | host_memory_guard_1.5_GiB_available |

Two passes copied inactive generated Python environments and Rust build caches
with checksum verification from RAM-backed `/tmp` to SSD: the first before the
headroom retries, and the second during the PSR run before the final converter
attempt.
Original paths remain usable through linked dependency/build directories, and the
source worktrees stay clean. This recovered several GiB of RAM without
removing source or simulation data. See [the relocation record](ram-reclamation.json).
The `headroom` attempts record their starting available RAM and swap usage.
No result is inferred from an interrupted run.

## Validation and provenance

The final focused CPU run passes **209 tests**, including the dense-preparation,
regional-preparation, automatic-termination, clock, engine, modal, and device
protocol checks. **22 single-GPU CUDA checks**, static checking, formatting, and
package build results are recorded in [validation.json](validation.json) and its logs.
At source head `c5907b6e`, the required CI trust gate passes. The evidence job
reports **2,025 passed, 1 skipped, 480 deselected, and 1 expected failure**, with
**86% coverage**. Python 3.10/3.12/3.13/3.14 compatibility, strict documentation,
packaging, CPU sharding, quality, and CUDA 12 component-build checks also pass.
The H100 hardware job was skipped. See [the source-head CI snapshot](ci-source-checks.json)
and [evidence summary](ci-evidence-summary.txt).
These local checks are not the entire repository suite or multi-GPU/Hopper
validation.

The initial CI compatibility run exposed three layout tests that unconditionally
required optional GDSFactory. Those tests now follow the existing absence-only
skip policy; the other experiment tests still run. Local checks pass **11/11 with
the extra**, and **8 pass / 3 skip with its import blocked**. A broken installed
dependency still fails. The broader evidence suite also exposed two assertions
comparing deembedded modal power with interpolation-attenuated raw flux. The
integration test now retains the **2% physical power propagation check** and
compares sampled reconstructed flux against raw monitor flux at **both planes**,
with the same **2% tolerance**. All four grid/direction cases pass. These test-only
fixes change no solver, extraction, or benchmark setting.

[runs.json](runs.json) retains every completed run, its executing revision and
working-tree patch hash, spectra, runtime, memory, decay, and raw-field hash.
[attempts.json](attempts.json) retains the supervised commands and stopped runs.
[spectra/](spectra/) contains compact complex S parameters, modal amplitudes,
projection diagnostics, flux checks, and grid edges. Full raw DFT fields and
geometry/field PNGs remain in the absolute local paths recorded in `runs.json`;
they are not all committed. `collect.py` verifies raw hashes before collection.

Early runs precede the final clock/reuse commits; their exact revisions and
nonempty [source patches](source-patches/) are retained byte for byte as gzip files. The matched MMI run establishes bitwise
unchanged raw observations for the reuse change. The typing-only commit adds
Python `cast()` annotations and changes no array computation. Subsequent runs include deferred result construction (`652d0d06`).

[environment.json](environment.json) records package versions, GPU/driver,
CUDA ABI 21, and the rebuilt native extension hash. The optional extension was
compiled from this checkout for SM86 with CUDA 13.3 and precise math. CI also
passed the CUDA 12 component build; local runtime behavior was not
revalidated against the oldest supported toolkit. `requirements.txt` records
the local environment, including its editable checkout path. Earlier PR-control campaign
timings involved GPU sharing and are not controlled speedup baselines. Some
current CPU preparation/analysis overlapped CPU-only checks; FDTD simulations
ran one at a time on the RTX 3090.

## Reproduction

Install this checkout and the CUDA component as described in
[the CUDA README](../../../../cuda/README.md). This campaign used JAX 0.9.0,
NumPy 2.4.1, SciPy 1.17.0, GDSFactory 8.18.2 and KLayout 0.30.6.
For this editable checkout, the freshly built optional CUDA extension was copied
from `.venv/lib/python3.11/site-packages/beamz/_cuda*.so` into `beamz/`, because
the editable package path did not include the installed component directory.
That generated binary is ignored, not committed; its hash is in `environment.json`.
Use a fresh output directory for each run:

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.investigate_passive_soi mmi2x2 --ppw 15 \
  --backend cuda_streamed --exact-center \
  --output validation-artifacts/pr245-repeat/mmi-15
```

For the ring add `--run-time-ps 51.2` and use `ring_resonator --ppw 6`;
for its matched short control use `ring_bus --ppw 6 --run-time-ps 6.4`.
Exact supervised commands are in `attempts.json`. Recollect compact evidence:

```sh
.venv/bin/python tests/differential/results/pr245-main-refresh/collect.py \
  validation-artifacts/pr245-main-refresh
```

AI-assisted merge resolution, implementation, tests, experiments, and analysis:
OpenAI Codex. Human review and ownership remain required by `CONTRIBUTING.md`.
