# Realistic CUDA temporal integration — RTX3090, 2026-09-18

Consistent 9 GCUPS across realistic domains remains unachieved. The bare periodic
16x8x16 two-step winner now runs inside the real backend, with mode sources,
12-cell CPML and monitors. Its coupling cost makes it slower than the established
field schedule on most tested realistic cases. It remains an explicit experiment.

The useful monitor optimization is separate: gather both timesteps once per
monitor point, share interpolation geometry, reuse samples across frequencies,
and update the DFT accumulators in chronological order. This improves large
monitor workloads; compact mode monitors were already a small fraction of time.
No dispatch defaults were promoted, and nothing was pushed.

## Implementation

- Replaced the old full-volume shared-memory two-step core with rolling H1, E1,
  H2 planes and direct E2 output. The default 16x8x16 tile uses 13,008 bytes of
  dynamic shared memory, plus source-intersection flags. Two wider variants are
  available for controlled comparisons.
- The initial field bank stays frozen while a third bank receives final fields.
  Only the coupling rim and conservative monitor regions publish intermediate
  H1/E1 fields. A 16x8x1 logical-region map includes every component and
  interpolation neighbor; padding never changes physical coordinates.
- Sources retain their H/E phase, waveform time and order at both substeps.
  Source/CTA intersection checks avoid per-cell source work on unrelated tiles.
  Grid-constant descriptors prevent the compiler's per-thread metadata copies:
  the corrected rolling kernel has 55 registers and zero stack bytes on SM86.
- A compact queue computes the two-cell core/CPML coupling band. CPML itself
  still evolves at every substep using its established recurrence kernels.
  **The absorber's auxiliary state is not temporally blocked.**
- Paired monitor processing prepares both phases, gathers both field states once
  per point, and reuses the cached samples across all frequencies. It preserves
  sampling intervals, masks, windows, normalization, odd tails and continuation.
- `BEAMZ_CUDA_PAIR_TILE=single` pairs monitor work while retaining the existing
  automatic field schedule, including the separate H/E path on narrow domains.
  This avoids the rolling core's coupling band.
- ABI 17 adds the optional publication-map input and paired-sample scratch output.
  The source-only path retains its prior arity. Scratch pointers and tile choices
  participate in native graph-cache keys.

The source is in [temporal_pair.cuh](https://github.com/beamzorg/beamz/blob/fa151d28/cuda/src/temporal_pair.cuh),
[program.cu](https://github.com/beamzorg/beamz/blob/fa151d28/cuda/src/program.cu), [io.cu](https://github.com/beamzorg/beamz/blob/fa151d28/cuda/src/io.cu) and
[runtime.py](https://github.com/beamzorg/beamz/blob/fa151d28/beamz/simulation/cuda/runtime.py).

## Measurements

All reported application cases use FP32 fields and CPML state, **12 CPML cells**,
lossless materials, a mode source and two monitors. Standard cases use fixed
mode-monitor apertures and three frequencies. Full-field controls and their
frequency counts are explicitly identified. The smooth-material case varies
permittivity without adding conductivity.

Each case builds one simulation and initial state, then compiles variants from
that same program and coefficients. Four warmups precede six interleaved rounds
covering every ordering of three variants. GCUPS counts logical material-grid
cells, excluding storage padding. Timings synchronize the complete compiled
execution, including field updates, sources, CPML and monitors. Compilation,
mode solving and setup are excluded. These are steady execution measurements,
not total public-API setup latency.

The 3090 is shared with the existing notebook and desktop/T3 processes. Benchmark
children run sequentially with a 2 GiB free-VRAM guard and an 85°C temperature
guard. The existing 370 W limit remained enforced; these are not higher-power
results. No H100 performance claim is made.

The final study contains **17 cases, 12 physical shapes, 51 variant executions
and 306 timed samples**. All use the same final native binary. Values below are
median GCUPS; paired speedup statistics are preserved in the raw JSON and CSV.

| Domain (z×y×x) | Case | Existing | Rolling two-step | Paired monitors |
|---|---|---:|---:|---:|
| 1024×256×64 | mode, 3 frequencies | 6.80 | 4.65 | 6.80 |
| 1024×64×256 | mode, 3 frequencies | 8.23 | 6.72 | 8.24 |
| 256×1024×64 | mode, 3 frequencies | 6.76 | 4.71 | 6.77 |
| 256×64×1024 | mode, 3 frequencies | 8.54 | 7.93 | 8.54 |
| 64×1024×256 | mode, 3 frequencies | 7.14 | 6.45 | 7.15 |
| 64×256×1024 | mode, 3 frequencies | 8.02 | 7.29 | 7.85 |
| 1024×256×64 | mode, 3 frequencies; 1024 presteps | 6.69 | 4.57 | 6.71 |
| 1024×256×64 | field, 3 frequencies | 5.18 | 3.80 | 5.33 |
| 1024×256×64 | field, 11 frequencies | 3.62 | 3.69 | 5.02 |
| 71×479×503 | mode, 3 frequencies | 7.94 | 7.26 | 7.94 |
| 97×289×593 | mode, 3 frequencies | 8.50 | 7.72 | 8.51 |
| 1024×256×64 | mode, 3 frequencies; 1024 steps | 6.54 | 4.47 | 6.55 |
| 128×256×512 | mode, 3 frequencies; smooth ε | 7.84 | 7.23 | 7.81 |
| 128×256×256 | mode, 3 frequencies | 8.36 | 6.79 | 8.41 |
| 128×256×512 | mode, 3 frequencies | 8.73 | 7.71 | 8.72 |
| 128×256×768 | mode, 3 frequencies | 8.90 | 8.06 | 8.89 |
| 256×256×256 | mode, 3 frequencies | 8.79 | 6.88 | 8.79 |

The monitor-heavy 11-frequency case improves by 38.9% in paired timing ratios;
the three-frequency full-field control improves by 3.3%. Compact mode-monitor
cases show little systematic benefit from pairing monitor work. The rolling
schedule is slower on all compact-mode cases in this study.

Observed free VRAM stayed above 5.25 GiB; maximum GPU
temperature was 79°C. GPU utilization reached 100%.
The largest domain has 25.17 million logical cells. These are saturated-domain
experiments, not tiny launch-bound benchmarks.

Raw data: [first ten cases](rtx3090-2026-09-18-realistic-temporal-final/),
[remaining seven cases](rtx3090-2026-09-18-realistic-temporal-final-tail/),
[combined CSV](rtx3090-2026-09-18-realistic-integration-validation/results.csv),
[verification summary](rtx3090-2026-09-18-realistic-integration-validation/summary.json).
The first run stopped at the dense-material exact-equality check. Only that
comparison's gate changed after the isolated FMA diagnostic; the native binary
remained unchanged for all seventeen successful cases.

Native SHA256: `f306491d5fbf657fcc1b7f866525428bdb83beeba36f515e90023f4fd7671198`.


## What is limiting performance

Compact mode-source and mode-monitor work is too small to explain the missing
throughput. The large costs are field evolution, CPML recurrence traffic and,
for the rolling schedule, the extra coupling work. Fusing more monitor launches
alone does not remove these costs.

Shape matters even at constant cell count. Ignoring the small Yee extent
adjustments, CPML occupies about 30% of a 128x256x512 domain and 45% of a
1024x256x64 domain. The two-cell coupling margin leaves only about 66% and 49%,
respectively, for the current deep-interior two-step kernel. Increasing total
volume does not remove a persistent thin dimension's surface cost.

The bare ceiling also omits staggered component pitches, material lookup,
source checks, intermediate publication and absorber synchronization. Its
27–32% temporal gain therefore cannot be applied directly to application GCUPS.
The rolling implementation saves transfers but adds halo computation and stage
synchronization. GPU timeline attribution identifies time spent in these
kernels; it does not establish a DRAM-stall diagnosis without hardware counters.

For large monitor planes, the old path repeated interpolation indices, weights
and scattered field loads for every frequency and timestep. Pairing accumulator
stores alone saved little. Caching the gathered samples removes that repeated
work and is the significant monitor improvement in this change.

Final GPU timeline shares explain the different outcomes:

| Case / schedule | Interior | CPML shell | Coupling band | Monitor kernels |
|---|---:|---:|---:|---:|
| 128×256×512 / existing | 57.1% | 41.8% | — | 0.6% |
| 128×256×512 / rolling | 49.6% | 36.3% | 13.2% | 0.4% |
| 64×256×1024 / existing | 41.7% | 57.2% | — | 0.6% |
| 64×256×1024 / rolling | 36.9% | 53.4% | 8.8% | 0.3% |

Source launches account for approximately 0.1–0.3% in these cases; fused source
arithmetic is included in the field kernels. For the 1024×256×64 full-field,
11-frequency case, monitor kernels fall from **45.4% to 24.3%** of summed GPU
duration with paired gathering. Its ordinary combined queue includes both the
interior and shell, so that queue's entire time cannot be labeled CPML.

An independent twelve-round repeat measured 3.63 → 4.96 GCUPS for that
monitor-heavy case. Compact-mode controls measured 8.471 → 8.470 GCUPS on
128×256×512 and 8.125 → 8.106 on 64×256×1024. The latter's roughly 2% paired-monitor
regression in the primary study did not repeat at that magnitude.

These shares sum GPU event durations from two profiled replays per variant;
they are attribution, not uninstrumented wall-clock timings or hardware-counter
measurements. See the [repeat and traces](rtx3090-2026-09-18-realistic-temporal-profiles/)
and [kernel shares](rtx3090-2026-09-18-realistic-temporal-profiles/kernel-shares.json).
Duplicate profiler exports and large XPlane protobufs were moved to the local
cache; archive paths and hashes are recorded beside the retained Perfetto traces.

## Correctness and experimental controls

Large-run rolling results are compared against the fused one-step CUDA
arithmetic; paired-monitor results are compared against the ordinary automatic
schedule. Packed-material rolling comparisons and all paired-monitor comparisons require
exact numerical array equality of the complete state. The dense-material rolling
case uses the existing hardware tolerance, `rtol=3e-5` and
`atol=3e-6 + 3e-6 * leaf_peak`. It is the only non-exact optimized comparison.
The separate H/E and fused one-step kernels have an existing FP32 rounding
difference: a 256-step diagnostic measured up to 8.6 ppm of a leaf's peak between
them, while the binary-material rolling result matched the fused reference exactly. The early
narrow-case comparison failure against the other arithmetic path was diagnosed,
not hidden by increasing its tolerance.

For smooth permittivity, an isolated diagnostic build forced the multiply/FMA
order in `AdvanceYeeField` without changing geometry or scheduling. That removed
all differences between rolling and fused execution. The production build keeps
its original arithmetic. On a seeded 61x73x97, 256-step diagnostic, the production
rolling/core discrepancy reached 9.2 ppm of a state leaf's peak; paired monitor
processing remained exactly equal to the unchanged field schedule. The long JAX
comparison also showed an existing DFT discrepancy of about 96 ppm of peak in
both CUDA schedules: JAX advances its FP32 clock by repeated addition, whereas
the native graph evaluates time from a base and timestep offset. This is not
counted as exact JAX parity or attributed to the new monitor cache.

Independent JAX comparisons cover seeded nonzero fields, binary and smooth
lossless materials, padding, tile tails, rotated/coincident sources crossing the
CPML interface, odd/even continuation and full-field/mode monitors. Paired DFT
checks additionally cover ragged monitor planes, different frequency counts,
intervals 1–4, Hann windows, normalization, masked fields and inactive windows.

Final validation on the unchanged production binary:

- Hardware suite: **166 passed**, two H100-only tests skipped, and two lossy
  cases excluded, as requested (311.90 s).
- Runtime/ABI unit suite: **33 passed** (3.90 s); generated ABI schema check,
  changed-Python Ruff checks and whitespace checks passed.
- Standalone Compute Sanitizer memcheck: **zero errors**, covering all three
  rolling tiles and ordinary tiles, packed/dense coefficients, publication maps,
  coupling bands, ragged extents and padded/unpadded storage.
- Bounded standalone racecheck: **zero errors or warnings** on 37×41×61.
- Full JAX/native pipeline memcheck: both selected ragged paired-DFT tests passed
  with **zero memory errors**, using rolling and ordinary-field paired schedules.
  CUDA API-probe reporting was disabled for this check; memory checking remained
  enabled.

Logs and compiler resource usage are in the
[validation directory](rtx3090-2026-09-18-realistic-integration-validation/).
Across the seventeen benchmark cases, 33 optimized/reference comparisons were
exact and the one dense rolling comparison passed the stated numerical gate.
This establishes the tested cases; it does not prove all arbitrary shapes,
source layouts or monitor configurations.

Several experiments were rejected or retained only as explicit controls:

- Full intermediate-field publication: 6.68 GCUPS on 128x256x512 and 4.16 on
  1024x256x64; correct but too expensive.
- A first sparse implementation accidentally copied source descriptors to a
  1,456-byte per-thread stack and used 106 registers. It fell to 2.40 GCUPS on
  the wide case. This was fixed before final measurements.
- Field padding to 32 floats reduced the earlier integrated wide result from
  7.59 to 7.32 GCUPS, and the narrow result from 4.66 to 4.37. It is not enabled.
- Thin face-specific coupling kernels did not improve the result, especially
  on narrow x. The compact direct queue was retained instead.
- A 32x8x16 rolling tile helped the earlier wide test modestly (7.76 versus
  7.49 GCUPS), but lost on narrow x (4.22 versus 4.43). The bare winner remains
  the default experimental tile; no universal tile winner is claimed.
- Extending the ordinary fused core's z tile from 8 to 16 regressed the wide
  controls. That variant was removed.

## Controls and next engineering step

For paired monitor gathering with the established field schedule:

```bash
BEAMZ_CUDA_TEMPORAL_STEPS=2 BEAMZ_CUDA_PAIR_TILE=single <simulation command>
```

For the integrated rolling experiment:

```bash
BEAMZ_CUDA_TEMPORAL_STEPS=2 BEAMZ_CUDA_PAIR_TILE=16x8x16 <simulation command>
```

The default remains `BEAMZ_CUDA_TEMPORAL_STEPS=1`. The paired schedules require
uniform CPML, scalar H coefficients and no before-H source groups; unsupported
programs retain the ordinary path. See [CUDA controls](https://github.com/beamzorg/beamz/blob/fa151d28/cuda/README.md).

The next substantial temporal optimization must include CPML's auxiliary psi
state and reduce the core/absorber coupling passes. Simply batching two existing
CPML launches does not reuse that state. A correct overlapping-tile design needs
frozen old psi and a distinct final destination, plus intermediate psi in its
rolling stages; otherwise one tile can overwrite auxiliary values another tile
still needs. Face/edge/corner specialization could limit those shared values to
the active derivative directions. This work has not yet been implemented or
benchmarked, and the current results do not justify claiming shape-independent
9 GCUPS.
