# Temporal CPML implementation record

Objective: at least 9 logical GCUPS consistently across the large realistic
RTX3090 suite, with 12-cell CPML, FP32 state, lossless materials, mode sources
and monitors. This is an active development record, not a claim of completion.

## Ownership and state

A pair computes H1/psiH1, E1/psiE1, H2/psiH2, E2/psiE2 in that order.
Initial field and psi banks remain frozen throughout the pair. Overlapping
halos are recomputed independently; only a tile's owned rectangle writes global
final fields or psi. Intermediate fields are published only for monitor gather
regions. Intermediate psi stays in rolling shared planes.

The prototype uses 16x8x16 owned tiles, three field stages with two planes each,
and two psi stages with two planes each. Stage K has transverse extent
(X+3-K, Y+3-K), coordinate offset -2+floor((K+1)/2), and longitudinal coordinate
origin+wave-2-floor(K/2). Component-specific bounds and outer-wall derivatives
are preserved. Constrained field values are zeroed after psi advances, before
source injection, matching the established phase/source sequence.

There are three global field banks (frozen initial, sparse intermediate, final)
and two psi banks. Psi swaps once per pair, then once more for an odd trailing
step. Runtime return selection must use this new parity; field-bank selection
retains the existing paired rules. Continuation starts from the returned bank.
The opt-in flag is frozen into CUDA program flags and graph keys. ABI 18
separates this schedule from earlier components.

## Validation and measurements

Pending. Do not enable by default before the numerical, sanitizer and
large-domain performance gates pass.

## Progress: generalized and specialized pair kernels

The general full-domain rolling prototype passed 15 seeded state/continuation
cases, standalone memory/race checks, and the ABI/runtime unit suite (33 tests).
It measured 5.07 GCUPS on 128x256x512 and 4.13 on 1024x256x64, below the existing
8.48/6.80 schedules. Axis specialization improved those numbers to 6.60/4.96.
A compact geometric partition measured 6.35/5.10. An oriented tile experiment
measured 7.11/5.16 while preserving the 16x8x16 interior: x faces use 15x16x16,
y/z faces 32x4x16, edges/corners 16x8x4. None is promoted.

Compact partition tests passed 15 seeded cases; six short oriented/shallow
cases passed before the BF16 extension. All include 12-cell CPML. The new
long dense fixtures use the existing rotated-source tolerance (3 ppm of leaf
peak plus 30 ppm elementwise). Diagnostics reproduce the previous CUDA/JAX
rounding discrepancy; changing contraction/tiling changes a few cancellation
values. No earlier hardware-test tolerance was increased.

## BF16, requested during implementation

BF16 now stores auxiliary state in both global and shared memory. Each first
substep uses its unrounded FP32 recurrence value for the field correction, but
stores a BF16-rounded value for the second substep, matching existing semantics.
Fields, coefficients and recurrence arithmetic remain FP32. Nine short BF16
seeded cases passed; standalone memcheck passed for both precisions, packed and
dense coefficients, padding, and all three boundary tile choices.

The existing BF16 schedule initially measured 7.62 GCUPS on 1024x256x64. A large
256-step pair comparison exceeded the old 2%-of-leaf-peak gate in a few psi
values. Diagnostic timing explicitly records failures; it does not count as
validation. NumPy does not classify ml_dtypes BF16 as inexact, so comparison
helpers now cast BF16 to FP32 before applying their numerical gate.

A separate 128x96x256, 4096-step pulse experiment compared both schedules and
precisions. At the final checkpoint, BF16 existing/pair DFT relative L2 errors
versus FP32 existing were 0.0163%/0.0145%. Maximum forward modal-power errors were
0.00531%/0.00359%; backward modal-power errors were 0.601%/0.588%. The final field
snapshot's maximum component-peak errors were 0.444%/0.520%. Backward power
includes source/discretization effects and is not an isolated absorber reflection
coefficient. These are one case's results, not a universal accuracy claim.

Raw long-run accuracy data: `rtx3090-2026-09-18-cpml-pair-accuracy/long-pulse.json`.
Prototype measurements and failure logs are retained in the adjacent
`rtx3090-2026-09-18-cpml-pair-*` directories. The goal remains unmet.

## Single-step spatial fusion control

The same owned-region CPML implementation now supports a single H/E step,
keeping only the intermediate H planes in shared memory. This isolates temporal
halo and auxiliary-stage costs. It is opt-in through
`BEAMZ_CUDA_CPML_SPATIAL=1`, with graph-key isolation; the complete two-step
flag takes precedence. Defaults remain unchanged.

The first spatial tiles passed 30 seeded JAX/CUDA comparisons covering FP32 and
BF16, three irregular shapes and padding choices, odd/even step counts, and
continuation. The runtime contract suite passed 29 tests. Standalone memcheck
and racecheck reported zero errors/hazards for spatial and temporal launches.

On 128x256x512, FP32 spatial measured 7.25 GCUPS against 8.60 for the existing
schedule. On 1024x256x64, FP32 failed the large comparison in two of 1,597,440
values (maximum violating difference 5.12e-5); it has no accepted timing result.
BF16 diagnostic spatial timing was 7.75/5.48 GCUPS versus existing 8.97/7.50
for the wide/narrow cases. Spatial BF16 comparisons failed the existing gate;
these are diagnostic timings, not numerical validation. The earlier complete
BF16 pair measured 7.41/5.45 versus existing 9.21/7.63 in its interleaved study.

This control does not establish temporal storage as the sole cost. The fused
boundary computation also remains expensive without it. A follow-up tile trial
fits the single-step H halo into one 256-thread iteration: 31x7 interior and
y/z-face tiles, 15x15 x-face tiles, longitudinal depth 16. The two-step interior
remains 16x8x16. The first 13x15 face trial passed 30 cases; inspecting the staggered high face
showed it could need a mostly empty second tile, so the final x-face tile is
15x15. The final geometry passed 12 checks (three irregular shapes, both
precisions, ordinary and paired execution, 33 steps plus continuation) and
standalone memory checking with zero errors.

The revised BF16 spatial path measured 7.86/5.78 GCUPS, against existing
8.92/7.41 on the wide/narrow domains. Both spatial comparisons still exceeded
the auxiliary-state tolerance, so these remain diagnostic timings. Sources and
monitors each account for less than 1% of measured GPU kernel duration in these
cases. In the narrow spatial run, the interior takes 35.4% and x-only CPML faces
34.1%; all remaining CPML regions take about 30%. In the wide spatial run, the
interior takes 44.5%, z faces 16.3%, and the other CPML regions about 39%.
Timeline percentages do not identify hardware stall reasons.

Raw tuned results, comparison failures and per-region profile totals are in
`rtx3090-2026-09-18-cpml-spatial-tuned-bf16/`, including `kernel-summary.json`.
The control is retained for investigation and is not promoted. Further work
should target field/auxiliary layout and boundary execution; more batching of
these compact monitors cannot recover the missing throughput.

The current build also passed six oriented two-step regression cases (33 steps
plus continuation, all three shapes and both precisions), checking the shared
phase-descriptor changes independently of the new spatial path.

Final 15x15 spatial sanitizer racecheck completed with zero errors or warnings.

## Explicit arithmetic across existing schedules

A controlled rounding investigation fixed the field update first, then the CPML
recurrence/correction, and finally derivative scaling. Fixing the field update
alone did not pass the large BF16 comparison. Explicit CPML FMA order alone also
left the same 24 violating auxiliary values in the 128x256x512 case. These
failures are preserved in `rtx3090-2026-09-18-explicit-yee-bf16/` and
`rtx3090-2026-09-18-explicit-cpml-bf16/`; neither trial is validation.

Rounding each scaled derivative before curl subtraction or CPML correction
removed the discrepancy. The shared CUDA primitives now specify a rounded
multiply for derivative scaling, a fixed multiply/FMA order for field and psi
updates, and a fixed CPML correction FMA. This prevents inlining and tile
specialization from contracting operations across different stages. It changes
floating-point rounding, not the equations, physical domain, or CPML thickness.

A seeded diagnostic compared ordinary CUDA against fused, spatial and temporal
schedules at 2 and 33 steps, for both FP32 and BF16 (12 comparisons). Every state
leaf, including fields, auxiliary state and monitor accumulators, was identical.
The existing seeded hardware fixture now requires exact CUDA schedule parity;
its independent JAX tolerance is unchanged.

Large BF16 runs with 12-cell CPML, a mode source and two three-frequency compact
mode monitors also passed with **zero maximum absolute difference on every
state leaf**. Six balanced timing rounds at 256 steps measured:

| Logical domain (z,y,x) | Existing one step | Paired monitor schedule | Complete CPML pair |
|---|---:|---:|---:|
| 128x256x512 | 9.064 | 8.844 | 7.139 |
| 1024x256x64 | 7.295 | 7.311 | 5.361 |

Units are logical GCUPS. Raw manifests, checksums, telemetry, timing samples and
state differences are in `rtx3090-2026-09-18-explicit-derivative-bf16/`. This is a
numerical validation improvement, not a speedup. The complete CPML pair is still
slower and remains opt-in. Historical timing differences do not isolate the
arithmetic change's cost; a direct before/after comparison is still needed.

The broad hardware regression run was intentionally interrupted to prioritize
the requested JAX/CUDA precision comparison after 41 passes. SIGINT arrived in
an XLA garbage-collection callback and pytest reported an unraisable-exception
warning as a failure; this is not an arithmetic mismatch or a completed suite.
Targeted 33-step schedule comparisons are rerun separately. The 29 runtime
contract checks passed. The subsequent precision comparison and recommendation
are recorded in `cpml-precision-default-comparison-2026-09-18.md`.

The final targeted hardware run passed all **30** 33-step seeded comparisons
(three irregular shapes/padding choices, both auxiliary precisions, and five
spatial/temporal tile schedules), including continuation. CUDA schedule parity
was bit-for-bit; independent JAX comparisons retained their existing tolerance.
Log: `.cache/perf/explicit-arithmetic-seeded-33.log` (109.75 s).
