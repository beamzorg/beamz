# CUDA streaming branch: measured outcome

FP32 remains the default for both JAX and CUDA. The established one-step CUDA
field schedule remains the default; complete CPML temporal blocking, monitor
pairing, alternative shell tiles and padding are experimental controls.
Consistent >=9 logical GCUPS for arbitrary realistic RTX3090 simulations has
not been achieved. No H100 result was measured locally.

Latest update: [automatic layout/shell selection](cuda-autotuning-2026-09-18.md)
now calibrates eligible large RTX3090 lossless CPML12 runs and caches a validated
choice. Matched warm throughput versus current main improves by 17.5% on
1024x256x64, 4.6% on 128x256x512, and 18.8% on 257x193x341 smooth. A 1,025-step
flat run improves by 17.4%. First-use calibration costs 33–37 seconds; long runs
calibrate one native chunk. Main-versus-branch CPML tolerance differences remain
unresolved. The earlier opt-in and canonical-layout measurements below describe
the preceding stages of the branch.

Latest follow-up: an opt-in internal cyclic storage layout now recovers narrow
domain performance without changing the physical problem. The controlled
1024x256x64 comparison improved from 6.75 to 8.38 GCUPS with bitwise complete-state
parity; automatic-fusion repeats were about 8.2–8.3. The table below still describes
default canonical storage. See the
[cyclic-storage study](cuda-cyclic-storage-study-2026-09-18.md) for the six-case
matrix, integration checks, and the remaining temporal-CPML gap.

A subsequent [oriented y-face refinement](cuda-cpml-yface-study-2026-09-18.md)
improves experimental complete CPML pairing by 11–13% on the tested flat
domains, to about 7.0 GCUPS in matched runs (6.78 in a final unpaired repeat).
It remains slower than the ordinary cyclic-storage schedule and stays opt-in.

## Latest FP32 performance

All three recorded precision-comparison cases below have 12-cell CPML, lossless materials, a
mode source, two compact mode monitors with three frequencies, and about
16.8–16.9 million logical cells. Eight balanced timing rounds follow warmup.
Timing includes the compiled simulation scan and synchronization, excluding
setup, compilation and modal postprocessing. These are not cold-start/public
API timings.

| Domain (z,y,x) | Material | CUDA GCUPS | JAX GCUPS |
|---|---|---:|---:|
| 128x256x512 | binary | 8.481 | 3.613 |
| 1024x256x64 | binary | 6.852 | 3.915 |
| 257x193x341 | smooth | 8.238 | 3.902 |

An earlier broader native build measured 6.54–8.90 GCUPS for the compact-mode
cases in a 17-case realistic integration study. Large full-field monitors are
more expensive: the 11-frequency narrow-domain example measured 3.62 GCUPS,
or 5.02 with paired monitor processing. Neither range is a guarantee for every
simulation. We have not measured a matched original-branch-to-final-branch
speedup over the complete final suite, so the user's earlier approximately
5-GCUPS observation must not be used as the denominator of a claimed uplift.

Latest raw data and accuracy comparison:
[precision report](cpml-precision-default-comparison-2026-09-18.md).
Broader earlier data:
[realistic integration](cuda-realistic-temporal-integration-2026-09-18.md).

## What the work delivered

- Merged the locally fetched `origin/main` into this worktree (merge b0a5ff32).
  Subsequent development remains local and uncommitted; nothing was pushed.
- Added reproducible, guarded benchmarks across 8.4–25.2 million-cell domains,
  strong aspect ratios and axis permutations, irregular widths, multiple
  source orientations, binary/smooth lossless materials, varying monitor loads,
  long runs and continuations. Physical CPML stays 12 cells and logical GCUPS
  excludes padding and redundant halo work.
- Implemented storage-only field padding and CPML queue tile variants. Padding
  did not produce a general gain. The 32x8 shell option improved short-x cases
  by roughly 4–7%, but regressed some other cases; FP32's 64x4 default remains.
- Built and integrated the rolling 16x8x16 two-step interior with source timing,
  selective intermediate-field publication and correct odd tails/continuation.
  A separate bare periodic benchmark measured 14.09–16.49 GCUPS, 27–32% above
  its best tested one-step controls. It omits CPML, sources and monitors and is
  not the realistic application rate. Tested four/eight-step designs lost.
- Implemented paired monitor gathering/accumulation. The heavy full-plane,
  11-frequency case improved from 3.62 to 5.02 GCUPS (about 39%); compact mode
  monitors benefited little. The option preserves sampling and DFT chronology.
- Implemented complete two-step CPML, with intermediate auxiliary state retained
  in shared memory and independent face/edge/corner ownership. Added oriented
  boundary tiles and a one-step spatial-fusion control. These remain slower
  than established execution and are not enabled by default.
- Added BF16 auxiliary storage experiments with FP32 fields/arithmetic. Latest
  large CUDA cases improved by 3–11% to 7.60–8.89 GCUPS; JAX slowed by 5–9%.
  Long-run modal and source-free packet tests measured a precision tradeoff.
  Fixed JAX's mixed-precision recurrence to round only stored auxiliary state.
  Both public defaults remain FP32, as requested.
- Fixed schedule-dependent CUDA floating-point contraction by specifying shared
  derivative rounding and field/CPML operation order. Full state was identical
  across ordinary/paired schedules in both large BF16 comparison cases, without
  relaxing tolerances. This is fixed-precision schedule parity, not FP32/BF16
  equivalence.
- Investigated the local CPU staircase scheduler and tested standalone GPU
  wavefront variants. They established correctness but were slower; that
  architecture is not being adopted.

## Bottleneck and verification limits

Shape-dependent boundary cost persists at GPU saturation. CPML occupies roughly
30% of 128x256x512 and 45% of 1024x256x64, before staggered-grid adjustments.
The realistic two-step schemes save field traffic but add halo recomputation,
shared-state pressure, synchronization and (in the earlier scheme) coupling
passes. Compact source and monitor work each used under 1% of kernel time in
the profiled spatial cases. These timelines do not prove a particular hardware
stall mechanism; hardware counters were unavailable.

The latest targeted hardware run passed all 30 seeded 33-step cases and their
continuations across three irregular shapes/padding choices, both auxiliary
precisions and five spatial/temporal schedules. CUDA schedule parity is exact;
independent JAX comparisons use their existing tolerances. The 29 runtime
contract checks and 34 selected lossless kernel tests also passed. Earlier
builds passed broader hardware suites and standalone memory/race checks; those
are recorded in their individual reports. The complete hardware suite and
multi-architecture release build have not been rerun after the final arithmetic
changes, so this is not a claim of full release validation.

The RTX3090 power limit stayed at 370 W; no higher-power results are claimed.
Recent precision runs stayed at or below 73 C with at least 14,976 MiB free.

## Matched current-main comparison

A subsequent [matched comparison against origin/main c5fe0d88](cuda-origin-main-comparison-2026-09-18.md) measures default-branch changes of -8.1% (1024x256x64), +4.1% (128x256x512), and +18.4% (257x193x341 smooth). The opt-in cyclic layout makes the narrow case +15.7% versus main. Inputs match exactly, but some CPML entries exceed the existing numerical tolerance in the two binary cases; fields and monitors pass. These results supersede using historical approximately 5-GCUPS records as a main baseline.
