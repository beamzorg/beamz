# CPML boundary execution refinements

These experiments retain FP32 defaults, the established field schedules,
12-cell physical CPML and the winning 16x8x16 two-step interior. They do not
adopt the standalone CPU-inspired GPU wavefront design.

## Compile-time axis masks in the existing queue

The combined CPML queue partitions z faces, then y faces excluding z boundaries,
then x faces excluding both earlier axes. Minimum staggered extents make the
inactive axes provable. A trial specialized those three regions with masks
7/6/4, eliminating inactive-axis CPML checks without changing ownership or
recurrence arithmetic.

Eighteen rotated-source/continuation hardware checks passed. A matched binary
comparison then used baseline/candidate/candidate/baseline ordering on three
large realistic domains. Every process had four warmups and nine samples of
256 steps, with a mode source, two three-frequency compact mode monitors,
lossless material and exactly 12-cell CPML. SHA256 digests of every final state
leaf matched across all four runs of each domain. The saved baseline already
includes the explicit arithmetic fix; this isolates the queue specialization.

| Domain (z,y,x) | Baseline mean of run medians | Specialized mean | Difference |
|---|---:|---:|---:|
| 1024x256x64 | 6.706 | 6.748 | +0.63% |
| 128x256x512 | 8.353 | 8.406 | +0.64% |
| 257x193x341, smooth material | 7.956 | 8.031 | +0.93% |

Units are logical GCUPS. These are descriptive differences, not established
speedups: the two baseline narrow-domain runs differed by about 3.4%, and the
individual run ranges overlap. Both representative queue kernels use 30
registers and no stack/shared allocation; occupancy was not improved by this
change. The trial was reverted instead of adding kernel code for a sub-percent,
unresolved benefit. Raw per-run samples, binary hashes, state digests and
telemetry are in `rtx3090-2026-09-18-cpml-queue-axis/`.

The real-workload benchmark now has `--state-digests` for exact cross-build
comparisons. Hashing and host transfers happen after timing. Both baseline and
candidate native binaries and source checkpoints remain under `.cache/perf/`.

## Oriented two-step x-face block size

The existing x-face tile owns 15x16x16 cells. Its four transverse stage sizes
are 18x19, 17x18, 16x17 and 15x16: 342, 306, 272 and 240 slots. A 256-thread CTA
requires a second loop iteration in the first three stages. The trial uses 384
threads for this tile only, keeping all spatial/temporal ownership and the
interior tile unchanged. Other face/edge/corner tiles remain at 256 threads.

The SM86 build uses 54/56 registers for dense/packed FP32 coefficients, compared
with 59/58 previously, and no stack or local-memory allocation. This is compiler
resource data, not a measurement of achieved occupancy. Numerical validation
and matched large-domain timing determine whether the change is retained.

The six oriented 33-step seeded hardware cases passed (three irregular
shapes/padding choices, FP32/BF16 storage, with continuation). The matched FP32
binary study used the same four-run ordering and workload as the queue trial;
all state-leaf digests matched exactly in all 12 runs.

| Domain (z,y,x) | Original 256-thread pair | 384-thread x-face pair | Difference |
|---|---:|---:|---:|
| 1024x256x64 | 5.343 | 5.831 | +9.14% |
| 128x256x512 | 7.192 | 7.333 | +1.95% |
| 257x193x341, smooth material | 6.585 | 6.824 | +3.63% |

These are means of two per-process medians, in logical GCUPS. The narrow-domain
result repeats clearly in both candidate runs. The smaller wide-domain result
has overlapping per-run ranges and should not be treated as a precise speedup.
Raw data: `rtx3090-2026-09-18-cpml-face384/`. The change is retained in the opt-in
oriented CPML pair; the default one-step schedule and FP32 precision are unchanged.
The full standalone memory check completed with zero errors.

A separate 224-thread interior candidate was compiled into
`.cache/perf/cpml-core224-build/`, retaining the same 16x8x16 tile. It has not been
installed, numerically validated or timed. It is not part of the retained result.

The standalone small-domain race check also completed with zero reported
hazards or warnings (`.cache/perf/cpml-face384-racecheck.log`).
