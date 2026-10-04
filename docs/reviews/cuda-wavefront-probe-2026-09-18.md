# CUDA wavefront scheduling probe

This follows the local CPU FDTD comparison. The new code establishes a numerical
and scheduling reference for non-overlapping temporal work; it is not integrated
into BeamZ, and its timings do not satisfy the realistic-simulation goal.

Implemented:

- `scripts/verify_cuda_wavefront_schedule.py`: integer field-version oracle for
  every H/E stencil read and each 12-cell PML update. Source and monitor event
  times must remain chronological. Forty randomized schedules pass, with three
  domain orientations, depths 4/8, multiple passes, and cube/rectangular tiles.
  Removing temporal dependency edges is a negative control that fails.
- `scripts/verify_cuda_wavefront_conflicts.py`: enumerates complete physical
  read/write footprints and DAG ancestors. All 67,344 unordered task pairs in
  three fixtures have disjoint conflicting footprints. This is stronger than
  testing a few serial execution orders, but does not verify CUDA memory ordering.
- `cuda/tests/cpml_wavefront_probe.cu`: common-extent three-dimensional Yee-curl
  diagnostic with six FP32 fields, twelve packed FP32 auxiliary arrays, exactly
  12-cell CPML, heterogeneous lossless update coefficients, analytic planar
  excitation and a point monitor with three DFT bins. Both global phase sweeps
  and wavefront kernels call identical explicitly rounded arithmetic.

A task advances a rectangular 16x8x16 (z,y,x) window repeatedly, shifting it
backward after each timestep. Both unit shifts and shifts of tile_size/depth
passed. The CPU's spatial predecessor and previous-pass positive-diagonal edges
are sufficient for these tested schedules, even with rectangular windows: they
can conservatively overconstrain execution. The earlier review correctly called
for verification; rectangular shape alone did not invalidate the edge set.

The prototype groups tasks by `x+y+z+4*pass`. Each layer is a kernel launch with
one CTA per task; stream ordering supplies global visibility between layers.
Within each CTA, H updates complete before E updates through block barriers.
All source/DFT time indices use `segment_start + pass*depth + local_step`.
Clipped tasks remain in the topology. Odd tails use ordinary H/E sweeps, and
continuation starts a new segment from the same in-place state.

The first CUDA execution passed 16 numerical fixtures, all bit-identical for
every field value, auxiliary value and DFT accumulator. Shapes were 29x31x37,
37x29x31, 31x37x29 and 65x49x97; depths 2/4/8; total lengths 35/67/131; split
points 17/33/65. The final formatted build reran the same fixtures successfully.
Full standalone memory checking reported zero errors. The CUDA-graph timing path
also passed a separate small memory-check run. Synchronization checking is
recorded in `.cache/perf/cpml-wavefront-synccheck.log`.

Two 16.8-million-cell diagnostic domains were then measured for 128 steps.
Both variants use CUDA graph replay, two warmups and six interleaved timing
rounds, with CUDA events. Allocation, task construction and host comparisons
are outside timing. Complete state is compared before timing and again after
all replays, preventing numerical drift from being hidden by timing-only runs.

| Domain (z,y,x) | Shift | Global phase sweeps | Eight-step wavefront |
|---|---|---:|---:|
| 128x256x512 | tile_size/depth | 8.925 | 3.027 |
| 128x256x512 | one cell | 8.891 | 3.075 |
| 1024x256x64 | tile_size/depth | 8.031 | 3.022 |
| 1024x256x64 | one cell | 7.845 | 3.056 |

Units are diagnostic logical GCUPS, counting each common-extent cell timestep
once. This fixture is not the production staggered Yee layout, does not use an
eigenmode-solver source, and does not implement the production interpolated mode
monitor. These measurements must not be substituted for the required realistic
benchmark suite. Raw samples, telemetry and source/binary hashes are retained in
`rtx3090-2026-09-18-wavefront-probe/`.

The wavefront kernels use 42 registers with zero stack/local storage and no
shared storage; the reference phases use 34/38 registers, also without spills.
Thus spilling is not the explanation. The current wavefront still reads and
writes global arrays at every substep, uses many block barriers and exposes
uneven parallelism across layers. It does not explicitly hold the evolving tile
in shared memory. Timings reject this straightforward cache-only port; they do
not isolate which hardware stall dominates and do not rule out a better
wavefront implementation.

The next architectural experiment should use this implementation as the
correctness reference while retaining an evolving field/CPML working set in
shared memory or registers. Choose the shared footprint and CTA parallelism
before increasing temporal depth. The full swept union of a large tile may not
fit efficiently, so smaller internal space-time tiles or rolling ownership are
needed. Persistent task execution is a separate later comparison, with device
publication ordering and deadlock avoidance explicitly validated.

Reproduce the oracles:

```sh
.venv/bin/python scripts/verify_cuda_wavefront_schedule.py --output docs/reviews/rtx3090-wavefront-schedule-oracle.json
.venv/bin/python scripts/verify_cuda_wavefront_conflicts.py
```

Build and run the numerical probe with the CUDA toolkit's `nvcc`:

```sh
nvcc -std=c++17 -O3 -arch=sm_86 cuda/tests/cpml_wavefront_probe.cu -o /tmp/cpml-wavefront-probe
/tmp/cpml-wavefront-probe
/tmp/cpml-wavefront-probe --bench 128 256 512 128
```

Large runs used the established 2 GiB free-memory and 85 C temperature guards.
No production defaults changed, no CPU-reference files changed, and nothing was
pushed. Consistent 9 GCUPS on realistic arbitrary domains remains unachieved.

## Explicit shared-memory staging

The probe now also stages the complete swept field read region and the owned
CPML state in shared memory. Fields use a rectangular backing region; CPML uses
the smaller owned bounding region and allocates only the active-axis records.
Loads are restricted to the union of owned cells and required neighbor cells;
stores are restricted to owned cells. Thus staging does not introduce extra
global read/write conflicts with tasks that are otherwise independent.

The tested internal tile is 8x4x8 (z,y,x), with unit shifts and depths two/four.
Its shared footprints, in bytes, are:

| Depth | Interior | Face | Edge | Corner |
|---|---:|---:|---:|---:|
| 2 | 20,328 | 26,808 | 33,288 | 39,768 |
| 4 | 36,504 | 50,056 | 63,608 | 77,160 |

Host preprocessing groups each wavefront layer by its active-axis mask; the
CUDA graph records the corresponding specialized launches. This permits
smaller shared allocations for interior and face work, but introduces multiple
launches per layer. There is no persistent ready queue in this version.

All 24 CUDA fixtures pass bit-for-bit, including the original 16 global-memory
wavefront cases and eight shared-memory cases across four shapes. Full memcheck
reports zero errors; racecheck on the small fixture reports zero shared-memory
hazards. The expanded integer oracle passes 64 randomized schedules, and the
expanded footprint oracle checks 455,276 unordered task pairs without a conflict.
These checks still cover a simplified common-extent diagnostic, not BeamZ's
production staggered fields or interpolated mode monitors.

Large measurements use the same 128-step, CUDA-graph, interleaved protocol:

| Domain | Depth | Reference sweeps | Shared wavefront |
|---|---:|---:|---:|
| 128x256x512 | 2 | 9.006 | 3.347 |
| 128x256x512 | 4 | 9.013 | 4.370 |
| 1024x256x64 | 2 | 7.806 | 3.425 |
| 1024x256x64 | 4 | 7.827 | 4.169 |

All fields, packed auxiliary values and DFT accumulators match bit-for-bit before
and after the timing replays. Raw timings, telemetry and provenance are under
`rtx3090-2026-09-18-wavefront-shared/`. The shared implementation improves on the
roughly 3-GCUPS cache-only wavefront, but is still substantially slower than its
own diagnostic sweep reference. It remains outside production.

Compiled shared kernels use 34 registers for interior work, generally 40 for
boundary work, and 48 for the depth-two corner; all have zero stack/local storage.
The dynamic shared-memory allocation is additional to the resource listing.
The depth-four shared footprint limits concurrent CTAs, especially at edges and
corners. Timings do not establish the relative contributions of occupancy,
shared-bank conflicts, launch serialization, indexing, and memory transactions.

A follow-up should reduce storage for the swept region rather than blindly
increase depth. For example, an 8-cubed, depth-eight unit-shift task owns 1,695
unique cells and reads 2,415 distinct cells including the union of nearest
neighbors. Compact row maps plus FP32 state would require approximately 64 KB
for an interior task and 91 KB for a face, versus an infeasible rectangular
field allocation of 118 KB before PML. Edges/corners would still exceed the
available per-CTA budget and need another execution strategy. This is a sizing
calculation, not an implemented optimization or a speedup prediction.

Run `--bench-shared Z Y X STEPS` to exercise the new diagnostic variants. All
previous `--bench` and correctness modes remain available. The installed BeamZ
extension and its defaults are unchanged by these standalone experiments.
