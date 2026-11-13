# CUDA optimization run, 2026-11-11

Started 10:20 UTC with a four-hour budget. The benchmark convention is exactly
12 CPML cells per face at every resolution. This is an engineering log; early
observations are not final performance claims.

## Accepted native result

Balanced confirmation finished at 13:23 UTC. The selected CUDA mstep build is
11–16% higher throughput than an equally tuned, corrected control on the three
realistic workloads, across both material preparation orders. All 24 fresh
processes match all thirty returned reference leaves exactly.

| Shape (z,y,x) | Byte-first control GCUPS | Selected GCUPS | Throughput gain | Header-first gain |
|---|---:|---:|---:|---:|
| 160×320×384 | 8.513 | 9.749 | 14.52% | 14.54% |
| 256³ | 8.266 | 9.560 | 15.65% | 15.13% |
| 256×512×512 | 10.427 | 11.600 | 11.25% | 11.16% |

These are throughput increases, not the same numerical percentage reduction in
runtime. Each group uses ABBA fresh-process order, two processes per build,
seven synchronized solve samples per process, four initial warmups and at least
two warmup seconds. Observed process/paired-ratio ranges are retained; two
processes per build do not establish a strong confidence interval. Both builds
were screened over whole/32/64-plane segments with 16×32 tiles before the
balanced run. Historical uses segment 32 for both, cube uses 64 for both, and the
large case uses whole columns for control and segment 64 for candidate. Settings were retained for
both preparation workflows. This is a bounded tuning result, not a universal
autotuning optimum.

The source-matched control and selected `full-typed-traced-v11` build share
source SHA `9dbc587cb4b7757f33645afc28c7b89c22ed1f0b18cd30909adc0994fa41276e`.
The frozen worker SHA is
`609f42b7fa7e8fae661f6e337bc050eb0825ab2e12e8cf18ed85f02974bb2d5b`.
Native hashes, actual coefficient shapes/types, all samples, launch settings and
raw artifact paths are in the [durable results](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-final-results.json).
The [offline chart](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-final-results.html) and
[PNG](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-final-results.png) use those records.

Retained defaults combine full-bank intermediate edges, proven homogeneous
material layouts, exact edge indexing, a redundant terminal-barrier removal,
typed CPML/wall handling and generated Yee arithmetic. Generated arithmetic
matched manual arithmetic speed; tracing alone is not credited with the gain.
The register-history candidate stays off: its improved 43-register allocation
still lost against the matched ordinary core at 8×16/segment32. Core publication,
axis-mask expansion and early prefetch waits also remain off.

The selected binary passed all 158 hardware gates, followed by a new tiny
CPML12 memory-check case. The complete graph-driven race/sync tool attempts
remain inconclusive after bounded timeouts, including a matched-control timeout.
A direct harness compiles the exact frozen production core source and effective
options with separable device compilation, bypassing graph capture. It passes
bitwise staggered/partial-mask/guard/input checks and memory, race and sync
tools. This is core-only coverage; see the
[native review](cuda-tracing-2026-11-11/native-review.md) and
[direct validation](cuda-tracing-2026-11-11/direct-native-core-validation.md).

## Final nanoparticle curve

The 680 nm cube includes twelve CPML cells per face at every resolution. It
contains a 100 nm-radius, nondispersive n=2 sphere in vacuum, 500 nm finite-
aperture illumination and six six-component field-DFT faces. Each solve covers
30 fs at CFL fraction 0.95. This is an explicit Mie-like throughput recipe, not
an exact reconstruction of the creator's unspecified hardware/workload or an
analytic scattering-cross-section validation. No other FDTD engine was run.

| Resolution | Grid | Streamed solve s | Tuned control s | Selected mstep s | Mstep throughput gain |
|---:|---|---:|---:|---:|---:|
| 5 nm | 136³ | 1.588 | 1.834 | 1.561 | 17.49% |
| 4 nm | 170³ | 3.763 | 4.072 | 3.479 | 17.05% |
| 2.5 nm | 272³ | 21.677 | 19.418 | 17.031 | 14.02% |
| 2 nm | 340³ | 51.058 | 43.612 | 38.503 | 13.27% |

All twelve points pass finite/nonzero-signal and clock checks. All eight mstep
comparisons against streamed CUDA report zero measured full-state relative
error, including CPML/DFT state. Each point is one fresh process with two warmups,
three synchronized solve samples and one synchronized public-call sample;
there is no confidence interval. Both mstep builds use 16×32 tiles and segment 64, chosen
from the bounded cube tuning screen. The 5 nm selected/streamed difference is
only about 1.7%, so treat it as near parity rather than a firm crossover result.
The 1 nm case was not run or extrapolated.

The selected 2 nm public call takes 38.529 s, versus 43.615 s control and 51.139 s
streamed. Scene construction to first public result takes 45.570/50.744/57.975 s
respectively; warm throughput gains are not interchangeable with startup gains.
Sampled device-global peaks before validation are unchanged across the three
curves: about 1.667, 2.151, 5.187 and 9.194 GiB in resolution order. These include
driver reservations and other GPU processes, and may miss spikes shorter than 50 ms.
The compact workspace remains allocated; no memory reduction is claimed.

The [complete curve records](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-mie-final-results.json),
[interactive chart](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-mie-final-results.html)
and [PNG](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-mie-final-results.png)
retain separate solve, public and first-result timings. Generated datasets,
figures and logs live under `.cache/` following the evidence-index convention;
the narrative and reproduction tools remain in source.

## Final profile and installed component

Matched, separately instrumented large calls reduce summed GPU kernel activity
from 1.6386 to 1.4752 s: core 0.8467→0.7566, first edges 0.3978→0.3805, second
edges 0.3818→0.3260. H/E/H/E launch order and counts are verified. These compare
the selected large settings (control whole/candidate segment 64), so core gains include
tuning. Activity sums exclude host gaps/copies and are not unprofiled wall-time
shares or measured DRAM bandwidth. The [profile record](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-final-profiles.json)
includes hashes, arguments and raw traces.

The final native component compiles for SM80/86/89/90 and is installed locally:
version 0.30.0, ABI 30, SHA
`34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`.
Its source SHA is
`f355a62b3cab7f76a69ad955b9c08e24f04587af27bf493f5601d30e586182a9`.
Relative to the measured source, only CMake defaults, a comment and an unused
declaration guard changed. All 159 hardware tests and exact thirty-leaf replays
of all three fixtures pass on this packaged build. The final CPU CUDA/memory
suite passes 471 tests. Ruff lint/format, generated-header freshness and
`git diff --check` pass. See the [packaging record](../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-production-package-validation.json).

Performance evidence remains specific to RTX 3090. Streamed fallback and automatic
schedule-selection limits remain in place. The 10× complete-engine target is
still open; the next controlled cycle is specified at the end of this report.

## Controls and provenance

- Local RTX 3090, 24 GiB, fixed existing 250 W power limit. No device settings
  changed. Archived October measurements explicitly used 370 W and recorded
  power near that limit; compare new candidates with present-day controls.
- The original component and Python/CUDA source were frozen before edits.
  Raw controls and full-state references live in
  `.cache/cuda-autoresearch-2026-11-11/baseline-run/`; the original `/tmp` path
  is a symlink after moving the files off RAM-backed tmpfs.
- Each candidate build stages an immutable CUDA source tree, records its hash,
  build commands/options and binary SHA, and never silently replaces the active
  component. Candidate installation and GPU ownership are serialized.
- Hardware performance counters are restricted to administrators on this
  machine. GPU activity traces are available; event-duration shares do not
  establish DRAM bandwidth or unprofiled wall-time shares.

## Initial CPML12 controls

Current mstep again wins on its favorable large workloads. Initial measured
rates below preceded the packed-layout fix described next; their ordinary
zero-state fixtures passed full-state comparisons, but these are retained as
historical controls rather than the final acceptance baseline.

| Workload | Streamed GCUPS | Mstep GCUPS |
|---|---:|---:|
| Historical recipe, 160×320×384, binary | 7.001 | 8.816 |
| Cube, 256³, binary | 7.372 | 7.957 |
| Large, 256×512×512, binary | 7.463 | 10.554 |

The historical recipe uses CPML12, a mode source, two three-frequency mode
monitors and 256 steps. Its workload-builder source is unchanged, but modern
preparation/source sampling differs from the October inputs; this is not a
bit-identical replay of every historical prepared array. The power difference
also prevents attributing absolute-rate differences solely to code.

Corrected CPML12 nanoparticle cases retain the 680 nm box, sphere/source/monitor
assumptions and fixed 30 fs duration. Initial mstep loses on 136³ and 170³, but
wins on 272³, where useful temporal coverage rises to about 65%. See the
[CPML12 baseline report](cuda-tracing-2026-11-11/cpml12-baseline.md).

Separate profiling indicates edge/CPML/scratch phases remain important: about
57% of summed GPU kernel durations in the small nanoparticle case and about
49% in the large rectangular case. These are event sums, not wall shares.

## Packed-layout correctness regression

Stronger nonzero-state checks exposed an existing decoder bug. The CUDA
preparation cache shares constitutive arrays across streamed and mstep, but
packing can depend on which backend prepares them first. Streamed-first can
produce rank-two 32-cell constant-block headers; mstep-first can produce
rank-one byte IDs. The packed mstep fast core had discarded that distinction
and always decoded byte IDs.

The fix retains each component's packed layout and applies the existing exact
constant-block decoding semantics. It changes private kernel metadata, not the
external FFI ABI. The first fixed build passed all 48 selected GPU checks:
36 existing hardware cases, six forced-layout/cache-order regressions, and six
traced-queue comparisons. The original binary failed five of the six new
material-layout regressions and the existing mode/window test. No tests were
relaxed to admit the fix.

The fixed build is now the restoration and performance acceptance baseline.
Old candidate binaries without the fix must not be promoted. Further testing
will confirm startup, capacity, continuation, mixed materials and whole-workload
performance before any default optimization is retained.

The full gate then exposed a second instance of the same representation issue:
cyclic storage transforms transposed byte IDs, but retained rank-two constant-
block headers in their old flat order. The transform now expands those headers
exactly on the device, transposes the physical values, and uses the valid dense
core. It also resets the coefficient pool's former decay slot to scalar one.
The transposition remains inside the timed invocation, and memory admission
budgets the dense values and index workspace. Two CPU negative controls fail on
the old transform. With both fixes, all 157 current hardware cases pass on the
fixed control, including rotated byte and constant-block layouts.

## Independent optimization hypotheses

1. Replace dynamic row-coordinate division in compact edge queues with exact
   host-precomputed reciprocal division. An independent portable C++ oracle
   checks roughly six million divisions. Generated SASS removes the intended
   reciprocal/conversion sequence without spills.
2. Specialize uniform-CPML edge precision and wall handling, and bypass CPML
   work only for cells proven outside every component's absorbing slabs.
3. Remove or merge redundant packed-core barriers while preserving async input
   readiness and every cross-thread shared dependency. Run race/sync checks.
4. Specialize homogeneous packed layouts at host dispatch to avoid paying for
   dynamic layout branches in every sample after the correctness fix.
5. Generate production packed-core arithmetic from the Python cell tracer,
   keeping numerical operations and memory scheduling separate. First prove
   parity and no material regression; tracing alone is not a speedup.
6. Test fixed-XY one-shared-plane/register-history scheduling as a separate
   D2 core. A deeper production schedule requires new geometry, publication and
   observation orchestration; changing a temporal-depth constant is unsafe.

Each candidate is a separate compile-time opt-in during research. Balanced
fresh-process confirmation is required before accepting small improvements;
instruction/resource estimates alone are not a performance verdict.

### Rejected or still unconfirmed experiments

Single-order screens have not shown broad gains from uniform-CPML type/axis
specialization, early prefetch waits, or the production register-history core.
The eight-axis-mask CPML version grows important edge kernels roughly sixfold;
it loses on all three production fixtures. These switches remain off.
Homogeneous packed-layout specialization is the strongest small candidate so
far (roughly 2–6% in initial screens); exact index division and terminal barrier
removal are small enough to require balanced confirmation.

The first full-bank edge prototype published H1/E1 core values, copied them
into the first bank, and computed only noncore first-step edges. It passed all
157 hardware gates and exact full-state comparisons but lost 3–5% overall.
Matched large-case activity traces explain the rejection: core duration rose
15.6%, first-edge duration rose 3.6%, and second-edge duration fell 11.1%.
The publisher wrote 6–9 times more core points than actually required because
compact slots allocate whole quadrants. About 24% of large-case first-queue
warps mixed copy and update work. The activity sums are not unprofiled wall
time or direct bandwidth measurements.

At 12:11 UTC the next experiment removed publication entirely: recompute the
original first-step envelope into full field banks, keeping the core unchanged.
As in the original compact schedule, only the required E1 subset must be valid;
discarded outer E1 halo values may read stale H1. Independent version/taint
checks must prove those values cannot reach second-step outputs or observations.
This retains full-address edges without the core publisher's cost. The frozen
v6 pair passed all 157 hardware gates and all three exact full-state references.
Its first screen is a consistent 4–5% improvement:

| Workload | Fixed v6 control GCUPS | Full intermediates GCUPS | Change |
|---|---:|---:|---:|
| Historical recipe | 8.432 | 8.883 | +5.4% |
| 256 cubed | 7.608 | 7.983 | +4.9% |
| 256×512×512 | 10.400 | 10.843 | +4.3% |

These are single-order screens, not the final balanced result. The raw matrix is
`.cache/cuda-autoresearch-2026-11-11/native-ab-v6/matrix.json`. Both native builds
share source SHA `68578fb0c95e6063ace49fb855385bfd65425b7797643b8737b6322b436904db`;
only the compile option differs. Each fixture has seven synchronized samples,
four initial warmups and at least two seconds additional warmup. Source/binary
hashes, all samples and device telemetry accompany each run. A description-only
NVML scope edit occurred during this early screen; the worker is frozen for
subsequent balanced confirmation. This optimization remains opt-in while the
combined candidate, bounded sanitizers and equally tuned controls are checked.

The register-history prototype also has a compiler-allocation experiment. Making
wave-local history explicit reduced 8×16 registers from 71 to 67. Requesting four
resident blocks for that one instantiation then produced 43 registers, no local
spills, and about 10% fewer static instructions with the same floating operations
and loads. This is compiler evidence, not an application throughput result.

## Tracing and generated-kernel evidence

The cell IR supports explicit rounded arithmetic, typed scalar/profile/dense
coefficient reads, CPML auxiliary recurrences, BF16 storage rounding, structural
CSE, deterministic serialization, SSA emission and local liveness analysis.
The generated `yee_cell_generated.cuh` is a private production integration
boundary with a deterministic generator/check command.

Generated CPML CUDA passed bitwise comparisons with an independently written
C++ FMA oracle for all eight active-axis masks in FP32 and BF16. Generated
register-history kernels also passed tiny ragged periodic GPU comparisons in
all three tested history modes. These probes use common field extents and
deliberately simplified layouts; they do not establish production throughput.

The traced 13-offset publication envelope matches the conservative mstep
results exactly in the new GPU gate. The prototype removes redundant queue
entries while keeping allocation and ownership unchanged. Its expected gain
is small, so retain it as an experiment unless complete timing justifies it.

## Next controlled engineering cycle

1. Remove unused compact field workspace only after the native component reports
   the full-bank capability. Keep the six ABI buffers as 128-value placeholders;
   an older component must retain complete compact allocation. Make runtime
   allocation, memory admission and schedule reporting agree. Test retained and
   donated inputs, continuation, mixed materials, cache reuse and capacity before
   measuring fine-resolution memory. Do not infer 1 nm capacity from a lower
   bound or from successful smaller solves.
2. Test small byte-codebook preloading into registers against the existing hot
   lookup cache, with a strict register/spill budget. Restrict admission to a
   proven small pool and retain exact constant-block decoding. This is a
   hypothesis; the rejected register-history candidate demonstrates that better
   resource counts alone do not establish a speedup.
3. Profile and lower the existing complete temporal CPML path through the
   tracer, preserving its source/observation times and private auxiliary state.
   Test a one-plane/short-lived-register schedule against its existing two-plane
   controls, with explicit redundant-halo and address/profile accounting. The
   separate edge activity makes this essential for larger complete-solve gains.
4. Generate D4/D8 ownership and dependency plans only after rebuilding the
   two-step-specific psi offsets and source halos. Score useful updates, extra
   halo work, shared storage, registers, spills, parallelism, first-result cost
   and actual peak memory. Retain streamed/D2 fallbacks below measured size and
   aspect-ratio crossovers. Deeper blocking is a search dimension, not an assumed
   win.

Every round should freeze source/binary/worker hashes, reject numerical or
ownership failures before timing, tune the existing control equally, and retain
raw fresh-process repeats. Optimize synchronized complete solves and public
time-to-result. Synthetic kernel rates remain separate. The 10× complete-engine
target is still open; this run does not establish it.
