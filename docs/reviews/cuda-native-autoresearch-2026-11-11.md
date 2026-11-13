# Native CUDA and temporal scheduling: two-hour study

Budgeted local window: 2026-11-11, 16:07–18:07 UTC; validation closed at 17:57.
RTX3090, unchanged 250 W
limit. No RunPod work. This report supplements the [earlier optimization
study](cuda-autoresearch-2026-11-11.md) and [released FDTDX
comparison](fdtdx-comparison-2026-11-11.md); it does not benchmark the unpublished
FDTDX rewrite in the creator's post.

The implemented result is a JAX-independent prepared-input CUDA runner and a
numerically exact four-step tracing/scheduling prototype. **A multi-fold engine
speedup has not been achieved.** Removing JAX from execution leaves large-case
timings effectively tied. Overlapping depth-four tiles remain slower than the
existing depth-two solver. Full-pulse tuning confirms **15.2% throughput gain
at 5 nm**, but only **0.8% at 2.5 nm**. The latter is much smaller than the
initial fixed-256-step screen suggested.

## What now runs without JAX

The optional `_cuda_native` module owns a CUDA stream, buffers, clocks, reset,
state preservation, graph replay and output downloads. It accepts prepared NumPy
arrays and calls the existing complete source/H/E/CPML/DFT program. A native-only
CMake build skips JAX header discovery; the build and test environment has no
installed or imported JAX. Native and FFI frontends can embed exactly the same
compiled CUDA OBJECT-library files, so their comparison does not change kernels.

The actual 5 nm scattering workload ran for all **3,280 updates / 30 fs**, over
12×256+208 chunks, in that NumPy-only environment. All 30 public state leaves,
repeated/reset results and input immutability checks pass exactly. The one
functional timing sample was 1.5199 s with state-preserving copies included;
this is not a balanced performance campaign. It used 8×16/segment64 and one
warmup; the discrete duration is 30.00448 fs.

High-level scene/source/material preparation and the main BeamZ package still
depend on JAX. The existing pip component metadata also retains JAX. This is
independent execution of prepared complete workloads, not a claim that the
entire scene-to-result Python package has been migrated.

Build/API/ownership details: [native runtime](../../cuda/NATIVE_RUNTIME.md).

A nonzero-origin regression exposed one real native-runtime bug: fused time
accumulation differed by one FP32 bit from JAX's split multiply/add after a 3+3
continuation. The private native clock now explicitly rounds the multiply and
add separately. All other 29 leaves already matched. Corrected D2 and selected
D4 modules each pass 13 strict ownership/reset/repeat/continuation checks,
including that checkpoint and elapsed/origin clocks. The independent oracle and
JAX-free checker are retained in
[`validate_cuda_native_clocks.py`](../../scripts/validate_cuda_native_clocks.py).
Old failures remain under `nonzero-origin/`; corrected evidence is under
`nonzero-origin-clockfix/`. Earlier zero-origin timings remain valid. The
corrected default pair is `final-default-clockfix-pair/`, and corrected selected
D4 is `native-v13-single-clockfix/`; previous modules retain the known limitation.

## Matched native versus FFI execution

These trials use the same compiled core objects, the same prepared persistent
and read-only input bytes, segment64,
256 updates, complete physics, and synchronized Python-call wall timing. Two
fresh processes per backend run in balanced order, each with two warmups and
three samples. The table gives the median of process medians. Observed ranges
are retained; two processes do not establish a strong confidence interval.

| Complete workload | Ownership | JAX FFI | Direct native | FFI / native |
| --- | --- | ---: | ---: | ---: |
| 256³ mode-source/monitor | preserve | 0.455915 s | 0.454222 s | 1.0037× |
| 272³ scattering/six DFT faces | preserve | 0.625374 s | 0.626534 s | 0.9981× |
| 256³ mode-source/monitor | owned/donated | 0.441298 s | 0.443516 s | 0.9950× |
| 272³ scattering/six DFT faces | owned/donated | 0.603786 s | 0.614572 s | 0.9824× |

Preserving calls include the matching persistent-state copies. Owned/donated
timings exclude explicit host-to-device reset on both paths. Setup, compilation
and downloads are excluded from warm solve timing. Native explicitly reuses
final/phase storage without a fill and zeros required scratch per chunk. The
JAX wrapper creates workspaces inside JIT, where XLA controls their allocation
and fill elimination. These have equivalent state semantics, not identical
allocation lifecycles; unused workspace contents are don't-care. All 30 leaves
pass exact finite array equality: dtype/shape match, `np.array_equal`, and finite
values. This detects nonzero FP32 differences but treats +0 and −0 as equal;
it is not a raw-byte identity claim. The separate moving-field CUDA probe uses
`memcmp` on fields and padding. The result rules out a large JAX dispatch penalty
in this already graph-based CUDA path; it does not measure scene preparation.

Evidence: `.cache/cuda-native-autoresearch-2026-11-11-1607/`
`shared-pair-v5-results/compact-summary.json`, archived worker source and all
sixteen process results; `paired-v5/manifest.json` records identical core object
paths and hashes for both frontends.

The corrected same-core follow-up repeats the actual **272³, 6,560-update /
30 fs** scattering run: FFI process medians 16.9368 / 17.1008 s, native
16.9843 / 17.0383 s. Their median-of-process-medians is **17.01879 / 17.01131 s**,
respectively, a ratio of **1.00044×**. All 30 leaves, repeats and native host
immutability pass exactly. Two fresh processes per backend run in
FFI/native/native/FFI order with one warmup and two samples. Preserving copies
and all 26 chunks are included; setup, compilation and downloads are excluded.
This confirms the short-step finding at full physical duration. Evidence:
`full30fs-large-corrected/summary.json`.

A separate optional warmed-memory readback observes 2,330 MiB per PID for the
preserving JAX wrapper and 2,040 MiB for native on Mie272. The owned wrappers
observe 2,330 / 1,468 MiB, respectively, but the JAX benchmark retains extra
baseline device arrays for reset. These are wrapper residency observations,
including context, graphs, allocator and live arrays, **not peaks or intrinsic
runtime memory ratios**. Device-global idle usage was 410 MiB. Raw commands,
worker hashes and scopes are in `paired-v5-memory/summary.json`.

## Existing two-step tuning

All runs retain the 680 nm scene, CPML12 on every face, FP32, the same source
and every DFT observation. Each resolution executes a fixed 256 updates. Three
fresh processes per setting use Latin ABC/BCA/CAB order, two warmups and three
samples per process. Non-plan input bytes and all 30 output leaves match exactly.

| Resolution/grid | 16×32 whole column | 16×32 / 64 planes | 8×16 / 64 planes | Best gain |
| --- | ---: | ---: | ---: | ---: |
| 5 nm / 136³ | 0.135879 s | 0.120105 s | **0.117441 s** | **1.157×** |
| 2.5 nm / 272³ | 0.657517 s | **0.627004 s** | 0.668747 s | **1.049×** |

The winning tile differs by grid size. No universal new tile default is inferred
from these two cubes. Explicit reproduction controls are
`BEAMZ_CUDA_MSTEP_TILE=8x16` / `16x32` and
`BEAMZ_CUDA_MSTEP_SEGMENT=64`. Results, telemetry, source and exact input/output
checks are in `d2-schedule-tuning/summary.json` under the run cache.

The follow-up measures the **entire 30 fs pulse** on the same corrected FFI core,
with all source/DFT work and preserving copies. Each setting has two balanced
fresh processes, one warmup and two samples; both settings use the exact same
verified data and reference. Only execution flags change.

| Resolution/grid | Updates | Whole 16×32 | Tuned | Throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| 5 nm / 136³ | 3,280 | 1.754316 s | 1.522295 s, 8×16/64 | **1.1524×** |
| 2.5 nm / 272³ | 6,560 | 17.206502 s | 17.073979 s, 16×32/64 | **1.0078×** |

The 2.5 nm fixed-step gain does not carry through the full pulse. It must not be
reported as a 4.9% full-run improvement. Rates are 5.420 GCUPS at tuned 5 nm and
7.732 GCUPS at tuned 2.5 nm, counting material-grid cells including CPML times
complete H/E updates; staggered terminal sites are not added to the numerator.
The central index is `full30fs-confirmation-summary.json`, with each raw campaign
under `full30fs-{5nm-tuning,tuning}-corrected/`.

## Four-step generated CUDA: correct, slower

The new code generator lowers traced per-cell Yee/CPML equations into a
warp-X/shared-Y/Z temporal wavefront. CPML histories remain private across four
steps; H/E sources execute at their original phases. Full staggered extents and
wall semantics are retained. A separate packed-psi output bank avoids recurrence
aliasing. One-to-three-step tails use the original program with explicit offsets.
Clocks remain fixed within each requested chunk and advance once at its end.

Sparse observation staging deduplicates actual native component/cell keys,
captures raw FP32 samples at each substep, remaps original gather indices and
reuses the original chronological DFT arithmetic. It preserves zero-weight reads,
invalid-index handling and interpolation/FMA order. It does not materialize
four whole intermediate field volumes or omit monitor updates.

The complete v6 candidate matches all 30 state leaves for nonzero small cubes
and rectangles, odd+odd continuation, 3+3+1, 1+1+5, 259 updates, and both large
workloads. With the same kernels and v7 host cleanup, on the seeded 64×96×128
seven-step fixture, native-only CUDA memcheck,
synccheck and racecheck each report zero errors; racecheck reports zero warnings.
These are bounded kernel checks, not a proof of every possible input.

| Same-binary large-case control | D2 | D4 v6 | D4 time / D2 |
| --- | ---: | ---: | ---: |
| 256³ mode-source/monitor | 0.4387 s | 1.9662 s | 4.48× slower |
| 272³ scattering | 0.6045 s | 2.6176 s | 4.33× slower |

Later typed source/material lowering removes about 35% of static bulk SASS
instructions while retaining 91 registers/thread and exact arithmetic. It
deliberately admits only the supported Hz/Ey source signature, scalar H and
byte-packed E coefficients; rectangles with a different signature are explicit
unsupported controls. Doubling shared frames reduces barriers but introduces
local-memory traffic on CPML faces. Both remain research candidates and are
disabled by default.

Balanced six-process confirmation uses identical Mie272 prepared inputs,
preserving copies, one warmup and two samples per process, in
v6/v8/v10/v10/v8/v6 order:

| D4 variant | Median of process medians | Speedup over v6 |
| --- | ---: | ---: |
| Generic v6 | 2.60844 s | 1.000× |
| Typed single-frame v8 | **2.16905 s** | **1.2026×** |
| Typed double-frame v10 | 2.53659 s | 1.0283× |

All 30 leaves pass exactly. The final research header selects v8 geometry plus
reasonful host admission checks (v12); unsupported inputs reject before private
D4 allocations or capture. D4 remains over three times slower than D2 in its
same-binary screen. The 20.3% improvement is **within a losing D4 prototype**, not
an engine speedup. Source, telemetry, hashes and raw results are in
`deep-balanced-results/summary.json`.

The selected v12 single-frame build also passes native-only memcheck and
synccheck with zero errors, and racecheck with zero errors/warnings, on seeded
Mie136³/7 including continuation and repeat checks. The final build does not
inherit the rejected two-frame barrier change.

Detailed generation/proofs/admission:
[deep temporal implementation](cuda-tracing-2026-11-11/deep-temporal.md).

## Why temporal depth alone loses

An instrumented 272³ solve attributes the D2 kernel sum to core 33.6%, edges
56.5%, DFT 7.8% and sources 2.1%. D4 v6 spends 49.2% of its kernel sum in
mask-zero bulk alone: 1.2875 s, compared with 0.6101 s for the entire D2 kernel
sum. Corner work is only 2.2% of D4 time, so repairing corner spills cannot
reverse the result. These are instrumented kernel sums, not unprofiled wall
shares, hardware DRAM counters or confidence intervals.

Actual 272³ geometry executes about **2.09 field expressions per useful field
expression**, excluding auxiliary/source/publication work. The D4 bulk uses
480 threads, 91 registers/thread and one resident CTA, versus 46.875% theoretical
thread occupancy for the profiled D2 core. The D4 occupancy is 31.25%. This
combination of redundant work, generated descriptor/address instructions and
limited residency is a concrete explanation for further schedule work.

DFT staging is useful independently: its instrumented DFT time is 0.0189 s
versus 0.0474 s for the old path. Since DFT is only 7.8% of that D2 budget, it
cannot establish a multi-fold whole-solver gain. Hardware performance counters
were not collected; bandwidth and cache-miss claims remain unverified.

## Safe memory improvement and rejected small change

An additive FFI capability target explicitly promises full-bank intermediates.
Only components advertising it receive six 128-float compact scratch
placeholders. Older components retain full scratch allocations and the old
target. The new target also gives XLA a separate compilation identity.

Declared unused compact scratch drops from 58,957,824 to 3,072 bytes at 136³,
and from 339,631,104 to 3,072 bytes at 272³. This is a workspace allocation
reduction, not a claimed equivalent reduction in device-global sampled peak.
The initial root change passes 70 CPU tests and 24 targeted GPU tests, including material
layouts, FP32/BF16 CPML precision, mixed cases, ownership and continuation. A
NumPy-only full-physics run with the placeholders passes all 30 leaves and CUDA
memcheck with zero errors. A separate JAX-mediated sanitizer attempt reported
CUDA symbol-probe API errors; that failed/inconclusive log is retained.

The final default paired build (both frontends, D4 OFF, D2 coefficient research
OFF) also passes all 24 targeted hardware tests. The final focused CPU suite
passes 92 tests, including native fixture contracts, generated deep header and
moving-schedule negative controls. ABI generation and whitespace checks pass.

Optional D2 coefficient specialization retains generic fallbacks for dense H,
mixed E layouts and unsupported data. All gates pass exactly. Its balanced
screen observes only 0.5% / 0.7% improvement on the two large fixtures, using
two processes per variant. That is insufficient evidence for a new default;
the option remains OFF.

## Next architecture and reproducibility

The next substantial optimization is a moving-subdomain communication schedule,
not another increase of overlapping temporal depth. A standalone CUDA mechanism
now works with staggered fields and unique updates. At N256/T8 its relative
addressing version takes about 30 ms versus 65 ms for a simple two-phase CUDA
reference. **This is field-only, with no CPML, sources, material geometry or
DFT; its control is not the optimized production D2 engine.** It is not a
twofold BeamZ gain. The measured benefit grows between N128 and N256, consistent
with the user's larger-domain hypothesis for temporal blocking.

The field-only four-frontier ring reduces boundary-history payload 16.75× at
N256/T8. Its first implementation spills and is about 1.6× slower than direct.
Hoisting eight neighbor bases/owner checks once per CTA removes those spills:
196 registers, zero local storage, and about 37.00 ms versus direct's
29.59–29.87 ms. It still loses by about 25%, so it remains a memory experiment.
Both mechanisms pass exact field checks, deliberately broken retirement
controls and bounded CUDA sanitizers. The direct version remains the research
performance control. A separate CPU lifetime model checks 95,476 consumer values
across 12 traced cases and rejects early epoch reuse and wrong history versions.

The
[systolic design](../../cuda/SYSTOLIC_SCHEDULE.md) gives the affine transform,
source/material/CPML/observation obligations, finite-storage questions and safe
CUDA execution plan. Its CPU dependency/ownership proof checks twelve staggered
Yee/CPML cases and rejects missing dependencies. No multi-fold speedup is inferred
from that proof or the paper's stripped-down H100 result.

The installed corrected native module additionally passes one full 5 nm pulse
with **raw-byte equality on all 30 state leaves**, including signed zeros and
unchanged caller arrays. This stronger single-case check is separate from the
array-equality gates used throughout the timing campaigns. Evidence:
`installed-native-fullpulse-bytes.json`.

All source/build snapshots, hashes, inputs, raw states, command lines, telemetry
and failed candidates remain under
`.cache/cuda-native-autoresearch-2026-11-11-1607/`. Source code and review documents
are uncommitted. The installed production component remains the backed-up
baseline; experiments use immutable, process-local modules. The corrected native
runner is installed only in the task's isolated NumPy environment, where
`native-only/bin/python -c 'import _cuda_native'` works without JAX. Its source
and installed hashes are recorded in `native-only-install.json`. No GitHub
mutation or cloud execution was performed.

Primary inspectable artifacts:

- [Full-duration confirmation index](../../.cache/cuda-native-autoresearch-2026-11-11-1607/full30fs-confirmation-summary.json)
- [Matched full-run FFI/native campaign](../../.cache/cuda-native-autoresearch-2026-11-11-1607/full30fs-large-corrected/summary.json)
- [D4 balanced rejection evidence](../../.cache/cuda-native-autoresearch-2026-11-11-1607/deep-balanced-results/summary.json)
- [Moving-box implementation and reproduction](cuda-tracing-2026-11-11/moving-box-probe.md)
- [Standalone source-backed charts](../../.cache/cuda-native-autoresearch-2026-11-11-1607/report/cuda-native-study.html)
- [Final handoff/checksum index](../../.cache/cuda-native-autoresearch-2026-11-11-1607/final-handoff.json)
