# CUDA cell tracing and temporal scheduling: first engineering plan

Date: 2026-11-11. This document separates existing measured evidence from new
engineering hypotheses. The objective is a substantially faster CUDA engine,
with a reproducible Mie-scattering workload and a Python definition of local
physics that can generate efficient temporal kernels. A 10x gain is a research
target, not a result or a forecast.

## Revised execution plan: fixed 12-cell CPML

User decision: **always use exactly 12 CPML cells per face, at every resolution**.
The nanoparticle runner now enforces this, including its default sweep. Physical
thickness changes with spacing; record it as metadata rather than holding it
constant. The initial 80 nm datasets below remain immutable historical records
and are superseded as the benchmark convention. They cannot be relabeled or
used as the acceptance baseline for the revised experiments.

1. **Recover a trustworthy baseline first.** Replay historical winning fixtures
   with their exact dimensions, CPML12, materials, sources, monitors, precision,
   schedule flags and timing method. Use today's source-built CUDA component;
   if results differ materially, compare a matching historical source build on
   the same machine. Resolve regressions or configuration differences before
   attributing results to temporal blocking or rewriting kernels.
2. **Freeze two benchmark tracks.** Rerun the 680 nm nanoparticle resolution
   sweep with CPML12, fixed physical duration and identical source/monitor
   settings. Independently test large cubes and rectangular domains through
   the memory-feasible range, with CPML12, both complete workloads and bare/PEC
   diagnostic controls. Preserve streamed and current mstep as controls. Label
   fixed-step throughput trials separately from fixed-duration solves.
3. **Profile the corrected workloads.** Measure useful temporal coverage,
   scratch traffic, DRAM/L2 activity where counters permit, kernel occupancy,
   barriers and boundary/monitor work. Test existing tile and z-segment knobs
   on large cases before new native code. The earlier 62% edge-event finding
   is a lead to remeasure, not an assumption for CPML12 or larger domains.
4. **Turn tracing into a production-compatible CUDA frontend.** Keep the
   restricted Python cell DSL independent of JAX/PyTorch. Extend the graph to
   production staggered extents, coefficient reads, CPML recurrences, source
   effects and observation times. First generate one-step kernels and compare
   every state leaf with the current CUDA engine, preserving explicit rounding.
   Avoid a separate Python object graph for every physical cell.
5. **Generate and measure temporal schedules.** Add stage liveness, minimal
   rings, register/shared placement and exact publication footprints. Compare
   depth two with deeper rolling kernels, then skewed/non-overlap ownership if
   overlap work is the limiting cost. Include CPML/source/monitor semantics as
   measurements justify; gate persistent queues on viable storage/residency.
   Rank candidates by complete large-workload runtime, not isolated core rate.
6. **Run bounded autoresearch after authorization.** Every experiment records
   hypothesis, single controlled change, exact source/binary/fixture hashes,
   measurements, memory and numerical verdict. Retain the best verified control;
   reject correctness failures and unsupported generalizations. Promote only
   with large-grid gains, acceptable memory and startup, and preserved small-grid
   streamed fallback. Treat 10x as a target against the best matched existing
   CUDA baseline, never against a deliberately poor forced schedule.

Acceptance gates include full H/E, CPML, DFT and clock comparisons; independent
physics checks; ragged grids, odd tails and continuation; and GPU memory/race/
synchronization checks. The nanoparticle fixture still needs a vacuum reference,
incident-intensity calibration and a convergence study before claiming analytic
Mie scattering accuracy. No new GPU trials were started for this plan revision.

The immediate prototype is an opt-in research tool. It does not change
`Simulation` dispatch, add a production backend, or establish equivalence with
the unpublished FDTDX rewrite described in the supplied post.

## Evidence and baseline

### Current whole-column multistep implementation

- [`mstep.py`](../../beamz/simulation/cuda/mstep.py) proves which 8x16
  quadrants can advance twice, constructs compact intermediate dependencies,
  and allocates boundary/source/monitor scratch. Source intersections currently
  exclude whole column quadrants and their halos from the fast core.
- [`mstep.cuh`](../../cuda/src/mstep.cuh) contains the general column core and
  the packed-E/scalar-H specialization. The latter already hoists metadata,
  increments addresses, carries scalar coefficients, and asynchronously
  prefetches two input planes. These are baseline capabilities, not proposed
  new optimizations.
- [`program.cu`](../../cuda/src/program.cu) schedules two-step core execution,
  first-step boundaries and observations, second-step boundaries and
  observations, and ordinary odd tails. Whole-program sources and monitors
  remain part of the measured work.
- [`schedule.py`](../../beamz/simulation/cuda/schedule.py) preserves streamed
  execution when `schedule` is omitted. Explicit `schedule="auto"` uses a
  conservative measured RTX 3090 cost envelope, not online timing trials.

The selector stores these historical calibration anchors for its current model:

| Physical cells | Boundaries | Streamed GCUPS | Mstep GCUPS |
|---:|---|---:|---:|
| 16,777,216 | CPML on all faces | 8.2242 | 9.0441 |
| 67,108,864 | CPML on all faces | 8.5313 | 10.7144 |
| 16,777,216 | CPML on one face | 8.1159 | 12.4346 |
| 16,777,216 | PEC | 8.4208 | 13.5445 |

These are stored RTX 3090 measurements, not fresh measurements of this
worktree, a Mie benchmark, or a prediction for another GPU. The model is
restricted by coefficient representation, geometry, monitor workload, startup
amortization, and memory headroom.

The [October integration validation](pr334-post-merge-validation-2026-10-17.md)
demonstrates why current best CUDA must be measured separately from any one
schedule: mstep improved 8.641 to 9.921 GCUPS on a 160x320x384 binary case,
but reduced 7.403 to 4.997 on a 512x256x96 smooth-material case. The full cosine
crossing was 13.9% slower with mstep. Existing small/narrow regressions must not
be hidden by choosing only large cubic workloads.

### Earlier experiments: useful references, not the current baseline

The [September realistic temporal study](cuda-realistic-temporal-integration-2026-09-18.md)
integrated rolling two-step overlap tiles with production source, CPML, and
monitor semantics. They lost to the established schedule on the compact-monitor
cases. Intermediate publication, halo updates, and a coupling band consumed
the gains from interior reuse. CPML accounted for roughly 42–57% of summed GPU
durations in selected profiles; compact sources and monitors individually
accounted for less than 1%. Those profiles predate current mstep and must be
repeated before attributing today's bottleneck.

The [wavefront probe](cuda-wavefront-probe-2026-09-18.md) already explored
non-overlapping ownership and depths greater than two:

- [`verify_cuda_wavefront_schedule.py`](../../scripts/verify_cuda_wavefront_schedule.py)
  checks every modeled H/E stencil read against integer time versions, CPML
  local recurrence times, unique visits, and source/monitor chronology. Removing
  temporal dependency edges is a negative control that fails.
- [`verify_cuda_wavefront_conflicts.py`](../../scripts/verify_cuda_wavefront_conflicts.py)
  checks conflicting read/write footprints against task dependencies.
- [`cpml_wavefront_probe.cu`](../../cuda/tests/cpml_wavefront_probe.cu) includes
  numerical global-memory and shared-memory wavefront references with identical
  explicit arithmetic. It uses common field extents and simplified sources and
  monitors, not the production staggered layout.

Layer-scheduled depth-eight global-memory wavefronts measured about 3 GCUPS;
depth-four shared-memory variants measured about 4.2–4.4, versus their own
7.8–9.0 sweep references. They passed the documented numerical fixtures but
did not produce a useful speedup. The shared variant stages a large swept
region and specializes launches by active CPML axes. It has no persistent
ready queue. Repeating this design with a larger depth is not a new hypothesis.

## Benchmark contract and missing reproduction details

The supplied FDTDX post fixes a 680 nm cubic scene, a nanoparticle-scattering
description, and resolution points 5, 4, 2.5, 2, and 1 nm. It does not provide
enough information for an exact independent reproduction.

Record explicitly before interpreting a cross-engine speed ratio:

- Whether 680 nm includes PML; sphere radius, location, dispersion, and material
  values; background index; geometry rasterization and subpixel averaging.
- Source wavelength/bandwidth, polarization, propagation direction, waveform,
  injection formulation, and placement; total physical simulation time and CFL.
- PML formulation, physical thickness or cell thickness, profiles, and target
  reflection; monitor geometry, components, frequency count, sampling interval,
  and any near-to-far or incident-reference solve.
- GPU, power settings, software and compiler revisions, precision, warmup,
  synchronization, repetitions, peak allocation, and exactly what runtime
  includes. Compilation, reference solves, and analysis must have separate
  accounting rather than silently disappearing from an end-to-end claim.
- Which BeamZ revision/backend was benchmarked in the post; whether a dotted
  segment is a timing extrapolation; and the exact allocation that caused OOM.

Our initial fixture should declare reasonable choices for all these unknowns
and call itself a similar workload. It must not claim to reproduce FDTDX's
runtime or physics exactly. Keep the requested physical duration constant as
resolution changes. With a fixed box and CFL, cell count scales approximately
as resolution^-3 and step count as resolution^-1: complete update work therefore
scales approximately as resolution^-4. A fixed-step microbenchmark answers a
different question and should be reported separately.

The user's follow-up specifically calls out the larger-simulation regime.
The 680 nm plot is a reproduction anchor, not the sole optimization target.
Keep a second, separate size matrix at fixed resolution and **12 CPML cells**:
e.g. 128/256/384-cell cubes at 5 nm (640/1280/1920 nm). Include bare/PEC controls and complete
source/CPML/monitor cases. Expand to 512 cells only after memory admission;
do not time an OOM-prone allocation as a throughput result. Report both fixed
physical-time application solves and explicitly labeled fixed-step controls.
Preserve a streamed fallback for the small grids where temporal blocking loses.

Maintain two linked fixtures: a realistic scattering scene with analytical Mie
or converged scattering checks, and a bounded kernel fixture suitable for rapid
full-state comparisons. A scattering-shaped execution without a validated
incident/scattered-field measurement is not yet a validated Mie analysis.

## Proposed compiler boundary

Use a restricted Python cell-expression tracer first. Operator-overloaded
field proxies can capture a stencil without requiring PyTorch or depending on
JAX tracing. A later PyTorch adapter could feed the same IR if that materially
improves ergonomics; interception alone does not improve CUDA scheduling.

The IR should record:

1. Field identity and component, constant affine spatial offsets in (z,y,x),
   and explicit H/E phase and complete-timestep versions.
2. Scalar or spatial coefficient reads, immutable material data, and region
   signatures for homogeneous, packed, dense, lossy, and tensor materials.
3. Arithmetic with explicit FP32 rounding boundaries, rather than algebraically
   interchangeable expressions that permit unreviewed reassociation.
4. CPML recurrence state and active derivative axes, including storage precision
   and conversion points. Auxiliary state is part of the dependence graph.
5. Source writes at exact H/E phases and absolute times; monitor observations
   and chronological reductions as explicit effects, not disposable outputs.
6. Boundary semantics, staggered extents, physical/storage coordinate maps, and
   unsupported operations that produce clear diagnostics.

Trace once per update/region signature, not one Python object per physical
cell. Hash-cons expressions, serialize the IR deterministically, and cache by
equations, precision, target architecture, layout, and specialization choices.
Keep local physics and scheduling separate: the same traced update should feed
one-step, overlapping temporal, and non-overlapping temporal schedules.

The numerical reference is
[`yee_primitives.cuh`](../../cuda/src/yee_primitives.cuh): derivatives round
with `__fmul_rn`, field updates use `__fmaf_rn` with an explicitly rounded decay
product, and CPML has explicit recurrence/correction rounding. Initially emit
these same operations. A future migration to one canonical source of physics
must demonstrate equivalence before replacing the existing definitions.

## Resource model and limits

For temporal depth D and an owned transverse tile X by Y, a naive extension of
the current rolling mstep implementation keeps 2D-1 intermediate stages. Each
stage stores three components and two z planes. With FP32 data and two-plane
input prefetch, the modeled shared-memory footprint is:

```
S(D,X,Y) = 24 * [sum(k=0..2D-2) (X+2D-1-k)(Y+2D-1-k)
                  + (X+2D)(Y+2D)] bytes
```

Ignoring z prologue/epilogue, clipping, masks, and component differences, its
transverse update-envelope ratio is:

```
R(D,X,Y) = sum(k=0..2D-1) (X+2D-1-k)(Y+2D-1-k) / (2DXY)
```

| D | X by Y | Modeled shared bytes | Transverse work ratio |
|---:|---|---:|---:|
| 2 | 32 by 16 | 61,392 | 1.147 |
| 4 | 32 by 16 | 144,672 | 1.362 |
| 4 | 16 by 8 | 50,208 | 1.793 |
| 8 | 16 by 8 | 163,392 | 3.012 |

The first footprint matches the current specialized kernel. These are sizing
equations, not kernel measurements. They omit CPML, extra coefficient state,
register allocation, and scheduler metadata. Depth four at the original tile
size already exceeds SM86's per-block shared-memory capacity. Smaller tiles
fit at the cost of more halo work. Depth eight requires a different storage or
ownership design, not a template argument alone.

Derive stage liveness and communication before generating kernels: last use of
each value, minimum plane-ring length, register/shared placement, exact owned
outputs, and the intermediate samples demanded by neighboring tiles and
monitors. Compare warp-register/shuffle microtiles, rolling shared planes, and
non-overlapping skewed macrotiles. Brick-local layout is an experiment; retain
coalesced component accesses and explicit logical/storage coordinate maps.

A useful idealized field-traffic floor is 48/D bytes per cell per timestep:
six FP32 fields read and written once per D-step chunk. Actual traffic exceeds
it through halos, coefficients, auxiliary state, sample output, repeated cache
loads, and scheduling. This floor is not a DRAM prediction or an achievable
speedup. For measured bandwidth B and actual bytes b, throughput is bounded by
B/b; compute instruction throughput, shared-memory bandwidth, and available
parallelism supply additional bounds. Measure actual traffic before calling
the engine bandwidth-bound.

Amdahl's law applies to the complete solve: if fraction f is unchanged and the
rest improves by s, overall speedup is 1/[f+(1-f)/s]. An unchanged 40% boundary
cost caps speedup at 2.5x even with a free interior. Historical CPML shares make
whole-domain temporal scheduling a central requirement, while leaving their
current values to be measured. A target of 10x from an 8–10 GCUPS CUDA baseline
means 80–100 GCUPS; achieving it would require multiple bottlenecks to move.

## Stages, experiments, and acceptance gates

### 1. Freeze the workload and establish today's best CUDA

Run streamed, current mstep, experimental paired interior, and complete paired
CPML on identical prepared inputs. Record full-state parity, model dimensions,
physical duration, source/monitor configuration, extra scratch, peak allocation,
compilation/capture/startup times, warm execution, and public-call times. Use
the same device and precision; time with completion synchronization. Alternate
variant order and retain individual samples, not only a reported median.

Gate: the harness must distinguish actual timing, unsupported configuration,
OOM, unavailable GPU, and failed numerical validation. No CPU fallback may
appear as a CUDA result. No claimed cross-engine ratio without matched inputs.

### 2. Prove tracing and generated one-step physics

Generate one-step CUDA equations and dependency manifests from the same trace.
Check H/E signs, component offsets, versions, FP32 arithmetic, and closed bounds.
Use seeded nonzero fields so zero-state agreement cannot hide missing updates.
Retain the existing CUDA sweep as the production reference.

Gate: full fields, auxiliary values when supported, and observations agree
under a predetermined arithmetic contract. Report the exact implemented subset.
A synthetic shared-box parity probe is a compiler smoke test, not an integrated
BeamZ backend or scattering correctness result.

### 3. Make depth two cheaper before making it deeper

Have the compiler derive ring liveness, sparse publication, and resource estimates.
Compare a generated depth-two kernel directly with the current specialized
mstep core and with the best complete application schedule. Record registers,
spills, dynamic shared bytes, useful/redundant updates, and kernel durations.

Gate: reduce a measured resource bottleneck or complete runtime. Reject added
abstraction that increases metadata, address recomputation, or scratch traffic
without an offsetting benefit. Do not promote a bare-core win that loses once
edges and observations are restored.

### 4. Compare deeper overlap with non-overlap ownership

Generate depth-four overlap variants only where their resource models fit.
In parallel architectural work, use the prior wavefront version/conflict oracles
to validate non-overlap skewed ownership, with rolling/register microtiles rather
than a full rectangular swept-region allocation. Include cross-pass pipelining
so single-layer parallelism does not define the architecture's ceiling.

Gate: a new diagnostic must first beat its own ordinary reference with numerical
parity, then beat the current best complete CUDA workload. Depth eight is gated
on a viable working set and measured parallelism, not requested in isolation.

### 5. Bring CPML, sources, and monitors into the same local-time model

Specialize CPML faces/edges/corners by active axes with fixed local offsets.
Use FP32 first, preserving the 12-cell thickness and reference profiles. Keep source
injection at its original substep; buffer samples if observations are drained
at a coherent chunk boundary, preserving chronological DFT accumulation.

Gate: full-state parity for asymmetric PML, material transitions, source/monitor
intersections, ragged extents, odd tails, and split continuation. Reduced
precision is a separately validated experiment, not a hidden part of a speedup.

### 6. Evaluate persistent scheduling and an independent CUDA execution boundary

Once task footprints and kernel residency are viable, compare CUDA graph layer
execution with a persistent ready-only queue. A resident CTA must never occupy
the device waiting for an unscheduled producer. Specify device publication and
acquire/release semantics, termination, and bounded queue storage explicitly.

Keep numerical/device execution APIs independent of the current JAX host
adapter. A native CUDA allocation/launch adapter or PyTorch adapter can be
considered after identifying host overhead. Replacing the host library alone
does not remove field traffic or fix temporal dependencies.

Gate: randomized scheduling/version checks, full read/write conflict checks,
Compute Sanitizer memcheck/racecheck/synccheck where applicable, continuation
stress, and no deadlock under occupancy-limited launches. Measure scheduling
overhead separately before claiming it improves kernel performance.

### 7. Promotion and continued research

Maintain a matrix covering Mie-sized cubes, large cubes, narrow and permuted
domains, irregular dimensions, packed/dense/lossy coefficients, CPML faces and
corners, sources, compact and large monitors, and long continuations. Report
logical cell timesteps, excluding storage padding and duplicate halo updates.
Separate setup, compilation, execution, and physical analysis. Retain a known
best control and numerical evidence for each accepted change.

Promote defaults only inside a measured supported envelope. Stop or redirect an
experiment when complete runtime, numerical semantics, or resource measurements
contradict the hypothesis; keep failed experiments documented so subsequent
research does not repeat them.

## Early work and results

### Implemented research artifacts

- [`benchmark_cuda_mie.py`](../../scripts/benchmark_cuda_mie.py) defines the
  physical scene, launches fresh processes for each resolution/schedule,
  synchronizes executable and public-call samples, records software/native
  hashes, and checks fields, clocks, CPML and DFT state. The default sweep is
  5/4/2.5/2/1 nm. CUDA only; the existing JAX FFI remains a host adapter.
- [`cuda_stencil_trace.py`](../../scripts/cuda_stencil_trace.py) records
  versioned affine field reads and explicit rounded arithmetic through Python
  operator overloading. It serializes a deterministic semantic DAG, extracts
  exact multistep initial-field dependencies, and emits scalar CUDA. Direct
  data-dependent branches/comparisons and dynamic indices fail explicitly.
- [`cuda_stencil_rolling.py`](../../scripts/cuda_stencil_rolling.py) generates
  staged rolling-plane CUDA at an arbitrary positive temporal depth, from the
  traced H/E equations. It uses a verified Yee schedule template; it does not
  yet synthesize an arbitrary schedule from arbitrary dependency graphs.
- [`probe_cuda_stencil_trace.py`](../../scripts/probe_cuda_stencil_trace.py)
  compiles and runs a full-box or rolling shared-memory kernel at depths 1/2/4
  and an unblocked two-kernel control. All use the same explicit arithmetic.
  An independently written C++ periodic Yee solver checks all six outputs.
- [`plot_cuda_mie.py`](../../scripts/plot_cuda_mie.py) makes an offline HTML
  chart with selectable warm executable, warm public, and scene-to-first-result
  metrics. It plots measured successful cases without extrapolation.

These are opt-in scripts, without new dependencies or production dispatch
changes. They do not yet support production staggered extents, CPML, material
coefficient layouts, source injection, or monitor effects in generated kernels.
The traced Python equations duplicate the canonical CUDA equations temporarily;
host and independent GPU checks are migration evidence, not a reason to remove
the existing physics primitives now. Structural hash-consing, source/observer
effects, region composition, and automatic liveness allocation remain work.

### Physical fixture and first cached-component baseline

The fixture assumes a 680 nm cube **including** 80 nm of CPML per face, a
100 nm radius lossless nondispersive n=2 sphere in vacuum, a 500 nm Gaussian
finite-aperture plane source, six six-component single-frequency DFT box faces,
30 fs duration, and CFL fraction 0.95. The binary cell-center rasterization has
no subpixel smoothing. This is nanoparticle throughput with nonzero finite
signals, not validated analytic Mie scattering: vacuum-reference complex-field
subtraction, incident-intensity calibration, and an illumination/convergence
study remain required before asserting a scattering cross section.

Initial measurements used the available ABI-30 component 0.30.0 wheel. Its
native SHA is recorded separately from the checkout source SHA; the wheel
does not expose its build commit. CUDA source has no commits after the cached
wheel's November 1 artifact date, but this alone does not prove binary/source
equivalence. Keep this dataset as cached-component evidence.

On the local RTX 3090 (24 GiB, 250 W limit, unrelated idle ComfyUI allocation
about 0.4 GiB), three synchronized executable samples after one warmup gave:

| Resolution | Grid | Steps | Streamed seconds | Mstep seconds | Eligible core |
|---:|---|---:|---:|---:|---:|
| 5 nm | 136 cubed | 3,280 | 1.6571 | 2.0889 | 27.65% |
| 4 nm | 170 cubed | 4,100 | 4.1533 | 4.5373 | 29.26% |
| 2.5 nm | 272 cubed | 6,560 | 25.4195 | 28.1893 | 30.95% |

See the [raw baseline](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-cached-component-baseline.json)
for exact steps, all samples, memory and provenance. Each mstep validation has
zero measured maximum leaf-relative error against streamed CUDA. Public-call
times are also recorded. One fresh process per schedule/resolution and three
samples are initial measurements, not confidence intervals. The 2 and 1 nm
cases are configured but were not run in this first pass.

### First profile and resulting priorities

A separate extra warmed 5 nm mstep call produced real GPU profiler events:

| Event group | Sum of GPU event duration |
|---|---:|
| Edge phases, including CPML and scratch | 61.86% |
| Two-step column core | 30.52% |
| DFT preparation/accumulation | 5.92% |
| Source injection | 1.52% |

These are sums of profiled events with instrumentation overhead, excluding CPU
gaps, **not** unprofiled wall-time shares or a pure-CPML attribution. Preserve
the [grouped kernel data](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-cached-profile-kernel-shares.json)
and [Perfetto trace](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-cached-perfetto.json.gz).
The core uses 61,392 shared bytes, 52 registers and 720 threads per block in
this profile; its launch grid is 5 by 9 by 1, with only 18 active tiles according
to geometry planning. The profiler suggests a minimum grid of 82 for this
kernel. More column segmentation or smaller tiles merit a controlled trial.

Fixed physical PML occupies 55.29% of this cube by volume. That is a geometry
fraction, not a timing fraction. The [geometry diagnostic](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-geometry.json)
also distinguishes source exclusions: removing sources leaves 5 nm eligibility
unchanged at 27.65%, while 4 nm rises from 29.26% to 34.14%, and 2.5 nm from
30.95% to 34.05%. Therefore source-aware tiling alone cannot repair the 5 nm
regression. Prioritize boundary/PML temporal reuse, compact scratch, tile
parallelism, and useful coverage before increasing bare-core depth.

### Generated CUDA validation and first resource experiment

Host tests validate six equations independently with NumPy, multistep impulse
influence with heterogeneous coefficients, serialization, input rejection,
staging versions, ring bounds, and output ownership. The ring suite includes
15 tile/depth cases through depth eight, plus negative controls for a single
plane ring and a missing E coordinate shift.

Fresh generated CUDA compiled with CUDA 13.3 for the RTX 3090. Depths 1/2/4
and the global sweep match the independent C++ solver bit for bit on a
23 by 29 by 37 ragged grid for 9 steps, a 2 by 3 by 7 periodic grid for 5 steps,
and a 64 by 64 by 128 grid for 17 steps, including odd tails. Compute Sanitizer
memcheck, racecheck, and synccheck report zero errors/hazards on a small ragged
case. These checks cover the synthetic common-extent scalar-coefficient
schedule only, not production geometry, coefficients, source or PML semantics.

For a 32 by 8 by 16 owned rolling tile, depths 1/2/4 use 3,672/13,008/40,992
shared bytes. On the synthetic 64 by 64 by 128, 17-step probe, the unblocked
control measured 10.23 GCUPS, depth two 12.48, and depth four 11.80 (seven
event-timed samples after two warmups). All had zero byte mismatches with the
independent CPU output. See the [probe report](../../.cache/cuda-autoresearch-2026-11-11/reports/traced-rolling32-report.json).
This is a small codegen/control experiment, with host submissions inside the
event interval and no CPML, sources, monitors, or production coefficients. It
does not establish an application speedup. A depth-four template still incurs
about twice the useful phase work in this configuration; increasing depth is
not sufficient by itself.

### Reproduction commands

Use a CUDA-enabled environment with the optional native component installed:

```bash
python scripts/benchmark_cuda_mie.py --dry-run --output /tmp/mie-plan.json
python scripts/benchmark_cuda_mie.py --resolutions-nm 5 4 2.5 \
  --samples 3 --public-samples 1 --warmups 1 --validate-schedules \
  --output /tmp/mie-baseline.json
python scripts/plot_cuda_mie.py /tmp/mie-baseline.json --output /tmp/mie.html
python scripts/cuda_stencil_trace.py --depth 4 --output /tmp/cell-trace.json
python scripts/probe_cuda_stencil_trace.py --layout rolling --tile 32 8 16 \
  --shape 64 64 128 --steps 17 --samples 7 --output-dir /tmp/rolling-probe
python -m pytest tests/unit/test_cuda_stencil_trace.py \
  tests/unit/test_cuda_stencil_rolling.py tests/unit/test_cuda_mie_benchmark.py
```

The probe needs `nvcc` on PATH or `--nvcc /path/to/nvcc`; generated runs use
`--arch native` by default. The native engine was also built directly from the
checkout using CMake/Ninja, precise defaults, CUDA 13.3 and architecture 86.
Fresh-source application replay completed successfully:

| Resolution | Streamed executable seconds | Mstep executable seconds | Streamed public seconds | Mstep public seconds |
|---:|---:|---:|---:|---:|
| 5 nm | 1.6653 | 2.0953 | 1.6884 | 2.1206 |
| 4 nm | 4.1380 | 4.5474 | 4.1489 | 4.6021 |
| 2.5 nm | 25.2309 | 28.0226 | 25.4208 | 28.0671 |

The [fresh-source raw report](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-source-build-baseline.json)
is authoritative for this checkout. All six cases have finite nonzero signals;
all three mstep state comparisons have zero measured maximum leaf-relative
error. The native binary SHA is
`5f0be93afd786be9f935f6b2f8a8b2c5029b0d9c55068092f8fdd973f915cc76`.
The [build configuration](../../.cache/cuda-autoresearch-2026-11-11/reports/native-build-info.json)
records precise Release/SM86 settings. The [offline chart](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-baseline.html)
uses this fresh-source dataset. These results confirm the initial regression,
without claiming equivalence to the unspecified FDTDX benchmark.

Validation completed: 76 focused host tests pass, Ruff check/format pass, all
22 native CMake/Ninja build steps succeed, and the three generated-kernel
Compute Sanitizer tools report zero errors/hazards on the recorded small case.
No production kernel or dispatch was replaced, and no 10x gain is claimed.

### Larger-domain control requested during the first pass

We also enlarged the physical cube to 1280 nm at 5 nm resolution: 256 cubed
cells, with the same 80 nm PML per face, 100 nm sphere, source wavelength,
six monitor faces and 30 fs duration. Three warm executable samples after one
warmup measured 8.9966 s streamed and 9.4806 s mstep (6.1167 and 5.8044 GCUPS).
Mstep eligibility rose to 47.66%; its final state again has zero measured
maximum leaf-relative error against streamed. Public-call samples were disabled
for this additional control. Both schedules use the fresh source-built binary.
See the [raw larger-domain report](../../.cache/cuda-autoresearch-2026-11-11/reports/mie-large-source-build.json).

Thus the mstep deficit shrank from about 26% to about 5%, but this case still
does not show a win. Temporal blocking's larger-grid opportunity remains a
research hypothesis here, supported by earlier large-grid calibrations rather
than established by this new fixture. Continue to larger feasible domains,
bare/PEC controls, and source/CPML-aware schedules; do not extrapolate these
measurements to a 10x claim. No further GPU experiments were left running.
