# Five-hour CUDA tracing and temporal scheduling study

Research window: 2026-11-13, 10:59:05–15:59:05 UTC. Local RTX3090, existing
250 W limit, FP32, 12 CPML cells on every face at every resolution.

The final native CUDA composition gives **1.034132× throughput at 512³**
and **1.037139× at 564³** versus the temporal backend at the start of
this session, in fresh direct A/B confirmations. Earlier two-pair ABBA stages
separately confirm the source/profile-cache improvement and the small geometry
increment. **Another 2× improvement was not achieved.** The persistent tile and
half-step pipelines are correct after the startup repair but slower in screens.

The measured implementation is an opt-in native module. Production defaults
and the installed extension remain unchanged. No RunPod resources were used.
Owned GPU jobs finished and were released by **15:51:59 UTC**. The installed
CUDA component hash and the 250 W limit were verified unchanged.

## Complete simulation measurements

Both fixtures simulate a 680 nm cube for 30 fs, with a 100 nm radius index-2
sphere, a 500 nm source and six-face DFT monitors. The 512³ case has 12,347
timesteps; the 564³ case has 13,601. Prepared execution includes preserving
copies, all fields and CPML history, sources, raw observations, chronological
DFT, residual ordinary steps and clock advance. Construction, upload and
readback are outside the warmed execution metric.

The final composition was also compared directly with the frozen starting
control. Each case below uses **one fresh A/B process pair**, with a full first
call and one complete timed repeat per process. Both first and repeat outputs
match all 30 reference leaves bytewise and preserve caller inputs. These are
single-pair confirmations, not confidence intervals.

| Complete workload | Starting temporal control | Final source/profile/geometry composition | Throughput |
| --- | ---: | ---: | ---: |
| 512³ | 111.918213 s | 108.224266 s | 1.034132× |
| 564³ | 158.323692 s | 152.654338 s | 1.037139× |

The two staged tables below each use four fresh processes in ABBA order,
two opposite-order pairs, and one timed repeat after the checked first call.
Their observed ranges are not confidence intervals.

| Complete workload | Starting temporal control | Source/profile-cache combination | Paired throughput | Pair range |
| --- | ---: | ---: | ---: | ---: |
| 512³ | 112.017693 s | 108.703020 s | 1.030493× | 1.030432–1.030554× |
| 564³ | 158.381684 s | 152.969676 s | 1.035380× | 1.034849–1.035911× |

The starting control is module
`52865a3ac8371e2390749953eb3af99746df108fe7bcb00fc4be8b35f4259b92`.
The source/profile-cache combination is
`5fb26e4f476a87bfa861554bec2184e867be8bfbff9e58465dd19583a23617b6`.
These are direct comparisons with the current temporal backend. Pinned-main results in the preceding study are historical; main and FDTDX
were not rerun in this window.

Changing N also changes source/tile alignment and policy cohorts. These two
observations do not establish a general size-scaling law.

The geometry specialization was then compared directly with that stronger
source/profile-cache combination, again using fresh ABBA processes:

| Complete workload | Source/profile-cache control | With geometry specialization | Additional paired throughput | Pair range |
| --- | ---: | ---: | ---: | ---: |
| 512³ | 108.629348 s | 108.255552 s | 1.003453× | 1.003276–1.003630× |
| 564³ | 153.094093 s | 152.663818 s | 1.002819× | 1.001680–1.003959× |

The geometry module is
`1a91ce344d012a9a22b38e6ebc8c3d57248fb0bbdd6d7fc2683d9ac63a13828e`.
Both pair directions favor it, but the increment is small and limited to the
two common shapes. The gains in these two tables are separate measured
comparisons; multiplying them is not used as a final direct measurement.

## Changes that improved complete runs

The first accepted pass extends an exact host certificate for uniform electric
material coefficients to two source-policy families. It retains the ordered
floating-point source additions and every field update. Policy 4 is excluded
because the corresponding compiler output spills registers.

The second pass stages all 36 independent CPML coefficient profiles in shared
memory for certified tiles with neither sources nor observations (NoIO): 36 profiles ×24 FP32
values, or 3,456 bytes. It reuses the existing startup barrier and raises the
shared allocation from 51,748 to 55,204 bytes. H/E, term and coefficient-kind
profiles remain independent; equal-profile assumptions are unnecessary.

The combined build is measured directly. Its gain is not the product of two
isolated screening ratios. All 62 moving entries retain zero compiler
stack/local memory/spill traffic.

The final candidate adds four ahead-of-time entries for 512³/564³ and temporal
depths 4/16. It specializes support predicates, CPML packed addressing, canonical
strides and derivative bounds while preserving runtime physics coefficients.
Other shapes retain the existing fallback. This is common-shape compiler
specialization; it does not trace arbitrary Python physics into a new solver.
All 66 moving entries pass the strict zero-stack/local/spill gate. The 512³
activity witness observes 2,048 new-entry launches and 279,232 certified CTAs,
with 64 registers, 55,204 shared bytes, zero reported local bytes and no dropped
activity records. Noncertified NoIO tiles retain the generic fallback.

## Temporal blocking and tracing architecture

The starting native algorithm already keeps a moving 8×16×32 field cache and
advances up to 16 complete Yee timesteps. Each tile owns 7×15×31 cells and
passes boundary trails to dependent tiles. Because the shortest owned extent
is 7 and the window moves, a physical point receives at most 7 owned updates in
one tile. Increasing the launch's temporal depth alone does not increase that
point's reuse to 16.

The cell tracer records field versions, constant spatial offsets, coefficient
axes, CPML recurrences and explicit rounding operations. Geometry and lifetime
passes derive legal support predicates, storage ownership and guarded runtime
facts. The measured path does not import JAX. Global schedules and kernel
storage still use reviewed templates and hash-bound lowering passes; a general
automatic CUDA scheduler remains future work.

The new persistent prototype replaces many rank launches with 82 resident
1024-thread workers. A prepared topological queue supplies complete tile jobs.
Seven producer dependencies protect reads; producer ownership and seven old
consumers also protect buffer reuse. Epochs, readiness, final completion and
field banks remain distinct. A thread block never overwrites a slot while an
old consumer can still read it.

Further variants publish readiness every timestep, every four timesteps, and
at H/E half steps. The final H/E variant publishes local H before waiting for
upstream H, removing a false H→H dependency. Warp publication uses a full-warp
join followed by an acquire/release chain of 32 warp leaders. E batching changes
publication frequency, never arithmetic or observation order.

## Scheduling results

These are complete-output checks for a **256-step prefix**, not 30 fs solve
measurements. Ratios above1 mean higher throughput than the starting control.
The strongest persistent screen remains below1.

| Persistent method | Throughput ratio | Result |
| --- | ---: | --- |
| Original five-barrier tile worker | 0.617694× | Slower |
| Publication every 4 steps, relaxed polling and shared upstream floor | 0.724977× | Slower |
| Specialized bulk and NoIO, global atomics | 0.944632× | Closest fenced version |
| Same version with cumulative release | 0.949412× | Fence removal changes little |
| Cumulative release plus later three-parent waits | 0.947319× | Exact, essentially unchanged runtime |
| H/E publication with early H and cumulative release | 0.726026× | Slower |
| Same, E publication every 2 steps | 0.732002× | Slower |
| Same, E publication every 4 steps | 0.735057× | Slower |

Specialization reduces the original persistent worker's runtime substantially,
but does not overcome its dependency polling, payload access and scheduling
costs. Deep Morton locality is particularly poor on this queue. These tests
do not establish which hardware stall category dominates; performance counters
were unavailable under the existing driver configuration.

The closest persistent versions use an explicitly declared exploratory spill
budget:8 bytes of stack with4-byte static spill stores/loads. The phase variants
use16 bytes of stack with12-byte stores/loads. These are separate from the
strict zero-spill accepted source/profile-cache build.

Other screens cover shared field storage, private CPML history, hybrid Y
mapping, contiguous family storage, warp-scoped publication, axis rotation,
concurrent NoIO axis cohorts and rolled register-bank selection. None supplies
a replicated complete-run win beyond the accepted source/profile changes.
Several compiler candidates were not executed because their resource use
exceeded the declared admission gate; that is not a measured runtime loss.

## Correctness and memory ordering

The campaign records 35 matched prefix screens and six complete-workload
comparisons, with ten fresh full-workload process pairs in total. Focused CPU
suites pass 270 persistent/control tests and 44 selected-pass/benchmark-contract
tests; these are separate suite counts rather than a unique total.

All accepted timings require bytewise equality of all 30 logical outputs on
the first call and every timed repeat, finite state and unchanged caller
inputs. Output checks include six fields, twelve CPML histories, monitor
counters/timestamps, scalar and vector DFT accumulators and related state.
Additional tests cover unequal H/E profiles, nonisotropic materials, signed
zero, source counts and geometry, rectangular fallback and clock advancement.

An early half-step prototype confused a valid ready value 0 with failure. Its
startup wait skipped later producer-tag checks; the GPU diagnosed tag/base
errors. The fix tests the explicit abort flag instead. A delayed-parent CPU
regression reproduces the old failure and checks the repaired implementation.
No timing from that failed prototype is accepted.

Final readiness and completion cannot be merged. A half-step producer can
publish its final owned E update before it finishes importing the upstream E
ghost cells. Consumers may read the publication, but buffer reuse must still
wait for the separate completion epoch.

The cumulative-release pass removes writer fences only after all writers have
joined through an existing block barrier or a verified warp-leader atomic
chain. It retains the GPU release/acquire operations and consumer gate. CUDA
stress tests perform 40.96 million payload checks across two SMs with zero
errors. This supports the implementation alongside the memory-model argument;
it is not a proof of every possible hardware execution.

The review follows the scoped release/acquire, barrier and causality rules in
[NVIDIA's PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).
[CUTLASS's barrier implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/barrier.h)
provides a useful reference for leader publication after participating threads
have joined. These rules justify the particular joins retained in this pass;
they do not permit arbitrary fence deletion.

The accepted source/profile-cache build passes scoped memcheck, synccheck and
racecheck with zero errors/hazards. Its sanitizer workload is a seeded
64×96×128 domain,17 timesteps, synthetic full-plane source and direct DFT,
with an independent ordinary Program reference. Fixed-shape entries need their
own execution witnesses and sanitizer coverage because small-domain fallback
tests do not exercise them.

All four fixed-shape entries now have nonvacuous CUPTI/ordinary-Program checks:
512³ and 564³, each at depths 4 and 16, with two reset/replays. Every certified
CTA is accounted for, activity records are complete, and all 30 outputs match.
The original 564³ probe worker exceeded host RAM while retaining two full
reference states. Its replacement fingerprints and releases the quiet state
before constructing the full expected state; it retains the original bytewise
candidate comparison.

Ten scoped sanitizer runs are clean: memcheck for all four shape/depth
entries, plus synccheck and racecheck for 512³/depth4, 512³/depth16 and
564³/depth4. The remaining 564³/depth16 synchronization runs were not completed
so the final complete-simulation timing comparisons could finish within the
authorized window. No clean result is claimed for those two runs.

The first large graph-enabled memcheck attempt exceeded 240 seconds. Final
bounded sanitizer checks prepare and hash the independent ordinary reference
outside the tool, then mmap those exact leaves during the instrumented
candidate run. They disable both graph-cache flags for validation and request
one exact FixedCube kernel launch, at a low or late-start high corner. Their
scope is the selected direct launch, rather than whole-grid or graph-mode
sanitizer coverage. The complete timed solves retain graph-enabled execution.

## What the measurements suggest next

The [systolic FDTD paper](https://arxiv.org/html/2502.20610v1#S4)
uses moving subdomains and delayed boundary-trail consumption, the same broad
communication strategy already present here. Its reported implementation uses
16-bit fields and an absorber based on complex permittivity rather than CPML;
its timing setup omits sources and output fields. Its performance therefore
does not predict a speedup for this FP32/CPML12/full-monitor comparison.
[AN5D](https://arxiv.org/html/2001.01473v4#S4) emphasizes register allocation,
double buffering and model-guided CUDA generation. My interpretation is that
those storage and schedule choices are the useful next compiler work, rather
than a new frontend that emits the same kernel instructions.

The current evidence favors reducing work inside a tile over adding more
global readiness machinery. A useful next compiler should emit several legal
tile shapes and thread layouts from the same typed trace, compute exact value
lifetimes and communication cuts, then reject poor register/shared-memory
plans before compiling. It should measure the surviving plans on full
workloads, including CPML, sources and DFT.

A concrete next trace experiment is a short steady-state phase cycle with
explicit register bindings, plus separately generated head/tail cases for
late windows. That can expose bank selection and last-use information without
copying entire private register arrays. Code size belongs in the cost model:
AN5D reports that fully unrolling its repetitive inner loop increased
instruction-fetch latency. This phase-cycle lowering has not been measured
here and should be tested as 2/4/8-phase alternatives rather than assumed faster.

The shortest owned extent7 limits reuse in the present layout. Increasing it
requires a different allocation of the six staggered fields and auxiliary
state, with an explicit occupancy/communication tradeoff. Earlier larger-cube
and512-thread experiments were slower, so this remains a design problem rather
than a demonstrated speedup.

Persistent scheduling should remain a separate research backend until it beats
the simpler control. For the full-step protocol, the final experiment
implements a support-aware transitive reduction from seven producers to three
nearest-axis parents for later waits. It retains all seven startup checks and
all WAR dependencies. The actual controller passes 144 epoch cases; an
independent review checks 23,247 mutable-state handoffs and 2,059 supported
diagonal packets. A concrete late-start counterexample rejects reducing
startup checks. This pass does not apply to the early H/E protocol.

The GPU screen passes all 30 first/repeat checks but takes 2.363456 s versus
2.238947 s for the starting control:0.947319× throughput. Its seven-parent
predecessor takes about 2.365 s in its earlier screen. Removing those four
later checks produces no visible gain; it supplies no basis for predicting 2×.

## Reproduction and evidence

The [source bundle and patch instructions](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/README.md)
collect complete source snapshots plus two small reviewable patches in the
campaign artifact folder. The [CSV](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/complete-results.csv),
[JSON](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/complete-results.json),
[PNG](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/complete-runtime.png) and
[SVG](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/complete-runtime.svg) provide the plotted
complete-run data and exportable figure.

The [validation summary](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/validation-summary.json)
binds the ten clean scoped runs and four actual-entry witnesses. The
[build provenance](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-artifacts/build-provenance.json)
records compiler resources and source identities; the
[final evidence audit](../../.cache/cuda-trace-five-hour-2026-11-13-1059/final-evidence-audit/audit.json)
checks module, fixture, report and output identities.

Campaign root: `.cache/cuda-trace-five-hour-2026-11-13-1059/`.
Frozen module manifests bind source hashes, compiler resources and binary
identities. Each benchmark plan freezes workers and dependency snapshots;
comparisons bind the expected control SHA and identical physical packets.

- `source-uniform-profiles-full512/prefix-comparison.json` and
  `source-uniform-profiles-full564/prefix-comparison.json`: complete30fs ABBA
  results despite the historical `prefix-` filename convention.
- `cumulative-gpu-sixth/`: 256-step persistent screens and exact-output checks.
- `validation-gpu-third/`: scoped sanitizer and independent-profile results.
- `persistent-phase-ready-independent-review.json` and
  `phase-startup-zero-negative/`: startup regression and repair.
- `persistent-fence-coalescing-independent-review.json` and
  `fence-coalescing-stress/result.json`: cumulative memory-order review and
  hardware stress.
- `cta-diagnostic/`: tile lifetime diagnostics. CTA presence is not instruction
  utilization and the diagnostic's extra barrier prevents performance claims.

The research build harness is
[`scripts/cuda_trace_research_build.py`](../../scripts/cuda_trace_research_build.py).
The benchmark orchestrator is
[`scripts/run_cuda_plane_experiment.py`](../../scripts/run_cuda_plane_experiment.py).
The persistent host adapter is
[`scripts/patch_cuda_native_persistent_session.py`](../../scripts/patch_cuda_native_persistent_session.py).
