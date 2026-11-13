# Two-hour CUDA tracing and plane-retirement study

Research window: 2026-11-12 22:11:39 UTC to 2026-11-13 00:11:39 UTC.
Local RTX3090, existing 250 W power cap, FP32, exactly 12 CPML cells per face.
The installed component and backend defaults remain unchanged. No RunPod or
other remote compute was used.

The fastest observed **experimental** build is **131.634 s** on the complete
512³ Mie solve: **1.18210× throughput** over the previous temporal median and
**2.09147×** the earlier pinned-main default-CUDA observation. It passes all
30 first/repeat leaves bytewise, but has only **one fresh process**. Memory
and synchronization checks pass; its broad racecheck reached the time limit
without completing. It remains a provisional research build. The fully
replicated, broadly checked fallback below is **138.112 s**.

The fully replicated candidate combines canonical support facts with immutable
uniform-E coefficient specialization in eligible CPML/boundary tiles. The
confirmed full 512³, 30 fs Mie result is **138.112 s**, versus
**155.604 s** for the previous temporal implementation:
**1.12665× throughput**, or **11.24% less execution time**.
Four fresh processes in A–B–B–A order pass all 30 first/repeat state checks
bytewise. Relative to the earlier same-fixture pinned-main measurements,
throughput is **1.993× default CUDA** and
**1.417× explicit Mstep**.
This round adds 12.67% throughput over the previous temporal build; it does
not establish several-fold acceleration over that already improved control.

## Benchmark contract

The primary fixture is the same one used for the direct stock-main comparison:
680 nm cube, 512³ physical Yee cells (134,217,728 cells), 1.328125 nm mesh,
100 nm radius index-2 sphere, 500 nm source, six-face DFT and 12,347 complete
six-field updates, reaching 30.00148 fs. FP32, all material arrays, source
waveforms, CPML memories, detector clocks and chronological DFT are retained.
Ownership is preserving, with 48 chunks of 256 steps followed by 59 steps.
Both temporal implementations use depth 16, including ordinary residual steps.

Times cover prepared solver execution, including preserving copies, temporal
kernels, sources, observations, DFT, tails and clock advancement. They exclude
scene construction, compilation, upload and readback. Setup and the first
execution are recorded separately; first execution includes material
certification and graph construction. Each full run checks all 30 reference
state leaves bytewise after its first and repeated solve and checks immutable
host inputs. The reference is frozen independently of the candidate; small
gates also execute an ordinary native CUDA Program as an independent oracle.

The four-process confirmation uses A–B–B–A, two fresh processes per variant,
one validated first call plus one timed repeat per process. Process pairs are
the repeat units. Two pairs cannot establish a reliable confidence interval;
the observed range is reported as a range.

Raw commands, exact module hashes, frozen worker, fixture hash, process records
and strict comparisons are in
`.cache/cuda-plane-retirement-2026-11-12/full-abba/`.

## Confirmed full-domain result

| Implementation | First execution, median of two processes | Warmed execution, median of two processes |
|---|---:|---:|
| Previous temporal depth16 |165.735 s|155.604 s|
| **New temporal depth16: NoIO + support facts** |**148.132 s**|**138.112 s**|

The warmed process observations are 155.612478/155.595973 s for the
control and 138.104463/138.120049 s for the candidate. Opposite-order
pair ratios are 1.126773705 and 1.126527060; their median is
1.126650382. This small observed range is not a confidence interval.
Candidate throughput is **11.999 billion complete Yee-cell updates/s**.
Setup ranges from 1.758–4.223 s for the control and
1.921–2.751 s for the candidate; these sparse setup observations
are not a comparative setup-speed claim.

The earlier untouched-main runs at `d5282752db8b258ebbaa5b1b7927026223116b74`
measured 275.308 s for default streamed CUDA and 195.666 s for explicit default
Mstep on this same fixture. Strict identical-physics comparisons against the
new selected binary also pass. Those are historical single-process controls,
not additional A–B–B–A repeats. The new candidate is approximately twice as
fast as that default-CUDA measurement, with 49.83% less warmed solver time.
See the [original direct-main audit](cuda-autoresearch-four-hour-2026-11-11.md#direct-stock-main-comparison-2026-11-12).

[Interactive comparison](cuda-plane-retirement-study-2026-11-12.html) includes
warmed and first-execution views and the separate guide workload. Accepted
numbers and hashes are in `.cache/cuda-plane-retirement-2026-11-12/results.json`;
`full-abba/root-strict-comparison.json` independently checks the paired contract.

## What changed in the selected candidate

The selected module is `noio-support-combined-v1`, SHA256
`4b8c628d0d8261597b625bc292c2c4b5e40b833d671515a38e83d8ceea6eeffd`.
It keeps the previous diagonal moving cache: cached ZYX 8×16×32, owned
7×15×31, depth 16, private field slots and shared phase exchange. All 48
compiled consumers have zero stack and zero allocator spills. Fast consumers
use 64 registers and 1024 threads; general consumers use 122–128 registers
and 512 threads. CPML recurrence, source arithmetic and observation order are
unchanged.

Two compiler changes compose:

1. **Reuse canonical support facts.** Host admission proves all six staggered
   component shapes. The generated consumer reuses the three physical extents
   for output support checks, derivative-wall conditions, CPML extents and
   packed E addresses. The H forward-wall check is unreachable after the
   canonical output-support check; E endpoint conditions remain. This changes
   integer metadata work, not floating-point expressions.
2. **Extend exact uniform-E specialization to eligible NoIO tiles.** NoIO is
   a tile policy with no local source or observation; these tiles can still
   contain CPML. An immutable certificate proves the actual E coefficient
   bits over each supported space-time trajectory. Eligible lossless isotropic
   H tiles reuse those coefficients instead of repeatedly decoding material
   values. The certificate excludes empty unsupported trajectories. Admitted
   rank-2 encodings and nonlossless/nonisotropic H retain their unspecialized
   consumer, as do tiles whose decoded coefficients fail the certificate. Fully
   unsupported features/layouts are rejected by the existing admission gates.
   The complete simulation still executes all sources and detectors.

The full Mie run certifies 17,452 NoIO tiles in addition to 24,745 already
certified bulk tiles, out of 49,248 scheduled tiles. Its resident payload is
13,768,123,836 bytes, including the existing temporal trails and observation
storage; this change adds no new field cache. The existing coefficient
certificate uses a temporary 542,130,192-byte host prefix allocation.

The important result is that tracing helps outside the bulk kernel. The
previous whole-run profile contains 40.45 s when CPML/boundary kernels are
the only active classified group, versus 20.66 s for bulk alone. Because graph
branches overlap, summed kernel durations are not wall-clock shares: the
338.05 s sum covers a 155.04 s union, with as many as five concurrent kernels.
The new specialization targets a measured complete-workload bottleneck.

Implementation entry points:

- [Support-fact lowering](../../scripts/patch_cuda_native_support_facts.py)
- [NoIO uniform-E lowering](../../scripts/patch_cuda_native_noio_uniform_e.py)
- [Complete candidate orchestration](../../scripts/run_cuda_plane_experiment.py)
- [Numerical and branch validation](../../scripts/validate_cuda_plane_retirement.py)

These patchers assemble isolated research source copies. The exact assembled
CUDA/C++ source, manifests, CMake invocation, build log and resource records
are under `noio-support-combined-v1/` in the research cache. They do not install
or select the candidate as BeamZ's default.

## Lifetime-aware tracing: implemented, correct, slower here

The [previous plan](cuda-plane-retirement-next-experiment.md) is now implemented
and tested. The larger cache uses ZYX 16×16×32, owns 15×15×31 cells and uses
512 threads. It retains H in 48 private floats per thread and retires E
through nine short-lived floats instead of retaining both 48-float families.
Its shared allocation is 101,080 bytes. All 48 actual complete consumers
compile without stack or spills, using 110–128 registers.

The lifetime proof is derived from the real canonical Yee reads, including
field versions, physical circular coordinates, private old-H ownership,
shared-plane last uses, boundary ghosts and phase-stage reuse. The full cache
case checks 4,392,834 tokens through 17 versions. Eleven deliberately invalid
schedules are rejected. A separate cross-warp test covers shared-stage reuse.
The proof found a real error in the first barrier-fusion proposal: the final
two E results must remain pending until the last H readers retire. Reusing
the consumed last-E vector repairs this without another persistent vector.

The conservative schedule has 36 barriers per update. The proved fusion
reduces this to 18, preserving the two final retirement/publication barriers.
All numerical gates pass, but neither design beats the smaller cache. The
larger tile admits only one 512-thread CTA, or 16 warps, versus 32 warps in
the current Fast consumer. Reduced logical boundary payload does not prove
reduced total runtime: synchronization, instruction work and available warps
also matter. The measured result rejects this particular mapping, not temporal
blocking or lifetime analysis as general techniques.

## Experiment outcomes

Unless marked full, these are exploratory complete-physics **256-step prefixes
of the same 512³ Mie fixture**, one fresh process per variant and one timed
sample each. They retain full source waveforms, CPML12, DFT and preserving
ownership. Prefix results screen candidates; they are not extrapolated 30 fs
times. Each screen has its own nearby previous-temporal control.

| Candidate | Time | Result |
|---|---:|---|
| Large E-retirement, 36 barriers |4.324 s|0.731× control; rejected|
| Large E-retirement, 18 barriers |4.164 s|0.762× control; rejected|
| Rolled E loop, C++ selector |Not timed|All 48 kernels place a 192-byte H array in local memory; rejected before GPU run|
| Rolled E loop, literal PTX selection |4.175 s|Correct, no stack/spills, about 31% slower than control|
| Depth 8 on previous small cache |3.307 s|0.960× control; rejected|
| Canonical support facts alone |153.171 s full|1.0165× earlier full control; small gain|
| NoIO uniform-E alone |2.827 s|1.1224× prefix control|
| **NoIO uniform-E + support facts** |**2.794 s**|**1.1341× prefix control; selected for full confirmation**|
| E-only private CPML state, on NoIO parent |2.865 s|Slower than its own 2.827 s parent; rejected|
| Whole-cache initial-support certificate |2.797 s|Ties selected candidate; not added|
| Initial address simplification on that certificate |2.794 s|Ties selected candidate within 0.02%; not added|

The C++ rolled-loop result is local-array placement with zero allocator spill
loads/stores; these are different resource outcomes. A literal PTX selector
restored scalar storage without improving runtime. Its net loss does not
isolate instruction-cache behavior from selector cost.

The E-only CPML experiment retains two E-psi arrays in private slots for
certified single-axis, uniform-material CPML tiles; H-psi remains global. It
passes numerical checks and a validation-only atomic witness proves actual
non-first private loads, including reset → tail-only → reset behavior. Its
timing is worse than its own NoIO parent. The witness build is never used for
performance claims. The result does not establish that CPML traffic is free;
the admission, refresh and publication scheme simply did not repay its cost.

The initial-support experiment proves the entire initial halo against all six
component supports and selects unchecked loads only for admitted tiles. The
address derivative also proves dense canonical strides, equal initial/final
shapes and int32-safe indexing. Independent checks include 3,904 address cases
near int32 limits. Both variants are exact, but their timings tie the simpler
selected candidate, so neither is included in the final winner.

## Late experiment: dedicated 1024-thread NoIO kernel

A final resource probe found an important coupling in the existing mapping:
1024 threads were selected through the `Fast` trait, which also means no CPML
and no physical support checks. Eligible uniform-E NoIO tiles need those
checks, so they had retained the 512-thread mapping even after coefficient
specialization. A separate admitted kernel decouples mapping from that trait.

The new body keeps `Fast=false`, CPML, wall checks, active time windows,
late-start initialization and all four phase barriers. It uses 1024 threads,
four fixed Z slots per thread and Z stride two, reducing permanent six-field
state from 48 to 24 floats/thread. Its two depth4/16 kernels fit **64 registers,
zero stack/local/spills**, with the same 51,748-byte shared allocation. The
linked runtime has 50 kernels, all zero stack/local/spills. The change allows
32 warps per eligible NoIO CTA instead of 16; resource fit alone does not
establish a speed gain.

The original 512-thread kernel and new 1024-thread kernel receive the same
NoIO tile list. Complementary immutable, CTA-uniform gates select exactly one
body **before** owner writes or barriers. Both launches stay on the same
existing policy stream before its completion event, preserving rank joins.
This temporary integration launches inactive blocks rather than adding a new
host cohort; their overhead is included in timing.

Independent source binding proves that reversing only the T expression
restores the exact previous body, while environment, import and stage helpers
remain unchanged. Both thread mappings cover all 4,096 slots once; 33 wrapped
versions check 6,930 low-X staging points, and all 256 flag bytes select one
writer. All 97 primary states, eight seam probes and 12 fallback states pass
against the ordinary Program. A fresh paired 512³ prefix measures
**2.803090→2.658559 s**, or **1.05436×** over the stronger NoIO/support control.
This prefix is a screen, not a full-solve speed claim.

The subsequent **full 512³, 12,347-update/30 fs solve** measures
**131.633664 s warmed**, with first execution **141.743699 s** and setup
2.671496 s. All 30 first/repeat leaves are bytewise exact and host inputs
remain unchanged. This is **one fresh process and one timed repeat**, not a
second A–B–B–A study. Compared with the replicated NoIO/support median it is
1.04922× throughput; compared with the previous temporal A–B–B–A median it
is **1.18210× throughput** (15.40% less runtime). Against the earlier pinned
main-default observation it is **2.09147× throughput**. The strict complete
comparison, frozen worker and report are in `noio1024-validation/`.

Late validation status: memcheck completes with zero errors in 69.54 s and
synccheck completes with zero errors in 19.79 s, both at depth16/17 updates
with exact reset/repeat state. The broad all-Asymmetric racecheck times out
at 90 s without a completed findings summary. A narrower single-run check of
only the new 1024-thread wrapper also times out at 42 s; an intermediate
two-repeat narrow attempt was cancelled to fit the remaining window. None of
these race attempts counts as clean coverage. No CUPTI launch-witness run is
performed. The complete numerical result remains valid, but the new mapping
**is not promoted past the racecheck gate**. The preceding replicated build
retains its completed three-tool checks as the fallback. Full records and
cleanup status are in `noio1024-validation/gpu-validation-summary.json`.

The module is `noio-uniform-1024-runtime-v1`, SHA256
`4941e6728b10e803d03d0f09e137cbbecd3b187eb4ac9686fe9fa7161da18e80`.
The [copy-only patcher](../../scripts/patch_cuda_native_noio_uniform_1024.py)
reproduces all 50 copied source files bytewise; six negative controls and Ruff
pass. Its manifests and independent review are archived beside the binary.
The optional full profile of the replicated candidate was cancelled before
completion to give this runtime experiment the remaining GPU time; that
cancelled profile supplies no timing claim.
The guide and 272³ observations below belong to the replicated NoIO/support
generation, not this later thread mapping; their gains are not transferred
to an unmeasured variant.

## Supplementary waveguide result

A separate complete 512³ guide, 512 updates, mode source and two fixed-aperture
three-frequency monitors runs in **4.446652 s**, versus **5.120642 s** for the
stronger D2 UniformE-v2/segment64 control: **1.15157× throughput**, or 13.16%
less time. Both use FP32, CPML12, preserving ownership and two 256-step chunks.
First and repeated state checks are bytewise exact across all 30 leaves.

This is one fresh process per variant, three within-process timing samples
each. It is a separate guide fixture, not the Mie case, stock main, or a
multi-process confidence estimate. Evidence is in `guide-supplementary/`.
The first guide execution is **6.495 s** for the new temporal build versus
**5.125 s** for D2; setup is 2.518 s versus 3.123 s respectively. Thus the
warmed guide win does not imply a faster single short solve. Dispatch must
consider certification/graph reuse as well as domain size.

## Smaller Mie crossover: retain D2

The complete 272³ Mie scene, 6,560 updates/30 fs with CPML12 and the same
source/DFT features, gives the following warmed observations:

| Runtime | Full 272³ Mie |
|---|---:|
| Strong D2 UniformE-v2 |**16.354 s**|
| Previous moving temporal |25.099 s|
| New NoIO + support moving temporal |22.740 s|

The new compiler improves this temporal path by 1.1037×, but reaches only
0.7192× the strong D2 throughput. Therefore the larger-domain result does
not support selecting moving temporal blocking for every simulation. These
are one fresh process per variant, with exact first/repeat all-30 state and
host immutability; the two strict comparisons are in `mie272-crossover/`.
Scene geometry, material coverage, monitor footprint and reuse costs must
inform future selection as well as raw cell count.

## Validation and limits

The selected candidate passes the 97 primary seeded/material/clock cases plus
eight source/observer seam probes, including first/repeat state. Additional
mixed/constant rank-2 encodings, including negative constant-block headers,
and nonisotropic-H cases exercise fallback and zero eligible-NoIO counts.
All comparisons use all 30 state leaves,
including CPML, detector/DFT state and clocks, bytewise. Source and detector
coincidence, odd tails, nonzero time origin, owned/preserving state, continuation
and reset/replay are covered by the bounded suites.

All three bounded Compute Sanitizer checks complete cleanly on the exact
selected binary: memcheck zero errors, synccheck zero errors, racecheck zero
hazards. Scope is the actual Asymmetric kernels on a seeded 136³ small-Mie
fixture, depth 4, seven steps, two coincident point sources and direct
six-component DFT, with ordinary Program oracle and two reset/repeat checks.
Actual macros and uniform-NoIO branches are positive. No JAX, API-error filter
or error suppression is used. These are bounded checks, not a sanitizer run
of the whole 512³ simulation or every rejected design.

A second complete three-tool check covers the benchmark's **actual depth 16**
on seeded ZYX 64×96×128, 17 updates. Each reset/repeat state executes one
16-step macro plus one ordinary tail step, with 424 certified NoIO tiles.
Memcheck, synccheck and racecheck all complete with explicit zero summaries
and all 30 ordinary-Program reference leaves bytewise exact. Their respective
wall times are 64.8, 17.9 and 77.7 s; no tool times out or leaves a process
group running. Exact commands, counters and logs are in
`combo-sanitizers-depth16/`.

The final CPU-only targeted sweep passes **54 tests** covering retirement,
emitted-source admission, rolled/assembly selection, cross-warp stage reuse,
depth 8 and the initial-support certificate. Separate support-fact proofs
check 73,002 component-support bits, 25,740 valid derivative-wall cases and
7,425 packed-E addresses. Token proofs establish storage/version/order facts;
they do not replace floating-point parity, compiler resource checks or GPU
race/synchronization tests.

The native prepared-input execution imports no JAX. Scene construction and
independent reference export still use JAX; the benchmark environment itself
has no JAX installed. The frontend is a specialized
trace of BeamZ's canonical Yee operator, not a general arbitrary-PyTorch
tracer. Public API integration, backward/adjoint support, other GPU families
and systematic crossover selection are outside this bounded run.

## Architectural conclusion and next priorities

The next compiler investment should make support, material and phase facts
explicit in the common IR and generate admitted consumers directly, replacing
the current research source patch chain. The measured winner removes work
from CPML/boundary consumers while preserving the existing efficient cache.
It deserves integration gates before automatic selection, especially on small
domains and scenes with little uniform-material coverage.

Concretely, a certified uniform-E region should replace spatial coefficient
lookup leaves with runtime scalar leaves while retaining the exact coefficient
bits and every floating-point rounding node. Canonical geometry should lower
support/address predicates in the same pass. This removes lookup/control
subgraphs and their metadata lifetimes from the IR, instead of depending on
callback templates to erase them after code generation.

The late NoIO experiment also exposes a compiler-model problem: semantic
admission and thread mapping should be separate choices. `SupportSafe`, CPML,
coefficient uniformity and H classes determine legal operations; T/Slots/ZStep
determine where those operations execute. A single `Fast` switch had coupled
them. Complete broader race coverage before promoting the new mapping. The
next integration should then generate its dedicated kernel and compact
certified tile list directly, removing the temporary complementary
inactive-block launches.

For a larger future jump, the next experiment should address the mapped
instruction/synchronization cost of a larger tile: component-specific last
uses and warp-local exchanges, fewer cross-warp publication events, and a
register/shared allocation chosen together. Retrying the same 16-plane design
with deeper time blocking is not supported by this round's measurements.
Profile active warps and instruction/issue stalls before another large layout
sweep; do not use logical requested-byte reductions as a speedup forecast.
One candidate for a later proof is a two-component shared frame: with warps
along X, only components 0 and 2 cross Y, while component 1 can remain private
and use warp exchange along X. A cached 8×32×32 tile would preserve 32 warps,
but its estimated field frontier is 38 floats/thread. The current Fast mapping
has only 24: its 1024 threads split the eight Z slots into two four-slot groups.
The proposal therefore needs at least 14 fewer metadata/temporary registers
to fit the same 64-register launch budget, plus complete ghost/import and
collective-convergence proofs. Shared-memory fit alone does not admit it.
The paper's geometry remains useful, but its source/output-free, non-CPML
fixture and reduced-width transport do not supply a matched FP32 BeamZ speed
claim. Those conditions are explicit in
[the paper's implementation section](https://arxiv.org/html/2502.20610v1#S4);
see also the [paper audit](cuda-large-domain-first-principles-2026-11-11.md#paper-audit-what-is-actually-specified).
The paper specifies a space-time schedule and cache geometry, rather than a
Python tracing frontend or a register allocation recipe. Those compiler and
storage decisions still require independent derivation and measured gates.

All raw measurements, proofs, exact copied sources and command plans are under
`.cache/cuda-plane-retirement-2026-11-12/`. Production binary SHA256 before and
after this run remains
`34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`.

GPU research stopped at 2026-11-13 00:06:04 UTC. All owned CUDA/sanitizer
process groups exited; the pre-existing idle ComfyUI process was left alone.
No further engineering runs are queued. The final report and visualization
were assembled within the original two-hour window.
