# Five-hour temporal CUDA and tracing study

Authorized research window: **2026-11-13 00:15:40–05:15:40 UTC**.
All owned GPU jobs finished by **04:56:39 UTC**.

Local RTX3090, unchanged 250 W power cap, FP32, exactly **12 CPML cells per
face at every resolution**. No RunPod or other remote compute. All research
modules are isolated; the installed CUDA component and production selection
defaults remain unchanged.

## Complete-workload results

The selected native CUDA source-storage hybrid completes the full **512³ Mie
solve in 116.079 s**, versus **124.584 s** for the strongest verified prior
candidate. The balanced A–B–B–A comparison gives **1.073262× throughput**,
or **6.826% less solve time**. Each variant has two fresh processes, each with
an independently validated first execution and one timed repeat. Every first
and repeat result matches all 30 reference leaves bytewise and preserves caller
inputs. Opposite-order pair ratios are **1.073064×** and **1.073461×**; this
observed range is not a confidence interval. Throughput is **14.276 billion
complete Yee-cell updates/s**.

The fresh pinned-main default streamed-CUDA control takes 276.397 s on the same
fixture, so the selected candidate has **2.381× observed throughput**. The
full **564³** confirmation gives **182.222→166.083 s**, **1.097178×**
throughput or **8.857% less solve time**. All four first/repeated solves match
all 30 leaves bytewise and preserve inputs. The larger-case observed pair
ratios are 1.097353× and 1.097004×. Its earlier comparison confirmed observation
specialization at 195.615→182.159 s, 1.073872×.

| Full workload | Control | Candidate | Status |
| --- | ---: | ---: | --- |
| 512³, previous verified fallback → starting NoIO1024 | 138.136 s | 131.491 s | Four fresh A–B–B–A processes; median paired throughput 1.050534× |
| 564³, starting NoIO1024 → observation specialization | 195.615 s | 182.159 s | Four fresh A–B–B–A processes; median paired throughput 1.073872× |
| 512³, observation specialization → bundled axis specialization | 125.622 s | 124.604 s | Four fresh A–B–B–A processes; median paired throughput 1.008164×; scoped positive memory/sync/race checks pass |
| **512³, strongest bundled-axis control → selected source hybrid** | **124.584 s** | **116.079 s** | **Four fresh A–B–B–A processes; median paired throughput 1.073262×; changed-source tools clean** |
| **564³, observation control → selected source hybrid** | **182.222 s** | **166.083 s** | **Four fresh A–B–B–A processes; median paired throughput 1.097178×** |

For the selected full512 comparison, the control first-call mean is 134.837 s
and the candidate first-call mean is 126.191 s. Control warmed observations are
124.586461/124.580796 s; candidate observations are 116.103517/116.055248 s.
Native setup/upload is separately recorded: 1.897–3.380 s for the control and
2.532–2.914 s for the candidate. These sparse setup observations do not support
a setup-speed claim. First executions include certification and graph work;
the warmed comparison includes normal prepared solver execution.

A **fresh pinned-main default streamed-CUDA control** completes the same full
N512 solve in **276.396958 s**, versus 275.546057 s on its first call. Both
first/repeat states match all 30 reference leaves bytewise and host inputs are
unchanged. It uses main `d5282752db8b258ebbaa5b1b7927026223116b74`, JAX 0.9.0's
compiled FFI call and the default streamed schedule. The optimized research
runners use native CUDA without JAX in their execution path. Compilation,
construction/upload and readback are excluded in both. The generic worker's
`method=mstep` label does not override its explicit `stock_schedule=streamed`.

The 116.079 s candidate therefore has **2.381× observed throughput** versus
this fresh main control, and about **1.686×** versus the earlier pinned-main
explicit-Mstep observation of 195.666 s. The fresh default control is one
process, not an A–B–B–A comparison with the candidate; the historical explicit
Mstep result is separately labeled. Raw evidence is in
`fresh-stock-main-default/`. Neither comparison establishes a 10× gain.

The N564 fixture retains the 680 nm cube, 100 nm radius index-2 sphere,
500 nm source and 30 fs duration. Its spacing is 680/564 nm, with 13,601 full
six-field timesteps, approximately 2.4401 trillion complete Yee-cell updates,
and six-face DFT. This is about 34% more cells than N512. The independent
ordinary reference completed within existing host/GPU memory guards;
all reference leaves are finite and all six DFT faces are nonzero. Peak sampled
device-wide residency during the one-time ordinary reference export was
22,573 MiB; this is not a selected-native per-process peak or memory-saving
claim. Larger proposals that failed the existing guards
were rejected without bypassing them.

For N564, control repeats are 182.160410/182.283948 s and candidate repeats
165.999831/166.165270 s. First-call means are 199.389/182.975 s respectively.
The larger gain is not a universal size-scaling result: the H/E-phase source
plane X coordinates are 104/105 at N512 and 115/116 at N564. Over versions1–16,
the skewed coordinate X+version stays within one owned-X block for N512 but
crosses the 124-cell boundary for N564. This static alignment changes source
cohort counts, including a nonempty both-IO cohort at N564; it is not a
quantitative causal decomposition of the measured gain.

A **272³** full30fs crossover check does **not** establish a win over strong
tuned D2: 16.482343 s versus 16.659733 s for the selected hybrid, **0.989352×**
throughput. This is one fresh matched pair, not replicated evidence of a small
regression. First/repeat all30 checks and immutable inputs pass. Keep selection
domain-aware; the large-domain winner should not replace every small-domain
configuration. Raw evidence: `selected-source-v12-app-only/full272-crossover-fresh-pair/`.

N512 is the established same-physics fixture: 512³ physical cells,
1.328125 nm mesh, 12,347 timesteps and 30.00148 fs. Timings cover prepared
execution, preserving copies, temporal updates, CPML, sources, observations,
chronological DFT, residual ordinary steps and clocks. Scene construction,
compilation, upload and readback are excluded. Setup and first execution are
recorded separately. Preserving ownership and 256-step chunks are identical
within each comparison.

Raw full comparisons are under `.cache/cuda-five-hour-2026-11-13/`.
Historical `full512-ABBA/` and `mie564/full-ABBA/` have `compact-summary.json`;
`full512-Bundle-Obs-ABBA/` has `root-summary.json`. The final
`selected-source-v12-app-only/{full512,full564}-ABBA/` directories each have
`curated-summary.json`. Their older generic compact-summary adapter has null
parity fields because it expected different key names. The raw four reports
and strict comparator supply the actual bytewise checks. Every campaign freezes its worker, fixture, source and
module hashes. New campaigns also require the expected control module hash
and verify the actual bytes at its recorded path before execution/comparison.

[Interactive comparison](cuda-five-hour-study-2026-11-13.html) separates domain
sizes and offers seconds/GCUPS views. Accepted numbers and original-report,
fixture and module hashes are assembled in the cache's `results.json`;
`finalize_results.py` independently rechecks each completed full comparison.

## Selected implementation: still genuine temporal blocking

The starting NoIO1024 implementation keeps a moving 8×16×32 cache, with a
7×15×31 owned region, and advances it through 16 complete Yee steps before
leaving the temporal kernel. It exchanges boundary trails between diagonally
ordered tiles. Field interiors stay private/shared across those steps. This
is multi-stepping and temporal blocking, rather than sixteen ordinary global
updates grouped only for launch savings.

NoIO means a tile has no local source or observation; it does **not** imply
that CPML is absent. The actual CPML recurrence and support/boundary checks
remain. The starting module's previously incomplete race gate now completes
cleanly on a positive rectangular NoIO case with a 17-step depth16-plus-tail
run. Its memory and synchronization gates were already clean.

The observation-aware specialization extends exact uniform electric-material
facts to source-free observation tiles. Admission examines supported points
over the complete owned space-time trajectory and compares coefficient bits;
it does not assume a material value or resolution. The new path retains
observation packet stores and chronological DFT. Failed certificates, other
layouts and unsupported magnetic facts retain their original consumers.
Its planned numerical suite and positive memory, synchronization and race checks
pass; the positive test proves an actual eligible observation tile executes
and verifies sensitivity to the source-driven DFT.

The symbolic tracer is a small typed Yee/CPML operation graph, not JAX tracing
or PyTorch FX. Runtime fact lowering substitutes proven readonly leaves while
preserving every RN multiply/FMA, operand order, support predicate and mutable
state identity. [The compiler report](cuda-tracing-runtime-facts-2026-11-13.md)
describes runtime uniform facts, paired phase generation, exact axis pruning,
guarded auxiliary-state interference and resource failures.

## Larger cache: storage schedule and compiler architecture

The new experiment doubles the cached Y dimension: **8×32×32**, with
**7×31×31** owned cells. H remains private. Six of each eight Ey planes use
shared storage and two remain private. Ex/Ez reuse the current Hx/Hz exchange
planes only after their last readers finish. Pending E values reuse retired
scratch. The mapping uses 26 persistent field floats plus two current Ex/Ez
values per 1024-thread thread, 100,984 shared bytes and eleven barriers per
step. Generic consumers use 512 threads; source-heavy fallbacks use 256.
The point operator's generated bytes are unchanged.

An independent actual-stencil integer-token oracle checks 29.8 million field
reads across 130 configurations, including wrapped heads, 31/64 versions,
late starts, Y groups, randomized warp run-ahead and deliberately broken
publication/retirement schedules. See the
[lifetime audit](cuda-partial-ey-lifetimes-2026-11-13.md). This establishes
storage-routing evidence, separately from floating-point and CUDA race tests.

An important full-engine discrepancy was isolated: an otherwise identical
non-RDC resource probe used zero local memory, while the engine's RDC build
introduced outlined shuffle convergence helper calls and live-state spills.
A controlled pair using the actual descriptors, full headers, namespace,
entry points and flags reproduces the difference. Narrow inline PTX shuffles
did not remove it. The repair gives header-contained moving kernels their own
whole-program CUDA translation unit while ordinary core/runtime code retain
RDC. Actual generated Ninja rules verify inherited source definitions and
extended-lambda flags; intended CMake settings alone are insufficient.

The full repaired module has **52 moving entries, all with zero stack, local
memory and allocator spill loads/stores**. Registers are 64 for the uniform
1024-thread entries, 128 for 512-thread classes and 221–235 for 256-thread
fallbacks. Unchanged ordinary/DFT kernel resource records are outside this
moving-only gate. The repaired module then passes its full base suite plus
28 new geometry/clock states, all 30 leaves bytewise. The additional cases use
a separate pinned observation-parent ordinary-Program process and place
sources/detectors at Y15/16/31/32/46/47, with shifted step/origin/elapsed clocks.

Its first proper matched prefix pair is **slower**: 2.533421 s for the
observation-aware control versus 3.660589 s for the larger layout, **0.692080×**
throughput. It is rejected from full-solve promotion. The 50% tile-count and
9.73% resident-trail reductions do not compensate for the changed synchronization,
resource mapping and memory hierarchy in this screen. Those are plausible
costs, not a hardware-counter attribution.

A follow-up derives the actual six-component source-presence mask from
validated private descriptor counts. Only per-component source graphs proven globally empty are
removed. For mask 20 (Hz/Ey), active coefficient products and ordered RN adds
remain unchanged, including the two-ULP spatial variation in Ey source
coefficients. A separate four-byte workspace fact enters the graph key and is
independently checked against the prepared source packet. Other masks retain
the general source path. This allows 512-thread source variants at 119/124/128
registers instead of 256-thread source-only/both-IO fallbacks at 221–225.
The full follow-up has 58 zero-local moving entries. After a host-only fix to
its diagnostic Python tuple conversion, it passes the base and geometry/clock
suite plus 88 source-presence cases. Its proper prefix control is
2.530122 s versus **3.091449 s**, **0.818426×** throughput. This recovers part
of the source cost but still loses; no full-solve promotion. The failed
metadata run is retained and provides no numerical result. Instrumented
prefix follow-ups use a warmed second call; their zero-warmup first calls
include graph construction and must not be presented as warmed times.

The subsequent hybrid keeps the existing small 8×16×32 cache and the original
bulk, NoIO and source-free-observation policy schedules. Policies 3/4/5 receive a different storage
schedule: all Ey stays shared, H stays private, Ex/Ez pending values stay
private, and separate Hx/Hy snapshots cover opposite Z chunks. At 1024 threads
this uses 16 field floats, seven barriers and 58,680 shared bytes, compared
with the original kernels' four barriers and 51,748 bytes. The point graph,
global trails, rank order and observation packets are identical. This frees
enough registers for source-mask20 kernels at 1024 threads (64/64/60 registers
for source-only/both-IO/generic policies), with zero spills and no electric
material-uniformity assumption. Admission requires byte-packed layout 1,
lossless/isotropic H (all three decays are +1 and the three scalar H curl gains
match bitwise), global source-presence mask 20 and compiled depth 4 or 16. Other
admitted layouts/H facts/masks use their original moving consumers.

The hybrid's whole moving-TU compilation mode also changes the generated
assembly of the original kernels despite unchanged source bodies. An
independent SASS audit records that fact. A hidden immutable host selector,
included in metadata and the graph key, permits an on/off control produced by
host-only relinking: all CUDA objects are retained. That control separates
source scheduling from the compilation-mode change. The hardware prefix is
**2.527920→2.328743 s**, **1.085535×** throughput with byte-identical CUDA
payloads. A separate proper observation-parent comparison gives
**2.530121→2.317182 s**, **1.091896×**. Both are short screens, not full solves.

The final five-barrier source variant batches two Z planes per publication.
Four old-own Ex/Ez values and four previous-pair results coexist with twelve H
values, so the source-level bound is 20 field floats. It retains independent
opposite-Z snapshots and the final reader-retirement/publication barriers.
All three compiled policy entries fit at 64/64/62 registers with zero spills.
Its proper prefix is **2.53469→2.32068 s**, **1.09221×**. This ties the
seven-barrier variant within these observations; fewer barriers do not establish
an additional speedup.

The selected variant passes **289 exact state/clock comparisons and 16 explicit
rejection checks**: 117 base, 52 geometry/source/DFT seams, 104 source-guard
cases and 16 observation cases, plus four unsupported-admission conditions on
four fixtures. Every planned command passes. The earlier nominal “297” label
mixed categories; the final ledger recounts actual hashed outputs. Source
tests include both sides of each Z-chunk snapshot, shifted clocks,
split/reset/preserving execution, mask/layout/H fallbacks, caller mutation,
zero coefficients retaining source presence, and live non-dyadic products with
coefficients differing by exactly two FP32 ULPs.

Direct CUPTI evidence records 864 actual new source-kernel launches when enabled
and zero when disabled, with no dropped records and all 30 first/final leaves
bytewise exact. In the selected five-barrier build those are 112 source-only,
272 both-IO and 480 generic-policy launches, all using 1024 threads. Generic
policy 5 also includes boundary tiles without a local source; their original
per-tile source/observation/CPML guards remain. The enabled/disabled modules
contain identical CUDA sections; only immutable host selection differs.

Memory, synchronization and race checks complete cleanly for every changed
source policy on 35 updates: two depth16 macros plus three ordinary residual
steps. All 30 leaves match fresh independent ordinary-Program children, with
live-source and DFT sensitivity. Instrumentation covers the candidate parent
application and `AsymmetricSourcePending20` kernels; unchanged oracle children
are explicitly outside the tool scope. The earlier all-child attempt is retained
as cancelled/incomplete, not a pass. Numerical reset/preserving coverage is
separate from this scoped owned-solve tool coverage.

One failed prefix process received SIGBUS because an audit command rewrote the
loaded ELF in place while dumping its device section. The final bytes were
unchanged, but the rewrite invalidated an instruction-fetch mapping. That run
has no numerical/performance result. Execution modules are now copied once to
never-built read-only paths, and a tested [read-only ELF auditor](../../scripts/audit_cuda_fatbin_identity.py)
verifies device identity without binary-tool mutation. The same-hash retry is
the accepted prefix result. Raw failure/backtrace and mtime evidence remain.

Final source work is in `source-pending-paired-on-v12/`, its matched Off sibling,
`frozen-execution/`, `source-selector-v12-kernel-witness/` and
`selected-source-v12-app-only/` under the study cache. The complete-solve
confirmation passes on both full N512 and N564 workloads.

Reproducible producers and gates:

- [Source-only storage hybrid](../../scripts/patch_cuda_native_source_pending_hybrid.py)
- [Paired source-plane schedule](../../scripts/patch_cuda_native_source_pending_pairs.py)
- [Immutable matched-control selector](../../scripts/patch_cuda_native_source_pending_selector.py)
- [Partial-Ey cache producer](../../scripts/patch_cuda_native_partial_ey_frame.py)
- [Resource-class mapping](../../scripts/patch_cuda_native_partial_thread_classes.py)
- [Bulk partition/dispatch proof](../../scripts/prove_cuda_native_bulk_dispatch.py)
- [Moving-only compilation boundary](../../scripts/patch_cuda_native_separate_moving_tu.py)
- [Actual compiler/resource contract checker](../../scripts/check_cuda_moving_build.py)

Exact source/build/resource evidence is in
`.cache/cuda-five-hour-2026-11-13/partial-ey-separate-tu-v7/`.

## Rejected or unpromoted experiments

All ratios in this table are **short prefix screens**, not complete-solve
claims. Each uses the correct guarded parent/control after the control audit.

| Experiment | Candidate/control throughput | Decision |
| --- | ---: | --- |
| Early synchronous low-X prefetch | 0.9402× | Reject |
| Async low-X transfer | 0.9142× | Reject despite zero local memory and real async instructions |
| Paired CPML phase/common profile loads | 0.9941× | Fewer static loads did not improve time |
| Depth32 | 0.9416× | Reject; larger trails and more expensive policy intersections |
| Serial axis-specific launch grids | 0.8638× | Reject; small grids fragment execution |
| Reduced exact cohort dependency DAG | 0.9750× | Correct, but slower |
| Whole-trajectory support certificate | 1.0021× | Tie; no promotion |
| Bundled CTA-uniform axis specialization | about 1.0115× | Full512 confirms a smaller 1.008164× gain; scoped positive sanitizers pass |

The dependency DAG tracks seven-neighbor field/psi RAW edges, actual recycled
slot WAW edges and every old-reader WAR edge. Its CPU proof includes millions
of hazards and adversarial schedules; CUDA instructions remain identical to
the parent. Transitive reduction cuts cross-policy waits from 4,416 to 1,339
per full512 macro, but graph freedom did not improve the matched screen.
This prevents mistaking a correct scheduling proof for a performance result.

Two early screens had a compact-control binary mislabeled as the stronger
observation control. They were detected before promotion, archived as invalid,
and rerun against the correct actual module. The comparison tool and campaign
runner now fail closed on the expected control hash and actual binary bytes.
Those invalid screens contribute no accepted numbers here.

Bundle memory/sync/race checks complete on a source/DFT-live seeded rectangular
17-step depth16-plus-tail case. Each tool's actual nested witness contains 423
certified NoIO tiles and one macro; all 30 leaves match. CPU/source-bound
reconstruction identifies productive Z/Y/X-only counts of 50/30/6 and 337
multi-axis tiles. These are verified routing counts, not per-axis device
counters. The tool scope is one owned solve; separate numerical tests cover
reset/preserving behavior. A legacy summary adapter read nonexistent root
keys and falsely printed `positive_noio=false`; raw outputs are retained and
`bundle-positive-sanitizers/summary-scoped.json` is reproducibly derived from
the actual nested witnesses, source/fixture hashes and clean tool logs.

## How this relates to the paper

[The systolic FDTD paper](https://arxiv.org/html/2502.20610v1#S4) describes
upwind-moving cached subdomains and downwind boundary trails, so interiors can
advance locally without redundant growing halos. BeamZ already implements
that basic architecture. Its practical H100 example stores 32×16×16 cells,
owns 31×15×15, and reports approximately 5.5× over its own naive baseline.
That experiment uses FP16, no sources/output and an absorber rather than PML;
its reported multiplier is not a prediction for BeamZ's FP32 CPML12 Mie solve.

The next compiler task is to derive feasible storage/synchronization schedules
from the actual read graph and hardware budgets, then measure complete solves.
Simply increasing depth or counting fewer global-load sites has already failed
several matched experiments in this round. The new moving-only compilation
boundary removes a demonstrated resource obstacle without changing physics.

## Next register-budget escape: shared auxiliary state

The private Eψ point-cache probe still spills 24 bytes per thread, even with a
single field/import loop and typed point callbacks. A new lowering moves that
same guarded two-color state into a volatile shared bank. The eight logical
values per thread become 8,192 words indexed by `(color*4+slot)*1024+thread`;
the existing NoIO mapping uses even/odd Z ownership. The appended bank occupies
`[51748,84516)` bytes, separate from every field/metadata/staging allocation.
Each cell has one fixed thread owner; warp addresses cover all 32 banks, and
the existing first-owned reload and high-face/final flush rules are unchanged.
No new inter-thread barrier is required.

CPU checks cover active supported warm points, index/stride/color collisions,
reload/flush negatives and arbitrary cross-thread interleavings. Both depth4/16
actual-context resource entries use 64 registers with zero stack/spills.
Private→shared static SASS removes local access sites, adds 20 shared-load and
20 shared-store sites, and leaves global-load/store site counts unchanged.
These static counts are not traffic or execution measurements.

This is **resource-only work**, not an integrated or GPU-validated speedup.
Its current parent requires privately captured equal-profile and disjoint-state
facts; it is not a drop-in substitution into the selected source engine.
The 84,516-byte allocation may require a larger shared-memory carveout and reduce
available L1; this was not measured by the resource probe. The next experiment must measure that trade against the verified
full engine, with fallback and numerical/tool gates intact. A derivative using
the canonical scalar point graph can remove the equal-profile assumption.
Evidence is in `noio-epsi-shared-bank-resource-v1/` and the
[runtime-facts report](cuda-tracing-runtime-facts-2026-11-13.md).

## Final selection and remaining work

The selected research binary is `source-pending-paired-on-v12`, SHA256
`a9394c3031d6635a875d673d50e7e09123807b7d9d2fe353c49fa05063b5941b`.
Its execution copy is under `frozen-execution/source-pending-paired-on-v12/`.
The matched Off module is
`12b330c790bb34f270d4efcf2fd01f0fdd0cac8a01ce8e7f7a8295f1671d518a`.
All 58 moving kernels have zero stack/local memory and allocator spills.
The larger-domain complete confirmation also passes: 166.083 s over
13,601 updates, or 14.692 billion complete Yee-cell updates/s. The installed component is unchanged at SHA256
`34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`.

The specialization is bounded by existing Cartesian FP32, packed-material,
CPML12 and source/admission contracts. Admitted mask/layout/H-fact mismatches
retain the original moving consumers. Existing unsupported cases such as
single-source/pre-E packets remain rejected by this research interface; this
study does not claim general frontend or adjoint integration. No experimental
candidate is installed or selected by default.

The next work should proceed in this order:

1. **Integrate the selected path and compose the verified NoIO axis lowering.**
   First measure the combination against this exact winner; retain original
   consumers and admission decisions. Keep the prepared native CUDA execution
   boundary independent of JAX while connecting it to normal BeamZ preparation.
2. **Hardware-test shared Eψ caching.** Retain the current profile/disjointness
   facts or derive it directly from the canonical point graph. Test the L1/shared
   trade on the complete N512/N564 scenes, with exact psi/field/DFT checks and
   positive memory/race/sync coverage. The clean resource probe is a feasibility
   result, not a predicted gain.
3. **Lift more proven source facts into the graph.** The two-source shape,
   spatially uniform Hz coefficient rows, and repeated timestep waveform leaves
   offer bounded unrolling/hoisting opportunities. Preserve the Ey coefficient
   variation and separate product/add rounding. Extend exact material-E
   certification to source classes independently of source-coefficient facts.

More temporal depth, larger tiles, async transfers and wider graph freedom
already lost matched experiments here. The productive direction is selecting
storage and specialization from actual dependencies, immutable input facts,
compiler resource budgets and complete-workload measurements.
