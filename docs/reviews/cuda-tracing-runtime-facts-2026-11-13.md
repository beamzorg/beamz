# Runtime facts, phase commoning and guarded storage

These experiments use the existing symbolic Yee cell tracer and actual native
moving-cache consumer. They do not use PyTorch FX or JAX tracing. All CUDA
modules below are private immutable builds; production defaults are unchanged.
Hardware correctness and timing are separate gates from CPU/compiler evidence.

## What the compiler now does

[The runtime region pass](../../scripts/cuda_trace_runtime_uniform.py) replaces
spatial electric-source coefficient leaves with three typed runtime scalars only
after a complete supported owned-trajectory certificate. The certificate decodes
the actual packed material tables and compares FP32 bits, including signed zero.
The kernel/class identity contains no material value, timestep or resolution bits.
Existing magnetic facts similarly prove decay 1 and one shared runtime H source.
No arithmetic node, operand order, evolving field dependency or storage round
changes during these leaf substitutions.

[Readonly profile binding](../../scripts/cuda_readonly_profile_bindings.py)
preserves twelve distinct logical psi IDs but maps equal same-phase/same-axis
profile leaves to eighteen runtime profile bindings instead of thirty-six.
The native constructor privately copies all 36 profile vectors before uploading
those same bytes. A later caller mutation therefore cannot change the fact used
for the already-uploaded packet. Unequal bits, signed-zero mismatches, unsupported
shapes/layouts and nonfinite profiles select the ordinary fallback. H/E profiles
may differ; there is no cross-phase alias.

Pointer binding alone did not remove instructions. The actual improvement is
[phase-wide commoning](../../scripts/generate_cuda_native_phase.py):

1. Recognize all six canonical guarded recurrences in one H or E phase.
2. Prove their old mutable states belong to independent component groups.
3. Schedule paired terms in Z, Y, X order. Load one a/b/inverse-kappa triplet per
   active pair and retire those three values before the next axis.
4. Reconstruct each output from its original ordered signed corrections and
   explicit RN multiply/FMA nodes. Emit Hx/Ex as soon as both corrections exist.

Inactive CPML retains exactly the signed raw derivative; identity-profile FMAs
are not substituted. Each component retains its own actual support predicate.
The admitted class has no sources or observations, so reordering independent psi
stores cannot reorder source or DFT side effects. A new host predicate checks the
actual prepared twelve mutable psi ranges are pairwise disjoint and do not
overlap any of the thirty-six readonly profile ranges. Existing native admission
also requires inplace FP32 packed CPML12 slabs and canonical Yee staggering.

Only the certified NoIO1024/profile class uses this generated phase functor.
Every other class keeps its prior scalar component path. The profile fact bit is
part of the existing graph key; no launch parameter layout was changed.

## Compiler evidence

Artifacts are under `.cache/cuda-five-hour-2026-11-13/`.

| Build | Result |
| --- | --- |
| `runtime-scalar-cell-v1`, module `c437d58f…ce8a65` | All 50 moving kernels have identical SASS instructions and offsets to the parent. Useful frontend/control, no execution improvement. |
| `runtime-profile-alias-v1`, module `a8636d21…98fa7c` | All 52 kernels have zero stack/spills; scalar and alias NoIO wrappers have equal instruction/load counts. |
| `runtime-joint-phase-v1`, module `ddb149d9…a265d2f` | All 52 kernels have zero stack/spills; both new depth 4/16 NoIO wrappers use 64 registers. |

Within the **same joint-phase binary**, scalar versus phase wrappers contain:

| Static SASS metric | Scalar | Joint phase |
| --- | ---: | ---: |
| Instructions | 4736 | 4400 |
| Global-load sites | 330 | 258 |
| Branch sites | 265 | 283 |
| Shared-load sites | 101 | 101 |
| Shared-store sites | 47 | 47 |
| Global-store sites | 97 | 97 |

These are static sites, not executed counters, DRAM traffic or speedups. Profiles
are small and may already hit cache. Extra branches can offset fewer loads.
GPU all-state parity, sanitizer and complete-workload timing remain required
before promoting this lowering. The baseline agent owns those gates.

The joint build archives source, commands, graph/fact hashes, resource logs,
SASS, and a self-contained Python producer/test bundle. Its 60 targeted CPU tests
also pass from that frozen bundle. Host-compiled independent equations check
all fields and logical psi, staggered boundaries, inactive CPML, signed zeros,
PEC and pure point-local Finish functions. This does not prove packed CUDA
addresses or inter-thread execution; those require native hardware gates.

## Guarded private auxiliary storage

[The guard-interference pass](../../scripts/cuda_trace_guarded_storage.py)
derives each E-psi axis and actual target support from the operator. It solves
integer-version intersections of the two active CPML slabs with the owned box
`[origin-n, origin+B-n)`. Every interference edge has an explicit physical point
and version witness; six-node coloring gives a resource-independent lower bound.

For two-color caching, term colors are `{0,1,3}` and `{2,4,5}`. A fixed physical
point has time-independent support and CPML activation. Its cache index is
`(point-origin) mod(cacheDimensions)`, so thread/private-slot identity stays fixed
while owned. First-owned/window-start reloads prevent recycled-slot aliasing;
outgoing-high-face or final-depth stores publish the last value. If a clipped
valid window ends early, every remaining valid point must leave on a high face.

This is stronger than demanding one axis bit over the entire swept region.
For example, N512/originZYX(21,495,248)/owned(7,15,31)/T16 visits high-Y early and
low-Z late without a same-point collision. Its supported trajectory also avoids
pinned outer E nodes. Actual material/NoIO certification is still a separate gate.
Ten tests include independent exhaustive enumeration of 35 randomized regions,
cache-token wrap/reload/flush checks and deliberately incorrect two-color corners.

The actual 1024-thread mapping has four Z slots per thread, so two E-psi colors
need eight private floats. The old 512-thread mapping needed sixteen. Actual
emitted one- and two-color cache callbacks pass 600 host checks, including
H-global fallback. The full copied runtime accessors additionally pass 768
host-compiled cases, including signed zero and all twelve packed slab routes.

Resource-only experiments, never GPU-timed:

| Candidate | Resource result at depths 4/16 |
| --- | --- |
| Joint phase + E-psi, runtime axis | 64 registers, 16 B stack/spill stores/spill loads: rejected. |
| Joint + constexpr axis + E-psi | Y/X:64 registers, zero stack/spills. Z: 16 B stack,24 B spill stores,20 B loads. |
| Z plus canonical dense psi strides | Same spills: rejected; integer helper inverse/admission proof retained. |
| Z with one cached color and the other recurrence global | 64 registers, zero stack/spills at both depths. |
| Full single-wrapper Z/Y/X budgeted bundle, module `0cef36cd…d0c13b` | 64 registers but 64 B stack, 96 B spill stores and 120 B spill loads: rejected before GPU use. |

The last candidate keeps the original certified NoIO grid and host-selects its
entry only when both private-profile equality and prepared-state disjointness
hold. Its CTA-uniform axis branch selects Z's one-color cache, Y/X's two-color
cache, or the joint/global fallback. It changes no host sorting, launch count,
rank ordering, events or workspace. Nevertheless, separately feasible bodies did
not remain feasible when merged into one kernel.

An exact-layout depth-16 diagnosis isolated Z, Y, X, generic and bundled bodies.
Z/X/generic had zero local memory, Y used 8 B stack with 4 B spill stores/loads,
and the bundle reproduced the full runtime's 64 B/96 B/120 B result. Thus merged
body lifetimes are a larger resource problem than the small accessor/layout
difference. This is compiler evidence, not an execution-time explanation.

No private cache module has been GPU-tested or promoted. The next copied
candidate retains one H/import/barrier/field loop and selects only the E point
functor; an optional H point-axis specialization leaves H psi global. This
targets duplicated-body liveness while preserving the same placement proof.
Its resource and hardware gates are still pending. More retained values are not
automatically a faster kernel.

## Additional CPU-only facts

[Immutable axis guards](../../scripts/cuda_trace_axis_guards.py) now prune the
actual canonical graph before optional phase emission. An absent axis removes
its unreachable local recurrence reads/writes and readonly profiles; it does
not substitute identity coefficients. Its correction remains exactly the
original RN signed derivative multiply. All eight axis subsets match independently
re-traced region graphs after runtime H/E coefficient specialization; every one
of 256 native flag bytes is checked against its axis class. The native planner
must still conservatively cover the whole actual H/E owned trajectory.

`generate(axis_guard=AxisGuardClass((axis,)))` in the phase generator emits only
the retained pair: four logical psi load/store sites and six profile-load sites
across H/E. The separate namespace and required `AdmittedNativeAxisFlags`
template argument make the class selection explicit. Eight host-compiled cases
check every field and auxiliary value, including preserved inactive psi, PEC
walls and signed zeros. These optional headers have not been CUDA-built or
installed in a native consumer. The default frozen header and proof hashes are
unchanged. Source and strengthened geometry-bound manifests are in
`axis-pruned-phase-cpu-v2`. A negative test rejects consistently mislabeled
profile axes when they disagree with the actual traced derivative axis.

[Common E scalar equality](../../scripts/cuda_trace_common_e.py) can alias three
already-certified runtime scalars to one without baking values. Ten tests reject
stale equality, signed-zero mismatches and missing spatial certificates. It has
not yet been installed in a full consumer.

[Cached-read speculation](../../scripts/cuda_trace_cached_read_facts.py) binds the
exact private/shared read callbacks, Slot helper and pure Difference helper of
the frozen joint consumer. All 147456 actual fixed-cache read addresses are
bounded, including both swizzles, independently of physical component validity.
An optional lowering evaluates only raw field derivatives outside Valid; profile,
psi, Finish and output guards remain. Seven CPU tests pass. This relies on CUDA's
absence of observable floating trap/exception-flag side effects and is not an
admission for arbitrary callbacks. The default frozen header remains unchanged;
the optional header has not been CUDA-built or hardware-tested.

Reproduce CPU checks with `/home/quentinwach/.cache/beamz-pr233-ci/bin/python -m pytest -q` and the targeted `test_cuda_*runtime*`, `test_cuda_native_phase*`,
`test_cuda_native_profile_fact.py`, `test_cuda_trace_guarded_storage.py`,
`test_cuda_native_epsi1024.py` and `test_cuda_trace_cached_read_facts.py` files.
Axis pruning has 30 additional focused checks in `test_cuda_trace_axis_guards.py`.
The exact resource commands and parent/source hashes are in each artifact's
`manifest.json` or `manifest-before-build.json`.

## Remaining compiler work, in evidence order

First resolve the larger consumer's exact build context and numerical gate;
an isolated clean object is insufficient. For guarded auxiliary placement, the
pending common-loop point-dispatch experiment is the next resource test: it
retains one H/import/field/barrier body instead of merging four whole bodies.
Actual private-load witnesses must accompany later all-state hardware gates.

If that path approaches the register limit, compose the already-proved axis
pruning and runtime common-E scalar class individually. The latter uses the
constructor's captured global `common_e[3]` tuple and can be host-selected over
the whole existing NoIO prefix; it needs no new spatial subcohort. Both keep
fallbacks and runtime values, including resolution-dependent values. Their
benefit is unmeasured. Cached-read speculation is lower priority because its
broader floating-environment/source-binding proof buys only branch elimination.

Any resource gate should retain the exact descriptor layout, RDC/device-link
mode, macros, full translation unit and actual entry specialization. Check local
memory as well as register count and record static instruction/call sites.
Runtime load counts and complete-source/CPML/DFT clocks remain distinct from
these compiler facts; none of the CPU proofs establishes an engine speedup.

The larger partial-Ey consumer demonstrates why this matters. An exact paired
compile isolated RDC as the cause of its local-memory regression. Explicit
shuffle PTX did not repair the compiler-generated convergence support paths.
The full `partial-ey-separate-tu-v7` module (`a0daf779…fd4ad`) now passes all
52 moving resource checks with zero stack/local/spills by compiling only the
header-contained moving translation unit without RDC. The same v3 consumer,
canonical point arithmetic and ordinary core/runtime RDC remain. Actual Ninja
definitions/options are independently asserted; the earlier empty-TU CMake
propagation mistake is excluded. Native GPU numerical/sanitizer/performance
gates are still pending; this is not a speed result.

## Exclusive-owner shared auxiliary bank

The next bounded placement proposal moves the point-cache E-psi colors from
eight private floats into 8192 shared floats appended after the old 51,748-byte
allocation, for 84,516 total bytes. Its word index is
`((color*4+slot)*1024+threadIdx.x)`. The old NoIO1024 map owns even/odd Z planes:
`z=(tid/512)+2*slot`, not the contiguous four-plane source-body map.

[The bank oracle](../../scripts/prove_cuda_shared_epsi_bank.py) proves all 8192
indices distinct/bounded and disjoint from field frames, base pointers, clocks
and low-stage storage. Each full-warp access selects 32 consecutive words and
32 distinct shared banks. The base's 36-byte offset modulo128 does not create a
shared-bank conflict. Wrong thread, slot stride and color formulas have concrete
collisions between simultaneously warm, valid, active one-axis recurrences.

For a fixed physical point while owned, cache position and thread/slot/color
identity remain fixed. The existing first-owned reload, outgoing-high-face or
final-depth flush, actual Valid/LocalActive guards and multi-axis global fallback
remain mandatory. H psi remains global. Eighteen lifecycle cases compare shared
and private tokens across 403,466 active reads, including 282,096 warm reads;
26 tests include stale reload/flush failures and randomized legal thread orders.
No other thread accesses a bank word, so this placement adds no synchronization
edge or CTA barrier. The proof covers either legal color subset; an actual Z
dispatch may still leave its second recurrence global.

Volatile references discourage register retention, but CUDA volatile is not a
synchronization primitive and does not guarantee hardware operation counts.
[NVIDIA's volatile rules](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#volatile-qualified-variables)
therefore do not establish exact LDS/STS width/count or zero local storage;
those need the compiled resource/SASS gate. The larger shared allocation can
force a 100 KiB carveout and reduce L1 capacity. The exact emitted resource probe (`22726fb6…ec246`) now passes both depths
4/16 under RDC with the real compact headers: 64 registers, zero stack and
zero spill loads/stores, with the required 84,516-byte dynamic shared allocation.
The source diff preserves the first-owned/flush and guarded recurrence rules.
This establishes compiler feasibility for those two entries, not native runtime
admission, hardware correctness, occupancy or performance. A future integration
must retain captured equal-profile/disjoint-state facts and all original class
admission, then pass numerical and sanitizer gates.

The exact compiled comparison removes all local-memory sites (8 LDL / 5 STL
to zero), replacing them with additional shared sites (LDS 177→197, STS
47→67). Global load/store sites remain 318/121; total static instructions
change from 6696 to 6712. These are static SASS counts, not executed traffic,
and the shared-capacity/L1 tradeoff remains unmeasured. This resource-only
artifact is not integrated into the native runtime in this window.
