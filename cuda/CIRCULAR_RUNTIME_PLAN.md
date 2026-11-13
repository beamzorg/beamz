# Integrating the circular cache into the complete CUDA runtime

The new field-only circular cache is a different algorithm from the existing
overlapping D4 implementation. Its large-domain crossover is evidence for the
cache schedule, not yet an acceleration of the complete engine. The immediate
performance gate is the strongest matched rolling D2 kernel: a 5.5% field-only
gain at 640³/T16 provides little headroom for integration overhead. First improve
that kernel margin, then carry the passing schedule into the native runtime
while retaining CPML 12, materials, source timing and chronological DFT.

Implementation status and raw measurements belong in the
[larger-domain study](../docs/reviews/cuda-large-domain-study-2026-11-11.md).
The independent derivations are in the
[dependency and traffic audit](../docs/reviews/cuda-large-domain-first-principles-2026-11-11.md).
This document describes the proposed integration; it does not claim it is built.

## Runtime and ownership

The immediate engineering sequence, after the measured experiments, is:

1. Use the validated 16×32/64-column preset for the measured 512³ guide case.
   Broaden complete-workload evidence before changing automatic defaults.
2. Instrument CUDA tile phases to locate time in stencil arithmetic, trail
   indexing/import, publication and frontier scheduling. The source payload
   estimates alone did not explain why the lower-traffic asymmetric cache ties D2.
3. Extend tracing with explicit field/offset storage classification and affine
   lifetime limits. The asymmetric prototype proves that a component can remain
   in registers while others remain permanently shared. The phase-counter fix
   proves metadata lifetime can materially change compiled register allocation.
   Generate these decisions from the cell DAG and verify the resulting schedule.
4. Test communication layout and scheduling changes against strongest D2 before
   porting more features. Keep the successful proofs and reject losing kernels.
5. Once a kernel has repeatable useful headroom, integrate the native descriptors,
   packed CPML, materials, clocks and sparse observations as described below.

The asymmetric cache owns 5×15×31 points, caches 6×16×32, and retains only Hy/Ey
in registers. Hx/Hz/Ex/Ez remain permanently shared; this removes whole-family
copies. At 512³/T16 its 64-register/two-block version takes 159.05 ms against 157.29 ms
for matched D2. It is a validated storage/compiler experiment, not an accepted
engine replacement. The full-paper cache still exceeds the practical register
budget in the tested lowering even after shared communication is reduced.

Add an experimental `run_systolic(chunks)` entry point to the native session,
disabled by default. Keep the existing ABI packet and D2 runner as the reference
and fallback. No JAX import is required in the execution process. The Python
frontend can continue to prepare geometry and packets independently of execution.

Replace the probe's single cubic field allocation with six existing field
descriptors, preserving their actual staggered dimensions and strides. A
rectangular grid uses independent z/y/x extents. Each timestep window owns a
diagonal sequence of frontier launches; a kernel finishes before the next
frontier reuses trail storage. Do not introduce a persistent global spin barrier.

Initially retain separate initial/final field banks. There is an independent
proof that an owned bank can alias them, but that is a separate CUDA experiment.
Preserving a caller's input always requires a private mutable state copy.

## Field cache and geometry

Start with the measured owned z/y/x tile 7×7×15 and cached 8×8×16 tile. The six FP32
fields consume 24 KiB; incoming base offsets add 64 bytes. The compiled prototype
uses 56 registers/thread, zero local bytes, and admits three 256-thread blocks
per SM on the local SM86 GPU. These are measured prototype resources, not a
resource guarantee for the integrated kernel.

The retained physical coordinate keeps its circular cache slot. H updates owned
points, then a barrier retires its old-E reads. Import current-version low faces,
then a barrier makes current H available. E updates owned points, then a barrier
protects outgoing publication. Gate publication loads themselves to the outgoing
high-face union, except the final full scatter. Retain the `q <735` tail guard.

A uniform interior specialization may remove physical-wall tests only if the
entire imported cache trajectory is valid. For window depth T, the conservative
common-domain lower bound is `origin >= T+1`, not `origin >= T`; the final imported
low halo reaches `origin-T-1`. Component-specific staggered supports can admit
more cells later. Keep a checked boundary path.

Select depth from useful-work, footprint and measured runtime. T16 cuts cold
initialization/final-scatter cost relative to T8, but deeper windows increase
invalid box visits and trail capacity. T32 is not automatically an improvement.
The admission model must count the actual harness separately from solver state.

## CPML auxiliary state can use a different storage policy

CPML psi is a local recurrence at a fixed physical coordinate; it has no spatial
neighbors. For point p and timestep n, the owning box is
`floor((p+n)/B)` componentwise. Its coordinates never decrease. A change of owner
therefore strictly increases frontier rank. Two distinct boxes on the same
frontier cannot update the same physical psi word at different timesteps.

This permits a candidate implementation that updates packed psi in global memory
in place, without psi trails or twelve dense shared psi arrays. H/E barriers
order successive versions within a block; frontier kernel completion orders
different owners. Use ordinary mutable global loads/stores, not read-only
`__ldg` loads. Each field correction consumes the newly computed psi register;
it must not reread a value after another version can replace it.

The address proof must include actual packed faces, component staggering, term
identity and H/E buffer separation. The recurrence must preserve canonical FP32
operation order, including the existing distinction between stored auxiliary
precision and the unrounded value used by the field correction. FP32 is the
first promotion target. No reduction of the 12-cell CPML thickness is allowed.

For a cubic N³ domain and12 cells on each side, twelve auxiliary slabs occupy
approximately `12*24*N²*4` bytes. Their read+write payload is approximately
`2304/N` bytes per base-cell timestep:4.5 bytes at N=512. This is an asymptotic
requested-byte estimate, not measured DRAM traffic. Exact descriptors determine
the allocation. It makes a packed global recurrence a practical first candidate
even before an auxiliary cache is considered.

## Immutable material coefficients

Initially read existing coefficients through the canonical descriptor/palette
helpers. That supplies a complete correctness gate before adding more caching.
For the supported lossless packed case, a later specialization can retain three
E-material byte codes per physical cache point and keep scalar H coefficients
uniform. Only arriving codes need global reads; immutable codes need no outgoing
trails. The extra cache is 3 KiB for 8×8×16 points, rather than six dense FP32
coefficient channels. Re-query compiled occupancy after this addition.

Reject unsupported dispersive/lossy layouts explicitly and use D2. A fast path
must never silently change physical coefficients or use a different loss model.

## Sources and observations

Apply sources inside their H/E phase before publishing fields. Reuse existing
source descriptors and arithmetic. Absolute source clocks must preserve explicit
multiply-then-add rounding; use the window's starting step and substep n−1.

Determine source and monitor activity conservatively for each tile trajectory.
Most interior tiles should bypass source loops and observation searches entirely.
The conservative physical footprint includes the initial and imported low halos.
An inactive flag may suppress optional work, never a required trail producer.

Reuse sparse observation keys and raw substep staging initially. Publish exactly
one value per observed field/physical point/substep. After the window finishes,
replay DFT accumulation in chronological order using the existing CUDA reduction
and clock semantics. Accumulating directly in frontier order is not equivalent
to chronological floating-point summation. Observation staging is O(T×keys),
not T whole-volume field snapshots.

## Custom tracing and lowering

Keep the versioned Python cell DAG as the source of arithmetic semantics. The
lowering chooses storage independently for each class of read:

| Value | First integrated storage policy |
|---|---|
| Retained H/E and curls | Physical circular shared cache |
| Incoming H/E at a box boundary | Versioned four-frontier global trail |
| Packed CPML psi | Mutable global in-place local recurrence |
| Material coefficients | Existing immutable descriptor/palette reads |
| Source samples | Existing absolute-clock source descriptors |
| Monitor samples | Sparse raw substep staging, then ordered DFT |

Generate exact scalar operations and constant phase/component specializations;
do not trace one Python object per Yee cell. The compiler's separate schedule
analysis must validate read versions, owner rank, cache liveness, publication
coverage and ring reuse. A correct cell DAG alone does not prove the CUDA schedule.

## Gates and promotion order

1. Descriptor-backed field cache on rectangular/staggered domains: exact fields,
   repeat execution, all allocation guards, and small mem/race/sync sanitizers.
2. Real packed materials and CPML 12: all field and psi leaves exact against D2,
   including faces, edges, corners, partial tiles and nonzero initial auxiliary
   state. Add negative stale-version and read-only-psi-cache controls where
   they can be made deterministic.
3. Sources and observation staging: exact all 30 state leaves, including nonzero
   start clocks, chunk tails and both monitor signals. Run an entire pulse.
4. Matched full-workload measurements:256³/384³/512³ guide scaling and the
   fixed 680 nm Mie recipe with 12 CPML cells at every resolution. Keep ownership,
   dtype, physical timestep count and outputs identical. Use fresh balanced
   processes, warmed CUDA timings, and separate compilation/setup/reset costs.
5. Optimize only the passing complete implementation: material arrival cache,
   interior specialization, trail layout/coalescing, depth, then owned global
   field-bank aliasing. Re-query resource usage after each change.

The promotion target is a repeatable several-fold complete-workload improvement
over the best matched D2 control. A field-only improvement is a useful milestone,
not a substitute for that gate. Hardware counter restrictions currently prevent
measuring DRAM sectors on this machine; label source traffic models accordingly.
No RunPod execution is authorized for this work.
