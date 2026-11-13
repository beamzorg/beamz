# Partial shared Ey: independent lifetime audit

The audited body is [cuda_native_partial_ey_frame.cuh](../../scripts/cuda_native_partial_ey_frame.cuh),
SHA `d7d93cce8f5472c2f540a2730258322080da2a98258c1139f306478f0d0eb4cc`.
Its canonical point arithmetic remains the existing generated Yee cell. This
review checks storage routing, reader retirement and collective participation;
it does not establish native numerical parity or performance.

The independent oracle is [prove_cuda_partial_ey_lifetimes.py](../../scripts/prove_cuda_partial_ey_lifetimes.py).
It obtains the actual field/version/offset dependencies from the typed operator,
then uses distinct integer tags for every field, version and cache coordinate.
Fast warps may publish the next H plane while another warp still reads the
current plane. The oracle does not copy the CUDA point equations or calculate
floating field values.

## The read matrix that permits reuse

Centers accompany each listed neighbor. Every component also reads its own old
value, and no other component reads that same-family old value.

| Output | First opposite-family neighbor | Second neighbor |
| --- | --- | --- |
| Hx | old Ez(+Y) | old Ey(+Z) |
| Hy | old Ex(+Z) | old Ez(+X) |
| Hz | old Ey(+X) | old Ex(+Y) |
| Ex | current Hz(-Y) | current Hy(-Z) |
| Ey | current Hx(-Z) | current Hz(-X) |
| Ez | current Hy(-X) | current Hx(-Y) |

Consequently only Hx/Hy need cross-chunk Z snapshots. Ez has no Z neighbor.
Hy(-X) uses an unconditional full-warp shuffle. Shared current Hx/Hz satisfy
the remaining XY readers; their Y neighbors cross warps.

Old Ey can occupy shared planes 0–5 and private planes 6–7 in each eight-plane
group: Hx's positive-Z reader reaches private planes 6/7 in the same thread,
and reaches shared plane 0 when crossing a chunk boundary. Private Ey(+X)
uses an unconditional full-warp shuffle. Predicate-excluded ghost lanes remain
necessary providers for owned neighboring lanes.

## Snapshot ownership and pending E

For the 16×16×32 cache with 1024 threads, each thread owns eight contiguous Z
slots in one of two chunks. Producers store Hx/Hy at global cache planes 7/15
into producer-indexed scratch. P0 publishes those writes. Exactly one opposite
chunk thread at the same XY reads each snapshot, during its E plane 0.

That consumer subsequently owns the same scratch cells as pending Ex/Ez.
The source uses `scratch_key = oppositeChunk*XY + y*32 + x` consistently for
snapshot reads, pending writes and pending restores. It finishes Ex, Ey and Ez
before replacing either snapshot cell. Replacing the Hx cell immediately after
Ex would destroy the snapshot still needed by Ey; the negative control detects
this error.

After the consumer finishes its E plane 0, no other thread needs that snapshot.
The original private H values remain live for the next H update's own-state
FMA; they cannot become E scratch merely because current E readers finish.

The 8×32×32 cache needs no cross-Z snapshot: a thread owns all eight Z planes
for each of its Y groups. Its scratch is pending-only after the first E plane.

## Why the eleven barriers suffice

R0 retires every old-E reader in the H phase and publishes H low-X imports.
Current E ghosts may then replace old ghosts: no owned E update consumes a
ghost as its own old state. Current H ghosts are available before snapshots
and H publication.

For plane j, each thread first saves its own old Ex/Ez and then publishes
current Hx/Hz. Pj makes all XY providers visible. For j>0, Pj also retires all
previous-plane H readers, so pending E(j-1) may replace H(j-1). Current E(j)
gets its Z neighbors from private H or the plane-0 snapshot, never the restored
shared previous plane. Thus a fast warp may publish H(j+1) while a slower warp
still reads H(j): the planes are disjoint.

After the last plane, RL retires its remaining H readers before final pending
E restoration. RE publishes all current E, retires scratch readers and publishes
the next phase counter before another H phase begins. The shared Ey update is
safe after R0 because E has only same-thread old-Ey readers.

The count is R0 + LocalZ Pj + RL + RE. LocalZ=8 gives eleven barriers. The
16×16 cache's generic 512-thread mapping has LocalZ=16 and therefore nineteen;
it must not be described as an eleven-barrier mapping.

For 8×32 generic mappings, each 512-thread thread owns two Y groups; a 256-thread
thread owns four. All groups publish H at the current Z **before** Pj. Otherwise
a reader at wrapped Y=0 could see an unpublished provider at Y=31.

## Evidence and limits

The archived `partial-ey-lifetime-proof/oracle-v2.json` covers 130 configurations
and 29,817,060 traced field reads. It includes all sixteen Z-head positions,
additional wrapped versions 31/64, randomized whole-warp run-ahead orders,
16×16 mappings at 1024/512 threads, and 8×32 mappings at 1024/512/256 threads.
Eight simulated unsafe schedules fail. Four additional reader/writer witnesses
cover publication, the final E barrier, snapshots and missing Y-group providers;
32 ghost-lane examples reject putting either required shuffle inside Owned.

The 38 focused unit checks also cover exact private Ey `si+1` indices, all
snapshot consumers, and late clipped starts. At a late start, initialized-zero
previous-owned cells are physically outside every component support; every valid
target's old/current read routes to the exact physical coordinate and version.
The integer-tag simulation treats valid field reads as a superset; actual
component validity can remove readers, not introduce new ones.

| Mapping | Persistent field floats/thread | Saved Ex/Ez | Barriers/step |
| --- | ---: | ---: | ---: |
| 16×16×32, 1024 threads | 26 | 2 | 11 |
| 16×16×32, 512 threads | 52 | 2 | 19 |
| 8×32×32, 1024 threads | 26 | 2 | 11 |
| 8×32×32, 512 threads | 52 | 4 | 11 |
| 8×32×32, 256 threads | 104 | 8 | 11 |

These are source-level storage counts, not register predictions. Shared allocation
is 101,080 bytes for 16×16×32 and 100,984 bytes for 8×32×32. Resource feasibility,
packed CPML/source/DFT parity, actual CUDA warp convergence and race/sync/memory
sanitizers remain separate required gates. No concrete lifetime defect was found
in the audited body.

The CLI binds the exact manually reviewed body SHA and rejects a later source
edit until review is repeated. It records the actual serialized operator and
oracle source hashes. This is an explicit audit binding, not automatic semantic
extraction of CUDA into the integer model.

## Resource context correction

The isolated resource probe is insufficient for admission: the subsequent full
partial-Ey v3 module used local memory (NoIO 48 B stack, uniform bulk 56 B), so
it has not been run on the GPU. Independent source comparison found identical
compact descriptor, Source and Parameters definitions. The original standalone
command omitted `-rdc=true`; the full CMake CUDA rule includes it. Whole
translation-unit and instantiated-entry contexts also differ. The source/command
comparison is archived in
`partial-ey-resources-v1/independent-full-context-audit.json`; it does not by
itself isolate which compiler-context change caused the extra local storage.

The full NoIO16 SASS contains four absolute device-call sites whose ELF
relocations target `__cuda_sm70_shflsync_idx_p`. They are divergence slow paths
beside ordinary inline shuffle sites, not outlined user lambdas. The non-RDC
probe has no such calls or local-load/store sites. Static call/local evidence is
archived in `partial-ey-resources-v1/noio16-call-and-local-audit.json`;
whether these slow paths execute was not measured. Avoiding a support-call ABI
at independently proven full-warp sites is a narrower hypothesis than changing
the entire moving translation unit's compilation mode.

## Narrow width-32 shuffle contract

The proposed primitive is `shfl.sync.idx.b32 d,a,b,31,0xffffffff`.
PTX defines the segment mask as bits 12:8 of `c` and its clamp as bits 4:0.
With `c=31`, the segment mask is zero, the maximum source lane is 31, and
indexed mode copies from `b & 31`. The full member mask requires every
non-exited lane to reach the same operation; sourcing a nonparticipating lane
is undefined. This exchanges register data, not shared-memory publication.
These semantics come from NVIDIA's
[PTX `shfl.sync` specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync).

The CUDA API's width 32 has the same whole-warp indexed-lane behavior.
All named live lanes must call with the same mask, and a selected source lane
must participate, including lanes whose physical field point is a ghost.
[CUDA warp shuffle and participation rules](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)
therefore forbid moving either required call inside Owned/Valid.

Installed CUDA 13.3 `sm_30_intrinsics.hpp`, SHA
`c97d1e6dddbae6d27e04386f96e3f606169d3ee597a5164f23679e40516de72b`,
lines 417–424, converts float to integer bits, passes
`c=((32-width)<<8)|31` to `__nvvm_shfl_idx_sync`, and converts the returned
bits back to float. At width 32 this is `c=31`. A direct primitive must likewise
copy all 32 bits without FP conversion or arithmetic: signed zeros, infinities
and NaN payloads are data. Its integer-register input/output constraints must
be paired with bit reinterpretation, not numeric float/int casts.

The two admitted sites are old private Ey(+X), with source `(sx+1)&31`, and
current private Hy(-X), with source `(sx+31)&31`. Every lane reaches them before
Owned/Valid; any private-Ey-versus-shared-Ey selection is uniform within a warp.
Startup/imports initialize all provider slots. Existing R0/Pj/RL/RE barriers
remain responsible for shared visibility and retirement. The primitive does
not justify deleting them. Its bit-pattern microtest, actual SASS participation
audit and whole-engine numeric/race/sync gates remain separate requirements.

The explicit-bitcast helper was tested as an RDC resource control in
`partial-ey-inline-shuffle-resource-v5`. It retained the same NoIO 48 B and
uniform-bulk 56 B stack/spill failure: the compiler still generated convergence
support paths. It is a rejected lowering, not a successful repair. A separately
compiled moving-only non-RDC translation unit is the subsequent hypothesis;
its actual flags, kernel count and local-memory resources must be checked before
hardware admission. Ordinary core/runtime RDC need not change.

The subsequent full `partial-ey-separate-tu-v7` build passed the resource gate:
module `a0daf779081324e31c09538e961d639c5321310fff49acaee84f8346a53fd4ad`
has all 52 moving entries with zero stack/local storage and zero ptxas spill
stores/loads. Its consumer remains exactly `12486a02…5ace12d`, the original v3
body and admission. Only the moving translation unit is non-RDC; ordinary core
and runtime retain RDC. `compile-contract.json` verifies the actual Ninja
definitions/options, including source-property propagation and extended lambdas.
This is now a full-build compiler result, not just an isolated resource probe.
GPU all-state, synchronization and performance gates are pending.

## Bounded last-two-plane fusion

The next CPU-proved proposal fuses only P6/P7 in the 8×32×32 cache, only for
the existing uniform 1024-thread classes. Save old Ex/Ez for both physical
planes, publish Hx/Hz for both, then P67 publishes both planes and retires E5's
H5 readers. Restore pending E5. Compute E6 and E7 in their original order;
their negative-Z H operands remain private. Both planes' H XY values survive
until RL, which retires every remaining cross-warp reader. Restore both E
planes after RL, then retain RE unchanged.

The four saved old-own slots become the same four output slots. E6's two
results survive E7 in their own slots; there is no additional pending pair.
Thus the source-level peak increases from two to four Ex/Ez floats per Y group:
**+2 field floats/thread** for the admitted 1024-thread class, zero shared bytes,
and eleven to ten barriers. This is not a ptxas register prediction. Generic
512/256-thread code keeps its original loop byte-for-byte. The H-phase private
Ey6/7(+X) shuffles are untouched. Both E-phase Hy(-X) shuffles remain full-mask
and precede Owned/Valid; all Y groups of both H planes precede P67.

[The tail-pair oracle](../../scripts/prove_cuda_partial_ey_tail_pair.py) uses the
actual traced field reads and permits a fast warp to publish both tail planes
while a laggard still reads H5, and to finish E7 while a laggard reads H6.
Missing H7 publication, saving old E after H overwrites it, early restoration
before RL, and overwriting E6's retained pair all fail with explicit reader
witnesses. E6/H6 shared Z access is not required or introduced. The 56 focused
tests include 38 unchanged-schedule checks and 18 tail-fusion checks. The
archived `partial-ey-tail-pair-v1/oracle.json` covers 192 configurations and
43,466,112 traced reads, with all 32 wrapped Y/X heads and shuffled warp orders.

[The private producer](../../scripts/patch_cuda_native_partial_ey_tail_pair.py)
binds exact parent consumer `12486a02…5ace12d` and canonical cell `bef1f611…865c5`.
Copied candidate `partial-ey-tail-pair-v1/source` has consumer
`27f0bb5f946e8d1bc9b8b57e8eb71ddb5ecb9ba552724038ac4c2cc960bb5f0f`.
An inverse recovers the complete parent. Each copied point body differs only
in its saved-slot names and omission of the single-plane pending store.
No CUDA compiler or GPU gate has run for this candidate.
`cpu-admission-binding.json` binds the exact candidate/canonical cell, both
oracle sources, archived oracle output and inverse-transform certificate.

The remaining barriers have concrete readers under the existing storage:
R0 retires old-E cross-warp H reads before frame reuse; every surviving Pj
publishes current H Y neighbors; RL protects both tail H planes; RE publishes
all restored E before the next H phase. Deleting any of those without another
storage/ordering mechanism is unsafe. Batching more planes requires retaining
more old-own/output values and is outside this bounded experiment.

## Smaller cache with permanent shared Ey

The larger v7 resource pass did not imply a speed win. Its first identical-input
Mie512/256-step prefix screen, archived in
`partial-ey-v7-validation-guarded/prefix-comparison.json`, measured only
0.6921× the strong Obs control's throughput in one fresh-process pair. That is a
screening loss, not a confidence interval or a full-duration result.

The next storage proposal returns to cache8×16×32/owned7×15×31. Ey occupies one
permanent shared plane family. H remains private; two private Ex/Ez pending
scalars replace the shared pending area. Independent Hx/Hy boundary snapshots
keep 2×1024 shared scalars and never alias pending storage.

At 1024 threads, two Z chunks own four contiguous planes each. Hx/Hy at cache
Z3/7 are published by P0 and read by exactly the opposite chunk's same-XY E0
consumer. Every later Z neighbor is private. At 512 threads, one chunk owns all
eight planes and needs no cross-chunk read; reserved snapshot storage is unused.
R0 retires every old shared Ey reader before ghost imports or E own-state
updates. Every thread, including ghosts, writes private pending at E0 before
any later iteration restores it. Each Pj retires previous-plane H XY readers;
RL and RE retain their original roles.

| Mapping | Private H | Current Ex/Ez | Pending Ex/Ez | Source field total | Barriers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small shared-Ey, 1024 threads | 12 | 2 | 2 | 16 | 7 |
| Small shared-Ey, 512 threads | 24 | 2 | 2 | 28 | 11 |

The shared allocation is 58,680 bytes. These field counts exclude coefficient,
address, control and compiler temporaries. They do not establish register count,
occupancy or execution speed relative to the old four-barrier cache.

The CPU oracle is [prove_cuda_shared_ey_private_pending.py](../../scripts/prove_cuda_shared_ey_private_pending.py):
128 configurations, 14,072,064 actual-trace reads, seven negative schedules,
late/ghost/repeat/reset checks and 28 focused tests pass. Missing, previous-version
or wrong-chunk snapshots all fail. No CUDA compiler or GPU gate is implied.
Copied source `shared-ey-private-pending-v1/source` has consumer
`f9d10fbd0f1b493aa9335922faffcd3adab234d5483cd75458fef8c16797f9ef`;
the new body is `83c46665…d8312`. Its producer binds the unchanged canonical
cell, exact original consumer inverse and v7 non-RDC moving TU context.

A policy hybrid can preserve every old NoIO/Bulk/Obs four-barrier kernel and
use this body only for an admitted source class if resources pass. Independent
comparison found identical Parameters, FieldPacket/ImportPacket, Environment,
Observe and Surface with old Obs9f6a. **Do not copy the entire v7 host header:**
its extra `uniform_bulk_counts_host` field is absent from old Obs's workspace.
Append only uniquely named helpers/body/wrapper, retain old host ABI/rank joins,
and use 58,680 bytes solely for the new kernel while old kernels keep 51,748.
The exact comparison is archived as `shared-ey-private-pending-v1/hybrid-abi-review.json`.

## Two-plane groups in the small source body

The next bounded CPU proposal publishes physical planes [0,1] and [2,3] in two
groups for 1024-thread/four-Z-slot chunks. Each group first saves both old Ex/Ez
pairs, publishes both Hx/Hz planes and synchronizes. That barrier also retires
all previous-group H XY readers before restoring the previous group's pending
E. E0/E1 or E2/E3 execute in their original per-point order. Separate snapshots
still supply the first plane's opposite-chunk Hx/Hy Z neighbors. Hy(-X) shuffles
remain unconditional and all Ey remains shared.

The source-level peak is H12 + current old-or-new Ex/Ez4 + previous pending4 =
20 floats, four more than the seven-barrier body. Current slots become results;
there is no additional result array beyond those slots. Shared memory stays
58,680 bytes and per-step barriers become R0 + P01 + P23 + RL + RE = five.
No register/occupancy claim follows from this bound.

[The paired oracle](../../scripts/prove_cuda_source_pending_pairs.py) passes
64 wrapped-head/interleaving configurations and 7,036,032 traced reads. Six
negative controls reject early restoration, lost first pending values,
overwritten second-plane old values, missing H3 publication and missing/stale
snapshots. Fourteen focused tests also check the exact source-class guards.

If only Policy5 fits the eventual resource budget, its generic point guards
can safely consume Policy3/4 IDs: actual low-six flags are 16/17, so all axis
bits are absent, all CPML recurrences remain inactive, and existing no-boundary
trajectory bounds prove added Valid/wall checks do not alter the computation.
Source/observer flags select the same side effects. This requires the existing
strict-interior fact; merely relabeling an arbitrary source tile is insufficient.

The resource-only copied source `source-pending-pairs-resource-v1/source`,
consumer `855a83a8…ff1ed`, appends `AsymmetricSourcePendingPaired20` but changes no
host route. New body `6ed18659…43db2c` has point-copy and H/import/bootstrap/RE
inverse checks against frozen hybrid v10. `cpu-admission-binding.json` binds
the source and oracle evidence. No CUDA compilation or hardware gate has run.
