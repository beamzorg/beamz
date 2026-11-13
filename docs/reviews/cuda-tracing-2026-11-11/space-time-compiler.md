# Tracing to an owned CUDA schedule

The tracer now feeds executable schedule/storage lowerings, rather than only
printing scalar arithmetic. All paths here are experimental. Default production
selection is unchanged. Source-level payload/liveness and static SASS counts
are kept separate from measured GPU performance.

## Implemented passes

`scripts/cuda_space_time_ir.py` consumes the existing typed cell DAG and checks:

- Version/phase legality and affine moving ownership `u=p+version`. Every source
  points componentwise to the current or a lower owner; streamed Z needs only
  current/previous transformed planes. Unsupported custom signatures fail.
- The physical-XY thread union has width `BX+T−1`, height `BY+T−1`. T4 owned
  29×13 fits 32×16 threads. Negative E and positive initial H aprons are external
  reads, not extra updates or secretly computed halo threads.
- Each operand is classified as same-thread register, full-mask warp shuffle,
  shared exchange, initial/external global value, or local mutable recurrence.
- Last cross-Z consumers determine the retained field frontier: `6T−4` scalars,
  hence 20 at T4. Hz at the final stage has no previous-plane consumer and is
  omitted. This is not a compiler register count.
- Readonly scalar/profile/material coefficients remain stationary. Packed local
  CPML is admitted only as same-point, previous-version, same-phase recurrence;
  no psi halo or dense auxiliary cache is introduced.
- Guard-aware interference allocates four persistent private values for twelve
  logical ψ states under a verified single-axis class; H/E histories stay
  distinct. See [the storage proof and resource evidence](stationary-psi-cache.md).

Provider materialization unions inverse-stencil source footprints. It admits
only requested points, proves transformed coordinates stay in `[-1,B−1]`, and
reports the extra apron outside the thread union. A packet callback can compute
lower-neighbor bits/surface/base once for 2–3 scalar fields. Raw external read
edges are compared with unique provider loads, explicitly before cell CSE/cache
reuse; these counts are not measured DRAM traffic.

## Streamed storage and synchronization

The device helper sweeps transformed Z upward. At each plane it performs
H1/E1 through HT/ET with physical `z=u_z−n`. H needs previous-plane E center/XY
and current-plane E at +Z. E needs current H center/XY and previous H at −Z.
Fixed physical-XY ownership keeps old/current center and Z values thread-local.
Only components x/z need cross-warp Y exchange; X uses full-mask shuffles.

Two shared exchange frames suffice, independent of T: 8192 bytes at 512 threads.
Immediately before Hn, previous E is republished from retained private history.
Barrier A publishes that frame and retires the preceding E's H-frame readers.
Hn then publishes current H; barrier B publishes H and retires H's E-frame
readers. En follows. Reusing an idealized per-stage bank without this actual
republish/retire protocol is rejected by the token interpreter.

Every external source is tested before local access. Version-zero registers
start at zero because the first previous-Z frontier is external. No undefined
inactive-lane input is passed to a shuffle. Phi/component handoffs occur only
after the old value's last consumer; sources finish before opposite-family use.

The optional provider lowering fills external provider slots before exchange.
Consumers inside the materialized union no longer repeat owner/surface address
work; true out-of-union aprons retain explicit global fallback.

## Compact control flow and metadata

`scripts/cuda_space_time_codegen.py` emits a complete device helper behind an
Environment contract. Relative callbacks receive transformed `u−origin` directly:
`ReadRelative`, `ReadPacketRelative`, and a six-field `PublishPacketRelative`.
This avoids reconstructing physical coordinates merely to subtract origins
again, and computes publication metadata once per owned point. Native physical
support remains the Environment's responsibility; an interior specialization
needs an independent once-per-CTA support admission.

The `--loop-stages` lowering retains statically named scalar queues and emits one
runtime H/E stage body, rather than T copies. Queue shifts are explicit scalar
assignments; no dynamic register-array indexing is introduced. This exchanges
queue-move/spill cost for a smaller code footprint. The arithmetic and ownership
order are unchanged. `--metadata-remat` is a separate readonly-origin diagnostic;
it regressed the uniform control's spills and is not the selected lowering.

`scripts/cuda_codegen_audit.py` reports static source/callback sites and selected
SASS functions. It rejects an absent/empty selection. Review budgets of 2048
instructions and 128 branch sites are screening thresholds, not performance
predictions or automatic production admission. Static instruction sites are not
dynamic execution counters or proof of instruction-cache misses.

## Native arithmetic and guarded CPML

`scripts/generate_cuda_native_cell.py` emits the actual point functor for the
owned asymmetric cache:

```cpp
beamz::cuda::trace_native::AdvanceNativeCell<Phase, Component>(
    env, read, version, z, y, x);
```

Field IDs are 0Hx..5Ez. The separate local-term namespace is 0H0..5H5, 6E0..11E5.
`read(field, macroLocalVersion, dz, dy, dx)` supplies cached n−1/n values.
Version n is 1..Depth; Env adds macro_offset for source/DFT exactly once.
Coefficient kinds 0/1 select decay/source by field ID; 2/3/4 select a/b/inv-kappa
by local-term ID. Optional support-check removal requires the caller's proof.

`Difference` receives actual center/adjacent stage values and applies canonical
ForwardDifference/BoundaryDifference branches and explicit RN scaling; it must
not reload stale global fields. Non-PEC E derivatives at coordinate zero or the
opposite field extent are zero. `Finish` masks PEC after CPML state evolution,
then injects point-local H/E sources. Pre-E injection remains excluded until its
separate input lifecycle is implemented.

`scripts/cuda_trace_cpml_lower.py` recognizes canonical recurrence/correction
DAGs. It proves each old psi's last uses and emits a short per-term active guard.
Inactive correction executes exactly `sign*derivative`, with no profile/psi
loads, recurrence FMA or store. Identity-profile FMA substitution is avoided
because it can change signed zero. Active correction uses unrounded next psi;
only stored BF16 state is rounded. Unsupported shifted/cross-state patterns fail.

Psi pointers are ordinary mutable global storage, never read-only cached loads.
Unique physical owners update a point in version order. Existing phase barriers
order within-CTA stores before future reads; rank-kernel completion orders owner
transitions. Distinct term/family storage and no historical/spatial psi readers
are required. Immutable caller snapshots and observations are separate lifetimes.

Root's `cuda_trace_specialize.py` can supply a verified specialized Operator.
Exact FP32 scalar coefficient facts replace only readonly leaves, preserving
rounding nodes and frontier dependencies. Dispatch must verify actual packet
bits/shape/dtype; the generic operator is the fallback.

The optional `--scalar-h-class` native variant instead admits equal runtime H
source scalars and lossless H decay. Only decay 1 is baked; the three source
leaves alias a single scalar callback with zero coordinates. Its hash does not
change with timestep/source value, so different resolutions need no recompilation
for that class. Every actual packet must still pass the class predicate.

`scripts/cuda_stationary_storage.py` places immutable Layout1 E material byte IDs
beside retained physical cache cells, leaving arithmetic and exact FP32 codebook
lookups unchanged. Three component IDs pack into one word. A cache slot keeps
the same physical point until it joins the recycled low-face union, so only that
union refreshes after startup. At cache 6×16×32/T16 the source model requires six
private words per thread and 300 extra low-X staging bytes; logical vector-ID
loads drop from 37,200 to 15,024. These are not DRAM transactions or a performance
claim. The pass rejects Layout2/shifted material reads and binds its manifest to
the unchanged operator hash. Native integration is a separate experiment.

`scripts/prove_cuda_native_scalar_queue.py` independently checks a compact native
Z loop: six fixed-scalar Hy/Ey queues rotate after every j, including ghost slots.
H consumes Ey q0/q1; E consumes Hy q0/q5. Six rotations restore physical storage
before imports/publication. Missing ghost rotation, wrong previous-Z direction,
and excluded shuffle source lanes fail. This supports one loop body instead of
six copies without dynamically indexed private arrays; ptxas/runtime gates remain.

## Evidence and limits

The CPU SIMT shim executes the actual emitted C++ body with CTA barriers and
full-warp exchange. An independent scalar oracle checks bitwise staggered fields,
point sources, native walls/PEC and logical CPML/BF16, including alternating ±0
with all CPML inactive. In that negative-access case profile/psi callbacks abort
if reached. The T4 looped case compares 1386 values; the native point functor's
BF16/PEC and inactive-zero cases compare 924 values each. Tiny logical CPML slabs
use thickness 1 for semantic coverage; this does not change benchmark CPML12.
The frozen native v3 consumer subsequently passed all 30 output leaves bytewise
on seeded/rectangular/depth16/chunk/clock fixtures and large complete workloads,
including packed CPML, actual sources, and chronological DFT. Its exploratory
complete screens were 1.49–2.89× slower than matched native D2, so it is rejected
for speed promotion. Evidence is in `moving-native-v3/complete-summary.json`
under `.cache/cuda-engine-autoresearch-2026-11-11-2032/`. Later queue/import
variants require their own complete gates and matched scoring.

The same uniform Env/wrapper/flags/relative callbacks produced:

| T4 lowering | Static instructions | Branch sites | Code bytes | Registers | Spill store/load bytes |
|---|---:|---:|---:|---:|---:|
|Unrolled|2888|251|46208|64|48/20|
|Runtime stage queues|888|83|14208|64|132/72|

These are min-two-CTA compiler diagnostics, not native timings. The actual native
unrolled direct helper was much larger (6911 instructions/~110 KB) and 128
registers; native callback/resource results cannot be inferred from uniform Env.
Provider materialization reduced abstract address/read duplication but increased
spills in initial controls. All candidates therefore need matched native scoring.

Artifacts are under `.cache/cuda-large-domains-2026-11-11/stream-ir/`, including
`stage-control-comparison.json`, frozen sources/manifests and host-oracle logs.
Native point artifacts are under `.cache/autoresearch-4h-2026-11-11/native-cell/`.
Main production selection is unchanged.

## Reproduction

```sh
python scripts/cuda_space_time_ir.py --output /tmp/stream-plan.json
python scripts/cuda_space_time_codegen.py --loop-stages --relative-callbacks \
  --output /tmp/stream.cuh --manifest /tmp/stream.json
python scripts/generate_cuda_native_cell.py \
  --output /tmp/native-cell.cuh --manifest /tmp/native-cell.json
python scripts/generate_cuda_native_cell.py --check --output /tmp/native-cell.cuh
python scripts/validate_cuda_streamed_host.py --depth 4 --loop-stages \
  --relative-callbacks --native-boundaries --cpml --bf16 --pec-mask 63 \
  --output-dir /tmp/stream-host
python scripts/validate_cuda_streamed_host.py --native-cell --cpml --bf16 \
  --inactive-cpml --zero-fields --output-dir /tmp/native-cell-host
/home/quentinwach/.cache/beamz-pr233-ci/bin/python -m pytest -q \
  tests/unit/test_cuda_space_time_ir.py tests/unit/test_cuda_space_time_codegen.py \
  tests/unit/test_cuda_trace_specialize.py
```
