# Native temporal-session memory plan

CPU/source audit, 2026-11-13. No allocation mode, CUDA kernel, production default,
or GPU experiment was changed by this study. Exact accounting comes from the
frozen V12 and current composite reports; larger-domain numbers are estimates.

## Current allocations and what is live

The current constructor allocates and uploads **every prepared input**, including
legacy Mstep workspaces and its plan. It then creates device snapshots of
persistent state when `preserve_inputs=true`. The constructor, `BindInitial` /
`BindOutputs`, and `RunMoving` are byte-identical between V12 and the selected
source/axis/CG composite. Hashes are in the source-binding artifact below.

For cubic physical extent N, one complete staggered six-field bank has

`F(N) = 12*N*(N+1)*(2*N+1)` bytes.

Twelve packed FP32 CPML slabs use `P(N) = 1152*N*(N+1)` bytes. One field bank is
3,230,668,800 bytes at N512 and 4,317,205,680 bytes at N564.

The approximately 13.77 GB N512 allocation is **not four full field banks**:

| Allocation | Prepared indices | Moving/tail liveness |
| --- | --- | --- |
| Six input/working fields A | 0–5 | Live |
| Six final fields B | 74–79 | Live for current out-of-place macro ping-pong |
| Six compact Mstep phase arrays S | 80–85 | Unused by moving and its ordinary tails |
| Legacy Mstep plan: bounds, masks, slots, first/edge cells | 129–133 | Unused by moving and ordinary tails |
| Packed ψ, coefficient/code arrays, sources, monitor state | Other inputs | Live, subject to narrower observations below |
| Preserved state | Private device snapshots | Live for current device-to-device restoring policy |
| Moving trails, owner metadata, raw observations and lookup | Added by moving preparation | Live |

Indices 68–73 are metric-width arrays, **not the Mstep plan**. They have zero
bytes in both Cartesian fixtures. `source_offset=86`, `monitor_offset=113` in
this 134-input ABI.

The compact scratch S is only 1,327,288,320 bytes at N512 (0.411 field banks),
not another full field bank. Its shape is six arrays of 55,303,680 FP32 values.
The legacy plan adds 232,620,640 bytes at N512 and 302,162,552 bytes at N564.

`RunMoving` uses A and B for full temporal macros and swaps their descriptors
once per macro. Its residual steps call `BeamzEnqueueProgramStepAt` with
`mstep=false`: ordinary H, post-H sources, E, post-E sources and DFT execute
in-place on the **current** bank. There is no tail bank swap, phase scratch use,
or legacy plan read. Thus these tails require neither lazy nor streamed D2
scratch. An optional optimized Mstep tail would be a separate capability.

The generic metadata strings about scratch zeroing describe the Mstep method,
not the moving execution: `RunMoving` reports `scratch_zero_bytes=0` and never
runs the Mstep scratch memset loop. Allocation bytes and zeroing traffic must
not be conflated.

## Exact same-fixture capacity savings

Requested device payloads are shown below. GB here means 10^9 bytes; these are
not sampled driver/context/graph VRAM peaks.

| Preserving session | N512 bytes | N564 bytes |
| --- | ---: | ---: |
| Current requested payload | 13,768,123,836 | 17,877,324,367 |
| Remove unused S and legacy plan | 12,208,214,876 | 15,992,642,519 |
| Additionally alias working initial/final fields | 8,977,546,076 | 11,675,436,839 |

The corresponding GiB values are 12.823→11.370→8.361 at N512 and
16.650→14.894→10.874 at N564. The second saving is exactly one F bank; it is
separate from the S/plan removal and is not implemented.

Without device preserve snapshots, the same proposed in-place payloads would
be 5,427,431,100 and 6,970,757,079 bytes (5.055/6.492 GiB). Those are **owned-mode**
figures, not a claim that preserving semantics can simply discard snapshots.
Current snapshots include fields, ψ, DFT accumulators and clocks:
3,550,114,976 bytes at N512 and 4,704,679,760 at N564.

As another trade, private host copies already support `Reset`. A future
preserving mode could restore from those host copies instead of retaining a
GPU snapshot. This preserves values but changes the restore transfer to H2D;
its cost must remain inside execution timing and be measured. It is not the
same implementation or performance contract as today's D2D restore.

Another proposed restore strategy is **constant-state rematerialization**.
When the privately captured initial bytes of a large state leaf are all zero,
its pristine GPU copy could be replaced by an exact zero-fill at reset/run
preparation. Certification must inspect bits, so negative zero or a nonzero
state selects a copy fallback. Capture the fact from the same initial bytes
that are uploaded, before any solve or caller mutation. Keep the fill inside
execution timing and preserve continuation/reset semantics. This proposal was
not implemented or benchmarked here; it can avoid field-alias shader changes
for zero-initialized cases while saving the zero leaves' snapshot allocation.

For the compiled sparse-observation path, the old GPU sorted-key and publication
word arrays are also unused by its point lookup; the sparse map already returns
the raw key index. Together these account for 168,299,172 additional bytes at
N512. They are retained in all estimates here. Any removal needs an explicit
sparse-only allocation/validation path; the general observer fallback still
needs them.

## N680 estimate: promising, not admission

A preserving in-place session keeps two F banks: one working field bank and
one immutable field snapshot. At N680 these alone use 14.087 GiB. Retaining
current observation allocation and trailing protocols gives this model:

| Item | GiB |
| --- | ---: |
| Working fields plus preserved fields | 14.087 |
| Working plus preserved ψ | 0.994 |
| Byte-packed material codes | 0.881 |
| Legacy observation publication mask plus sparse coarse map | 0.304 |
| Four-epoch depth16 trails | 1.137 |
| Owner/tile metadata | 0.001 |
| Remaining planar source/monitor/observation data, estimated | 1.006 |
| Total model | 18.409 |

Adding 25% to the estimated planar remainder gives 18.661 GiB. With the existing
2 GiB benchmark reserve, the model requests 20.661 GiB, versus 23.159 GiB initially
free in the recorded runs. This **does not admit N680**: sparse-map alignment,
real detector geometry, waveform lengths, graph/runtime allocation and host
memory must be measured by the actual planner and unchanged guards. No reserve
was reduced. The model scales the N564 planar remainder by N² and computes
field/ψ/code/mask/coarse/trail terms explicitly; it is not a confidence bound.

Without field aliasing, preserving two working banks plus a field snapshot adds
another 7.044 GiB at N680, so merely dropping S/plan does not reach this model's
capacity. Host staging/copies also remain material: persistent input state is
privately copied on the host even in owned mode to support reset.

## Same-buffer field theorem and actual read audit

The independent [field-bank proof](cuda-field-bank-alias-2026-11-13.md) and its
CPU artifact are the starting point. For tile origin O and owned width B, the
actual first-start cache reads lie in `[O−1,O+B−1]`; terminal field writes lie in
`[O−T,O+B−T−1]`. Clipping to actual staggered supports shrinks those boxes.

For another tile C of the same or higher rank, some origin coordinate is at
least O+B. Its initial minimum is at least O+B−1, strictly greater than this
tile's final maximum when T≥1. Earlier-rank initial reads finish before this
rank starts under the **current complete rank joins**. A tile's own startup
barrier retires all its initial reads before any final scatter. Empty tiles
write no fields; late-start tiles bootstrap from retained trails, not version0
fields; early-last tiles cannot own valid terminal field points.

The actual scalar and paired source bodies satisfy these read restrictions:

- `Initial` uses ordinary coherent field loads only during startup. No field
  `__ldg` or field `__restrict__` promise was found in these moving consumers.
- Later halo operands come from trails. Material/profile/source reads use
  separate storage. Point-local sources change the current local value and do
  not expand the field write box.
- `Observe` publishes the local post-source value into raw staging. Its access
  to `FinalField` uses shape metadata, not a field-data reload.
- DFT runs after all rank joins on remapped raw rows. The current native monitor
  implementation uses post-H/post-E values, not deferred initial-H averaging.
- Ordinary tails already use one-field-bank chronological phase launches.

This supports a **field-storage anti-dependency theorem**, not CUDA execution,
API correctness, a racecheck pass or a speedup. It does not automatically cover
rank-overlap/cohort-DAG execution: such a scheduler would need additional
initial-reader anti-dependencies. It also excludes future deferred version0
observers, arbitrary overlapping component/material/ψ/source buffers, and
ordinary temporal-Mstep D2 aliasing.

## Implementation requirements before any experiment

1. Prefer a moving-only 117-input program packet (74 graph +27 source +16
   monitor inputs), with private final allocation, or an explicitly lazy
   134-input mode. Current `EnsureMovingResources` requires `mstep_` and 12
   workspaces even though moving/tails do not use Mstep. Those structural
   assumptions, source/monitor offsets, output indices and resource queries
   need a deliberate new path.
2. Skip allocation **and upload** of unused buffers. For field aliasing, skip
   the B workspace upload/clear entirely: uploading its arbitrary contents onto
   A destroys the seeded initial state before snapshot construction. Merely
   aliasing descriptors while retaining the old allocations saves no memory.
3. Preserve a separate immutable snapshot for current D2D replay, and retain
   private host initial state for reset/caller-mutation semantics. BindInitial,
   macro parity, graph-capture/replay bookkeeping and metadata must match the
   new storage policy. The same-pointer case must not cause a stale-bank commit.
4. Replace both existing alias rejections (session admission and CUDA host
   `Prepare`) with a narrow explicit mode requiring exact same-component alias,
   canonical injective dense pitches, and disjoint component/other-state byte
   ranges. Retain the disjoint-bank mode. Bind storage mode and all pointers in
   graph identity. Do not add a dual-restrict or noncoherent initial-load promise.
5. Do not expose null/unmaterialized scratch as downloadable raw outputs. A
   moving-only API should return the complete observable state directly. A lazy general
   session must materialize/validate Mstep scratch and plan before `Run` uses
   Mstep, preserve immutable plan contents, and manage graph lifetimes before
   freeing their backing allocations. Ordinary tails can stay scratch-free.
6. Validate seeded nonzero fields/ψ, partial high/low windows, actual source and
   six-face DFT sensitivity, odd/even macro counts, depth4/16, ordinary-only
   tails, split chunks, reset/replay, owned continuation and preserving mode
   against an independent ordinary Program. Add explicit alias admission
   negatives and positive runtime branch witnesses, then memory/race/sync tools.

The algorithm still performs one initial field read and one terminal field
write per macro. Aliasing saves capacity; it does not by itself remove that
traffic or preserve-copy volume. No performance gain is established.

CPU evidence: `.cache/cuda-two-hour-2026-11-13-0819/moving-memory-study/`
contains `capacity.json`, its descriptor-only calculator and `source-binding.json`.
The actual methods inspected are frozen `native_extension.cc` constructor,
RunMoving, BindInitial/BindOutputs; `native_launch.cc` constructs the in-place
ordinary phases; `program.cu:BeamzEnqueueProgramStepAt` orders tails; and
`asymmetric_native.cuh` contains the startup-only initial reads, raw publication
and explicit current alias rejection. No CUDA compile or GPU job was run.
