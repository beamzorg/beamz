# Two-hour CUDA tracing continuation

Authorized research window: **2026-11-13 08:19:09–10:19:09 UTC**.
All owned GPU jobs finished and the GPU was explicitly released at
**10:05:21 UTC**. Report/chart verification followed after timing ended.

All experiments use the local RTX3090 at the existing **250 W** limit,
**FP32** and **12 CPML cells per face at every resolution**. No remote compute
is used. The installed component and production defaults remain unchanged.

## Complete-workload result

The selected axis/source-topology/trail-load combination completes the full
**512³ Mie solve in 111.940 s**, versus **116.220 s** for frozen V12. The two
opposite-order pairs give **1.038232× throughput**, or **3.682% less solve
time**. Every first execution and timed repeat matches all 30 reference leaves
bytewise and preserves caller inputs. This is an incremental improvement to
the existing temporal backend; further several-fold gains remain unproven.
The larger 564³ confirmation gives **166.111→158.263 s**, **1.049585×
throughput** or **4.724% less solve time**, with the same all 30 first/repeat
and immutable-input checks.

| Complete workload | Frozen V12 | Selected | Paired throughput |
| --- | ---: | ---: | ---: |
| 512³, 12,347 timesteps | 116.219960 s | 111.940281 s | 1.038232× |
| 564³, 13,601 timesteps | 166.110788 s | 158.263368 s | 1.049585× |

The two warmed control observations are 116.182741/116.257180 s and candidate
observations 111.982384/111.898178 s. Opposite-order pair ratios are
1.037509×/1.038955×; this observed range is not a confidence interval.
First-call means are 126.270124/122.009349 s. Selected throughput is 14.8042
billion complete Yee-cell updates/s. Construction/upload/readback remain
outside the warmed execution metric.

For 564³, control repeats are 166.067940/166.153636 s and selected repeats
158.236604/158.290131 s. Opposite-order ratios are 1.049491×/1.049678×.
First-call means are 183.214655/175.270222 s; selected throughput is 15.4180
GCUPS. The two cases differ in source/tile alignment and source-policy
cohorts; these observations do not establish a universal size-scaling law.

Against the previous study's **historical pinned-main** default-CUDA control
of 276.396958 s, the selected 512³ observation is 2.469× faster. Main was not
rerun in this window, so this is not a fresh paired main comparison.

Selected module: `52865a3ac8371e2390749953eb3af99746df108fe7bcb00fc4be8b35f4259b92`.
All62 moving kernels have zero compiler stack/local/spills. Its matched hidden
Plane-Off control retains exactly the same CUDA payload; the accepted full
comparison uses frozen V12, not the off control.

## Control and measurement contract

The frozen control is the preceding paired-source native temporal CUDA winner,
module `a9394c3031d6635a875d673d50e7e09123807b7d9d2fe353c49fa05063b5941b`.
The complete Mie fixtures retain a 680 nm cube, 100 nm radius index-2 sphere,
500 nm source, 30 fs duration, coincident source arithmetic and six-face DFT.
The 512³ fixture has 12,347 timesteps; the 564³ fixture has 13,601.

Prepared execution includes preserving copies, temporal field/CPML updates,
sources, observation publication, chronological DFT, ordinary residual steps
and clock advance. Compilation, construction/upload and readback are separate.
Each first execution and timed repeat must match all 30 reference leaves
bytewise. Workers and comparisons require the actual frozen control hash;
variant names alone are insufficient. The prefix screen covers 256 timesteps
and cannot establish a complete-simulation speedup.

The preceding full512 result was 116.079 s for V12 versus 124.584 s for the
strongest prior temporal control. Its 2.381× comparison with pinned-main's
276.397 s default CUDA result is historical context, not a fresh main benchmark
in this study. See [the preceding report](cuda-five-hour-study-2026-11-13.md).

## What is being optimized

The native solver already performs real temporal blocking: a moving 8×16×32
field cache, 7×15×31 owned cells and up to 16 complete Yee timesteps, with
dependency-ordered boundary trails between tiles. These experiments retain
that algorithm and its scalar arithmetic. The tracer supplies typed cell
dependencies, exact support/CPML guards, runtime facts and mutable-state
lifetimes; it is independent of JAX and PyTorch FX.

This remains a compiler for the admitted Yee/CPML operator and reviewed kernel
families. Ordered cell arithmetic is generated from the typed graph; storage
schedules and runtime-fact lowerings use separately reviewed templates and
hash-bound passes. It does not yet trace arbitrary Python physics into an
automatically optimized global CUDA schedule.

Depth16 does not mean each physical point receives16 owned updates in one tile.
Because the owned Z extent is7 and the window moves one cell per timestep,
each point receives at most 7 owned updates in that tile. The steady geometric
volume/publication-surface ratio is 3255/735≈4.43. A 15³ owned cube raises it to
3375/631≈5.35. These ratios explain a reuse opportunity; they are not measured
memory traffic or predicted speedups.

## Isolated 256-step screens

Each row is one fresh matched pair against frozen V12 on the 512³ fixture,
with first/repeat all 30 bytewise checks. Ratios above1 indicate higher
throughput. Small differences are screens, not replicated full-solve results.

| Candidate | Throughput ratio | Decision |
| --- | ---: | --- |
| Full shared Eψ bank | 0.982935× | Slower |
| Compact X-only shared Eψ | 0.993157× | Slower |
| Paired NoIO field storage | 0.968315× | Slower |
| Paired storage + private8 Eψ | 0.970846× | Slower |
| Original storage + private4/colorA Eψ | 0.998499× | Tie/slight loss |
| Coincident source geometry | 1.004873× | No clear win |
| Same source pass, no source-loop unrolling | 1.002012× | No clear win |
| NoIO axis specialization | 1.011586× | Included in composition |
| Trail-load cache policy | 1.019238× | Included in composition |
| Source-plane topology | 1.005382× | Included in composition; isolated difference small |
| **Actual combined build** | **1.035075×** | **Selected; full solves verified** |
| Late16³ cube geometry | 0.936078× | Slower; no full promotion |

The combined gain is directly measured. Multiplying isolated ratios is not
used. Compiler-clean ψ placement is feasible, but none of the tested ψ-cache
variants improves this workload. This does not disprove temporal blocking:
both the control and every candidate already use it.

## Trace-guided methods

### Canonical shared electric CPML history

The first implementation moves eight electric auxiliary-state values per
thread into an exclusive-owner shared bank. The original interleaved Z owner
mapping remains: `zs=tid/512+2*slot`, not the contiguous source-kernel mapping.
The bank is SoA: `(color*4+slot)*1024+tid`. It needs 8,192 FP32 words and raises
dynamic shared allocation from 51,748 to 84,516 bytes.

Only certified source/observation-free tiles with one CPML axis cache electric
history. Magnetic history and other axis classes retain global storage.
First-owned points/window starts reload global state; outgoing high faces and
the last timestep flush it. Four existing field barriers remain. Each original
profile load and RN arithmetic node remains, so unequal profiles are supported.
The twelve prepared ψ ranges must be pairwise disjoint and disjoint from the
36 readonly profile ranges; profiles may alias one another. A hidden
immutable host selector and its fact are part of graph identity; a C++-only
relink supplies a control with the same CUDA payload.

A nonvacuous execution check caught the first host inventory being calculated
before tile-list compaction. The repaired host-only build preserves the entire
CUDA payload. Its later diagnostic derivative marks actual retained reads,
including disappearance on a tail-only run; diagnostic timings are discarded.

Private placement was rechecked in the actual current non-RDC/scalar-cell
compilation context. The original eight-value private cache reports eight bytes
of stack and eight bytes of static spill stores/loads and is rejected before
GPU timing. These compiler counts are not measured traffic. It fits without spills when
combined with the paired field-storage body, at 58,680 shared bytes/five
barriers. A second design caches only colorA terms0/1/3, leaving colorB terms2/4/5
global. Its four values fit without spills in the original four-barrier body
with 51,748 shared bytes. These are separate candidates; neither compiler fit
alone establishes an execution improvement.

### Compact X-only history

For a single active X axis, two colors ×8 Z slots ×16 Y slots ×12 CPML columns
need only 3,072 words: **64,036 total dynamic shared bytes**. The index is
`((color*8+zs)*16+ys)*12+packed_x%12`. It is formed only after the actual scalar
cell's valid/active guard. Other axes use global history on the same grid.

For physical Nx≥64 and depth4/16, opposite-side points sharing a packed column
are separated by Nx−11≥53 cells. An owned-X31 trajectory spans at most
31+depth−2≤45 cells, so those points cannot both occur in one macro. Surviving
points retain their original thread owner; first-owned reload and final/high
flush still apply. The CPU oracle checks actual IR-derived supports and
recurrences, with explicit alias, missing-reload and missing-flush negatives.
This is a capacity/lifetime proof, not a speedup or occupancy measurement.

### Source and field-storage experiments

The source-geometry pass reuses the native constructor's existing coincident
start fact. For the SourceMask20 specialization only, coordinates, bounds and
clamped time move outside the source loop. Every original coefficient/waveform
product and ordered RN addition remains. The general callback is unchanged.
The constructor compares private start snapshots that are also uploaded to the
GPU, so caller mutation cannot invalidate this fact. The actual C++ callback
oracle passes 47,174 cases plus 384 noncoincident negative witnesses.

The paired-field experiment applies the existing source storage body to the
certified NoIO class. It retains 20 source-level private field values instead
of 24, uses directional H exchange and five barriers instead of four, and needs
58,680 shared bytes. The source body itself is byte-identical. This is a
separate experiment; less private state alone does not guarantee faster runs.

The same-grid axis bundle composes the previously successful exact axis guard
specialization with V12. SourceMask and ActiveAxes remain distinct template
arguments. The complete paired-source body, all host launches, queues, clocks,
graph keys and canonical cell arithmetic remain unchanged.

A separate cache-policy experiment changes only the six guarded, same-address
trail import loads to `__ldcg`. It retains producer ordering, support checks,
payload bits and all floating-point operations.

### Stronger source-topology tracing

The source-plane pass derives a narrower class from privately captured input
topology: both active groups have exactly two coincident sources, one X column,
Z/Y starts at 12, and transverse dimensions equal to the physical grid minus
24. No waveform or coefficient value is baked into code. For the existing
CPML-free source policies3/4, the planner already proves every owned update lies
inside `[12,N−13]`. The transverse source bounds are therefore always true.

The resulting callback checks only the runtime X coordinate, computes the
transverse offset, and emits two separate ordered RN multiply/add pairs.
Policy5, other source counts/footprints and failed facts keep V12's callback.
The fact bit and immutable selector are bound to graph identity; four new
kernel entries retain the paired field schedule. This is a stronger tracing
transformation than grouping several ordinary launches: it removes operations
whose predicates are proved from actual input topology and the tile trajectory.

## Cube geometry: correct, resource-clean, slower

A separate actual-read oracle compares the existing cache with a 16³ cube.
Both cache 4,096 points, but the cube owns 3,375 rather than 3,255. Four Z groups
of four planes can retain the same 20 private field values, 2,048 snapshot
floats and five barriers in the paired schedule. The predecessor snapshot must
use `(group−1)%4`; the old two-group XOR is invalid. X exchange needs 16-lane
shuffle width, executed by all lanes before ownership predicates.

The independent control/cube oracle covers 128 schedules and 14,302,464 actual
affine/version reads, with deliberate wrong-group, wrong-shuffle and early
restore failures. The complete experimental CUDA lowering changes the host
plan, trail packing and every field consumer together. Its first build fails
the strict resource gate in two block-packed/isotropic/NoIO fallbacks. A second
build keeps that admission and gives only those fallbacks256 threads/16 private
point slots; all 58 moving entries then have zero stack/local/spills. It is
sealed as `56c77f7ee0e99e8202b8151c64689feba53dabf08fccc2f671ef72d212ed77f2`.
The late GPU gate completes 40 positive state/clock cases and 8 explicit
rejections. Its one fresh512³/256-step pair is slower:2.311445→2.469287 s,
**0.936078× throughput**. It receives no full-run promotion or general
performance claim. At512³/depth16, its frontier raises four-epoch trail capacity
by about 29%, despite lower modeled transfer per cell. Larger geometric reuse
and compiler fit did not produce a win here; dynamic instruction/coalescing
costs are not isolated by this measurement.

## Evidence and boundaries

The selected build completes **289 positive state/clock comparisons plus 16
explicit admission-rejection checks**; all 14 planned validation commands pass.
The additional 80 targeted source-plane comparisons also complete successfully.
The count is recounted from completed reports in
`selected-combined/correctness-evidence-ledger.json`, rather than reusing the
older nominal 297 label. The regression set includes source/mask, packed-layout,
material, geometry, observation, continuation and clock cases.

Separate synthetic64×96×128/17-step diagnostics retain FP32/CPML12, coincident
full-plane sources and a deliberately source-sensitive DFT. They are not used
for benchmark timings. Actual CUPTI records 4 newSourcePlaneP3 launches, 4 P4
launches and 23 NoIOUniform launches, with 0 dropped records. The actual host
planner, material certificate and both compactions prove 377 certified NoIO
tiles, including productive one-hot Z/Y/X counts 34/18/9 in both H/E recurrences;
the source fact has 5 P3 and 4 P4 tiles. All134 input hashes and physical witnesses
are bound in `positive-small/cpu-plan-v2/proof.json`.

The full512 benchmark's prepared source-plane routing is[P3=2077,P4=0]; its
actual new hot path isP3. The full564 routing is[P3=4301,P4=799], so both new
source classes are present there. These counts are prepared static routing
metadata, not instrumented full-solve per-kernel launch counts.

Selected positive memcheck and synccheck complete with 0 errors and all 30 exact
against fresh separate ordinary Program references. A completed same-scope
racecheck retry takes about 73 seconds and reports 0 hazards/errors/warnings,
with all 30 exact. The first 65-second attempt is retained as incomplete; its
coverage gap is closed by the completed broad retry, not a reduced launch
filter. The narrower backup filter was prepared but was not needed.

Device-wide power/clock/temperature/memory and compute-process sampling begins
at 09:19:42 UTC, every 10 seconds. Initial power/device checks and spot checks
precede that, but continuous telemetry does not cover 09:14→sampling start.
All 273 samples through 10:05:11 record the unchanged 250 W power limit. The
complete564³ matrix is sampled. Sampling can miss short spikes; it is not a
continuous-from-start claim.

Raw sources, sealed modules, compile contracts, exact-inverse source proofs,
CPU oracles, prefix comparisons and hardware checks live under
`.cache/cuda-two-hour-2026-11-13-0819/`. Every execution copy is separate from its
build output. Read-only ELF auditing checks matched CUDA payloads without
rewriting loaded modules. Moving CUDA uses the verified non-RDC compilation
mode; the ordinary core/runtime retain RDC. Zero stack/local/spills are required
before a candidate reaches GPU timing.

## Next architectural work

1. **Source-aware internal axis layout.** The
   [ordered cyclic-axis study](cuda-cyclic-axis-compiler-2026-11-13.md) proves
   cell/CPML graph equivalence under proper cyclic permutations. The actual
   planner reduces source-tagged field-update work by 31.6% at 512³ and 66.0% at 564³
   when the source-normal direction uses the shorter owned dimension. Total
   physical Yee work remains unchanged; these are not speedups. A backend
   transform needs new source-mask admission and must publish observations in
   original world key/reduction order. A normalized-weight FP32 counterexample
   shows why re-sorting DFT gathers by internal coordinates would break parity.
2. **Remove unused native allocations, then implement proven field aliasing.**
   Moving execution and its ordinary in-place tails do not use the legacy
   compact Mstep scratch or D2 plan buffers. Their lazy allocation would save
   about 1.560 GB at 512³. The
   [field-bank proof](cuda-field-bank-alias-2026-11-13.md) separately permits
   corresponding initial/final fields to share storage under the current full
   rank joins, potentially removing another 3.231 GB. Neither change is enabled.
   Spare-bank upload/clearing, ownership/reset, graph identity and tail/API
   behavior must be implemented and tested; a future overlapping-rank schedule
   needs explicit initial-read anti-dependencies. The
   [native memory plan](cuda-native-memory-plan-2026-11-13.md) accounts for the
   current allocations and estimates N680 capacity without weakening guards;
   it also considers host/constant-state restore strategies. These remain
   unimplemented capacity proposals, not speed or admission guarantees.
3. **Use measured layout/instruction costs in schedule search.** The compiled
   cube passes correctness but loses speed. Its actual-read proof remains
   useful; future candidates should combine that proof with measured global
   access/coalescing and instruction costs rather than optimizing geometric
   reuse alone.

The source-backed [interactive comparison](cuda-tracing-continuation-2026-11-13.html)
separates full solves from prefix screens. `results.json` and its independent
`finalize_results.py` recheck actual module/report/fixture hashes, all 30 first
and repeat checks and each strict comparison. The selected evidence index is
`selected-combined/evidence-index.json`; `GPU-RELEASE.json` records cleanup and
the unchanged installed component hash. Research paths retain their existing
admission limits; these results do not establish generic frontend,
dispersive-material or adjoint support, and do not change installed selection.
