# Trace-driven CUDA scheduling: two-hour study, 13 November 2026

**Completed.** The selected source/lifetime composition improves full prepared
forward throughput over the immediately preceding temporal winner by
**512³: 104.825 → 103.642 s (1.011417×)**, and
**564³: 147.636 → 146.217 s (1.009700×)**.
Both fresh A–B and B–A pairs pass all 30 first/repeat outputs bytewise and finite.
This study delivers a small further gain, not another 2×. The new rotated
ownership model remains CPU-only; the executable column schedules were slower
or tied, or failed resource admission.

The study started at 19:34:09 UTC with a 21:34:09 UTC deadline and a five-minute
GPU-work reserve. All execution uses the local RTX3090 at its unchanged 250 W
limit. No RunPod resources, commits, pushes or GitHub mutations are involved.
The installed extension and production defaults remain unchanged. GPU work
finished before the 21:29 UTC cutoff; all recorded campaign processes have exited.
The cleanup snapshot reports 0% GPU utilization, with an unrelated resident
ComfyUI process left untouched. [Final state](../../.cache/cuda-trace-systolic-2026-11-13-1934/final-state.json).

## What the tracer can now derive

[`cuda_trace_systolic_compiler.py`](../../scripts/cuda_trace_systolic_compiler.py)
consumes the existing typed Python Yee/CPML cell DAG. It derives private-register,
warp-shuffle, shared-Y and chunk-boundary-Z communication routes; tracks source,
observer, recurrence and publication effects; and checks the required shared
reader-retirement/publication order. Compact provider boxes and their address
maps come from the actual shifted reads. The emitted frontier helpers are also
executed as ordinary C++ with unique value tokens and deliberately wrong-provider
negative controls. Those checks complement the model; they do not prove CUDA
memory visibility.

[`cuda_trace_affine_ownership.py`](../../scripts/cuda_trace_affine_ownership.py)
solves integer difference constraints for minimum coordinate-wise time skew and
per-component ownership offsets. It rediscovers the current box schedule
`u=(z+n,y+n,x+n)` and finds a legal rotated x–y alternative:

```
u_z = z+n
u_+ = y+x+n
u_- = x-y+n+r_field
r_Hx=r_Hz=r_Ey=0; r_Hy=r_Ex=r_Ez=1
```

CPML auxiliary ownership is tied to its primary field. A finite independent
checker verifies unique physical/version ownership, producer-before-consumer
order and an inverse-stencil publication set. The rotated basis has determinant
two, so a future CUDA lowering must honor its lattice parity and staggered
component geometry. These offsets change execution ownership, not physical
timesteps or source clocks.

**The rotated model is CPU-only and diamond-inspired. It is not an implemented
DiamondCandy CUDA solver.** It does not yet establish packed in-place CPML,
source/monitor execution or CUDA visibility for that ownership scheme. The
executable column candidates retain the existing global moving-box ownership.

This is a restricted trace-guided compiler for BeamZ's supported stencil and
effects. It is not a general arbitrary-Python/PyTorch tracing compiler. The CUDA
lowerings transform reviewed templates under an exact parent-source hash
guard. A new parent revision needs a reviewed binding; the prototype does not
optimize arbitrary CUDA or automatically discover every geometry.

## Executable schedules investigated

The column lowerings retain a Z column or private Z chunk per thread, exchange
X neighbours with full-warp shuffles, and share only tangential Y values and
required inter-chunk Z values. Distinct H/E banks permit two block barriers per
timestep instead of four. Shared provider packing and deduplication avoid
publishing values already available in another shared region. Publication can
be fused with computation when the opposite family has separate storage.

The schedules exercise one/two/four Z thread groups, 512/1024 threads, different
power-of-two cache shapes, and a two-block residency target. Shape changes use
the existing generic source-capable fallback rather than incorrectly retaining
the old shape-specific paired-source pipeline. The coarse observation masks
retain their original physical geometry. Complete physics is checked after
every admitted build.

These designs trade fewer barriers/shared reads against persistent registers,
address/predicate work, halo communication, block residency and source-path
cost. Lower barrier count or higher nominal occupancy alone did not establish a
whole-engine win. Numerical passes and resource rejections are recorded
separately from performance claims.

## Short screens and resource rejections

The study compiled 22 staged variants, including repaired and superseded builds.
Fifteen passed the zero-stack/local/spill gate, the coupled physics suite and
a fresh matched 512³/256-step screen. Seven were rejected before GPU execution.
These are build counts, not 22 independent algorithms. Each screen uses one
fresh control/candidate process pair with three timed samples per process;
small differences are selection evidence rather than replicated full-solve gains.

| Candidate | Screen throughput / control |
| --- | ---: |
| `bounded-plane-source-v1` | 1.005598× |
| `bounded-terminal-cache-v1` | 1.014032× |
| `bounded-terminal-cache-waves-v1` | 1.014081× |
| `cache-exit-v1` | 1.005834× |
| `column-compact-v1` | 0.884505× |
| `column-fused-fast-v1` | 0.993321× |
| `column-separate-v1` | 0.869991× |
| `column16x8-fused-all-v2` | 0.829342× |
| `column8x8-fused-all4-v1` | 0.882826× |
| `column8x8-two-block-v1` | 0.830893× |
| `terminal-cache-exit-v1` | 1.013121× |
| `terminal-reuse-layout1-v2` | 1.006470× |
| `zchunk2-fast-dedup-v1` | 1.001102× |
| `zchunk2-fast-v1` | 0.985916× |
| `zchunk2-generic-v1` | 0.974205× |

The full private-column version needs about twice the persistent field registers
(48 rather than 24 scalar field values per thread), while reducing thread count.
The Z-chunk versions retain the smaller private state but add frontier and
address work. The 16×8 transpose still has a shortest owned extent of seven,
and its source path falls back to the correct generic geometry. None yielded
a complete-engine gain that justified full-duration promotion.

Resource-rejected variants were `zchunk2-compact-v1`,
`zchunk2-noio-dedup-v1`, `terminal-reuse-v1`, `column16x8-fast-v1`,
`column16x8-fast-v2`, `column16x8-fused-all-v1` and
`cache-exit-peeled-v1`. Resource-gate failures are not numerical failures or
measured performance losses. The repaired Layout1 terminal pass and the
256-thread cold-layout column fallback are represented separately above.

## Selected finite-horizon and source passes

1. **Bounded plane sources in mixed/CPML tiles.** The existing host admission
   proves two coincident sources on a field-specific X plane and the transverse
   `[12,N-12)` rectangle. The new Policy5 specialization retains explicit Y/Z
   bounds, simplifies the source loop, preserves both ordered split-round
   additions and keeps independent coefficient/waveform arrays. Generic source
   fallbacks remain available. The actual emitted callback passes host
   ASan/UBSan checks: 49,152 FP cases, 1,380 source-positive cases and 22 rejected
   topology mutations.
2. **Reuse terminal H outputs.** Within a finite macro, terminal E has no
   evolving-state consumers. Terminal H is already stored in the final output
   bank and can supply downstream E/psi consumers after the existing rank
   dependency. The admitted Layout1 path reads those coherent final H values
   and removes all duplicate terminal trail stores. Layout2 keeps its previous
   communication cut. Raw observation staging and final E remain intact.
3. **Remove dead shared E publication.** Full-private-field bodies need their
   shared E cache only for the next in-CTA H phase. At the final actual local
   window iteration, the overwrite and its two retirement/publication barriers
   have no remaining shared reader. Global trails/final outputs and all
   source/observation effects retain their order. The paired source pipeline
   keeps its reviewed schedule.

The optional waveform-cache experiment hoists four CTA-uniform source values
using the existing startup, RLpair and RE barriers. Its emitted staging preserves
independent H/E clocks, waveform lengths and split-round source arithmetic. It
passes 49,152 host FP cases, 1,260 independent-wave checks and a wrong-field
negative control, but its short screen ties the simpler composition. It is not
part of the selected module.

## Fair comparison contract

The control is the immediately preceding frozen temporal CUDA winner:
`becc5f154df24dbe53b68335d9f13a682fcbb3f11742f601cf04a5229745afe7`.
The selected module is:
`e373c208ae47ce63cd3e9b2b9d24644913f09c31185d633ea545da335b7341f3`.
All 68 selected moving kernels pass zero compiler stack/local/spill admission.

Both complete cases use the same 680 nm cube including PML, 100 nm radius
index-2 sphere, 500 nm source, six-face one-frequency DFT and nominal 30 fs
propagation. They preserve **12 CPML cells per face at every resolution and
FP32 state**. The solver uses the existing 8×16×32 cache, 7×15×31 owned volume
and 16-step macro schedule. A physical cell's continuous residence in one tile
is not 16 updates; the shortest owned extent is seven.

Prepared execution includes preserving device copies, fields, all CPML history,
source additions, raw observations/DFT, ordinary residual steps and clocks.
Construction/upload/readback are excluded. Each full comparison uses fresh
A–B–B–A processes, with one full checked first call and one full checked timed
repeat per process. Both first/repeat results must match all 30 reference leaves
bytewise, be finite and preserve caller inputs. Pair ranges are observations,
not confidence intervals. No JAX execution is present in the native workers. These measurements
cover the prepared forward solve; no adjoint benchmark was run.

## Full-duration results

| Yee domain | Timesteps | Actual duration (fs) | Control (s) | Selected (s) | Paired throughput | Observed pair range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512³ | 12,347 | 30.001479 | 104.825124 | 103.641856 | 1.011417× | 1.011096–1.011738× |
| 564³ | 13,601 | 30.001496 | 147.635675 | 146.217335 | 1.009700× | 1.009486–1.009915× |

Control/selected times are medians of the two process-level timed repeats.
Throughput is the median of the two matched pair ratios. The range is not a
confidence interval. These are comparisons with the explicit previous temporal
control, not a fresh stock-main or FDTDX-release benchmark. The legacy filename
`prefix-comparison.json` is also used for these full fixtures; actual step counts,
fixture hashes and durations determine scope.

Cold construction and first call are reported separately:

| Domain | Build | Median setup (s) | Median first full call (s) |
| --- | --- | ---: | ---: |
| 512³ | Control | 2.822 | 114.888 |
| 512³ | Selected | 2.644 | 113.881 |
| 564³ | Control | 3.996 | 164.550 |
| 564³ | Selected | 4.244 | 163.180 |

[Comparison chart](../../.cache/cuda-trace-systolic-2026-11-13-1934/full-comparison.html),
[standalone PNG](../../.cache/cuda-trace-systolic-2026-11-13-1934/full-comparison.png)
and [recorded comparison data](../../.cache/cuda-trace-systolic-2026-11-13-1934/comparisons.json).

## Validation scope

The coupled suite covers seeded cube/rectangle, both macro depths, 35-step
wrap/continuation, the 259-step Mie case, nonzero clock origins and material/
source/probe sensitivity. The focused CPU suite passes **181 tests**. Emitted C++
frontier checks for three representative generated layouts use injective
field/location tokens (570,090, 569,842 and 318,990 reads) and a wrong-provider
negative control; source callbacks run under host ASan/UBSan. These checks and
their outputs are retained in the campaign directory.

The selected composition itself passes these deadline-bounded GPU diagnostics:

| Scope | Tool | Result | Process time (s) |
| --- | --- | --- | ---: |
| Exact depth16 NoIOUniform1024, one launch after skip8 | memcheck | pass, explicit zero summary | 87.020 |
| Exact depth16 NoIOUniform1024, one launch after skip8 | synccheck | pass, explicit zero summary | 42.466 |
| Exact depth16 NoIOUniform1024, one launch after skip8 | racecheck | pass, explicit zero summary | 42.317 |
| Exact depth16 SourcePlanePair Policy5, 35-step prefix, one launch after skip8 | memcheck | pass, explicit zero summary | 83.650 |

The NoIO packet is the seeded 136³ point-probe case, 35 steps, ordinary
Program oracle and reset/repeat parity. CUPTI separately records
**144 actual launches** of the exact selected
NoIO symbol for that same packet and worker. The source memcheck uses the
35-step prefix of the Mie plane-source packet, an independent ordinary Program
oracle and all 30 first/repeat outputs. CUPTI records
**120 exact Policy5 launches** for that
same packet and worker, with zero drops. The initial source command failed
before CUDA because of an NPZ/NPY reader mismatch. After a byte-identical
224-array conversion, the full 259-step diagnostic exceeded 180 seconds and is
**incomplete**, not a passing sanitizer run. The bounded 35-step check above uses
the compatible validation reader. Both unsuccessful attempts remain recorded;
full 259-step numerical validation passed separately. A separate 512³ activity
run records **1,712 bounded-source Policy5
launches**, with all 30 parity and zero dropped records. Activity timings are
not used as benchmark timings or as wall-time shares.

These are selected-launch diagnostic scopes, **not whole-grid sanitizer
certification**. Racecheck primarily addresses the instrumented shared-memory
accesses; it does not establish every inter-kernel global visibility invariant.
The typed ownership/publication checks and full numerical cases supply
complementary evidence. Earlier cache-exit-only diagnostic records remain
separate and are not substituted for the selected composition's results.

[Selected validation and module identity](../../.cache/cuda-trace-systolic-2026-11-13-1934/selected-final.json),
[confirmed diagnostic records and retained unsuccessful attempt](../../.cache/cuda-trace-systolic-2026-11-13-1934/selected-confirmation-diagnostics/confirmed-processes.json).

## Logical ownership cut: gate for the next layout

A separate CPU calculation counts the unique field values required across an
owner boundary in an unbounded homogeneous Yee interior. It enumerates the
actual integer lattice, field-specific retiming and both rotated parity phases.
It is an infinite-horizon logical cut, **not measured CUDA traffic or a runtime
prediction**; it excludes CPML, materials, sources, monitors, address work and
cache/coalescing behavior.

| Ownership | Physical cells/field/owner | Unique incoming field values/cell | Mean remaining updates in same owner | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Current 7×15×31 box | 3,255 | 1.901690 | 3.268817 | 7 |
| Rotated x–y, transformed 7×30×31 | 3,255 | 1.720430–1.722888 | 3.505376 | 7 |
| Balanced 15³ box reference | 3,375 | 1.573333 | 4.266667 | 15 |

At equal physical volume, the rotated example reduces this logical cut by
about 9.4–9.5%. The balanced reference reduces the **normalized** cut by 17.3%
but has 3.7% more useful cells. Neither alone supports a 2× claim. The longer
residence of a balanced tile is a reason to investigate a 16³ cache with
16-lane subwarp X communication, rather than another permutation of
8×16×32. Such a lowering must handle two Y rows per warp, inter-group Z edges,
source geometry and its register/shared allocation together. The current
full-warp column generator does not implement that mapping.

The calculation and explicit assumptions are retained as
`ownership-traffic-model.json` and `model_ownership_traffic.py` in the campaign
root. These field-only counts do not replace the full-physics benchmark.

## Artifacts and reproduction

Campaign root: `.cache/cuda-trace-systolic-2026-11-13-1934/`.
[Selected source archive](../../.cache/cuda-trace-systolic-2026-11-13-1934/selected-source.tar.gz),
[control-to-selected CUDA patch](../../.cache/cuda-trace-systolic-2026-11-13-1934/selected-cuda.patch),
[implementation file hashes](../../.cache/cuda-trace-systolic-2026-11-13-1934/implementation-manifest.json).
It retains the explicit control contract, immutable candidate sources/modules,
compiler resource gates, typed/finite proofs, actual emitted-helper host checks,
fresh-process logs, comparisons, sanitizer scopes and activity records.

The selected source can be regenerated with
[`compose_cuda_systolic_passes.py`](../../scripts/compose_cuda_systolic_passes.py)
from the control source, its explicit SHA and the passes
`bounded-source terminal-reuse cache-exit`. Build with
[`cuda_trace_research_build.py`](../../scripts/cuda_trace_research_build.py)
and `--expected-kernels 68`. Execute a matched fixture using
[`run_cuda_systolic_experiment.py`](../../scripts/run_cuda_systolic_experiment.py)
with the campaign's control contract. The source package contains all 51
selected source files, build/configuration commands and compiler/resource
provenance. A source-identical regeneration passes against all 51 hashes.
`selected-cuda.patch` changes only `src/asymmetric_native.cuh` relative to the
frozen control.

For example, from this worktree, with a new disjoint output directory:

```bash
python scripts/compose_cuda_systolic_passes.py \
  --source .cache/cuda-trace-150min-2026-11-13-1620/phase-grouped-sourcefacts-v1/source \
  --expected-consumer-sha256 cce32cfee8c4497d05c30585672935d03515bbf6fcf2bfc179c45865a2f3b7e1 \
  --output .cache/cuda-systolic-reproduction/selected/source \
  --passes bounded-source terminal-reuse cache-exit
python scripts/cuda_trace_research_build.py \
  --candidate .cache/cuda-systolic-reproduction/selected --expected-kernels 68
```

These research harnesses depend on frozen local source/toolchain artifacts;
they are not a packaged public frontend or a production-default change.
The run harness also requires an explicit current GPU deadline in its control
contract: the recorded two-hour contract intentionally expires after this
study. A later authorized run must use a new contract rather than modifying
the historical evidence.

## References and next architectural step

The [fdtd-z implementation](https://github.com/spinsphotonics/fdtdz/tree/ef4ea7a8ca4c4554c84402912657253cec3ac460)
uses a hand-designed span/skew/scan hierarchy with CUDA kernels behind JAX.
Its limited Z extent and boundary choices differ from this complete cubic
CPML12 workload. The [FDTDX creator's proposal](https://github.com/ymahlau/fdtdx/issues/104)
motivates expressing physics per cell and deriving ownership/schedules.
[DiamondCandy](https://library.keldysh.ru/preprint.asp?id=2018-225&lg=e),
[the full-3D systolic FDTD paper](https://arxiv.org/html/2502.20610v1),
[AN5D](https://arxiv.org/abs/2001.01473) and
[EBISU](https://arxiv.org/abs/2305.07390) supply related decomposition,
storage-lifetime and reuse ideas, not a transferable speedup guarantee.

The next implementation should make geometry and storage a joint compiler
choice:

1. Add a 16-lane X mapping and a balanced 16³ cache, with generated Y-row and
   Z-group ownership. Retain exact plane-source specialization for that shape
   rather than losing it to a generic fallback.
2. Lower the rotated affine example with parity-aware component placement and
   exact packed CPML regions. Compare its logical communication savings with
   actual address, register and scheduling costs before expanding it.
3. Allocate H/E and CPML storage from the complete live-range/effect graph;
   search phase ordering and tile geometry together. Include source/observer
   kernels in the objective, not just an interior-only throughput score.

Those lowerings still need independent address/publication checks, resource
admission, full-physics parity, positive CUDA diagnostic scopes and replicated
full benchmarks. The present study does not establish another 2× route or an
executed DiamondCandy engine.
