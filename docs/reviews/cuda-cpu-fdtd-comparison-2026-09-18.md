# CPU FDTD lessons for the RTX3090 backend

The local reference is `/home/quentinwach/Code/FDTD`, clean at commit
`455e73d` (`Make the non-PML simulation domain explicit`). This is a source
comparison, supplemented by an executable geometry model; no CPU throughput
benchmark was run and no CUDA speedup is claimed from this investigation.

The strongest transferable idea is **dependency-ordered temporal reuse without
recomputing overlapping halos**. The CPU solver has substantially more than a
larger temporal tile: its storage, execution order, PML ownership, source timing,
and monitor buffering all support the same in-place space-time schedule. Our
current two-step CUDA prototype buys independent CTAs by duplicating halo work
and maintaining frozen input banks. Increasing its temporal depth alone does not
reproduce the CPU design.

| Mechanism | Local CPU implementation | Current CUDA implementation | Implication |
|---|---|---|---|
| Temporal execution | Sliding staircase windows, normally 16 groups deep; asynchronous spatial and cross-pass dependencies | Independent overlapping two-step tiles; three field banks and two auxiliary banks | Investigate a GPU dependency schedule that avoids redundant halo updates |
| Within-block ordering | Precomputed, compressed skewed 4-D Morton order, time fastest | Regular rolling planes in shared memory | Keep regular coalesced GPU execution; borrow the space-time dependency structure, not the CPU interpreter |
| PML state | Group-local records containing only active-axis state; PML advances with the group | Separate packed global term arrays; intermediate pair state in shared planes | Test boundary-brick state with fixed offsets and coalesced term storage |
| Classification | Build-time runs of lean, material, seven PML masks, and special operations | Packed material lookup, axis-specialized CPML, runtime geometry/source tests | We already have some specialization; precompute more tile metadata and separate rare special work |
| Field layout | SIMD AoSoA: all six component vectors in one aligned cell group | Component SoA with different logical Yee extents and optional physical padding | Evaluate brick-local SoA, preserving contiguous lanes per component |
| Sources and monitors | Local group timestep; source injection in H/E order; raw samples drained every 256 steps | Source phase integration, paired monitor gathers and chronological DFT | Preserve local timestep semantics in a wavefront; deeper sample buffering is secondary for current compact monitors |
| Parallelism | Ready-task counters, cross-pass pipelining, cache-cluster routing, local continuation and stealing | CUDA launches/graphs and independent CTAs | GPU needs enough ready work and correct device memory ordering; CPU thread scheduling is not a drop-in solution |

Actual implementation references, rather than the stale “design only” banner in
`fdtd_block_schedule.h`:

- [Staircase scheduler](</home/quentinwach/Code/FDTD/src/fdtd_sim.cc:3330>):
  `RunStaircaseBlock` executes a prebuilt program, signals +X/+Y/+Z successors,
  and signals the (-X,-Y,-Z) successor in the next pass. Arrival counters use
  acquire/release ordering and parity slots. Temporal continuation is preferred
  when it stays in the worker's cache cluster.
- [Morton traversal construction](</home/quentinwach/Code/FDTD/src/fdtd_group_order.cc:100>):
  coordinates are skewed by local time before clipping. Morton order is over
  skewed coordinates, not arbitrary physical cell/time coordinates.
- [Operation classification](</home/quentinwach/Code/FDTD/src/fdtd_block_program.h:213>)
  and [specialized execution](</home/quentinwach/Code/FDTD/src/fdtd_kernel_body.inc:504>):
  lean work and PML masks are explicit instruction classes; the implementation
  uses specialized group kernels and a musttail interpreter.
- [PML recurrence layout](</home/quentinwach/Code/FDTD/src/fdtd_kernel_body.inc:185>):
  active axes have compile-time state offsets; inactive axes omit those loads.
- [Segmented execution](</home/quentinwach/Code/FDTD/src/fdtd_sim.cc:7900>) and
  [sample draining](</home/quentinwach/Code/FDTD/src/fdtd_monitor.cc:99>):
  globally coherent states are recovered at segment boundaries, with local
  counters driving sources and sampling inside the segment.
- [Temporal equivalence tests](</home/quentinwach/Code/FDTD/src/fdtd_sim_test.cc:770>):
  tests compare energy and monitor phasors bit-for-bit against single stepping,
  including multiple blocks and repeated parity reuse. These tests were read,
  not rerun; they are not proof of our GPU scheduler.

The geometry model in `scripts/analyze_cuda_wavefront_geometry.py` checks 630
one-dimensional clipped-window configurations, including non-multiple extents.
At every staircase local time, the shifted block windows cover each active cell
exactly once. Cartesian products inherit that coverage. This establishes the
coverage property only, not Yee dependencies, race freedom, or GPU performance.

For the current CUDA two-step implementation, an unclipped owned X×Y×Z tile
executes stage envelopes `(X+3-k)(Y+3-k)(Z+3-k)`, for k=0..3. Dividing their sum
by `4XYZ` gives:

| Owned tile (x,y,z) | Stage work slots / owned updates |
|---|---:|
| 16×8×16 interior/default boundary | 1.448 |
| 15×16×16 oriented x face | 1.333 |
| 32×4×16 oriented y/z face | 1.609 |
| 16×8×4 shallow edge/corner | 1.865 |

These are loop-envelope counts, not measured instruction counts or a prediction
of recoverable speedup. Component bounds, CPML tests, clipped regions, shared
memory and memory transactions change actual cost. They nevertheless explain
why simply increasing overlap depth is a poor substitute for the CPU schedule.
The winning bare interior kernel remains worth preserving as a control.

A second model counts equal-rank layers of a hypothetical tiled DAG. With
16×8×16 logical tiles, the 128×256×512 domain has at most 240 tiles in a
single-pass spatial layer; the 1024×256×64 domain has 128. Adding eight pipelined
passes with rank `x+y+z+4*pass` raises those counts to 1,528 and 1,024. These
counts exclude the CPU's extra far-edge blocks and are not measured ready-queue
occupancy. They show why a pass-by-pass wavefront can underfill a GPU and why
cross-pass pipelining must be part of the experiment from the start. The CPU's
four-edge graph is derived for cubes in *group coordinates* with temporal depth
equal to the group-block side. Our rectangular GPU microtiles do not automatically
satisfy that contract: either introduce a compatible outer macroblock or derive
and verify a different dependency graph. The model counts an illustrative graph;
it does not prove that the proposed rectangular GPU updates can use those edges.

**Benchmark accounting needs separation from architecture.** The user's 10+
GCUPS result may be from newer measurements or different hardware; it was not
found in this checkout. Its older `HANDOFF.md` records 1,152–1,795 Mcells/s
(1.152–1.795 GCUPS) for temporal runs, about 1.7–2.0× single stepping. The current
[benchmark](</home/quentinwach/Code/FDTD/src/fdtd_benchmark.cc:229>) computes
`num_cells_xyz product × steps / run_time`, counts a complete cell timestep once,
and uses the thread pool—not a single CPU core. `num_cells_xyz` returns the
allocated grid shape, which includes guards and alignment slack. Our target
counts logical cells and must continue excluding inactive storage padding.

The CPU benchmark really injects a port eigenmode and installs a port monitor,
but uses one monitored wavelength. Its monitor implementation automatically
subsamples according to source/monitor frequency, capped at stride 64. That is
not equivalent to silently dropping samples from our requested monitor schedule.
Its timestep is also a separate discretization choice: equal GCUPS alone does
not establish equal physical simulated time or accuracy.

[CPU grid alignment](</home/quentinwach/Code/FDTD/src/fdtd_grid_layout.h:18>) may
round PML thickness and insert physical alignment slack. We must retain exactly
12 physical CPML cells, the requested domain and original source/monitor
coordinates. Storage-only padding and inactive lanes are appropriate; moving the
absorber or changing its thickness is not. SIMD widths that do not divide 12
require mixed-lane treatment or separate boundary groups.

**Recommended next implementation sequence:**

1. Build a small GPU wavefront correctness prototype with explicit space-time
   ownership, using one shared numerical update definition. Start with dependency
   stages that can be audited, then a persistent ready queue that only dequeues
   runnable work. Never let resident CTAs occupy the GPU while waiting for
   unscheduled producer CTAs. Validate same-cell counts, cross-tile dependencies,
   odd lengths, clipped domains, and local source/monitor times.
2. Retain the proven 16×8×16 independent interior as a performance control.
   Compare regular skewed/diamond GPU tiles at temporal depths 2/4/8 against the
   existing overlap design. A CPU Morton bytecode interpreter should not be
   ported into per-lane GPU execution: divergence, indirect dispatch and loss of
   coalescing could outweigh reuse.
3. Bring 12-cell CPML into that ownership model with fixed per-brick state
   offsets and preclassified face/edge/corner descriptors. Test FP32 first, then
   BF16 auxiliary storage with FP32 arithmetic. Separately measure physical
   stride padding and compact state layout; neither is an assumed improvement.
4. Include sources and sample gathering at the exact substep. Drain buffered
   monitor samples in the established chronological order at coherent boundaries.
   Current profiles put sources and compact monitors below 1% each, so prioritize
   field/PML scheduling over further DFT tuning.
5. Gate promotion on the complete varied-size/aspect-ratio suite, including
   irregular widths, long continuations, mode sources, requested monitor
   frequencies, exact 12-cell CPML, full-state comparisons, physical modal
   accuracy, and sanitizer checks. Report logical GCUPS, redundant-work counts,
   achieved occupancy, storage footprint and end-to-end timing separately.

The new architecture is a testable hypothesis, not a promise of 9 GCUPS. Current
best established schedules still miss that target on narrow domains; the
experimental fused paths also have unresolved large-case numerical differences.
No defaults or kernel behavior were changed by this review, no CPU repository
files were modified, and nothing was pushed.
