# CUDA-native runtime and deep temporal blocking plan

Prepared 2026-11-11 after the released-FDTDX comparison. This is a plan; no new
GPU benchmark or runtime change was made while preparing it. RunPod stays stopped.
The intended outcome is a **3–5× research target against the best current BeamZ
CUDA solver on large complete workloads**, with 10× an exploratory stretch.
These are acceptance goals, not predictions or measured gains.

## Architecture decision

Make the forward solver independent of JAX: a C++ CUDA library owns state,
workspace, execution, clocks, and observations. Python defines scenes and local
physics and submits an entire solve. Framework integrations become optional
adapters to that same library. Build a new space-time scheduler alongside the
current two-step implementation; retain the latter as a measured control.

A framework rewrite by itself does not establish a speedup. The current
`cuda_mstep` path already calls native CUDA graphs through JAX FFI, in bounded
256-step chunks. Its heavy Yee/CPML computation is not a sequence of ordinary
JAX tensor expressions. Native ownership can remove constraints on allocation,
aliasing, layout conversion, and scheduling; the large throughput gain must
come from doing less off-chip work and advancing further on chip.

Current evidence:

- Completed RTX3090/250 W Mie timings: BeamZ 1.76/3.53/17.20 s at 5/4/2.5 nm,
  with CPML12, six field-DFT faces, FP32 fields, and 30 fs duration. The 2 nm
  case passed validation; its timing was stopped. This is the current comparison
  configuration, not proof that every launch setting is optimal.
- In the accepted large rectangular profile (256×512×512, CPML12), the core is
  51.28% of summed kernel durations; first/second edge work is 25.79%/22.10%.
  These instrumented event sums exclude CPU gaps/copies and are not wall shares.
  Eliminating core time alone would leave roughly half of this measured work.
- Production temporal depth is two. The current first-step edge envelope is
  recomputed into full field banks, followed by second-step edges. Source
  intersections can exclude whole column quadrants from the temporal core.
- At 272³ the comparison sampled 5.17 GiB device-global peak for BeamZ versus
  3.26 GiB for released FDTDX. This is a memory opportunity, not evidence that
  JAX alone owns the difference. Peaks include compilation and the idle context.
- A JAX-free scalar cell tracer, generated Yee arithmetic, CPML recurrence
  traces, rolling/register prototypes, and native launchers already exist.
  A direct harness validates the actual core without JAX, but is not a full
  solver or an overhead benchmark.
- Earlier register-core and deeper wavefront experiments lost in production
  or synthetic matched controls. Increasing the depth parameter and repeating
  those schedules is insufficient. New experiments must change data movement,
  intermediate publication, and ownership, and retain those losing controls.

Evidence: [optimization report](cuda-autoresearch-2026-11-11.md),
[tracing implementation](cuda-tracing-2026-11-11/custom-tracing.md),
[comparison report](fdtdx-comparison-2026-11-11.md), and raw profile
`.cache/cuda-autoresearch-2026-11-11/final-profiles/summary.json`.

## 1. Establish a complete native execution boundary

Split `cuda/CMakeLists.txt` into a framework-free `beamz_cuda_core` library,
a thin native Python binding, and an optional JAX FFI binding. Only the latter
requires JAX headers. Both bindings must link the same compiled core, so the
runtime A/B comparison does not accidentally compare different kernel builds.
Reuse `program.cu`, `graph.cu`, `update.cu`, `io.cu`, and the existing native ABI;
introduce a versioned ownership API above the raw launch descriptors.

The native runtime owns CUDA streams/events, field banks, CPML auxiliary state,
coefficient palettes, source tables, detector accumulators, and reusable work
buffers. A solve makes one Python-to-C++ call. Native code replays bounded CUDA
graphs and manages clocks, tail steps, continuation, and completion. Start with
safe graph phase boundaries; a persistent kernel is a later scheduling choice.

Support an exclusive mutable state and explicit snapshots/clones. Do not
silently overwrite externally retained input arrays. Compare mutable native
runs against existing donated/owned-state CUDA runs, and preserved-input runs
against preserved-input controls. Report initialization/reset costs separately;
removing an input-preservation requirement is not a kernel speedup.

First consume the existing frozen NumPy/common-data contracts without JAX.
Then extract NumPy host preparation and lazy package imports so a user can
install and run the forward solver without JAX. This requires packaging work:
`beamz/__init__.py` currently imports optimization/simulation eagerly, and the
base dependency list requires JAX. Merely adding another backend flag would
not make the package JAX-free. Geometry/source/monitor preparation must also
have a native/NumPy path before claiming complete JAX independence.

Expose optional device interop using DLPack with explicit stream and lifetime
handling. Material/geometry autodiff belongs in optional adapters. Existing
CUDA adjoint kernels remain available; gradient integration is a separate
acceptance track and must not block the forward scheduling experiment.

**Deliverable:** the entire scattering workload runs in an environment where
JAX is absent. Initially this can use a previously prepared input artifact;
complete scene-to-result independence is a separate, explicit milestone.

## 2. Measure the actual constraints before choosing depth

For the same native kernels, compare JAX FFI and direct native execution with
identical input bytes, layout, buffers, reset policy, streams, and observations.
Use CUDA events plus synchronized wall time. Separate setup, transfers,
conversion, copies/clears, graph replay, and useful kernel work. This determines
whether removing JAX gives launch savings, allocation savings, or both.

Profile matched 256³, 384³ and memory-admitted 512³ workloads, the 680 nm scene,
and rectangular controls. Keep CPML12 on every face. Measure DRAM bytes, L2
traffic, spills, registers, shared memory, active warps, barrier stalls, executed
versus useful updates, edge work, source/monitor work, and host gaps. Use Nsight
Compute counters where permitted; otherwise retain event attribution and static
byte accounting and state which conclusions remain unverified.

Use a useful-work roofline. For six FP32 field components, one ideal read/write
of a whole field state is 48 bytes per cell. A temporal tile advancing T full
steps has an ideal **field-state** traffic floor of 48/T bytes per useful cell
step: 24 at T=2, 12 at T=4, and 6 at T=8. This excludes halos, coefficients,
CPML state, observations, spills, and layout conversion. D2→D8 therefore offers
4× lower ideal field traffic, not a guaranteed 4× complete-solve speedup.

**Deliverable:** a component-time and byte budget that identifies what must
change to reach 2×, 3×, and 5× complete-solve targets. Do not multiply unrelated
microbenchmark speedups or call a slow forced schedule the baseline.

## 3. Turn the existing tracer into a scheduling compiler

Keep the existing restricted Python cell language independent of JAX and
PyTorch. Trace one equation/region/material signature, not one Python graph
per physical cell. Extend the existing IR with actual staggered supports,
field/auxiliary versions, coefficient layouts, and explicit source/observation
effects. Record phase clocks, bounds, ownership, and numerical rounding.

Generate exact dependency cones through 2T H/E phases. Derive liveness and
register/shared-plane reuse without expanding an exponentially large scalar
expression tree. Generate tile descriptors and CUDA from the same checked IR.
The code generator must decide memory placement and publication as well as
emit arithmetic; the current generated cell header alone cannot do that.

Specialize bulk dielectric, material-interface, source, observation, CPML-face,
edge, and corner tiles. Keep generic fallbacks for unsupported materials and
features. Generate only signatures actually used by the scene to limit code
size. Preserve real Yee extents and coefficient averaging; changing discretization
is not a performance optimization.

**Deliverable:** deterministic codegen and dependency/ownership proofs for T=4
and T=8, including negative controls that fail when a required read, auxiliary
update, publication, or observation is removed.

## 4. Replace two-step columns with deeper space-time tiles

First implement a practical overlapped temporal-brick schedule at T=4 and T=8:
load a frozen initial dependency footprint, stream planes through shared memory,
keep short-lived values in registers, and publish uniquely owned final cells.
Use double-buffered asynchronous prefetch where it reduces measured stalls.
Coefficient decoding should occur once per reusable local lifetime, not once
per field update. Avoid global intermediate field writes between substeps.

Derive the halo footprint from the staggered dependency graph. Measure halo
recomputation, spills, occupancy, barriers and available parallelism for every
candidate. More depth can make a tile slower. Allow mixed depths: a bulk tile
may advance eight steps while a complex interface tile advances fewer, provided
their versioned boundary exchanges are correct.

In parallel with that algorithm design, define skewed diamond/swept ownership:
advance each subdomain as far as its dependencies permit and exchange compact
time-front surfaces. This is the main alternative when overlap/recomputation
limits deeper bricks. The objective is to replace repeated whole-volume edge
sweeps with local work and surface communication.

After a graph-launched phase implementation is correct, compare a systolic or
persistent cooperative executor. Use a resident grid sized from actual compiled
register/shared resource usage; do not let CTAs spin on producers that cannot
be scheduled. Grid synchronization requires a legal cooperative launch. A
persistent ready queue additionally needs proved progress, bounded queues, and
correct device publication ordering. Neither persistence nor high occupancy
is an end in itself; select by complete-solve time.

The design must run on the local RTX3090 (SM86). Ampere asynchronous copy is
available; Hopper TMA/cluster features cannot be prerequisites. Preserve the
streamed fallback for small/narrow domains where temporal blocking loses.

**Deliverable:** a changed communication algorithm with measured useful work,
not another isolated interior-only depth-eight demonstration.

## 5. Integrate CPML, sources, and DFT into the time tiles

This work is central, because roughly 48% of the existing profiled large-case
kernel time is outside the core.

- Keep CPML recurrences private to their owning tile across substeps. Preserve
  configured auxiliary precision and every required rounding point. Exchange
  evolving field dependencies correctly at the bulk/PML interface. Reuse the
  existing complete temporal-CPML implementation as a reference; do not assume
  its currently slower schedule will win unchanged at a larger depth.
- Apply sources inside the exact H/E substep. Source regions must not force
  unrelated columns back to an ordinary timestep sweep. Tile classification
  should depend on local effects, and common source slabs get specialized code.
- Give monitor points unique ownership. Compute their original interpolation
  and chronological complex DFT updates while required substep fields are
  available. If local observation is too expensive, buffer compact surface
  samples for a separate reduction. Never materialize full intermediate volumes
  solely for a few detector planes. No monitor decimation or omitted components
  may enter the headline speedup.

Keep CPML12 and the existing six-face benchmark intact. Also test compact mode
monitors and multiple frequencies so the scheduler is not overfit to one scene.

## 6. Make memory scale with necessary state

Remove unused compact scratch from full-bank implementations using an explicit
capability/ABI contract; never shrink allocations for an older component that
still accesses them. Replace per-cell edge/slot queues with regional brick
metadata where possible. Keep material palettes and immutable coefficients
resident and shared across timesteps.

Deep temporal blocking must not allocate T full global field volumes. Aim for
at most the required global state banks plus on-chip stage rings and compact
communication fronts. Audit each array's size and lifetime. Independently test
whether legal ownership permits fewer full banks in exclusive-state mode.
Padded/blocked layouts are candidates after the reference native path exists;
score conversion cost and memory as well as kernel speed.

A 1 nm scene is a later capacity target. Admit it from actual allocation
accounting; do not assume reduced scratch alone makes it fit.

## Validation, promotion, and research order

Reuse the existing integer dependency/taint proofs and independent scalar
oracles. A scheduling-only kernel initially preserves the explicit FP32/FMA
semantics and is tested against the current CUDA engine for all fields, CPML
state, DFTs, clocks, and retained inputs. Cover nonzero initial states, odd
counts, ragged dimensions, all faces, source/monitor intersections, and split
continuation. Run memory/race/synchronization tools on small bounded fixtures.

Add an independent high-precision reference for selected long decays. The
released-solver comparison accepted a signal-normalized floor for weak residual
fields; it does not justify accepting a new schedule with 13–25% residual error
against the same-equation CUDA reference. Track divergence/charge and decay
behavior rather than relying only on integrated spectra. Mixed precision,
reassociation, and approximate math are separate experiments.

Benchmark the best existing streamed/mstep configuration with matched mutable
or preserved-state semantics. Record full physics and memory. Use balanced fresh
processes and uninstrumented measurements; profiling is separate. Promote a
candidate only after correctness and an application gain survive retesting.
Freeze the numerical contract before performance screening.

The first sustained engineering block should produce:

1. A full JAX-free forward runner using frozen inputs and the existing kernels.
2. A controlled native-versus-FFI breakdown and complete memory/traffic budget.
3. T4/T8 generated schedules with resource estimates and dependency proofs.
4. One integrated bulk-plus-CPML/source/observation prototype, or a documented
   resource/communication failure that selects the swept alternative.
5. A reproducible results ledger and a decision on the next architecture.

Do not spend that block on a broad API rewrite, giant blind flag searches, or
repeating previously rejected kernels. The eventual gates are 2× as the first
substantial complete-solve milestone, 3–5× as the program target, and 10× only
if measured bandwidth/communication limits support it. No speedup is promised
by a deadline. All work remains local unless separately instructed otherwise.

## Primary research informing the design

- [AN5D](https://arxiv.org/abs/2001.01473): model-guided CUDA stencil code
  generation with deeper temporal blocking and reduced register/shared pressure.
  This is general-stencil evidence, not a BeamZ speed forecast.
- [Systolic FDTD update scheme](https://arxiv.org/abs/2502.20610): FDTD-specific
  communication restructuring and on-chip reuse.
- [fdtd-z implementation](https://github.com/spinsphotonics/fdtdz/blob/main/README.md):
  shows that a JAX-facing interface can wrap a systolic CUDA engine. Its documented
  material, boundary and output restrictions prevent treating it as a drop-in
  replacement for BeamZ's full CPML/DFT workload.
- [NVIDIA asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
  and [cooperative groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html):
  the hardware and synchronization contracts the generated schedules must obey.
