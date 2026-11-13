# Moving-subdomain schedule: next CUDA architecture

This is an implementation design supported by CPU dependency proofs and an
isolated field-only CUDA mechanism probe. It is not a complete BeamZ backend or
a measured complete-engine speedup. The existing two-step solver remains the
control. Experimental depth-four CUDA is disabled by default.

## Why the next schedule must change communication

The four-step overlapping prototype is numerically exact, including CPML12,
source phases and all six DFT faces, but loses by more than four times on the
large complete fixtures. Its bulk kernel alone takes more than twice the time
of the entire two-step solve. The complete 272³ launch executes approximately
2.09 field expressions per useful field expression before auxiliary, source,
publication and observation costs. Its 480-thread bulk CTA uses 91 registers
per thread and achieves one resident CTA (31.25% theoretical thread occupancy).
Increasing temporal depth without changing that geometry is a poor next step.

The [systolic FDTD paper](https://arxiv.org/html/2502.20610v1) instead moves
subdomains diagonally through space and time and communicates boundary trails.
Its published H100 throughput uses 16-bit fields, an absorber other than CPML,
and no sources or output fields. Its mechanism is relevant; that throughput is
not a comparison against BeamZ's complete FP32 scattering workload.

Tile dimensions cannot be copied directly: 32×16×16 cells with six FP32 fields
alone require 196,608 bytes, before materials or auxiliaries. SM86 has 128 KiB
combined L1/shared capacity, 64K 32-bit registers and at most 48 resident warps.
The current 256-thread, high-register probe therefore has little latency-hiding
capacity even when shared storage fits. Resource limits and asynchronous-copy
capabilities are documented in the primary
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
The next mapping must reduce register/address lifetimes as well as global bytes;
forcing occupancy with spilled registers has already lost in this study.

## Acyclic coordinates from the actual cell graph

BeamZ's H update at integer version n reads E at version n−1, at the same
physical cell and positive adjacent cells. E at n reads H at n at the same
cell and negative adjacent cells. Old fields and CPML memories are at n−1.

For each field or auxiliary value, define transformed coordinates

```
u = (z + n, y + n, x + n).
```

Every evolving-state dependency has transformed coordinates componentwise less
than or equal to its destination. Partition u into fixed integer boxes. A box
then moves toward decreasing physical coordinates as n increases. It consumes
only its own values or values from lower boxes. Order boxes by the sum of their
three box indices; execute local versions in increasing order, H before E.
There is no cyclic wait between boxes and no need to recompute overlapping
physical field updates.

[`scripts/prove_cuda_systolic_schedule.py`](../scripts/prove_cuda_systolic_schedule.py)
checks the actual Yee and local CPML trace dependencies, distinct staggered
component extents, physical-wall reads, unique ownership and producer-before-
consumer ordering. Twelve cases cover two irregular geometries, D2/D4/D8 and
CPML/no-CPML. Unskewed boxes and a deliberately missing required boundary value
fail. A separate inverse-stencil enumeration checks producer destinations
against every consumer requirement, including version-zero initial reads.
A packed surface layout and its memory ordering remain to be implemented.
This model retains every version in host dictionaries. It proves neither CUDA
memory ordering nor floating-point execution. Edge/corner trails are intrinsic:
H reads transformed E values at offsets such as (−1,−1,0), while old local
field/psi values have offset (−1,−1,−1).

```bash
python -m scripts.prove_cuda_systolic_schedule --output /tmp/systolic-proof.json
pytest -q tests/unit/test_cuda_systolic_schedule.py
```

## First executable scheduler

1. Generate box tasks and exact incoming/outgoing field and auxiliary surfaces
   from the trace. Start with short temporal slabs, explicit CUDA launches per
   topological rank and no spinning CTAs. A same-rank frontier contains independent
   tasks. This supplies a progress guarantee even when the frontier exceeds GPU
   residency. Ordinary CUDA graph nodes can replay the finite launch sequence.
2. Give each task unique physical/version ownership. Keep its evolving interior
   in registers/shared storage; communicate only required surface values. Scatter
   the final slab state once. Preserve a global-state reference implementation
   until every field and CPML leaf matches exactly.
3. Derive face-trail lifetime before allocating storage. In the proved stencil,
   cross-box rank distance is at most three. Four rotating rank epochs now pass
   a logical address/lifetime model and the field-only CUDA probe below. The
   full CPML implementation still needs packed-slab addresses, activity changes
   and auxiliary capacity admission. Do not allocate a full field volume per
   local timestep.
4. Carry immutable packed material data with moving cells, fetching new material
   on incoming surfaces. Decode against the original codebooks and preserve
   arithmetic rounding. A stationary material array is not stationary relative
   to a moving tile; repeatedly decoding it at every substep can erase the gain.
5. Compute CPML activity from physical coordinates at each version. A moving box
   can cross a CPML interface within a slab. Psi remains attached to its physical
   cell and follows the same versioned ownership rules; a fixed per-box CPML mask
   is insufficient. Integrate faces, edges and corners before headline timing.
6. Evaluate sources with absolute H/E phase clocks at their owned versions.
   For observations, reuse deduplicated raw-cell staging and the original DFT
   interpolation/reduction order. A monitor gather can span boxes: separately
   computed partial sums would change FP32 rounding. Bound staging by a slab and
   retain chronological updates, including continuation and odd tails.

Only after this graph-launched version wins should a persistent/cooperative
executor be considered. A global ready queue or CTA spin loop does not establish
progress. Measure compiled resources on SM86 before choosing a resident grid;
Hopper clusters or TMA cannot be prerequisites for the local RTX3090.

## Promotion gates and useful-work budget

The 272³ D2 instrumented kernel sum is approximately 33.6% core, 56.5% edge
work, 7.8% DFT and 2.1% sources. These are kernel-time shares, not unprofiled
wall-time shares. Even eliminating the core entirely would give only about
1.5× for that measured budget. A threefold complete-solve target requires a
major reduction of both core and edge communication. It cannot follow from a
faster interior microbenchmark alone.

Admit a candidate only after all of the following:

- All six fields, twelve CPML arrays, full DFT state and clocks match the frozen
  reference; nonzero initial state, 3+3+1, odd/odd, 259-step boundaries and full
  pulse continuation are included. Keep all 12 CPML cells at every resolution.
- Unique update and publication proofs reject deliberate omissions. CUDA
  memcheck, synccheck and bounded racecheck exercise the actual emitted kernel.
- Repeated balanced fresh-process trials beat the best measured D2 setting,
  with matched precision, sources, monitors, ownership, initialization and reset.
  Include both a scattering cube and a rectangular mode-source case.
- Report useful updates, executed updates, registers/spills, shared bytes and
  required global traffic. When hardware counters are unavailable, mark traffic
  estimates as estimates. Reject a depth increase that merely expands halo work.

Keep the current solver as the small-domain and unsupported-feature fallback.
The initial production target is a demonstrated complete-workload gain, followed
by a measured multi-fold gain; neither is implied by the paper's headline rate.

## Implemented field-only mechanism

[`scripts/cuda_moving_box_probe.cu`](../scripts/cuda_moving_box_probe.cu) is a
standalone moving-box CUDA implementation. It uses real staggered component
supports, explicit topological-frontier launches, generated scalar Yee arithmetic
and a separately written ordinary two-phase CUDA reference. It has **no CPML,
sources, material geometry or DFTs**. Its reference is much simpler than BeamZ's
optimized production D2 control. Its results cannot be reported as engine gains.

Each box stores 4×8×32 cells in 24 KiB shared field storage. High-face union
storage has 373 sites per component/version. Small N17/N33 cases at T4/T8/T16
match all field/padding/guard bits and reject invalid trail reads. Schedule-aware
relative addressing removes repeated global-coordinate division, reducing the
first probe's 254 registers plus spills to 210 registers without spills. On
N256, a bounded screen measures roughly 2.17–2.20× versus the simple two-phase
reference. This is a field-only mechanism result with a limited control.

The four-frontier ring uses immutable tile-to-frontier slots and owner tags.
N256/T8 cuts trail payload from 1,468,271,232 to 87,657,984 bytes, but increases
register pressure/spills and is about 1.6× slower than direct trail storage.
The ring is a validated memory experiment, not the selected performance path.
An additional measured lowering hoists eight neighbor bases and owner checks
once per CTA into 64 shared bytes, removing spills (196 registers). Its N256/T8
time improves to about 37 ms, still slower than direct's 29.6–29.9 ms. It retains
the same four-epoch lifetime and 16.75× boundary-storage reduction. Both final
host harnesses recheck repeated output and initialized trail guards after timing.
An intentionally unsafe two-epoch control rejects stale owners and mismatches
the reference. Both direct-relative and four-epoch probes pass bounded CUDA
memcheck, synccheck and racecheck on N17/T8. These do not establish full-physics
correctness or a production-quality resource mapping.

The independent CPU ring model checks 12 Yee/local-CPML trace cases and 95,476
consumer values; three-epoch early reclamation and wrong-version reads fail.
Logical psi supports in this model are not the production packed CPML slabs.
The four-epoch lifetime bound covers field/psi stencil consumers. A monitor
reader deferred until the end of a slab can outlive those epochs. Independent
sample staging or an explicitly earlier observation task must satisfy that
reader before trail reuse; the existing bound does not prove observer lifetime.

```bash
python -m scripts.prove_cuda_systolic_ring --output /tmp/ring-proof.json
pytest -q tests/unit/test_cuda_systolic_ring.py
python scripts/probe_cuda_moving_box.py --nvcc /path/to/nvcc \
  --output-dir /tmp/moving-probe --run 33 8
```

The compile wrapper freezes source, generated arithmetic, flags and hashes;
execution is explicit through `--run`. Use `--trail-ring 4` for the memory
experiment. Preserve the direct path as the control. Raw experiments and
compiler-resource rejections are under
`.cache/cuda-autoresearch-2026-11-11/moving-box-*`.

## Next implementation gates

1. Extend the tracer's placement decisions beyond arithmetic. Classify address,
   coefficient and effect nodes by CTA/warp/cell/version uniformity, and hoist
   reusable metadata once. The measured eight-neighbor lowering is a concrete
   example. Reduce working-set lifetimes without forcing spills through an
   occupancy cap. Collect hardware traffic/stall counters when available.
2. Connect the moving task plan to the native runner with actual packed E
   materials and CPML12. Derive per-version physical activity and packed psi
   addresses from the checked IR; carry or fetch material data as cells move.
   Validate faces, edges, corners and crossing interfaces before timing.
3. Integrate phase-accurate source effects and independent raw observation
   publication. Reuse the existing DFT arithmetic; extend lifetime proofs to
   every deferred reader. Retain the full source/DFT workload and clock tests.
4. Compare complete scattering and rectangular workloads against their tuned
   D2 controls in balanced fresh processes. Select direct histories or the
   bounded ring by measured speed and admitted memory. Keep D2 for small or
   unsupported cases. Promote only a complete-workload win; the field-only
   twofold result and paper headline do not meet that gate.

The high-level NumPy scene compiler and an independent native wheel can follow
the prepared-input boundary. They improve independence and usability; the
measured FFI/native tie means their implementation is not the primary throughput
optimization in the current forward path.
