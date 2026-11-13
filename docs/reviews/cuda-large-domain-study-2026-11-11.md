# Larger domains and the circular-cache CUDA experiment

The complete 512³ guide benchmark improves from 5.984 s to 5.592 s with the existing
16×32 tile and 64-cell column segments: **7.0% more throughput**, with exact state
parity across three fresh processes. The paper-derived circular cache establishes
a separate modest field-only crossover:
**1.055× over the strongest matched two-step rolling control at 640³/T16**.
These are field-only kernel measurements, not a measured acceleration of the
complete BeamZ engine. The existing complete overlapping D4 path remains slower
at 256³/384³/512³. The larger register prototype is correct but remains slower.

The next integration architecture is described in
[CIRCULAR_RUNTIME_PLAN.md](../../cuda/CIRCULAR_RUNTIME_PLAN.md). The detailed
[dependency and traffic audit](cuda-large-domain-first-principles-2026-11-11.md)
records the paper study, independent equations and CPU proofs.

## Local machine and experimental scopes

All runs use the local RTX 3090, 24 GiB, SM86, driver 610.43.03. The 250 W power limit
was left unchanged. An idle ComfyUI context occupies approximately 410 MiB. GPU
experiments run serially. No RunPod resources were used.

The field probe uses deterministic nonzero initial fields, canonical generated
FP32 cell arithmetic, actual six-component staggered supports, and zero exterior
reads. It excludes CPML, source injection, material geometry and DFT accumulation.
The spatial control is a shared-memory tiled H/E implementation, substantially
faster than the original plain array kernel. Its fields update in place, while
the moving solver initially has separate input/output banks. Both execute the
same complete H/E update count and compare against an independent plain oracle.

CUDA event timings include the moving tile's initial cache loads, trail
communication and final scatter. Deterministic initialization, reference copies,
poisoning and host readback occur outside timing. Each process rotates plain,
spatial and moving timing positions. Raw samples and observed ranges are retained;
a small process range is not a statistical confidence interval.

The complete-workload scaling recipe fixes 80 nm resolution and grows physical
domain size. It uses a binary Si/SiO2 guide, FP32, **12 CPML cells at every size**,
one mode source, two fixed-aperture mode monitors 32/64 cells downstream, three
frequencies, and 512 complete timesteps. Both monitor signals are nonzero. This is
a separate scaling experiment from the fixed 680 nm nanoparticle Mie scene.

## Complete-workload controls

All four variants preserve the same immutable input packet and execute two
256-step chunks. The FFI/native D2 pair shares compiled CUDA code; generic D2/D4
share their own compiled module. All 30 state leaves, including fields, packed
CPML, monitor accumulators and clocks, match the independent reference **byte
for byte**, including repeated execution.

| Base domain | Yee cells | JAX→CUDA D2 | Native CUDA D2 | Generic D2 | Existing generic D4 |
|---|---:|---:|---:|---:|---:|
|256³|16.8M|0.9943s|0.9988s|1.0017s|3.9558s|
|384³|56.6M|2.8866s|2.8771s|2.8769s|11.2998s|
|512³|134.2M|6.0087s|5.9903s|5.9920s|23.9758s|

These are initial screens: one fresh process per variant, one warmup and two
within-process samples. No performance gain is promoted from sub-percent frontend
differences. Native D2 reaches approximately 11.47 billion base-cell updates/s
at 512³. Removing JAX from execution does not materially change this CUDA kernel's
runtime; it does reduce warmed process GPU residency here, from 16678 MiB for FFI
to 10748 MiB for native D2. Those figures include runtime/context allocations and
are distinct from exact solver payload.

The 512³ reference initially failed the stock workspace admission guard under
preserving ownership. The retry persisted the immutable packet first and donated
only the one-off reference state through the existing `build_scan` API. The stock
guard was retained. A 256³ control verified donated/preserved reference hashes for
all 30 leaves; the saved initial packet hashes were checked after donation.
All timed 512³ variants still preserve inputs. Reference peak JAX allocation was
14.55 GB and peak host RSS 8.83 GB; native timed-worker peak host RSS was 11.18 GB.

The new mmap fixture store, leafwise comparisons and `NativeSession.download_into`
avoid gathering all outputs into another full host copy. Their small gates cover
all 36 raw output buffers,30 logical leaves, invalid destinations, closed sessions
and concurrent serialization. Memory admission includes packed CPML, sparse
observations, private preservation copies and explicit reserve.

## Confirmed complete-engine improvement at 512³

The guide packet, module, preserving ownership and 512-step schedule stay fixed.
Only the existing tile/column-segment flags change. Three fresh processes per
setting use rotated/reversed order, one warmup and two timed samples each.
All 30 leaves and repeat results match byte for byte in all nine processes.

| Tile / column segment | Median runtime | Median paired throughput ratio |
|---|---:|---:|
|16×32 / whole column|5.983840s|1.000×|
|8×32 /64 cells|5.697356s|1.050×|
|16×32 /64 cells|5.592016s|1.070×|

The winning 16×32/64 observed process ratios are 1.06854–1.07028×. That is about
7.0% more throughput or 6.5% less runtime. The tested settings are available via
the existing configuration, before compiling/preparing the simulation:

```sh
export BEAMZ_CUDA_MSTEP_TILE=16x32
export BEAMZ_CUDA_MSTEP_SEGMENT=64
```

This confirms an improvement for this local 512³ workload, rather than an
automatic policy for every scene/GPU. The installed component/default selection
is unchanged. Raw fixtures, commands and the nine independent result hashes are
in `engine512-tiling-root-v1/confirmed.json`; `confirmed-index.json` records paired
ratios and all 30-output acceptance. No new deep-cache speedup is implied.

## First large field-only sweep

This first screen uses owned z/y/x 4×8×32 tiles and T8. Each row is one fresh
process, three balanced timing positions. Face caching stages boundary values
once per phase; circular caching retains physical points across timesteps.

| Base domain | Spatial control | Face cache | Circular cache | Face speedup |
|---|---:|---:|---:|---:|
|384³|63.9ms|55.2ms|54.8ms|1.16×|
|512³|152.8ms|119.9ms|125.7ms|1.27×|
|640³|306.9ms|224.8ms|230.4ms|1.37×|

640³ contains 262.1 million base Yee cells. At128³, these original moving variants
remain slower than the spatial control. The crossover is therefore measured,
rather than assumed from the traffic model.

## Correcting the cache geometry

The [paper's Figure 7](https://arxiv.org/html/2502.20610v1) led to a physical-slot
cache: a retained physical point keeps the same storage slot while the box moves.
The owned dimensions are one smaller than the cached dimensions. A scaled SM86
candidate owns 7×7×15 cells and caches 8×8×16. Power-of-two cache dimensions make
wrapping cheap, including negative physical wall coordinates.

The 735-point ownership region needs guarded partial thread tails. Incoming
producer coordinates require conditional wrap; the earlier owned-dimension
bitmask is invalid for 7/15. Independent proofs and GPU small gates cover both.
The circular publication load itself is restricted to outgoing high faces; an
unconditional interior load would race with the next H phase.

| Field-only candidate at 512³/T8 | Registers/thread | Shared bytes | Resident blocks/SM | Median paired speedup |
|---|---:|---:|---:|---:|
|Circular4×8×32, rolling heads|47|35704|2|1.38×|
|Circular8×8×16, rolling heads|45|33112|2|1.52×|
|Circular7×7×15, power-of-two cache|56|24640|3|1.83×|

These candidate screens use two fresh processes with opposite timing rotations,
three samples each. The paper-shaped candidate's observed process ratios range
from 1.72× to 1.95×. It uses no local memory/spills. Larger balanced confirmation,
depth selection and sanitizer results are recorded separately below when ready.
Block residency is the maximum returned by CUDA's occupancy API, not a hardware
counter measurement of achieved active blocks during execution.

## Strongest matched two-step control: the decisive comparison

The earlier spatial comparison uses a single-step control. BeamZ's D2 already
exploits temporal reuse, so that factor overstates progress against a strong
two-step algorithm. A new rolling control follows the production schedule while
using the probe's exact cell arithmetic, staggering, zero exterior reads and
initial state. It is an algorithm control, not a claim that the production ABI
kernel and this isolated implementation have identical cost.

Three justified geometries were screened. XY32×8 with 256 threads beats both
XY16×8/256 and XY32×16/512 at 512³. The selected comparison uses three fresh
processes per size/depth, four rotated timing samples per variant, with reversed
size/depth order. All 18 processes pass full bitwise fields, padding, guards and
repeat checks. The D2/circular binary passes memcheck, racecheck and synccheck on
the bounded33³/T4 case.

| Base domain | T8 cache / D2 speedup | T16 cache / D2 speedup |
|---|---:|---:|
|384³|0.880×|0.857×|
|512³|0.938×|0.987×|
|640³|0.953×|1.055×|

Values above 1 mean the circular cache is faster. At640³/T16 it takes 304.908 ms
versus 321.842 ms for D2; the observed paired process ratios are 1.054–1.082×.
This is the established crossover over the stronger baseline. No several-fold
engine gain is claimed from the spatial-control results. The regenerated
`large-domain-results.html` chart uses this D2 comparison as its default.

## Larger register cache and next integration gates

Larger cache geometry is the next architectural lever. Keeping one field family
in shared memory and both per-cell families in fixed-index registers could admit
an owned 15³/cache 16³ tile. The 256-thread version has 96 state registers before
scratch; compiled generic code actually uses 188 registers. Forcing 128 spills.
The 512-thread mapping halves state slots per thread. Making phase metadata
lifetimes explicit and using local register centers cuts it to 119 registers,
zero stack/spills, with 49220 bytes of dynamic shared memory. The opt-in source
and wrapper flag are retained and pass numerical gates and all three sanitizers.
Its 512³/T16 screen is 169.423 ms versus 156.712 ms for matched D2: rejected as a
performance candidate. A storage budget alone did not predict its runtime.

Separate interior specialization reduced register pressure but did not fit the
two-block budget; those root artifacts are explicitly compile-only, with GPU
execution disabled. Phase lifetime isolation saved approximately 40 registers;
the savings do not add linearly with other specializations. Source-level state
counts and static ptxas spill statistics are not measured dynamic traffic.

CPML has a separate storage opportunity: its recurrence is local in physical
space. The independent owner-chain proof permits a candidate packed global
in-place psi policy with ordered frontier launches, avoiding twelve dense shared
auxiliary caches. Sources still need absolute clocks; monitor samples still need
chronological accumulation. The runtime plan details those gates.

## Asymmetric storage: fitting useful state instead of forcing spills

A final prototype keeps Hx/Hz/Ex/Ez permanently in shared memory and Hy/Ey in
registers. The dependency graph permits this because Y derivatives never consume
the Y component; Z neighbors of the register fields stay on the same thread and
X neighbors use unconditional full-warp shuffles. The cache is 6×16×32, with
owned 5×15×31 points. There are only 12 retained field scalars per thread.

The corrected three-barrier phase-counter protocol is independently checked.
The compiled 512-thread kernel uses 64 registers, zero stack/spills and 49220 bytes
of shared memory, admitting two blocks per SM. Small/ragged and 512³/T16 numerical
gates, full padding/guards/repeats, memcheck, racecheck and synccheck all pass.

The two-block version takes 159.048 ms versus 157.292 ms for strongest matched D2:
essentially parity, with no clear win. The one-block 79-register version takes
197.873 ms. Lowering register pressure therefore improves the prototype itself
by about 24% throughput, but still does not justify engine promotion. These are
single-process screens with four rotated samples, not balanced confirmations.
`asymmetric-result-index.json` records raw paths, hashes and sanitizer scope.

The full-paper 15×15×31 warp-exchange prototype reduces shared requests further,
but its 128-register cap still produces a 184-byte stack and substantial spills.
It was rejected before GPU execution; adding per-iteration warp synchronization
did not fix the allocation. These results point to phase/address lifetime and
communication scheduling as the next compiler work, rather than simply deeper
unrolling. The next measurements should time CTA phases separately to distinguish
H/E arithmetic, trail addressing/import, publication and scheduling costs. A
requested-byte model alone has not predicted these runtimes.

## Evidence and reproduction

Raw results are retained under `.cache/cuda-large-domains-2026-11-11/`:

- `complete-controls/n{256,384}-steps512/summary.json`
- `complete-controls/n512-steps512-owned-reference/summary.json`
- `three-bank-large-T8-screen-v1/` — first 384³/512³/640³ sweep
- `paper-layout-small-and512-v1/summary.json` — small gates and two-process screens
- `result-index.json` — selected rows, raw paths, source/binary hashes and resources
- `matched-result-index.json` — decisive strongest-D2 comparison and sanitizers
- `paper-confirmed-root-v1/` — independent spatial confirmations and sanitizers
- `register-phase-metadata-durable-v1/` — reproducible 119-register research path
- `paper/systolic-fdtd.pdf` — paper and rendered figures inspected locally

The analytic models and CPU proofs run without GPU execution:

```sh
python scripts/model_cuda_systolic_traffic.py --sizes 384 512 640 --steps 8 \
  --output .cache/cuda-large-domains-2026-11-11/model-reproduction.json
python scripts/prove_cuda_circular_cache.py
python scripts/prove_cuda_circular_geometry.py
python scripts/prove_cuda_moving_inplace.py
python scripts/prove_cuda_moving_cpml_inplace.py
```

The model counts requested scalar payload, not measured DRAM traffic. NVIDIA
profiling counters require unavailable administrator permission on this machine;
the work does not change that policy. The installed production CUDA component
remains unchanged; experimental artifacts and losing D4 candidates stay separate.
