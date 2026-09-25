# CUDA performance diagnosis and H100 scaling targets

2026-09-25. This analysis uses the completed 53-configuration benchmark and six
GPU traces, plus inspection of the implementation at `2f41186`. The explicit
Hopper backend has now been removed. Supported execution implementations are
`jax` and `cuda_streamed`; `auto`, `cuda`, and spelling aliases select those
implementations rather than adding another execution path. Historical evidence
is retained in [the benchmark report](H100_BACKEND_COMPARISON.md).

## Findings

This is not primarily an H100 capacity problem. The measurements show three
different costs that require different changes: monitor accumulation, the
distributed update implementation, and communication/placement. Removing the
unused backend simplifies maintenance but does not accelerate streamed kernels.

| Workload | Streamed 1 GPU | Streamed 2 GPUs | Streamed 4 GPUs |
|---|---:|---:|---:|
| CPML + DFT, 128×256×384 | 19.581 | 4.937 | 6.541 |
| CPML + DFT, 512³ | 23.626 | 5.653 | 9.008 |
| Physical crossing, 36 nm | 4.484 | unmeasured | unmeasured |
| Physical crossing, 25 nm | 6.044 | unmeasured | unmeasured |

Values are warm aggregate GCUPS. Canonical trials preserve state; crossing
trials donate it. Compare implementations within each workload/protocol.
The canonical single-GPU cases already reach part of the requested 15–35 range;
the physical crossing and distributed execution do not.

### 1. DFT interpolation repeats across frequencies

**Confirmed:** `AccumulateDftGroups` in `cuda/src/io.cu` indexes CUDA threads by
point, frequency, and component. Each frequency thread repeats the same neighbor
indices/weights traversal and field interpolation. It then updates one complex
accumulator. The crossing has 101 frequencies. The JAX observation implementation
in `beamz/simulation/observe.py::_accumulate_dft` expresses the gather once before
the frequency outer product. This is a concrete algorithmic difference, not a
claim about measured DRAM bandwidth.

In the 36 nm streamed diagnostic, `AccumulateDftGroups` alone consumes 40.878 ms
of 54.868 ms summed GPU event duration: **74.5%**. H/E updates take 23.7%.
The default launch also pads every monitor to the largest aperture/frequency
count and launches six component lanes, even where masks disable components.
Those inactive lanes return, but still contribute scheduling overhead.

**First change to prototype:** a single-step gather stage shared by all
frequencies, followed by a contiguous complex-accumulator update; bucket active
monitor/component plans by aperture and frequency count. Preserve neighbor
summation order, windows, normalization, cadence, and component masks.
`GatherDftPair` / `AccumulateDftPair` already implement gather reuse in the temporal
pair path. Reuse that design for the ordinary path before inventing another
backend. Temporal pairing itself changes scheduling and needs its own control.

Using the diagnostic fraction as a rough Amdahl estimate, eliminating DFT cost
entirely would move 4.484 GCUPS to about **17.6 GCUPS** if other costs stayed
fixed. Reaching 15 through DFT optimization alone would require approximately a
17× reduction in that cost. Reaching 35 needs additional update/memory/scheduling
improvements. This is a prioritization estimate, not a performance prediction:
the trace is 32 steps with preservation, whereas crossing timings use 256 steps
with donation. Accumulator read/write traffic remains even after gather reuse.

### 2. Multi-GPU switches to a slower update implementation

**Confirmed:** `execute.py::build_scan` disables native multi-step graphs when
sharding is enabled. Distributed execution calls the phase-by-phase FFI wrapper
in `cuda/sharding.py` and the generic kernel in `cuda/src/sharded.cu`, rather than
running the optimized single-GPU queue kernels independently on local interiors.
Packed material codebooks are also disabled on the distributed path.

`sharded_schedule.h::UpdateThread` loops over three components, reconstructing
coordinates using runtime 64-bit divisions/remainders. `sharded_cell.h` performs
global-to-local checks, metric dispatch, boundary checks, and CPML term checks
inside the general cell update. Those operations are present in the source;
their individual instruction or bandwidth costs have not been measured.

The medium-grid four-GPU trace spends 104.389 ms of summed device event time in
native sharded updates: **40.1%**. Averaged over four devices that is 26.10 ms,
versus 18.43 ms of native update time on one GPU for the same global 32-step
workload. Ideal division of the single-GPU update work would be about 4.61 ms per
device. This exposes a large local-update problem even before communication is
optimized. Concurrent activity affects timings, so it does not isolate which
kernel instruction is responsible.

**Required design:** local owned interiors should use the proven streamed
uniform-grid kernels and local material packing. Handle physical boundaries,
CPML shells, and inter-rank interfaces separately. Specialize the common uniform
scalar/diagonal case at compile time; retain a correct generic streamed fallback
for rectilinear metrics and full tensors. Expose local geometry/ghost faces
without making every interior cell repeat general coordinate validation.

### 3. Communication is more than neighbor halos

**Confirmed from source:** each H/E phase exchanges both faces of all three input
field components (`exchange_halos`), concatenates halos into source buffers, masks
CPML ownership, and sums CPML slabs normal to the partition axis with `psum`.
The normal slabs and monitor state are replicated. Sources, boundary constraints,
and observation also execute through JAX between the native phases.

**Confirmed from the four-GPU trace:** summed GPU event time includes 42.25 ms
AllGather, 34.54 ms SendRecv, 10.80 ms ReduceScatter, and 5.36 ms AllReduce, about
**35.7%** in total. These are actual communication-kernel categories, not proof
that NVLink bandwidth is saturated. Their durations include waits, and sums across
devices/streams are not wall-time fractions. The existing trace does not map
every gather/reduction back to its source operation; an HLO dump is required
before attributing the AllGathers solely to monitors or CPML.

**Required design:** exchange only the tangential field faces needed for the
next curl; keep ghost buffers separate from owned interiors; batch neighboring
messages; compute interiors while interfaces are in flight. Give physical CPML
state a local owner instead of globally reducing it every phase. Accumulate DFT
points locally and assemble/reduce them when outputs are requested, preserving
the public continuation-state contract. Crossing interpolation at a partition
interface still needs neighbor support. Full-tensor constitutive coupling needs
separate treatment and must not silently use the scalar fast path.

Capture/reuse the resulting device/communication schedule only after ownership
and stream dependencies are correct. Restoring CUDA graphs alone cannot remove
the slower local kernel or unnecessary data movement.

### 4. Long-run clocks are numerically inconsistent

**Confirmed from source:** JAX's step updates `t = t + dt` in FP32. Native DFT
phase preparation evaluates an origin plus `(step_offset + 1) * dt`; native
chunk boundaries also reconstruct time from the run origin. Both feed DFT phases
and window decisions. Neither formula should be silently replaced without
defining continuation and sampling semantics.

A scalar FP32 reproduction using the actual 36 nm crossing's timestep gives:

| Quantity | Value |
|---|---:|
| Steps | 14,570 |
| FP32 dt | 6.863669437257166e-17 s |
| Repeated-addition final time | 9.999503395446974e-13 s |
| Indexed final time | 1.0000366420376272e-12 s |
| Difference | -8.630249292984615e-17 s |
| Phase difference at 1.31 µm | -0.1240945 rad |

This is a strong candidate for the measured 0.003459 dB transmission / 0.015949 dB
crosstalk difference. It is **not yet causal proof**: isolate only the clock/DFT
phase policy and repeat the full crossing before changing tolerances. CUDA
`sincosf`, arithmetic contraction, and reduction ordering can also differ. Do not
declare JAX or streamed physically correct merely because it is the comparison
reference. Prefer a documented integer-step/origin clock contract that is stable
across chunking, then validate both implementations against it.

### 5. Preparation costs are a separate regression surface

At 512³, sharded streamed XLA compilation takes 107–126 seconds, versus 4.72
seconds for one-GPU streamed and 20–22 seconds for multi-GPU JAX. Four-GPU streamed
public-path throughput is 6.206 GCUPS versus its 9.008 warm rate. This cannot
explain away the warm scaling failure, but materially affects real short runs.
No compiler-pass dump was collected, so attributing the compile time to constant
folding, replicated masks, or layout conversion is still a hypothesis. Collect
HLO/pass timings and tensor placement before changing preparation policies.

## Quantitative acceptance targets

Proposed interpretation of “close to linear”: at least **80% parallel efficiency**
on eligible large domains, with 90% as the stretch target. This is a proposed
engineering gate, not a measurement or an assertion that all small domains scale.
For fixed-domain strong scaling, use `efficiency = throughput_N / (N * throughput_1)`
against the fastest validated single-GPU streamed path.

For the existing 512³ baseline of 23.626 GCUPS:

| GPUs | 80% efficiency throughput gate | Maximum 256-step execution time |
|---|---:|---:|
| 1 | retain at least the baseline | 1.454 s |
| 2 | 37.802 GCUPS | 0.909 s |
| 4 | 75.603 GCUPS | 0.454 s |
| 8 | 151.206 GCUPS | 0.227 s |

The current four-GPU result of 9.008 needs about an **8.4× improvement** to meet
that four-GPU gate. For the medium domain, the eight-GPU 80% gate allows only
about 20 µs of extra overhead per timestep beyond ideal divided single-GPU work.
That illustrates why phase-level orchestration/collectives must be redesigned.

Measure strong and weak scaling separately. In weak scaling hold the local grid
size approximately fixed while increasing the global domain, explicitly reporting
how CPML surface fraction and monitor placement change. Also measure independent
simulations per GPU as a separate parameter-sweep throughput metric; that does
not establish scaling of one coupled simulation. Eight-GPU capacity was unavailable
in the initial run, so none of these eight-GPU gates has been measured.

## Implementation and validation order

1. **Backend removal (implemented):** remove the kernel/build target/FFI symbol,
   Python dispatch and aliases, active benchmark options, and obsolete tests.
   Reject old Hopper requests; do not silently alias them to a numerically
   different implementation. Filter obsolete registrations from older extensions.
2. **Clock contract:** isolate the long-run difference; gate full spectra and
   chunked continuation before numerical or scheduling optimizations.
3. **Single-GPU DFT:** shared gather and active-plan launches; compare 3/101
   frequencies, heterogeneous apertures, field/flux/mode monitors, windows, odd
   continuation splits, and donated/preserved ownership. Keep the current path
   as an experimental control until measured parity and speed pass.
4. **Local distributed kernels:** compare current sharded update against the
   optimized local-interior path without communication first, then add interfaces,
   all partition axes, uneven shapes, CPML ownership, and supported materials.
5. **Communication schedule:** map collectives with GPU HLO and timeline evidence;
   remove global assembly from the step loop, overlap interiors with face exchange,
   and retain exact Yee dependencies. Validate device races with Compute Sanitizer.
6. **Hardware gates:** paired/interleaved repetitions on 1/2/4/8 H100s of the same
   node type, at least five samples plus fresh-process repeats. Report warm rate,
   public latency, compilation, memory, raw variability, complete-state parity,
   long-run spectra, and continuation. Keep FP32/physical duration/monitor fidelity
   fixed. No gain from fewer frequencies or less CPML counts as an equivalent run.

Only removal is implemented in this change. No additional GPU rental was needed
for this analysis, and no unmeasured speedup is claimed. Local validation passed
76 focused tests, including compiling the production C++ FFI decoder with the
portable sharded cell implementation and executing 2/4-device CPU collectives.
That validates the removal and contracts, not CUDA device execution or H100
performance after removal; this Mac has no CUDA compiler or H100.
