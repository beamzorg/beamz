# H100 optimization results — 2026-09-25

The supported execution paths are **`cuda_streamed` and pure `jax`**. Hopper has
been removed. The retained changes bring the tested single-H100 CUDA crossing
and canonical workloads into roughly **15–24 GCUPS**. JAX improves, but some
workloads remain below 15 GCUPS. **Near-linear multi-GPU scaling is not achieved.**
Eight-device performance is unmeasured; no eight-H100 pod was available.

Code is committed on `bench/h100-backend-comparison`, based on origin/main
`2f41186`. The user's original checkout and notebook edits were preserved.
See the [initial diagnosis](H100_PERFORMANCE_ANALYSIS.md),
[59 measured trials](h100-optimization/measurements.csv),
[spectral validation](h100-optimization/spectral_validation.json), and
[artifact/cost manifest](h100-optimization/manifest.json).

## Retained changes and commits

| Commit | Change |
| --- | --- |
| `491c5b9` | Remove the Hopper execution backend. |
| `8b9906c` | Preserve initial measurements and define numerical/performance gates. |
| `b9178e6` | Derive JAX observation time from immutable run-entry time plus integer step. |
| `20bfb44` | Exchange only the two tangential faces required by each CUDA curl phase. |
| `7ddf7b8` | Specialize distributed CUDA arithmetic by phase/component; use validated 32-bit coordinate division. |
| `563121e` | Prevent stale native objects when building transferred source snapshots. |
| `3e2b792` | Omit inactive JAX DFT component gathers while retaining fused accumulator updates. |
| `7e30af1` | Reuse CUDA monitor samples across batches of eight frequencies. |
| `d8e2d6c` | Compute differentiable JAX CPML material scales once per scan. |
| `b93b144` | Build native objects on local scratch storage. |
| `a213759` | Record optimized HLO and NCCL environment settings. |
| `a570e7b` | Traverse the packed CUDA DFT arena for strongly mismatched apertures/frequency counts. |
| `052fa8d` | Compact distributed CUDA post-source boundary masks. |
| `8da4686` | Keep normal CUDA CPML slabs replicated between timesteps. |
| `0000f9a` | Keep unsharded JAX fields in a common scan layout when the layout API is available. |
| `7561a5e` | Extend crossing benchmarks to distributed full-spectrum runs. |

The clock fix eliminates repeated float32-addition drift within a run.
Continuation uses its supplied entry time; arbitrary chunk partitions are not
promised to be bitwise identical. Material-scale precomputation retains
conductivity/permittivity dependence and gradients. Older JAX releases without
the layout API retain automatic layout selection; measurements use JAX 0.9.0.

## Single-H100 measurements

A dedicated H100 SXM 80 GB was used for baseline and candidate trials. CUDA 12.8,
JAX 0.9.0, FP32 CPML and fast math OFF are used throughout. Throughput trials use
fresh processes, warmup, 256 steps, five timed samples and complete-state
synchronization. Canonical trials preserve state; crossing trials donate it.
Compare within a row: compilation/setup are excluded from kernel GCUPS and
recorded separately in the raw results.

| Workload | Backend | Same-device control GCUPS | Retained implementation GCUPS |
| --- | --- | ---: | ---: |
| Cosine crossing, 36 nm | CUDA | 4.475 | **15.288** |
| Cosine crossing, 36 nm | JAX | 6.254 | **7.385** |
| CPML + DFT, 128×256×384 | CUDA | 19.411 | **19.318** |
| CPML + DFT, 128×256×384 | JAX | 16.934 | **17.453** |
| CPML + DFT, 512³ | JAX | 10.100 (material scales, automatic layout) | **11.216** |
| Cosine crossing, 25 nm | JAX | 8.595 (material scales, automatic layout) | **11.093** |

The CUDA canonical difference is approximately −0.5%, within the observed run
variation; it is not evidence of a universal improvement. The strongest measured
change is the 3.42× CUDA crossing improvement. The finer 25 nm CUDA crossing
measures 17.178 GCUPS. That trial used build F, whose unsharded implementation is
identical to the retained code but whose separate distributed dispatch experiment
was rejected. A separate two-GPU node's single-device 512³ CUDA control measures
23.560 GCUPS; the optimized monitor path is not selected for its three-frequency
plan.

The final single-GPU 14,570-step, 1 ps crossing measures **16.654 CUDA GCUPS**
(about 6.985 seconds) and **7.474 JAX GCUPS** (15.563 seconds). Maximum backend
differences are **0.00002051 dB transmission** and **0.00002811 dB crosstalk**;
all fields are finite. Through/cross flux relative L2 differences are
1.97e-6 and 5.33e-6. These are full-run spectral comparisons, not just short
finite-field checks.

A separate 32-step CUDA crossing profile reduces summed DFT GPU event duration
from 40.878 ms initially to 6.030 ms with frequency batching, then **1.779 ms**
with packed traversal. The final profile spends 13.026 ms in H/E updates out of
15.667 ms total GPU event duration. Summed event durations are not wall-time
fractions, especially across devices.

## Two-H100 measurements

Both GPUs are H100 SXM with NV18 topology. Same-node one-device controls and
latest two-device measurements use the same canonical workload and timing
protocol. JAX one-device controls in this table predate the unsharded layout
change; its two-device path is unchanged by that change.

| Workload | Backend | 1 GPU control | 2 GPUs latest | Speedup over control |
| --- | --- | ---: | ---: | ---: |
| CPML + DFT, 128×256×384 | CUDA | 19.734 | **7.637** | 0.39× |
| CPML + DFT, 128×256×384 | JAX | 17.698 | **6.527** | 0.37× |
| CPML + DFT, 512³ | CUDA | 23.560 | **8.436** | 0.36× |
| CPML + DFT, 512³ | JAX | 10.099 | **9.629** | 0.95× |

CUDA improves over the initial run's two-device 4.937/5.653 GCUPS medium/large
results, but that initial comparison used a different node. The same-node
measurements above demonstrate that scaling remains poor. They do not justify
extrapolating to four or eight GPUs.

Boundary-mask compaction reduces CUDA compile time from **8.81 to 3.67 seconds**
on the medium case and **76.08 to 10.52 seconds** on 512³, while runtime stays
similar. Consistent normal-CPML placement further measures 3.10/9.56 seconds
compilation and 7.637/8.436 GCUPS. The latter runtime changes are modest.

The full two-GPU crossing measures **5.334 CUDA GCUPS** and **3.042 JAX GCUPS**.
JAX's spectra match the single-GPU JAX reference exactly. CUDA's maximum
transmission/crosstalk differences from that reference are 0.00000150 and
**0.00000323 dB**. All fields are finite. This strengthens numerical confidence;
it does not solve the performance deficit.

## What still prevents the target

1. **Distributed CUDA uses a substantially different local update kernel.**
   The optimized unsharded tiled/streamed update is not the kernel used inside
   the distributed phase. On one GPU, an isolated no-communication diagnostic
   improves from 2.661 to 3.838 GCUPS after specialization, but remains slow.
   Its native resource use falls from 48 registers/88 stack bytes to 39/24.
   In the latest two-GPU profile, `UpdateSharded` accounts for 63.800 ms of
   110.202 ms summed GPU event duration, approximately 58%. Faster communication
   alone cannot recover the lost local-update performance. The next substantial
   CUDA change should adapt the optimized streamed local update to distributed
   halo/CPML ownership, retaining the existing backend and a general fallback.
   Nsight Compute should distinguish address-arithmetic, occupancy and bandwidth
   limits before another kernel rewrite.
2. **CPML ownership and source placement still communicate beyond the stencil.**
   HLO confirms that normal CPML slabs were partitioned at scan entry, then
   gathered for a native phase requiring replication. Matching placement removes
   those slab all-gathers/reduce-scatters. Collective event time falls from
   16.272 to 13.008 ms in the paired profiles. CPML all-reduces remain, as does a
   source-patch all-gather. The monitor gathers are already sparse in this trace;
   blaming full-field monitor replication would be incorrect. A future ownership
   change should keep recurrences local throughout the scan and assemble the
   public continuation layout at its boundary.
3. **Pure JAX has substantial stencil/CPML communication and intermediate costs.**
   Its two-GPU profile records 84.496 ms in collective events out of 142.631 ms,
   including 80.447 ms in SendRecv. Such events can include waiting, not just
   transfer time. HLO shows CPML slice exchanges of five cells and source-related
   exchanges of 41 cells, in addition to one-cell stencil halos. An explicit
   ownership-aware JAX lowering is the next route to reducing those exchanges.
   Single-device material scales and common layouts help, but do not fuse away
   enough intermediate work to achieve 15 GCUPS on every tested workload.

These are concrete remaining implementation tasks, not claims that near-linear
scaling will follow automatically. Re-run full-state, continuation, gradients,
and full-spectrum gates before accepting such changes, then measure 1/2/4/8
GPU scaling on sufficiently large domains. Keep state ownership, physics,
frequency counts, and timing definitions fixed.

## Rejected experiments and evidence limits

- Separate JAX per-component DFT accumulator updates reduced crossing throughput
  to 5.972 GCUPS; the original fused accumulator structure was retained.
- A checked CUDA interior fast path reduced distributed-local throughput to
  3.279 GCUPS. Separate interior/boundary launches reduced it to 2.890.
- Increasing the CUDA frequency batch from 8 to 128 produced 13.030 crossing
  GCUPS versus 13.199; batch 8 remains.
- Splitting JAX bulk and CPML slab updates passed short numerical/gradient gates
  but reduced crossing/canonical throughput to 4.103/6.551 GCUPS.
- Indexed JAX CPML reconstruction passed numerical gates but measured
  7.309/16.026 GCUPS versus the common-layout control's 7.385/17.453.
- Separate native phase kernels measured 3.829 GCUPS in the isolated diagnostic;
  shared-coordinate component traversal measured 3.751. Neither improves on the
  retained 3.838 result. They were not committed.
- An early rebuild reused stale native objects because transferred timestamps
  predated them. The binary lacked the intended DFT kernel. That trial is a
  control, not evidence against sample reuse. Fresh builds, binary hashes and
  kernel-symbol inspection identify subsequent experiments.
- The first split-JAX upload came from the older checkout and failed on a missing
  context field before numerical testing. The corrected retry was source-hash
  verified. Both logs are preserved. A later verification script stopped on
  archive ownership restoration and was rerun with ownership restoration disabled.
- A baseline 64-step distributed-local diagnostic failed a strict near-zero CPML
  comparison (largest violating absolute error 0.0243; leaf magnitude 6274).
  The 16-step diagnostic passes without relaxing tolerances. It is a short-run
  gate, not a long-run distributed parity guarantee. Full-run spectra provide
  separate evidence, not proof of elementwise agreement of every final field.

## Validation and reproducibility

The final combined shipping code passed **43 CPU tests** across distributed
native contracts, CPML placement, material gradients and compiled integration
([log](h100-optimization/final-cpu-validation.log)). Initial changes also passed
100 focused backend/ABI/runtime/compiled-engine checks.
Production C++ CPU-FFI distributed tests cover two/four CPU devices, all axes,
CPML, tensor and rectilinear media, sources, monitors and continuation. The new
placement regression test checks replicated normal slabs and partitioned
transverse slabs explicitly. CPU evidence is a numerical gate, not GPU scaling.

New JAX tests cover analytical DFT updates, ragged arenas, masked non-finite
fields, and untouched storage. Material-scale tests compare fields and gradients
at conductivities 0, 1e3 and 1e6. The layout candidate passed 30 CPU integration/
gradient tests and four applicable single-H100 reference tests. The packed DFT
candidate passed 16 applicable hardware tests; four multi-GPU cases were skipped
on that one-device pod. Latest distributed ownership passed **22 hardware tests**;
eight four-device cases were skipped on the two-device pod. No tolerance was
relaxed to accept a performance change.

Raw JSON, logs, optimized HLO, traces, source snapshots and native binaries are
preserved in the two checksum-verified archives listed in the manifest. Remote
Git strings such as `20bfb44-dirty` are not exact experiment identifiers: use the
experiment directories, as-built snapshots, scripts and binary hashes. The
committed-source archive identifies shipping code at `7561a5e`; experimental
builds in the evidence include deliberately rejected code. The CSV distinguishes
isolated diagnostics from coupled runs and retains failed-experiment controls.

RunPod setup encountered network-filesystem stale handles during Rust builds;
local scratch storage resolved this. The two-GPU node independently reproduced
NCCL multicast initialization failure (CUDA error 401). `NCCL_NVLS_ENABLE=0`
passed a minimal all-reduce while NCCL reported P2P/direct-pointer transport.
That node-specific setting is recorded, not made a library default. A four-GPU
reservation failed after inventory changed; the last eight-H100 availability
read reported NONE.

Both optimization pods were terminated after archive verification, and follow-up
reads returned 404. Estimated compute cost, including setup, rejected experiments
and the previous initial run, is **$36.93 of the $100 cap**. This is an elapsed-time
estimate, not an invoice. No benchmark pod remains running.
