# H100 cube memory-capacity sweep

**Best: 27.317 GCUPS. Largest tested: 748³, 97.2% peak live VRAM, 26.692 GCUPS. The 30 GCUPS target was not reached. Peak memory is dominated by preparation; this is not a near-full active-runtime-memory test.**

## Workload and measurement

Each trial uses one Runpod H100 SXM 80 GB (81,559 MiB reported by NVML), driver 580.126.09,
700 W power limit, CUDA component 0.19.0, JAX 0.9.0. Production solver source is
unchanged from `dev/h100-preflight` at `130106c` (solver changes last at `4fd289e`).
The native binary is compiled for SM90 with the existing release arithmetic.

Each case is a cube at 80 nm spacing with a binary silicon/cladding waveguide,
exactly 12 CPML cells on all six faces, one solved TE mode source at 1550 nm, and
one fixed-aperture mode monitor with three frequencies. Source/monitor construction
is shared with `scripts/benchmark_cuda_realistic.py`. Growing the cube grows the
physical domain; this is not an optical-resolution convergence study.

All fields and CPML memories are FP32. The configuration is `cuda_streamed`,
original storage order `012`, queue tile `32x4`, H/E interior fusion off, padding
off, and temporal steps 1. No timestep-chunking or production kernel change is
made. GCUPS counts material-grid cells, including CPML, times complete H/E steps.

A fresh process builds each size. Three warmups precede seven synchronized timing
samples of 256 steps. Each invocation receives a freshly initialized state, with
allocation outside the timer and donation enabled. Preparation, XLA compilation,
state checks, and result transfer are excluded from GCUPS and recorded separately.
Every leaf of the last timed result is checked for finiteness; its final step
count must be 256.
This is a throughput benchmark, not a complete optical propagation/convergence
validation. The pulse need not reach the distant monitor within these 256 steps;
monitor accumulation remains scheduled at the same cadence in every case.

## Memory definitions

- **Peak live allocation** is JAX's cumulative `peak_bytes_in_use` across preparation
  and execution, divided by the actual NVML device capacity. It includes setup
  transients; it is not the warm solver working set.
- **Pool reservation** is separately reported when the allocator exposes it. It can
  remain high after temporary arrays have been freed and is not counted as live data.
- **Post-execution live allocation** comes from `bytes_in_use` after synchronization.
  The summary also reports this plus XLA's temporary-buffer size as an estimate of
  runtime allocation, not a sampled peak or exact DRAM working-set measurement.
- NVML telemetry is sampled every 200 ms throughout each process. Its GPU and memory
  utilization percentages are busy-time indicators, not measurements of peak
  compute throughput or achieved DRAM bandwidth.

The default BFC allocator uses preallocation off and a 0.99 memory fraction.
A separately labeled `cuda_async` series tests whether a default-allocator capacity
failure can be avoided without changing the solver. It includes paired 712³
(initial pod) and 720³ (finish pod) allocator controls. Within each pod, both
allocators use the same native binary and benchmark semantics.

## Reproduction

```bash
export PYTHONPATH=.
export NUMPY_MADVISE_HUGEPAGE=0
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.99
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8
export BEAMZ_DISABLE_JAX_PERSISTENT_CACHE=1 BEAMZ_RASTER_CACHE=0
python scripts/benchmark_hopper_cube.py --side 712 --output cube-712.json
# Allocator control, in a new process:
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async python scripts/benchmark_hopper_cube.py \
  --side 712 --output cube-712-async.json
python scripts/summarize_hopper_cubes.py benchmarks/results/hopper-cubes --plot
```

Results are retained in ignored `benchmarks/results/hopper-cubes/`. The initial
pod was deleted by its two-hour cost guard during a conversation pause; completed
JSON records already downloaded survived, but remaining remote logs and upper-size
results were lost. The finish pod repeats the control and upper points, with an
independent downloader copying artifacts every 30 seconds. It verifies every
file against a final SHA256 manifest before deleting the pod. Full raw logs,
telemetry, environment, source manifest, and measured harness are retained for
that continuation; `initial-pod-notes.json` identifies the earlier missing artifacts.

## Results

| Allocator | Cube | Cells (M) | Peak live GiB | Peak VRAM | Median GCUPS | Sample range |
|---|---:|---:|---:|---:|---:|---:|
| bfc | 512³ | 134.2 | 25.09 | 31.5% | 25.306 | 25.130–25.532 |
| bfc | 592³ | 207.5 | 38.46 | 48.3% | 26.680 | 26.428–26.684 |
| bfc | 640³ | 262.1 | 48.62 | 61.0% | 26.516 | 26.343–26.519 |
| bfc | 688³ | 325.7 | 60.23 | 75.6% | 26.955 | 26.844–26.956 |
| bfc | 712³ | 360.9 | 66.75 | 83.8% | 27.080 | 26.971–27.083 |
| bfc | 720³ | 373.2 | 69.03 | 86.7% | 27.317 | 27.314–27.319 |
| cuda_async | 712³ | 360.9 | 66.75 | 83.8% | 26.426 | 26.196–26.606 |
| cuda_async | 720³ | 373.2 | 69.02 | 86.7% | 26.704 | 26.397–26.805 |
| cuda_async | 728³ | 385.8 | 71.35 | 89.6% | 26.727 | 25.212–26.813 |
| cuda_async | 740³ | 405.2 | 74.93 | 94.1% | 26.377 | 25.892–26.739 |
| cuda_async | 748³ | 418.5 | 77.39 | 97.2% | 26.692 | 26.162–26.763 |

**Larger cubes did not reach 30 GCUPS.** The best completed result is 27.317 GCUPS at 720³ with BFC. The largest tested cube, 748³ (418.5 million cells), reaches 97.2% peak live allocation and 26.692 GCUPS with `cuda_async`. The default-allocator 512³→720³ increase is about 8%; the `cuda_async` 720³→748³ results are essentially flat. These are observations for this configuration, not a global kernel optimum.

**The 97.2% figure is a preparation peak, not 97.2% active runtime occupancy.** At 748³, peak live allocation is 77.39 GiB, while the largest NVML reading inside timed samples is 39.85 GiB (about 50% of the device). BFC retains its pool and therefore shows about 99% reservation during timing even though much of it is reusable space. This experiment does not cover a 50–100% active-runtime-memory sweep: normal preparation approaches device capacity first. Reducing its full-volume temporaries is needed to test that larger runtime range.

All five finish-pod cases have median sampled GPU busy time of 100%, median memory-controller busy time of 85–86%, and sampled SM clocks fixed at 1,980 MHz throughout timed windows. Hardware bandwidth/occupancy counters were not collected. This supports a size-scaling plateau for the current implementation, but does not prove peak H100 bandwidth saturation.

## Allocator and preparation findings

The initial BFC 728³ attempt reported failure allocating 2.88 GiB during mode-source launch diagnostics; its exception fallback then raised a profile/weight shape mismatch, (360,) versus (442,). No timing from that attempt is used. `cuda_async` successfully runs 728³, 740³, and 748³. That demonstrates an allocator-dependent preparation limit; it is consistent with fragmentation/reservation effects, without isolating their exact mechanism.

The finish pod repeats BFC 720³ at 27.317 GCUPS, matching the earlier 27.308 timing to about 0.03%. On the same finish pod, `cuda_async` 720³ measures 26.704 GCUPS, about 2.2% lower. Its upper points must therefore be interpreted as a separate series. No allocator or kernel default was promoted.

This workload starts from an explicit material grid and does not benchmark rasterization. Finish-pod preparation takes 315–371 seconds per case, versus 1.4–1.7 seconds for XLA trace/lowering/compilation. Periodic stack samples catch full-volume mode-source launch-power diagnostics and `np.unique` coefficient packing. These samples identify expensive preparation work but do not quantitatively divide setup time between stages. The largest case reaches about 208 GiB peak host RSS.

## Verification and resource cleanup

Eleven complete records contain seven timing samples each. GCUPS was independently recomputed from every median; the last timed result in every record has finite complete state and the expected step count. All five continuation processes exited successfully and used one native-binary hash. Field maxima match between the finish-pod 720³ allocator controls; that is a limited consistency check, not a full elementwise allocator parity test.

The initial native hash is `f0b580880260fa3a636120858c5ebbc10a05efef7d5aed843a489ec86355b0a9`; the rebuilt finish-pod hash is `d0ad76b378454ab7387346bfbe845c086b23cba0cbb6693b6ec4087069b3e775`. Both use the same production source snapshot. The new binary and its measured harness are downloaded.

All 31 finish-pod artifacts were verified against the remote SHA256 manifest before pod `srrm8ptzak2lwc` was deleted. The earlier pod `o4jmkfg3k1wvlx` was deleted by its cost guard. The account pod list is empty and current spend is zero. The observed account-balance decrease across both rentals is $9.22, including the first pod’s guard period and the repeated upper trials.

The committed [CSV](hopper-cubes/summary.csv) contains all derived values, record locations, sample ranges, memory estimates, and timed telemetry summaries. Initial and finish runs are identified separately.

Raw archive: `benchmarks/results/hopper-cubes-raw.tar.gz` (42 MiB). SHA256:
`927da2a25db7fbaf1abf809cb9272b525e5780589c591f5f76bb00a91e53cf98`.
