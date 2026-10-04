# H100 configuration search and split-interior experiment

**The >30 GCUPS reference target was not reached.** The best existing configuration is original storage order (012), 32x4 queue tile, H/E fusion disabled. Three fresh-process reference results were 22.442, 22.518 and 22.529 GCUPS. No backend default was changed.

## Fixed workload and timing

Same solver as commit 4fd289e; benchmark-only additions followed d303158. One H100 SXM 80 GB, 128x256x512 useful cells, FP32 fields/CPML, exactly 12 CPML cells on all faces, one solved TE mode source and one fixed-aperture mode monitor with three frequencies. Geometry and source come from benchmark_cuda_realistic.build_simulation, as specified in HOPPER_CPML12.md. No timestep-chunking change.

Production-style donation is enabled except in the explicit preservation control. Each invocation receives a fresh copy of the same pristine state, allocated and synchronized outside the timer. Compilation, exact complete-state comparisons, and four warmups are excluded. Eight synchronized samples per candidate use alternating rotated orders. GPU telemetry is sampled between rounds, not continuously; it cannot establish absence of transient throttling. Each validation case runs in a fresh process.

All tested states were finite and exactly equal to the unfused reference for that case, including fields, CPML memories and monitor accumulators. This is numerical parity evidence, not independent optical convergence validation.

## All 18 existing configurations

| Storage order | Queue tile | Fusion off GCUPS | Fusion on GCUPS |
|---|---|---:|---:|
| 012 | 64x4 | 21.797 | 18.771 |
| 012 | 32x8 | 21.935 | 18.882 |
| 012 | 32x4 | 22.442 | 19.092 |
| 120 | 64x4 | 17.309 | 14.898 |
| 120 | 32x8 | 17.677 | 15.177 |
| 120 | 32x4 | 17.984 | 15.276 |
| 201 | 64x4 | 19.889 | 16.998 |
| 201 | 32x8 | 20.091 | 17.158 |
| 201 | 32x4 | 20.611 | 17.381 |

Donation control: 21.797 GCUPS donated versus 21.767 preserved (about 0.14%). Buffer donation does not explain the target gap. Fusion used the existing 32x8x8 interior tile, the best of the preceding study. Other fused tiles were not retuned because no alternate layout made fusion competitive.

## Validation of the leading queue configurations

| Case | 64x4 | 32x8 | 32x4 |
|---|---:|---:|---:|
| repeat2 | 21.861 | 22.003 | 22.518 |
| repeat3 | 21.874 | 22.023 | 22.529 |
| large | 24.207 | 24.331 | 24.832 |
| irregular | 21.750 | 21.930 | 22.455 |
| developed | 21.839 | 21.988 | 22.492 |

Large is 256x512x512; irregular is 97x289x593. Developed starts after 1,024 propagation steps on the reference grid and measures a further 256. All validation files have the same native-binary SHA256 as the original search.

## Kernel prototype

A separate opt-in interior kernel removed queue-region selection and used direct 3D block indexing. It retained the existing per-cell arithmetic and CPML shell kernel. The graph key included the switch to prevent graph reuse across schedules. This prototype was tested on the reference grid, not promoted to production, and restored to the original binary before repeat/shape validation.

In a paired comparison, the best existing tile reached 22.476 GCUPS and its split version 22.690 (+0.95%). Every candidate passed exact state parity. This small benefit does not justify a new default or explain how to reach 30. The experimental source and patch are retained with the raw results.

A separate Nsight Systems trace of the split 64x4 configuration attributed about 52.5% of kernel time to interior H/E updates and 44.5% to CPML boundary queues; sources and monitors accounted for the remainder. Average interior E/H times were 211/192 microseconds per step, and boundary E/H times were 180/163 microseconds. These profiled timings are diagnostic, not the source of reported GCUPS.

## Limitations and remaining work

The provider denied hardware counters with ERR_NVGPUCTRPERM in a tiny CUDA probe before environment setup. Achieved DRAM bandwidth, traffic, cache behavior, occupancy and stalls remain unmeasured. We have not established a globally optimal configuration or completed the counter-guided kernel optimization stage. A profiling-enabled H100 is needed to discriminate the next hypotheses before a larger rewrite.

The original 45-minute guard bounded rental cost; starting account credit was about $3.75. No RTX3090 regression run was purchased because no production kernel or dispatch change was retained.

## Reproduction

```bash
export PYTHONPATH=.
export NUMPY_MADVISE_HUGEPAGE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8
python scripts/search_hopper_cpml12.py --output search.json
python scripts/search_hopper_cpml12.py --configs 012-64x4-f0 012-32x8-f0 012-32x4-f0 --output repeat.json
```

For explicit use of the measured configuration with the public API (no automatic selection):

```bash
export BEAMZ_CUDA_STORAGE_AXES=012
export BEAMZ_CUDA_CPML_SHELL_TILE=32x4
export BEAMZ_CUDA_CPML_CORE_FUSION=0
export BEAMZ_CUDA_CPML_PSI_PRECISION=fp32
export BEAMZ_CUDA_TEMPORAL_STEPS=1
```

Use backend="cuda_streamed" and retain the benchmark workload. This is not a universal recommendation for other shapes, material models or monitor workloads.

Raw artifacts are in ignored benchmarks/results/hopper-search/. All 29 remote files were checksum/byte verified before Pod p5dy97444208dn was deleted; the account pod list was empty afterward. Archive SHA256: `10e765e1b7b7d7bab645946219e4d826ef04f06151b9a90b02807b43346cd6ed`.
