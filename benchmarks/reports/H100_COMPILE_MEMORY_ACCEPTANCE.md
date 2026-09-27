# H100 compilation-memory acceptance

2026-09-27. Follow-up to PR #288 and the RTX 3090 memory-policy fixes
`4ce2335` / `345d68f`. The baseline is `d6cd7f3`; the fixed solver plus
multi-GPU acceptance harness is `3b561cb`.

## Scope and acceptance gates

The budget for this run is $25. Eight-H100 allocations were initially unavailable. The first two-H100 SXM pod
cost $6.98/hour and provided an NV18 link. After its 14 one-GPU and 8 two-GPU
correctness gates passed, an eight-H100 SXM allocation appeared in AP-IN-1.
It costs $27.92/hour and provides 2,013 GB host RAM; the two-GPU pod was deleted
after retrieving its evidence. The eight-GPU pod has a hard 10:46 UTC deadline. One-GPU measurements restrict visibility
to GPU 0. Other users' pods are not part of this experiment.

1. One H100: full-state CUDA/JAX parity, compilation and allocation peaks,
   baseline/fixed throughput, near-capacity continuation.
2. Two H100s: actual x/z partitions, source/monitor and CPML parity against
   unsharded JAX, repeated continuation, propagated 101-frequency spectra,
   and memory measurements on both devices.
3. Eight H100s: reproduce the original 15-billion-cell prepared failure and
   test the fixed version, then attempt the full public modal workload.

The fixed and baseline workers share the same native SM90 extension, JAX 0.9.0,
allocator settings and hardware. Fresh processes prevent compilation/cache and
allocator carry-over. Performance acceptance means less than 5% slower where
both versions fit; capacity-only successes are reported separately.

## Prepared versus public workloads

Public modal runs construct the binary waveguide material, 12-cell CPML, mode
source and two 101-frequency mode monitors using the simulation API. Propagated
spectral comparisons use 2,048 steps on a `(64,96,256)` grid, independently of
short stepping benchmarks. Their existing tolerances are unchanged.

Prepared probes allocate six FP32 Yee fields, three dense electric coefficient
arrays and 12-cell FP32 CPML directly on their destination GPUs. Nonzero pulses
cross actual partition interfaces. They isolate compilation and execution
capacity from full host material construction; they contain no mode source or
mode monitor and do not prove the public workload fits.

The two-GPU z-partitioned `(2500,1250,1200)` case contains 3.75 billion physical
cells. Its largest padded local field has the same dimensions as the original
eight-GPU `(10000,1250,1200)` 15-billion-cell target: 1,251 z planes per GPU.
This is a useful per-rank memory test, not an eight-GPU substitute.

## Evidence

Raw records, commands, source revisions, environment and native extension hash
are retained in [h100-accept25/](h100-accept25/). Source archives are initialized
as temporary Git repositories on the pod: their snapshot commit IDs differ
from the original revisions recorded in `manifest.json`.

## Completed gates

- 14 one-H100 capacity/continuation parity tests passed.
- 8 real two-H100 tests passed, covering both backends, x/z partitions,
  automatic/forced capacity policy, CPML, mode sources and 101-frequency monitors.
- Retrieved propagated spectra passed the existing tolerance comparison for
  one-H100 CUDA and two-H100 x-partitioned JAX against one-H100 JAX.
- The eight-H100 load check showed no thermal/throughput outlier: all eight
  devices completed between 25,371 and 26,412 identical matmuls in 45 seconds.

## Prepared 15-billion-cell result

Both baseline and fixed completed three consecutive 32-step calls and validated
all field/CPML leaves as finite, with nonzero physical fields. Every GPU held
64.60 GiB after preparation; the peak was 65.14 GiB. The executable needed
40.16 MiB scratch per GPU. Host peak was 21.94 GiB.

| Revision | Prepare | Compile | Warm throughput | Final step |
| --- | ---: | ---: | ---: | ---: |
| Baseline `d6cd7f3` | 14.26 s | 0.62 s | 154.09 GCUPS | 96 |
| Fixed `345d68f` solver | 14.93 s | 0.62 s | 153.95 GCUPS | 96 |

Throughput uses the median of the second and third synchronized segments;
first-call overhead is excluded. The measured runtime change is +0.09%, but
this is one process pair, not a full repeated performance acceptance sweep.
Crucially, the baseline also fits this simpler prepared fixture. It does **not**
reproduce the original failure with sources and monitors and cannot establish
that the compilation fix resolves that workload.

Two initial attempts failed inside the harness pulse seed, before solver
compilation. A global scatter gathered a 55.97 GiB field despite output sharding
constraints. Explicit local-coordinate seeding inside `shard_map` fixed it
(`4267525`), and a regression test checks actual local partition sizes. Both
successful revisions used the same corrected harness. The excluded attempts
are retained in `raw/initial-seed-gather/` and `raw/explicit-seed-gather/`.

The full public 15-billion-cell case is still running. Final results and
teardown accounting will be added after evidence retrieval.
