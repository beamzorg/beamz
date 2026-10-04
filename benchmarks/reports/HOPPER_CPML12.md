# H100 CPML12 mode-source / mode-monitor study

The 30 GCUPS target was **not reached** in this study. Keep the existing SM90
fusion guard: forcing the current GA102 fused kernels regresses H100 performance.

## Workload and method

Solver revision: `4fd289e7e438522b5b41d297b717f36b2bd325f2`.
One H100 SXM 80 GB, CUDA component 0.19.0, JAX/jaxlib 0.9.0, FP32 fields and CPML.
Use `scripts/benchmark_hopper_cpml12.py`, derived from the existing paired storage
study and `benchmark_cuda_realistic.build_simulation`.

The lossless silicon waveguide uses an 80 nm uniform grid, silicon index 3.48,
cladding index 1.44, a solved TE mode source centered at 1.55 micrometers, and one
mode monitor with a 1.6 by 0.8 micrometer aperture. Its three frequencies are
0.97, 1.00, and 1.03 times the source frequency. CPML is exactly 12 cells on all
faces. Geometry, source, monitor, and sampling stay identical between variants.
This is a performance workload, not an optical convergence claim.

Each executable advances 256 steps. Complete-state comparisons include every
JAX tree leaf: all fields, CPML state, monitor accumulators, and clocks. All tested
variants matched their case's unfused reference exactly; reference leaves were
finite. The developed case starts after 1,024 steps of propagation. Tests do not
establish independent physical correctness or full optical convergence.

Timings exclude setup, compilation, parity comparisons, and warmups. Three
additional warmups follow each candidate's parity run. Variants are interleaved
in forward and reverse rotated orders (eight samples each for fusion, six for
shell tiles), and GCUPS uses useful material cells and median synchronized time.
The executables preserve input state (`donate_state=False`), with the same policy
for every candidate; storage conversion is included. This differs from the
previous cosine benchmark's donation policy. No timestep chunking was changed.

## Unprofiled results

| Grid z × y × x | Unfused 64×4 | Fused 32×8×8 | Fused 64×4×8 | Fused 32×4×8 | Unfused 32×4 |
|---|---:|---:|---:|---:|---:|
| 128×256×512 | 21.74 | 18.73 | 18.37 | 17.58 | 22.40 |
| 256×512×512 | 24.16 | 20.57 | 20.07 | 18.94 | 24.78 |
| 97×289×593 | 21.67 | 18.98 | 18.91 | 18.09 | 22.37 |
| 128×256×512, after 1,024 steps | 21.77 | 18.74 | 18.37 | 17.59 | Not tested |

All entries are GCUPS. Shell tiles were tested in a separate paired experiment:
its 64×4 controls were 21.76, 24.17, and 21.67 GCUPS. Thus 32×4 gains about
2.5–3.3%, not enough to close the target gap. No backend default was changed.

At the reference size, 30 GCUPS requires 256 steps in 143.17 ms, or 559.24
microseconds per complete timestep. The best tested reference configuration is
22.40 GCUPS; the larger-grid 24.78 result is not a substitute for this target.

## Kernel timelines

Nsight Systems 2025.3.2 captured one warmed 256-step call using cudaProfilerApi
and CUDA graph node tracing. These profiled times are diagnostic only; reported
GCUPS above comes from separate runs without the profiler.

For the reference unfused path, the combined E and H update kernels account for
about 97.1% of kernel time (392 and 364 microseconds per timestep, respectively).
They include both interior and boundary work. Source kernels account for about
1.5%; DFT accumulation and phase preparation together account for about 1.4%.
Do not label the entire combined update time as CPML overhead.

For the fused 32×8×8 path, the interior kernel takes about 536 microseconds per
step, with E/H boundary queues adding about 181 and 163 microseconds. Fusion
reduces the queues' work but adds more interior time than it saves.

Nsight Compute returned `ERR_NVGPUCTRPERM`: the provider restricts hardware
counters. We therefore did not measure achieved DRAM bandwidth, occupancy,
cache efficiency, register spills, or hardware stalls. The timeline identifies
where time goes, but does not establish whether those kernels saturate bandwidth.

## Next engineering target

Optimize the combined update kernels, not source or monitor launches. Retain
SM90's unfused default. The small 32×4 gain is available for explicit experiments
through `BEAMZ_CUDA_CPML_SHELL_TILE=32x4`; it is not a validated universal default.
Before a major rewrite, obtain hardware counters on a profiling-enabled H100 to
distinguish excess global-memory traffic from indexing/instruction overhead,
occupancy limits, and cache effects. The existing fused-core implementation is
not evidence that fusion in general cannot help Hopper.

## Reproduction and artifacts

```bash
export PYTHONPATH=.
export NUMPY_MADVISE_HUGEPAGE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8
python scripts/benchmark_hopper_cpml12.py --shape 128 256 512 \
  --study fusion --output fusion.json
python scripts/benchmark_hopper_cpml12.py --shape 128 256 512 \
  --study shells --output shells.json
```

The ignored `benchmarks/results/hopper-cpml12/` directory contains all raw timing
samples, state hashes, exact executed harnesses, hardware/environment records,
Nsight reports and SQLite traces, profiler failure logs, and cleanup receipt.
The raw archive SHA256 is
`49aebc36b42a2e223a768e5f35704134f8201b7a5cf9aa047a357d0dc2a546d1`.
All 30 downloaded remote files were byte-verified against that archive before
Pod `5g1u9cae2pevqu` was deleted. No paid resource remains from this study.
