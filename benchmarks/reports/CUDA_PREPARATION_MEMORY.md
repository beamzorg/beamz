# CUDA preparation memory: local RTX3090 investigation

Measured 2026-09-24 on the local RTX3090 (24 GiB, driver 610.43.03),
Python 3.11.15, JAX 0.9.0/CUDA 13, native code compiled for SM86 with CUDA
13.3 and GCC 15. Production baseline: `36ad890`. Fixes: `a0ea8a6` and
`95488bb`, on `dev/h100-preflight`.

## Finding

The large GPU spike was in **mode-source preparation**, before XLA compiled
the time-stepping executable. The launch-power diagnostic reconstructed six
complex fields over the entire simulation volume, created full and masked
states, advanced them through Yee H/E updates on the GPU, and finally sampled
a small source plane. Host intermediates used complex128; the GPU calculation
used complex64. These temporary volumes were much larger than the local
calculation required.

The diagnostic now reuses the existing stagger-aware source crop and halo.
Residual indices and the source center are translated into that coordinate
system. The same reconstruction, phase de-embedding, and Yee-plane power
integration run on the cropped fields. It does not disable power normalization
or change the source, monitor, CPML, precision, or time-stepping schedule.
Unnecessary full-array host transfers when constructing zeros or slicing local
materials were also removed.

A second problem appeared on this 32 GB host: coefficient encoding used
`np.unique(..., return_inverse=True)` over each full-volume coefficient array.
Its sort/inverse arrays exceeded the benchmark's 14 GiB host-memory cgroup
limit at 624³. Encoding now discovers the exact sorted table in bounded blocks,
then writes packed IDs into the final buffer in a second bounded pass. This
only blocks **coefficient preparation**, not simulation time steps.

## Same-machine measurements

All rows below use `cuda_async`, fresh processes, FP32 fields and CPML, a
uniform 80 nm grid, 12-cell CPML on all faces, one solved TE mode source,
and one mode monitor with three frequencies. Queue tile 32×4, layout 012,
fusion disabled, 256 steps, three warmups and three timed samples. This isolates
memory behavior; it is not a search for the fastest RTX3090 kernel configuration.

| Code / cube | Preparation peak, live GPU GiB | Execution live + scratch GiB | Peak observed GPU use during timing, GiB | Host peak RSS GiB | Setup s | XLA s | GCUPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, 256³ | 3.119 | 1.666 | 2.523 | 10.662 | 9.869 | 0.349 | 8.476 |
| Local source diagnostic, 256³ | 1.353 | 1.666 | 2.539 | 2.545 | 5.743 | 0.391 | 8.536 |
| Local source diagnostic, 512³ | 10.787 | 12.725 | 13.654 | 10.145 | 12.538 | 0.383 | 8.283 |
| Local source diagnostic, 600³ | 17.350 | 20.336 | 21.338 | 14.267 | 18.118 | 0.443 | 8.405 |
| Both fixes, 620³ | 19.142 | 22.409 | 23.387 | 10.952 | 13.186 | 0.427 | 8.445 |

At 256³ the source diagnostic itself falls from 4.469 s to 0.150 s.
It takes 0.154 s at 620³: with a fixed aperture its cost no longer scales
with simulation volume. XLA compilation does not increase the preparation
GPU peak in these cases.

The completed 620³ case contains **238,328,000 cells** and reaches **97.44% of
physical VRAM use during execution**, including the desktop, driver/context,
and allocator overhead. BeamZ's peak live allocation is 22.409 GiB (93.37%
of the 24 GiB card). This is execution occupancy, not a preparation high-water
mark or a reserved BFC pool. The remaining memory cannot all be assigned to
fields: materials, CPML state, execution scratch, and system allocations count.

Host RSS includes shared mappings and can exceed the cgroup’s charged-memory
limit; these are different accounting measures.

Throughput is essentially unchanged at the common size. These fixes improve
capacity and preparation cost; they do not make the CUDA update kernels faster.

## The new limit

With both fixes, 624³ finishes preparation in 13.50 s and XLA compilation in
0.405 s, with 19.515 GiB peak live GPU allocation and 11.102 GiB peak host RSS.
It fails at the **first execution**: the required scratch allocation is
6,294,603,528 bytes (5.862 GiB), while CUDA reports only 6,096,551,936 bytes
(5.678 GiB) free. Preparation is no longer the limiting stage in this test.
620³ completes all iterations and full-state finiteness validation.

The initial 620³ attempt completed all six iterations, then hit the host limit
in benchmark validation. `np.asarray(jax_array)` caches a host copy on the JAX
array; deleting the local NumPy variable did not release it. Validation now
transfers small slices outside timing. The final table uses the successful
rerun with bounded validation. Both interrupted attempts remain in the raw data.

The default BFC allocator still fails at 620³ when allocating the 5.75 GiB
execution scratch buffer, both at memory fractions 0.95 and 0.99. At 0.99,
live allocations plus the requested scratch fit below BFC's configured byte
limit, but its pool cannot satisfy that allocation. This is consistent with
pool fragmentation/reuse constraints; allocator settings still matter near
capacity. The successful 620³ result uses `cuda_async`. No library-wide
allocator default was changed.

## Validation and scope

- 55 source-planning, source/monitor integration, and coefficient-codebook tests pass.
- Local versus dense diagnostics agree for all three axes, both launch directions,
  and source planes near an outer edge and in the interior (12 cases).
- All 30 final state leaves after 256 steps on the 64³ GPU parity case are
  **bit-for-bit identical** between baseline, source fix, and both fixes.
- Codebook tests check exact table/word agreement, values appearing in later
  blocks, final-word padding, and rejection of more than 256 values.
- The local crop applies to isotropic uniform grids. Rectilinear/nonuniform
  launches retain their dense path to preserve metric alignment; their large-grid
  preparation footprint is not fixed by this change.
- The H100 uses the same preparation code, but no post-fix H100 measurement was
  made in this investigation. No claim of >30 GCUPS follows from these results.

## Reproduction and records

Use a CUDA-enabled environment with this checkout's native extension built for
SM86. The runner only uses the local GPU, limits each process to 14 GiB of host
memory with no swap, and records 200 ms `nvidia-smi` telemetry plus stage traces:

```bash
.venv/bin/python scripts/run_cuda_memory_trials.py \
  --output benchmarks/results/local-memory-repeat \
  --allocator cuda_async --sides 64 256 620

PYTHONPATH=. .venv/bin/python scripts/summarize_cuda_preparation.py \
  benchmarks/results/local-memory-repeat --output /tmp/memory-summary.csv
```

The equivalent allocator environment, set before importing JAX, is:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

GPU use near capacity depends on other processes. The measured local desktop
used about 461 MiB before each trial. JAX allocator semantics also differ: a
memory fraction should not be interpreted as a portable hard cap across allocators.

[All trial summaries](cuda-preparation-memory/summary.csv), including failed and
partial trials, are committed. Raw local measurements, state snapshots, stage
traces, telemetry, logs, environment, and the production patch are in
`benchmarks/results/local-rtx3090-memory/`.

Two initial Runpod baselines (384³ and 480³) were downloaded before the user
corrected the machine choice. The pod was deleted immediately; no cloud resources
remain. Those records are kept separately in `benchmarks/results/rtx3090-memory/`
and are **not** mixed into the same-machine table above.

Verified raw archive: `benchmarks/results/local-rtx3090-memory-raw.tar.gz`
(90 files). SHA256: `3792c870f37d3d983b64705baa5d3ec3d82acc2e4da144fe8995e278dbf2dcc0`.
