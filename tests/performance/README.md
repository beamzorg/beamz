# Performance evidence

Performance tests record rasterization, tracing, compilation, warm execution,
memory, result extraction, and monitor overhead separately. Hard regression
gates belong on controlled hardware; shared runners should publish measurements
without making noisy pass/fail claims.

`benchmark_schema.py` defines the portable record and comparison policy. Every
record includes the BeamZ/JAX/JAXLIB/Python versions, resolved backend, processor
or accelerator, device count, field and CPML-state precision, native CUDA component
version and ABI, immutable CUDA flags, grid, timestep count, boundaries, sources,
monitors, compilation time, multiple warm samples, and peak memory. A comparison deliberately
returns an ungated result unless the caller declares that it ran on controlled
hardware. Both the compiled executable and the full public execution path are timed;
the record derives GCUPS from each median using
``prod(grid_dimensions) * timesteps`` cell updates.

`h100_workloads.py` owns the two canonical 3D profiles. `bare_3d` isolates the
field stencil. `realistic_3d` includes heterogeneous material, six-face CPML,
source injection, and a plane DFT monitor. Run either from the repository root:

```console
python scripts/benchmark_h100.py --workload realistic_3d \
  --backend jax \
  --output benchmarks/realistic-h100.json
```

Pass `--devices 4 --shard-axis auto` for a four-device JAX or explicit
`cuda_streamed` record. The runner derives device identity, count, and peak
memory only from devices that own the placed field state, so unrelated visible GPUs
do not change the record.

Use `compare_benchmarks()` for same-backend regression gates and
`compare_backend_speedup()` to compare JAX and streamed CUDA records for the
same physical workload and hardware.

`RTX3090.md` documents the controlled PR-versus-`origin/main` harness. Unlike the
H100 schema records, it deliberately checks out `origin/main` and emits a table,
raw JSON statistics, and a graph for the custom CUDA-kernel decision on one RTX
3090.

`benchmark_rtx3090_capacity.py` holds the physical 3D silicon waveguide fixed and
shrinks its regular-grid resolution until the first GPU out-of-memory result. Each
resolution runs in a fresh process with XLA preallocation disabled. The realistic
curve includes a solved mode source, heterogeneous material, and six-face CPML; a
matched source-free PEC curve exposes the CUDA field-update ceiling. The runner
emits raw JSON/CSV, bootstrap intervals, a static QA plot, and a portable-report
artifact input:

```console
python scripts/benchmark_rtx3090_capacity.py \
  --output-dir benchmarks/results/rtx3090-capacity
```

The deliberate OOM bounds local capacity, so this is an exploratory hardware run,
not a shared CI gate. Use representative points from its measured throughput
plateau for recurring regression tests.

The default `(nz, ny, nx) = (128, 256, 384)` grid and 500 timesteps are part of
the workload identity. Regression records with different features, precision,
backend policy, hardware, device count, shape, or timestep count are deliberately
not comparable.
See `H100.md` for measurement protocol and the pre-harness observations that
motivate the CUDA work.

For the realistic cosine-crossing resolution sweep, see [the cosine-crossing report](../../benchmarks/reports/COSINE_H100.md).

Historical GPU experiment reports and compact result tables live in
[`benchmarks/reports/`](../../benchmarks/reports/README.md); generated run artifacts
belong in the ignored `benchmarks/results/` directory.
