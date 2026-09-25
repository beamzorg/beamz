# GPU benchmark reports

These are historical measurements, not tests or performance guarantees. Each
report records its workload, code revision, hardware, limitations, and commands.
The benchmark tools remain in `scripts/`; automated contracts remain in
`tests/performance/`.

- [H100 backend comparison](H100_BACKEND_COMPARISON.md): streamed/Hopper/JAX
  comparison and 1/2/4-GPU scaling after PR #285, with a Hopper coefficient fix.
- [H100 performance analysis](H100_PERFORMANCE_ANALYSIS.md): removal of the
  experimental backend, measured bottlenecks, and single/multi-GPU targets.
- [CUDA preparation memory](CUDA_PREPARATION_MEMORY.md): the memory diagnosis,
  fixes, and local RTX3090 capacity measurements supporting PR #285.
- [Modal notebook parity](modal-notebook-memory-fixes/README.md): unchanged
  tutorial outputs on the local RTX3090 after the memory fixes.
- [Cosine crossing](COSINE_H100.md): realistic resolution sweep.
- [H100 CPML12 kernels](HOPPER_CPML12.md): unfused/fused kernels and queue tiles.
- [H100 configuration search](HOPPER_SEARCH.md): launch and workload search.
- [H100 cube sweep](HOPPER_CUBE_SWEEP.md): size and allocator measurements
  **before** the preparation memory fixes; peak setup memory is not runtime usage.

Small CSV summaries are versioned beside the reports. Raw telemetry, generated
plots, executed notebooks, and detailed comparison exports belong under the
ignored `benchmarks/results/` directory. Historical raw artifacts are local and
are not included in a fresh checkout. Reproduction commands generate new results.
