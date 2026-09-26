# H100 modal correctness, completion, and scaling

This follow-up measures a finite-pulse S-bend to completion separately from warm
stepping throughput on a size sweep. The acceptance targets remain 150 GCUPS for
`cuda_streamed` and 75 GCUPS for pure JAX on eight H100 SXMs. A short stepping
benchmark does not establish time to a converged optical spectrum.

## Correctness changes

The original 4,096-step, 128×512×1024 modal comparison exposed two independent
floating-point effects. XLA reassociated the DFT angular-frequency/time product;
rounding barriers on both products restore the native FP32 operation order.
Bounded CUDA graphs also restarted their floating-point observation clock at
each graph chunk. They now retain one invocation origin and an integer elapsed
step count. Continuation still respects the incoming state's time.

The CUDA interface is ABI 21 (`beamz-cuda-component` 0.21.0). Its new elapsed-step
buffer participates in graph-cache identity, so replay cannot bind another
invocation's offset buffer. Rebuild the native component with the Python changes.
The long single-GPU comparison and eight-GPU comparisons pass the existing
pointwise gate: `rtol=3e-5`, `atol=max(1e-12, 2e-6*max(abs(reference array)))`.
Tolerances were not relaxed. Bounded and unbounded CUDA graphs also produce
bitwise-identical DFT arrays in their focused hardware regression.

## Distributed monitor optimization

For supported DFT-only monitors, each rank accumulates its owned interpolation
contributions locally. The scan boundary sums the complex integrals; public
continuation arenas retain their original layout. Prior integrals enter rank
zero exactly once, and normalization weights count timesteps once. Unsupported
monitor combinations retain the prior path.

A 512³ global-domain, eight-GPU, 32-step CUDA profile reduced all-reduce GPU
events from 544 to 32. Median synchronized throughput rose from approximately
88.17 to 89.65 GCUPS, about 1.7%. Event-duration sums are not wall-time shares:
removing monitor reductions left substantial kernel and halo costs. This is a
small measured gain, not evidence of reaching the eight-GPU target.

## Continuation placement

The completion experiment also exposed work outside the timestep kernel.
Distributed continuation converted evolved CPML slabs to NumPy, then eager
placement could assemble differently partitioned public fields on the host.
Continuation now preserves device CPML arrays and uses a cached compiled identity
to reshard multi-device arrays on their existing device set. Host-built setup
arrays retain their original placement path. Hardware coverage explicitly
forbids device-to-host transfers while restoring and preparing continuation
state, in addition to comparing the resulting fields and monitors.

A two-GPU, four-segment CPU profile reduced cumulative state preparation from
9.00 to 3.30 seconds and the last two public calls from 6.08/5.40 to 2.77/2.65
seconds. Total profiled cold-run time increased from 66.78 to 213.28 seconds;
the profile attributed most of that increase to CPU symmetry/material scans
(whose code was unchanged), so these profiled runs are diagnostic rather than
accepted completion-speedup measurements. The final matrix fixes NumPy's huge
page advice setting consistently with the size-sweep harness.

## Complete optical workload

The S-bend uses a uniform 80 nm grid with shape `(z,y,x)=(128,512,1024)`:
67,108,864 cells over 10.24×40.96×81.92 µm. A 480×320 nm silicon core in silica
moves transversely by 2 µm. It has 12-cell CPML, a finite Gaussian mode source,
and two 101-frequency mode monitors around 1550 nm.

The source is zero after step 1,024. Runs advance 16,384 steps (2.398 ps) in
512-step segments. Convergence requires the last three checkpoints to satisfy
both a scaled field-norm ratio below 1e-6 and per-monitor raw complex-DFT relative
L2 change below 1e-4. The field norm is a decay proxy, not a material-weighted
physical energy integral. Checking raw integrals avoids dilution from the
public flux's running sample-count normalization. Final modal flux is evaluated
once after the decay checks.

Wall time includes construction, compilation, all public continuation calls,
checkpoint checks, and final mode decomposition; it excludes writing the final
NPZ. These are single fresh-process completion trials, not latency distributions.
Stepping rates from the separate sweep exclude cold setup and compilation.

## Completed S-bend results

All eight trials converged. Every CUDA/device-count candidate passed the original
pointwise comparison against one-GPU JAX, including raw real/imaginary DFT
integrals, weights, and both modal flux spectra. The largest relative L2 error
was 4.05e-6 for a raw DFT array and 3.08e-6 for flux.

| H100s | CUDA full wall (s) | JAX full wall (s) | CUDA warm 512-step call (s) | JAX warm 512-step call (s) |
|---:|---:|---:|---:|---:|
| 1 | 97.73 | 135.88 | 2.050 | 3.171 |
| 2 | 108.31 | 136.37 | 1.862 | 2.775 |
| 4 | 90.19 | 108.78 | 1.300 | 1.816 |
| 8 | 83.69 | 95.17 | 1.010 | 1.396 |

Warm columns are the median public-call latency after the first two segments;
they include state preparation, publication, and result construction. They are
not pure kernel timings. The first call took 28.7–39.7 seconds; donation can
compile a second executable on the next call. On this fixed 67-million-cell
domain, eight GPUs improve full-run wall time by only 1.17× for CUDA and 1.43×
for JAX. CUDA remains faster overall, but setup and repeated public-call overhead
limit the benefit of more GPUs.

The earlier completion trials use different runtime/protocol revisions and are
retained only as diagnostic evidence. They are excluded from this table.

## Evidence and reproduction

Final runtime source: `dbed19a` (native graph-cache fix: `ea5aa4d`). Source hashes and raw artifacts accompany the
compact evidence. Build native SM90 with fast math disabled and FP32 CPML memory.
Use `scripts/benchmark_device_completion.py` for completion and
`scripts/benchmark_modal_scaling.py` for the warm size sweep. The latter includes
12-cell CPML, one mode source, and two 101-frequency mode monitors, but its
256-step timing window may precede pulse arrival at distant ports. Its spectra
are not a replacement for the propagated completion comparison.

The machine has eight H100 SXM 80 GB GPUs, with NV18 links between every pair,
JAX 0.9.0, Python 3.12.3, and a CUDA 12.8 native build. `NCCL_NVLS_ENABLE=0`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, allocator fraction 0.80, and `NUMPY_MADVISE_HUGEPAGE=0` were set.

The 80 nm grid is a performance and backend-parity case. Pulse decay and backend
agreement do not establish spatial discretization accuracy; a separate mesh
refinement study is needed before interpreting the S-bend's transmission as a
converged physical prediction.
