# CUDA performance review — 2026-09-17

This is the initial source review. Subsequent local builds, profiling, implementation, and measurements are documented in [the RTX3090 optimization report](cuda-optimization-2026-09-17.md).

Reviewed merge commit `b0a5ff32`, incorporating `origin/main` at `14f8c47d`.

The likely explanation for the throughput spread is a combination of extra memory traffic, substantial absorber work, and abrupt changes in kernel selection. The implementation does not provide the same optimized execution path for arbitrary simulations. The reported 13 H100 GCUPS and 9 versus 5 RTX 3090 GCUPS are user observations; their raw benchmark records were not supplied, so this review cannot attribute their exact differences to individual kernels.

Evidence comprises current source inspection, a CPU scheduling probe, 40 passing targeted tests, and the repository's historical GPU experiment logs. No new GPU timing or hardware-counter measurement was collected. This checkout's virtual environment has CPU-only JAX and no CUDA extension. Another environment has GPU JAX, but its extension comes from a different, modified checkout; it was not reused. No `nvcc` or `ncu` executable was found in the searched locations. The available 3090 was also serving a notebook using approximately 9.3 GiB and showed concurrent activity. No H100 is locally available.

## Confirmed implementation limits

### 1. Material representation controls algorithm selection

`beamz/simulation/compile.py:148–191` packs coefficients only when each electric source coefficient is a rank-3 array with at most 256 distinct FP32 values, and all three decay coefficients are scalar unity. Packing is exact, not approximate. If any component fails, all three remain unpacked.

The combined CPML core/shell queue additionally requires an isotropic uniform grid, equal positive low/high absorber thickness for every H/E term, scalar H coefficients, and a nonempty interior (`beamz/simulation/cuda/runtime.py:530`). Conductivity, graded grids, asymmetric absorbers, or high material cardinality can therefore select the generic CPML kernel. Smooth geometry and subpixel material mixtures can increase coefficient cardinality; actual failing applications need their compiled coefficient counts inspected.

A CPU probe of the real packing and schedule-selection functions, using an otherwise eligible CPML context, returned:

| E coefficient representation | Packed | Program layout | Schedule flags |
| --- | --- | ---: | ---: |
| Scalar | No | 2, CPML in place | 133 |
| 256 unique values | Yes | 4, source/temporal CPML | 191 |
| 257 unique values | No | 2, CPML in place | 133 |

These synthetic arrays test dispatch only, not numerical performance. The scalar case is an unnecessary eligibility restriction: even a homogeneous CPML problem misses the combined queue because uniform-grid elision turns its coefficients into scalars before packing.

There is a second interaction: packing is attempted for all streamed 3D simulations, while `_temporal_yee_supported` accepts only rank-0/rank-3 coefficients. Packed low-cardinality heterogeneous PEC coefficients are rank 1, so they exclude full-step fusion. The historical heterogeneous-PEC fused timings should not be assumed to describe this revision's dispatch.

### 2. CPML does not actually fuse H and E

`LaunchTemporalCpmlProgram` in `cuda/src/program.cu` alternates field banks but still calls separate H and E phase kernels, with sources between them as required. The combined queue in `cuda/src/update.cu` combines shell and core blocks within each phase; it does not combine the two phases or multiple timesteps.

The full-step H→E shared-memory kernel is used by the eligible source-free, monitor-free PEC program. Adding a source or monitor selects a different schedule family and loses this fusion. Thus a small source can have a larger cost than its own injection kernel. The current CPML double-bank path pays for additional field storage without obtaining full-step temporal reuse. Whether removing that bank improves runtime requires an A/B measurement.

The combined queue's interior blocks avoid recurrence instructions, but share one compiled kernel with shell blocks. Register allocation can consequently constrain both. This is a profiling hypothesis, not a measured occupancy diagnosis.

### 3. Absorbers can occupy much of a realistic device domain

For a box with side lengths z, y, x and thickness p on every face, the approximate fraction touching CPML is:

`1 - (1 - 2p/z)(1 - 2p/y)(1 - 2p/x)`.

| Grid | p | CPML shell fraction |
| --- | ---: | ---: |
| 64 × 96 × 128 | 8 | 45.3% |
| 128 × 256 × 384 | 10 | 26.3% |
| 32 × 256 × 384 | 10 | 67.2% |

Thin photonic domains can spend most of their updates in the absorber. Those cells update recurrence state in addition to the fields. Shape also changes partially filled tiles, parallelism and monitor-area-to-volume ratios. A single cell count does not characterize the work.

Ignoring Yee edge corrections, FP32 recurrence reads+writes contribute approximately `64p(1/z + 1/y + 1/x)` bytes per material cell per full step: 17.3, 9.2 and 24.2 bytes respectively for these examples. This excludes CPML coefficient traffic and instructions. BF16 halves only recurrence storage traffic, not all field or monitor traffic, and requires application accuracy validation.

### 4. Some observation modes leave native multi-step execution

`beamz/simulation/execute.py:584–607` permits native grouped monitors only if every monitor is DFT-enabled, has positive frequency/point counts, has no recorder, and requests neither power nor frequency accumulation through the other paths. Enabling JAX x64 also disables this packed monitor route. Unsupported source batching similarly prevents native multi-step execution. CUDA phase kernels can still run, but surrounding operations return to the general scan path.

Within the native path, DFT work scales with points × frequencies × components × sampling cadence. `cuda/src/io.cu:482` launches over maximum point/frequency extents and masks inactive work. Ragged accumulator storage is compact, but launch geometry still contains padding. Phase caching already removes much repeated trigonometry; that optimization is present, not an unimplemented opportunity.

### 5. The Hopper experiment is not an integrated H100 fast path

`cuda_hopper` is explicit-only. Native multi-step graph eligibility checks specifically for `cuda_streamed`. `BeamzLaunchHopper` launches three separate component kernels per phase, each using a fixed 32 × 4 × 2 shared-memory tile. That means six update launches per step before source/monitor work, compared with two phase launches in the streamed CPML path.

The current Hopper kernel does not implement multi-step reuse, TMA, or thread-block cluster cooperation. Selecting it is not evidence that execution is tuned for H100, and its throughput cannot be inferred from the streamed path. Both backend and actual executed kernel names must accompany H100 results.

## What the existing measurements establish

The last recorded matrix in `tests/performance/RTX3090.md` is explicitly historical (candidate `6cd65d0`, 64 × 96 × 128, 160 steps):

| Workload | Historical GCUPS |
| --- | ---: |
| Uniform PEC | 11.837 |
| Heterogeneous PEC | 9.845 |
| Heterogeneous CPML | 5.763 |
| CPML + source | 5.192 |
| CPML + source + DFT | 5.094 |

This is strong evidence that boundary/update execution, rather than the small DFT alone, dominated those cases. The CPML+source+DFT result remained 5.100 GCUPS at 519 steps: merely extending the run did not remove its plateau. These values are not a fresh measurement of the merged branch or an explanation of the user's separate 9-GCUPS realistic result.

The recorded full-step fusion improvements are around 16–19% for eligible profiles. Graph reuse alone previously yielded about 2%. Earlier exact two-step designs lost performance because of halos, barriers and live state; see `tests/performance/CUDA_TEMPORAL_BLOCKING.md`. Repeating those designs unchanged is not a credible route to a 2.54× improvement.

## The bandwidth budget for 33 H100 GCUPS

At 13 GCUPS, reaching 33 requires 2.54× throughput, or 60.6% less time per cell. NVIDIA specifies 3.35 TB/s for H100 SXM and 2.00 TB/s for the 80-GB PCIe model. The exact variant matters:

| H100 | Peak-bandwidth byte budget at 33 GCUPS | Budget at an illustrative 80% of peak |
| --- | ---: | ---: |
| SXM | 101.5 bytes/cell/step | 81.2 |
| PCIe 80 GB | 60.6 bytes/cell/step | 48.5 |

Sources: [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [NVIDIA H100 PCIe board specification](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf). These are engineering budgets, not measured effective bandwidth.

In a large-grid streaming model, separately updating H and E needs about 72 field bytes/cell/step even with ideal spatial reuse: each phase reads its old target fields and the opposite fields and writes its targets. Interphase cache residency can reduce DRAM traffic, while redundant transactions can increase it. The source-level stencil issues more loads than this idealized model. Ideal full-step fusion approaches 48 field bytes before halos, coefficients, sources and absorbers.

For the canonical H100 shape, separate phases plus packed E IDs and FP32 recurrence traffic already suggest roughly `72 + 3 + 9.2 = 84.2` bytes/cell/step, excluding other overhead. At 33 GCUPS that is 2.78 TB/s. This is demanding on SXM and above the PCIe streaming budget. Dense E source coefficients replace the approximately 3 ID bytes with about 12 bytes, with additional decay traffic for nonuniform loss.

Consequently, 33 realistic GCUPS on SXM is a plausible optimization target, not established by current evidence. On PCIe, large realistic grids likely require substantial traffic reduction/cache reuse beyond the current separate-phase design. FLOP peak and tensor-core throughput do not resolve this stencil's memory and scheduling costs. Conversely, 13 GCUPS alone does not prove bandwidth saturation: low occupancy, excess instruction work, launch gaps or a slower backend could produce the same result.

## Recommended order of work

1. Add compiled-plan diagnostics: resolved backend, graph eligibility/rejection reason, field banks, fusion mode, material cardinalities/ranks, CPML thickness/shell fraction, and monitor work. Capture the actual 13/9/5-GCUPS cases with these diagnostics and the same timing boundary.
2. Profile those cases on idle hardware. Use a timeline to separate H/E, source, DFT, copies, graph instantiation and gaps; collect DRAM bytes per update, achieved bandwidth, L2 hit rate, register count, occupancy, spills and warp stalls for the dominant kernels. Record exact GPU variant, clocks, power, precision and binary provenance.
3. Decouple combined core/shell scheduling from the 8-bit material representation. Support scalar and dense lossless coefficients, then graded/asymmetric cases where profitable. Make packing compatible with PEC full-step fusion. Benchmark the 256→257 transition explicitly.
4. For H100, prioritize reducing realistic H/E field traffic and tuning core versus shell resource use. Integrate profitable Hopper kernels into the native program schedule. Prototype fusion only with correct source ordering and shell/halo dependencies; retain it only after parity and measured wins. Inspect whether the second CPML field bank is useful before keeping its capacity cost.
5. Optimize monitors when their measured share warrants it: native support for common fallback modes, less padded execution, and application-appropriate cadence. Do not expect a tiny-monitor improvement to turn 13 into 33.

Compare equal physical workloads and distinguish warm executable throughput from public-path time. The H100 harness records both; its “kernel” metric times the whole compiled executable, not a single CUDA kernel. The historical H100 document's 20–40-GCUPS ranges are estimates, not achieved results.

Validation: `tests/unit/test_cuda_runtime_contract.py`, `tests/unit/test_cuda_abi_schema.py`, `tests/performance/test_cuda_material_codebook.py`, `tests/performance/test_h100_workloads.py`, and `tests/performance/test_benchmark_schema.py`: **40 passed**. Hardware parity and new performance measurements remain unperformed.
