# RTX 3090: bounded compilation without an unconditional throughput penalty

2026-09-27. Follow-up to [the initial capacity fix](RTX3090_COMPILE_MEMORY.md)
and PR #288. Baseline is `d6cd7f3`; the initial fix was committed and pushed as
`4ce2335` before this follow-up. Hardware remains one RTX 3090 (24 GiB), driver
610.43.03, JAX/jaxlib 0.9.0, Python 3.11.15, native ABI 21 built for SM86.
No CUDA kernels or arithmetic precision were changed.

## Answers to the three follow-up questions

1. **Both backends now avoid empirical XLA GPU autotuning for ordinary simulation
   compilation.** This removes the observed compilation allocations and reduces
   compilation time. CUDA additionally switches to existing in-place kernels
   under memory pressure. Pure JAX still has substantial execution workspace;
   it does not reach CUDA's state-only cell capacity.
2. **The measured steady simulation penalty is below 5%.** In 48 fresh-process
   modal runs, the worst paired slowdown was **1.34% rounded upward for JAX** and
   **0.49% rounded upward for CUDA**. Donation no longer forces the slower CUDA
   schedule when the fast schedule fits. These measurements are not a universal
   performance guarantee: capacity mode can still trade throughput for memory
   when the fast schedule would not fit.
3. **The solution is general at the scheduling/compilation layer.** It uses the
   actual grid, CPML layout, device count, and allocator headroom; it contains no
   RTX-specific dimension thresholds or benchmark-workload detection. It reuses
   existing numerical kernels. The memory estimate is conservative, not an
   allocator reservation or a proof that every source/monitor configuration fits.

## Implementation

- Both backends use the per-executable option `xla_gpu_autotune_level=0`.
  XLA still optimizes and compiles the program, using default emitters instead
  of benchmarking alternative emitters with full-sized trial buffers. No
  process-wide XLA flags are changed.
- CUDA scheduling is independent of donation. `BEAMZ_CUDA_MEMORY_POLICY=auto`
  retains the fast temporal/layout schedule when estimated extra workspace plus
  64 MiB fits available allocator capacity. Otherwise it omits temporal banks,
  automatic x-shard rotation, and optional native storage permutations. The
  estimate accounts for field shapes, CPML slabs, temporal pairs, rotated
  materials, and retained input ownership. The choice is made when building the
  executable, with no timestep-time device query or benchmark.
- Explicit `speed` and `capacity` overrides are available. The policy participates
  in program-cache identity and does not disable the existing layout selector.
  If GPU allocator statistics are unavailable, `auto` retains the fast schedule;
  callers can select `capacity` explicitly.
- Donation continues to mean ownership transfer only. Non-donating calls preserve
  inputs even with the capacity schedule. Near-capacity runs need donation to
  avoid retaining another state.
- JAX rejects compiler options on nested jits. A small dispatch wrapper applies
  options when the scan owns compilation and traces the ordinary scan body under
  `jit`, `grad`, or another transformation. The enclosing executable then owns
  its compiler policy. Nested differentiation remains functional, but this fix
  does not automatically set options on an externally supplied outer jit.
- Older JAX versions without per-jit compiler options retain their existing
  compilation behavior. The allocation mitigation is verified on JAX 0.9.0.
- Initial-state construction previously copied `Hx` and `Ez` a second time just
  to obtain their dtypes. It now reads dtype metadata directly, eliminating two
  unnecessary full-field copies for both backends.

## Repeated performance measurements

The public modal workload has CPML, a mode source, two DFT mode monitors, and
three frequencies. Each process runs nine synchronized 128-step segments. The
first segment is excluded; the median of the eight warm segments is the process
measurement. Each baseline/fixed pair is repeated three times, reversing order
on the middle repetition. GPU runs do not overlap. Preallocation is off and
the native layout tuner is off in both versions, isolating the compiler/schedule
changes. All CUDA `auto` cases below retained the fast schedule.

`x` uses the actual sharded implementation on a forced **one-rank mesh**. It tests
local sharded lowering, not multi-GPU communication or scaling.

| Backend | Grid / path | Baseline warm 128 steps | Fixed warm 128 steps | Median paired change | Compile baseline → fixed |
| --- | --- | ---: | ---: | ---: | ---: |
| JAX | 128³ / native | 0.1254 s | 0.1254 s | +0.05% | 2.951 → 1.463 s |
| JAX | 256³ / native | 0.9362 s | 0.9418 s | +0.60% | 3.308 → 1.532 s |
| JAX | 128³ / x | 0.1399 s | 0.1403 s | +0.27% | 2.280 → 1.307 s |
| JAX | 256³ / x | 0.9013 s | 0.9133 s | +1.33% | 2.376 → 1.301 s |
| CUDA | 128³ / native | 0.0388 s | 0.0388 s | −0.04% | 0.067 → 0.068 s |
| CUDA | 256³ / native | 0.2440 s | 0.2439 s | +0.00% | 0.068 → 0.068 s |
| CUDA | 128³ / x | 0.0870 s | 0.0870 s | −0.00% | 2.037 → 1.051 s |
| CUDA | 256³ / x | 0.4928 s | 0.4932 s | +0.07% | 1.909 → 1.037 s |

Times are medians across the three processes. Percentages use the three paired
ratios before rounding, rather than ratios of displayed times. At 256³ native
JAX, compilation's additional allocator peak fell from **310.83 MiB to zero**;
for the 256³ JAX x path it fell from **520.31 MiB to zero**. CUDA's x path similarly
lost its compile-time allocation spike. Its already-fast native compilation
remains approximately 67 ms. Required *execution* workspace is unchanged in these
fast-schedule comparisons.

The [48 raw records](rtx3090-memory-policy/performance/) contain all timings,
memory snapshots, environment, extension hash, solver hash, and harness hash.
[The summary](rtx3090-memory-policy/performance-summary.json) retains every paired
ratio. Changes during measurement were limited to transformation compatibility,
capacity-only layout handling, and benchmark support; the measured fast kernels
and shapes stayed unchanged. Each record captures the exact working-tree source
hash rather than relying only on the parent commit ID.

## Capacity demonstration

Prepared fixtures load six FP32 Yee fields, three dense electric material arrays,
and twelve-cell FP32 CPML with a localized electric pulse. CUDA stores electric
update coefficients; JAX stores permittivity and computes its update scales.
Neither fixture benefits from homogeneous-material scalar compression. Each
successful case executes three donating 32-step continuations, checks the final
step counter, and checks finite, nonzero field/CPML state using bounded slabs.

| Backend / path | Cells | Requested inputs | Execution scratch | Compile | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| CUDA / native, auto | 868³ = 653,972,032 | 22.7775 GiB | 512 bytes | 0.0244 s | 96 steps; capacity selected automatically |
| CUDA / x, auto | 800³ = 512,000,000 | 17.8993 GiB | 19.64 MiB | 0.1813 s | 96 steps; original PR failed during compilation |
| JAX / native | 701³ = 344,472,101 | 12.1048 GiB | 10.6895 GiB | 0.6114 s | 96 steps; finite field/CPML state |

Native CUDA loads in **2.68 s**, lowers in **29 ms**, and compiles in **24 ms**.
Inputs plus execution workspace occupy **99.82%** of the measured allocator
budget of 24,502,263,547 bytes. Inputs alone are **94.91% of physical VRAM**.
For this dense CUDA fixture, 869³ exceeds that budget even without workspace,
as derived in the initial report. The original native CUDA schedule needed
another 15.45 GiB and could not run 868³.

JAX loads in **2.31 s**, lowers in **88 ms**, and compiles in **611 ms**. Its
compiler-reported inputs plus output minus aliases plus workspace require
24,475,191,324 bytes: **99.89%** of the same allocator budget. This percentage
includes JAX's workspace; it must not be interpreted as state-only efficiency.
The original JAX implementation also runs this size. Its compilation took
**1.570 s** and temporarily added **3.89 GiB** to the allocator high-water mark;
the fixed compilation adds no new allocator peak. Warm stepping is approximately
5.532 s per 32 steps in both versions in this single large-case comparison.

All native large probes used a preallocated pool with
`XLA_PYTHON_CLIENT_MEM_FRACTION=.97`. The x probe used `.90` to leave NCCL headroom.
Allocator counters at near-capacity can include an unsplit tail of the BFC pool;
the percentages above use requested buffer sizes and executable memory analysis,
not inflated allocator live-byte readings. Process peak host RSS stayed below
2.3 GiB for these prepared fixtures.

Raw large-case records and compact IR:
[CUDA 868³](rtx3090-memory-policy/cuda-868-auto.json),
[CUDA 800³ x](rtx3090-memory-policy/cuda-800-x-auto.json),
[JAX 701³](rtx3090-memory-policy/jax-701.json),
[original JAX 701³](rtx3090-memory-policy/jax-701-baseline.json).

## Correctness and scope

- 101 CPU tests passed across memory policy, CUDA runtime contracts, layout
  tuning, backend selection, and initial-state placement. These include actual
  nested JIT and gradient calculations, not only mocked option inspection.
- 14 GPU tests passed: binary/smooth materials, native/x/z paths, auto/capacity
  scheduling, storage permutations, temporal pairs, padding, mode-source and
  modal-monitor state, donation, and 17+31-step continuation versus 48 JAX steps.
  The two prepared-fixture tests also check non-donated input preservation.
- The sharp-pulse prepared fixture showed FP32 cancellation differences of up
  to 2.3 ppm of individual CPML-slab peaks at 48 steps. Its test uses a 3 ppm
  dynamic floor for CPML and a separate 2 ppm triplet-scale check for physical
  fields. The twelve modal tests retain their existing tolerances. The measured
  electric-field difference was below 0.4 ppm;
  [2/16/48-step diagnostics](rtx3090-memory-policy/prepared-parity.json) are retained.
- Ruff, targeted Pyright, and whitespace checks pass.

Full public material preparation near 868³ remains a separate host-memory
problem; these are prepared-input capacity demonstrations. Public modal setup
is included in the smaller performance cases. The small initial-state copy fix
does not eliminate the public material compiler's global host intermediates.
Neither 96 steps nor these parity tests establish optical convergence. No actual
multi-GPU or H100 run was performed. An externally jitted/differentiated wrapper,
an older JAX release, other GPU architectures, huge monitor buffers, or changing
memory pressure after an executable is cached require their own validation.

## Reproduce

Use the same Python environment and ABI-matched CUDA extension in each checkout.
Run GPU cases sequentially on an otherwise idle device.

```bash
PYTHONPATH=. python scripts/benchmark_memory_policy.py \
  --baseline ../beamz-capacity-baseline --output /tmp/policy-comparison

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.97 BEAMZ_CUDA_AUTOTUNE=off \
python scripts/benchmark_compile_capacity.py --backend cuda_streamed \
  --side 868 --steps 32 --samples 3 --output /tmp/cuda-868.json

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.97 BEAMZ_CUDA_AUTOTUNE=off \
python scripts/benchmark_compile_capacity.py --backend jax \
  --side 701 --steps 32 --samples 3 --output /tmp/jax-701.json

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false BEAMZ_CUDA_AUTOTUNE=off \
python -m pytest -q tests/hardware/test_cuda_capacity.py
```
