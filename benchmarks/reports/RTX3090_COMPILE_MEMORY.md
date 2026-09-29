# RTX 3090 compilation and capacity follow-up to PR #288

**Historical measurements for the initial fix (`4ce2335`).** The
[subsequent memory-policy report](RTX3090_MEMORY_POLICY.md) supersedes the
CUDA-only compiler policy and the unconditional donation/capacity coupling
described here. It covers both backends, repeated throughput comparisons, and
automatic capacity selection.

2026-09-27. Baseline: `d6cd7f3` (PR #288). Local branch:
`fix/rtx3090-compile-memory`. One RTX 3090, 24 GiB, driver 610.43.03,
JAX/jaxlib 0.9.0, Python 3.11.15. Native ABI 21 rebuilt from the PR's
unchanged CUDA sources for SM86 with precise arithmetic. No remote GPU used.

The local compiler OOM is reproduced and fixed. A **653,972,032-cell (868³)**
prepared dense-coefficient simulation now loads in **2.67 s**, lowers and compiles
in **47 ms**, and completes **96 steps** with finite, nonzero fields and active
FP32 CPML memories. Its requested inputs occupy **22.778 GiB**. This is **99.82%**
of the configured allocator budget and **94.91%** of the GPU's physical 24 GiB.
The baseline compiles this native case, but cannot execute it because it asks
for another **15.45 GiB** of workspace.

These are prepared-input execution results, not end-to-end preparation of a
868³ material grid through `Simulation.compile()`. Public modal preparation is
measured separately below. Neither short capacity runs nor these tests establish
optical completion. The original eight-H100 15B case has not been rerun.

## What was wrong

Three independent costs were conflated in the earlier failure:

1. **XLA compilation autotuning allocates real GPU buffers.** The x-partitioned
   CUDA path automatically rotates the field storage, producing large XLA
   transpose fusions around the native calls. XLA benchmarks candidate emitters
   using full-sized input/output buffers. These temporary allocations happen
   during `.compile()`, not execution, and do not appear in the executable's
   `memory_analysis()`. On the original PR, an 800³ prepared case fails in
   compilation with `Autotuning failed for HLO: %input_transpose_fusion.11`,
   attempting another **1.91 GiB** allocation despite the input fitting. The
   failed fusion's shape is `f32[800,801,801]`. This is concrete evidence of a
   compiler allocation, rather than inferring it from the timing of an OOM.
2. **Donation did not eliminate execution workspaces.** Single-GPU native CPML
   selected a second complete field/CPML bank even when the caller donated the
   state. The sharded x path also allocated rotated field buffers. Input/output
   aliasing alone does not remove these internal banks and transposes.
3. **Growing BFC pools can fragment during loading.** With preallocation off,
   the 864³ fixture failed on a 2.41 GiB allocation at 17.70 GiB live, although
   the final 22.47 GiB input fits the allocator budget. Reserving a contiguous
   pool first made the identical input succeed. This is separate from autotuning.

The PR's H100 log did not preserve the failing HLO. The local result establishes
a mechanism consistent with that symptom, not proof that the 15B z-partitioned
failure has no other cause.

## Changes

- `build_scan()` disables empirical XLA GPU autotuning **per CUDA executable**.
  The expensive FDTD kernels are already native CUDA; XLA still compiles and
  optimizes the surrounding operations using its default emitters. Pure-JAX
  compilation and process-wide settings are unchanged. Older JAX versions
  without per-executable options retain their prior behavior; this mitigation
  was verified on JAX 0.9.0.
- Donating CUDA execution chooses the existing in-place schedule and skips the
  automatic x-shard storage rotation. The non-donating schedule remains available
  for throughput-oriented runs. Native kernel code and FP32 precision are unchanged.
- The public donation documentation describes the capacity/throughput tradeoff.
  Regression tests cover scheduling, option scoping, older JAX compatibility,
  full-state CUDA/JAX parity, and executable alias/workspace sizes.
- `scripts/benchmark_compile_capacity.py` records setup, lowering, compilation,
  each continuation, validation, allocator snapshots, exact input sizes,
  executable memory, revision, and source/extension hashes. It writes compact IR
  and preserves failure records.

## Controlled before/after

Fresh Python processes, identical workload and extension, no persistent
compilation cache. Each successful case runs three donating 32-step segments.
Compilation below excludes lowering; scratch is XLA's execution workspace.
The x-path tests use the actual shard-map/native sharded kernels on a deliberately
constructed **one-rank GPU mesh**, isolating their local lowering without requiring
a second GPU. They do **not** validate real multi-GPU communication.

| Case | Original PR | Fixed |
|---|---:|---:|
| Modal 256³, x path: compile | 1.919 s | 0.734 s |
| Modal 256³, x path: compile peak above live inputs | 210.26 MiB | 0.018 MiB* |
| Modal 256³, x path: execution scratch | 435.30 MiB | 2.03 MiB |
| Modal 256³, native: compile | 69 ms | 33 ms |
| Modal 256³, native: execution scratch | 458.54 MiB | 1,552 bytes |
| Prepared 800³, x path | Compiler autotuning OOM (30.16 s) | Compiles in 181 ms; 96 steps pass |
| Prepared 868³, native | Execution OOM: extra 15.45 GiB requested | Compiles in 24 ms; 512-byte scratch; 96 steps pass |

\* Peak counters are cumulative. This tiny excess occurred during lowering;
compilation itself added no allocator allocations in the fixed measurements.

An isolation control keeps **all baseline code**, adding only
`XLA_FLAGS=--xla_gpu_autotune_level=0`: the 256³ modal x case compiles in **1.034 s**,
without compilation GPU allocations, but still needs **435.30 MiB** of execution
scratch. Thus disabling autotuning fixes the compiler spike; avoiding the
rotation separately fixes the execution workspace.

Warm 32-step modal segments changed from 0.1244 to 0.1291 s on the x path
(about 4% slower), and 0.0611 to 0.0695 s on native (about 14% slower).
These short samples document the tradeoff, not a general throughput benchmark.
The non-donating native schedule is preserved.

An additional **840³ (592.7M cells)** one-rank x-path case completes 96 steps,
compiles in 184 ms, and uses 16.17 MiB scratch with 20.68 GiB requested inputs.

## Exact capacity accounting

The prepared fixture uses a homogeneous dielectric stored deliberately as three
**dense** FP32 electric update coefficients, as in the sharded H100 representation.
It has six distinct FP32 Yee fields, 12 packed FP32 CPML recurrences with 12 cells
on both faces, scalar magnetic coefficients, no monitors, and a seeded electric
pulse near the CPML. It is not the modal waveguide benchmark. The fixture constructs
shape metadata and initializes device buffers without first building full host
material volumes. Standard BeamZ `build_scan()` and native kernels do all stepping.

For an unsharded cubic grid of side `n`, ignoring a few scalar bytes:

```text
Fields + three E coefficient grids = 36 n³ + 60 n² + 24 n bytes
12 packed CPML recurrences         = 1152 n (n + 1) bytes
Total                             = 36 n³ + 1212 n² + 1176 n bytes
```

For 868³ the formula gives **24,457,163,808 bytes**; measured input leaves add
44 scalar bytes. The allocator budget is **24,502,263,547 bytes**. The next cube,
869³, needs **24,540,733,800 bytes** before scalars, so **868 is the largest integer
cube allowed by this buffer model and budget**. This is not the maximum for all
material representations: scalar/codebook coefficients have a different budget.
BFC can charge small unsplit tails to live allocations, so allocator live bytes
are slightly larger than requested input bytes and can consume the whole pool.

All three timed continuations retained every coefficient and showed no live
memory growth. Final state validation scans bounded slabs. Coefficients are
released only **after the final timestep**, providing validation scratch. An early
868³ trial stepped successfully but failed while allocating an 11.51 MiB
validation slice; that record is preserved rather than called a successful run.
Host peak RSS for the final 868³ prepared test was about **2.14 GiB**.

## Public preparation and remaining limits

The full public modal workload includes a binary waveguide, mode source, two
three-frequency mode monitors, and 12-cell CPML. On 384³, public preparation plus
placement takes **8.58 s**, lowering **0.476 s**, compilation **0.053 s**, and peak
host RSS reaches **8.98 GiB**. It runs and validates all 96 steps. On 256³, setup
remains about 5.4–5.8 s both before and after this change.

**Global host material construction is still a separate scaling bottleneck.**
This change does not make the full public 868³ modal preparation fit this
machine's available host RAM. The prepared fixture deliberately isolates the
compiler/runtime limit from that unresolved material-setup cost. Multi-GPU
placement, communication scratch, large-domain spectra and the H100 thermal issue
also remain outside this single-GPU evidence. A next H100 trial must reserve NCCL
headroom and repeat numerical parity before retrying 15B.

## Reproduce

Build the native extension from this checkout (ABI 21; SM86 on the 3090). From
the repository root, using a Python environment with BeamZ's dependencies:

```bash
# Prepared near-limit native case on an otherwise idle RTX 3090.
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.97 BEAMZ_CUDA_AUTOTUNE=off \
python scripts/benchmark_compile_capacity.py \
  --side 868 --output /tmp/capacity-native.json

# Reserve more outside-pool memory for NCCL in the one-rank sharded probe.
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 BEAMZ_CUDA_AUTOTUNE=off \
python scripts/benchmark_compile_capacity.py \
  --side 800 --axis x --output /tmp/capacity-x.json

# Actual public setup, mode source and DFT monitors.
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false BEAMZ_CUDA_AUTOTUNE=off \
python scripts/benchmark_compile_capacity.py --workload modal \
  --side 384 --output /tmp/capacity-modal.json
```

For normal use, request `sim.advance(..., backend="cuda_streamed", donate_state=True)`
and do not reuse the donated input. Choose a pool fraction appropriate to other
GPU users and communication allocations. `.97` is a measured single-GPU probe
setting, not a universal default. With `.97`, the one-rank NCCL path failed
outside JAX's pool; `.90` allowed both baseline and fixed controlled comparisons.
No global allocator settings were added to the library.

To compare PR #288, run the same script with `PYTHONPATH` pointing to an unmodified
`d6cd7f3` checkout containing the same native extension. `BEAMZ_CUDA_AUTOTUNE=off`
disables BeamZ's calibration in both versions; it is distinct from XLA's compiler
autotuning, which is left at its default for the baseline comparisons.

## Evidence and validation

Raw stage JSON, compact lowered IR and failure logs are in
[rtx3090-compile-memory](rtx3090-compile-memory/). The main comparisons use the
`*-f90.json` x-path records and the `*-none.json` / `*-native.json` native records.
The failed `.97` NCCL attempts, fragmented load, and first validation OOM are
preserved with their failure stages. Hashes distinguish source changes from the
unchanged base commit identifier during local work.

- 34 runtime-contract tests; 7 sharding/host-placement tests; 75 broader CPU
  regressions (116 distinct CPU tests, excluding repeated runs).
- Six RTX 3090 full-state CUDA/JAX parity cases: binary/smooth materials, native
  and one-rank x/z paths, two mode monitors, seeded fields, all CPML terms, and
  17+31-step donating continuation. Native cases also require field aliasing and
  less than 32 KiB execution scratch; padding/temporal flags are covered.
- Ruff and targeted Pyright results are saved beside the measurement data.

The explanation is consistent with upstream documentation:
[XLA compile-time autotuning](https://openxla.org/xla/persisted_autotuning),
[per-function JAX compiler options](https://docs.jax.dev/en/latest/201/controlling-xla.html),
and [JAX allocator fragmentation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
The root-cause claim above rests on the local failing HLO and controlled toggle,
not documentation alone.
