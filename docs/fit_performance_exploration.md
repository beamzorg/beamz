# FIT interface performance findings and next exploration

Recorded 2026-09-30. Status: preserve the prototype and investigate performance
before extending it into the full photonics backend.

## Decision

The initial coupled dielectric operator improves average interface accuracy,
but its approximately eightfold timestep cost is a material concern. We have
not demonstrated a favourable accuracy-versus-runtime tradeoff. Treat that
tradeoff as the next acceptance gate before milestone 3 integration.

Keep the diagonal FIT path, the coupled prototype and the benchmark as
reproducible comparison tools. Avoid treating the coupled formulation as a
production default until this gate is resolved.

## What was implemented and verified

The work starts from BEAMZ v0.3.1, revision
`65b1adaf432edb67156b4dd98406483d020fdc24`. It adds a separate uniform-grid FIT
engine supporting 2D TE/TM and 3D, PEC boundaries, integrated electric/magnetic
quantities, diagonal materials, conductivity, impressed currents and native
field export. The initial milestone 2 extension adds exact planar dielectric
cell fractions, interface normals and a symmetric positive coupled electric
constitutive operator. Coupled mode currently requires lossless materials.

Validation ran on CUDA-enabled JAX 0.9.0 on an NVIDIA RTX 5070 Laptop GPU:
54 FIT core/interface tests and 177 existing baseline tests passed. The GPU is
available outside the execution sandbox; inside the sandbox CUDA initialization
fails. Explicitly select CUDA when reproducing GPU results so a CPU fallback
cannot be mistaken for GPU execution.

The dependency manifests already contained CUDA-JAX changes before this work.
Those user changes remain separate from the FIT implementation commit. A clean
checkout therefore needs a suitable CUDA-enabled JAX environment to reproduce
the GPU results. Dependency versions and manifest hashes are recorded in the
[GPU baseline](fit_baseline_gpu.json).

## Measured accuracy and runtime

The unit-square PEC TE spectral study compares staircasing, exact-fraction
scalar averaging and a coupled tensor treatment for relative permittivity 2/12.
It covers three interface angles, two translations per angle, three mesh
resolutions and the first three nonzero modes. An independent interface-fitted
P1 FEM reference was checked at 96 and 144 cells per axis; their maximum
relative frequency difference was 0.0157%.

At 28 cells per axis, mean relative frequency error was:

| Material treatment | Mean error |
| --- | ---: |
| Staircase with baseline native averaging | 0.2732% |
| Exact-fraction scalar averaging | 0.3427% |
| Coupled tensor | 0.1549% |

The tensor error was about 2.21 times lower than scalar-fraction error and
1.76 times lower than staircase error at that resolution. Some coarse staircase
cases had lower errors, so these findings do not establish universal superiority
or an accuracy order.

Separate float32 GPU timestep benchmarks measured the median of three
synchronized warm runs of 32 steps, without field recording:

| Mesh | Scalar fraction | Coupled tensor | Slowdown |
| --- | ---: | ---: | ---: |
| 2D TE, 48 × 64 | 0.989 ms | 8.233 ms | 8.33× |
| 3D, 12 × 16 × 20 | 1.832 ms | 14.369 ms | 7.84× |

These are small workloads on one GPU. The spectral and timestep experiments
use different precisions and mesh sizes; their ratios cannot be combined into
an equivalent-accuracy speedup or slowdown. Compilation, geometry preparation,
operator memory and raw case results are recorded in
[fit_interface_convergence.json](fit_interface_convergence.json).

## Why the CUDA extension does not explain this gap

Both sides of the measured eightfold comparison use this checkout's FIT engine
through GPU-enabled JAX. Neither side uses the newer upstream native CUDA
extension. Therefore the absence of that extension does not explain the
measured gap between these two variants.

The newer upstream combines JAX orchestration with direct CUDA Yee/CPML kernels
and CUDA-graph stepping through JAX FFI; see its
[CUDA documentation](https://github.com/beamzorg/beamz/blob/main/cuda/README.md).
Those kernels could affect comparisons with newer BEAMZ, but that is a separate
experiment. Upstream also has newer tensor interface smoothing that should be
included as an accuracy and performance comparator.

The coupled prototype assembles permittivity through cell-corner gather/tensor
application/scatter operations. Each Maxwell step evolves displacement and
uses matrix-free Jacobi-preconditioned conjugate gradient to recover electric
voltages. The scalar path instead uses diagonal division. This additional
iterative work is the leading explanation for the slowdown, inferred from the
implementation. We have not yet profiled its share of runtime or measured
iteration counts. Kernel launch overhead, reductions and scatter traffic may
also matter on these small meshes.

Custom CUDA fusion could reduce overhead, but does not eliminate the iterative
solve. Generic sparse-matrix acceleration is not the immediate question: the
prototype uses JAX stencils and matrix-free CG, without BCOO/CSR allocations or
host callbacks inside the compiled timestep loop.

## Next experiments

1. **Profile the current formulation.** Record CG iteration counts and actual
   residuals across contrast, angle, translation and resolution. Capture GPU
   traces to attribute time to constitutive application, scatter operations,
   reductions, launches and Maxwell curls. Separate compilation, warm stepping
   and geometry preparation; synchronize all timed GPU work.
2. **Scale the workloads.** Repeat 2D and 3D measurements over several larger
   meshes that fit available memory, physical simulation durations and material
   contrasts. Report distributions over enough repetitions, peak memory,
   precision, timestep, device and software revision. Include zero-field and
   representative excited-field cases since solve work depends on the state.
3. **Compare equivalent accuracy.** Choose a physical observable and target
   error, then refine each method independently until it reaches that target.
   Measure total runtime for the same physical duration, including the increased
   number of timesteps required by finer meshes. Include staircasing, scalar
   fractions, this tensor operator and upstream tensor smoothing. Use a separate
   checkout/environment for upstream to preserve this numerical baseline.
4. **Explore cheaper stable constitutive operators.** Investigate interface-only
   work, better preconditioning, block/local formulations and published explicit
   tensor discretizations. Require global symmetry, positivity, consistency,
   PEC compatibility, charge behavior and stable energy evolution. An effective
   local tensor alone does not establish the assembled scheme's stability.
   Relax solve tolerances only after quantifying physical error and long-run
   energy/charge effects.
5. **Consider native kernels after profiling.** Prototype fusion or JAX FFI
   only for demonstrated bottlenecks. Recheck CPU/GPU numerical agreement,
   residuals, energy, charge and convergence after every formulation change.
   Keep an unfused reference implementation for verification.

Acceptance requires a documented accuracy-versus-runtime and memory comparison
on representative photonic workloads. Agree on target errors and acceptable
costs before selecting the production formulation. If the substantial slowdown
persists at equivalent accuracy, reconsider the operator instead of proceeding
automatically with PML, ports and nonuniform mesh integration.

## Reproduction and related files

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m pytest tests/test_fit.py tests/test_fit_interfaces.py
.venv/bin/python scripts/record_fit_baseline.py --platform cuda
.venv/bin/python scripts/benchmark_fit_interfaces.py --platform cuda
JAX_PLATFORMS=cuda .venv/bin/python examples/2D_basics/6_fit_interface.py
```

The benchmark scripts overwrite their report destinations; use their `--output`
option to preserve the committed baseline when recording new experiments.
The FEM reference and spectral eigenanalysis run on the host; FIT operator
evaluation and timestep benchmarks run on CUDA.

See the [development plan](fit_backend_plan.md),
[detailed interface validation](fit_interface_validation.md),
[FIT usage guide](../beamz/simulation/fit/README.md), and the implementation in
`beamz/simulation/fit/`.
