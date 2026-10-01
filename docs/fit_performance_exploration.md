# FIT interface performance findings and next exploration

Recorded 2026-09-30. Status: preserve the prototype and investigate performance
before extending it into the full photonics backend.

## Objective and benchmark tracker

The acceptance goal is **better physical-observable accuracy at the same total
wall time as the non-FIT FDTD baseline, or the same accuracy faster**. Equal
per-cell throughput is not required. A costlier update can win through fewer
cells, fewer timesteps, or both, using improved interface treatment or nonuniform
meshing. The current results do not yet demonstrate this end-to-end win.

The tracker combines the existing benchmark reports:

- [Machine-readable tracker](fit_benchmark_tracker.json): runtime observations,
  separate accuracy evidence, source environments, and missing acceptance data.
- [Runtime spreadsheet](fit_benchmark_tracker.csv): one row per recorded workload.
- [Tracker generator](../scripts/fit_benchmark_tracker.py): regenerate derived
  metrics after rerunning any of the source benchmarks.

```bash
.venv/bin/python scripts/fit_benchmark_tracker.py
# Optional: select input report directory and separate output destination.
.venv/bin/python scripts/fit_benchmark_tracker.py --reports docs --output /tmp/fit_tracker.json
```

This command only reads recorded results and writes the tracker JSON/CSV; it
neither runs simulations nor overwrites the source benchmark reports.

| Metric | Definition | Purpose / current evidence |
| --- | --- | --- |
| Gcell updates/s | `cells * steps / warm_wall_seconds / 1e9` | Implementation throughput; derived for all saved timing workloads. One update means a complete Maxwell step, not one field component. |
| Wall time for a fixed physical duration | Record `duration = steps * dt` and synchronized wall time | Compare within the same physical workload. The tracker also reports `wall_seconds_per_simulated_ps`, a normalized warm rate, not a measured end-to-end run. |
| Wall time to target observable error | Minimum measured total runtime among refinements meeting the agreed error threshold | Primary acceptance metric. **Warm TM pilot measured** below. Isolated cold-start and observable-extraction timing remain outstanding. |

Timing ratios are meaningful only within each tracker `comparison_group`.
Different groups can differ in physical dimensions, polarization, initialization,
precision, step count, and measurement scope. In particular, the public-run
non-FIT baseline cannot be ranked against the newer isolated FIT loops simply
by comparing their throughput. The existing non-FIT comparison is this checkout's
JAX FDTD implementation, not the newer upstream native CUDA backend.

For example, the saved GPU baseline has the following matched timing-only pairs:

| Nominal mesh | Backend | Gcell updates/s | Simulated duration (ps) | Warm wall time (ms) |
| --- | --- | ---: | ---: | ---: |
| 48 × 64 | Non-FIT FDTD | 0.0642 | 0.1584 | 3.064 |
| 48 × 64 | Diagonal FIT | 0.0662 | 0.1584 | 2.968 |
| 16 × 20 × 24 | Non-FIT FDTD | 0.1257 | 0.1294 | 3.911 |
| 16 × 20 × 24 | Diagonal FIT | 0.1532 | 0.1294 | 3.209 |

These are the previously recorded single warm samples; they contain no measured
accuracy comparison and are not a new benchmark run. Native seed sampling and
field storage differ, as recorded in the original report.

### Acceptance experiment to populate the third metric

For each selected physical problem, hold geometry, materials, excitation,
boundaries, observable, and physical simulation duration fixed. Use an
independently converged reference and report its uncertainty. Agree on the
observable error threshold and whether it applies to the worst case, an average,
or every measured mode before selecting a winner; no target is assumed here.

Refine each candidate independently: non-FIT FDTD, scalar FIT, tensor-diagonal
FIT, coupled FIT, and explicit polynomial FIT. Include nonuniform meshes and
improved interface geometry when implemented. Record actual cells, smallest and
largest spacing, timestep, number of steps, precision, device/software versions,
peak memory, observable values, and reference errors for every run. Include
spatial and temporal error in the final physical observable.

Measure synchronized preparation, compilation, warm stepping, and end-to-end
runtime with repeated excited-field runs. Keep cold-start and reusable-compiled
scenarios separate; a first execution already includes stepping, so do not add
another warm-run duration to it. Existing preparation/first-execution fields are
partial measurements, not complete end-to-end totals.

Plot observable error against total wall time. At each agreed target, report the
fastest **measured** refinement that meets it and compare its total runtime with
the non-FIT baseline at that same target. Keep failing refinements visible; do
not interpolate a claimed win or multiply accuracy and speed ratios from
unrelated workloads.

A coarser uniform 3D mesh can reduce both cell count and the number of timesteps.
For a nonuniform mesh, the smallest cell generally limits the global timestep;
cell-count savings alone do not imply a timestep gain. If local time stepping
is introduced, count actual local cell updates and report the timestep schedule
instead of using `cells * global_steps` for throughput.

## Matched TM cavity pilot (2026-09-30)

**A material-sampling improvement achieves better accuracy at effectively the
same warm runtime in the existing non-FIT engine.** At 20 cells/axis, replacing
native raster ownership values with exact fractions centered on the electric
nodes reduces worst-mode error from **4.287% to 0.678%**, with summed warm runtime
changing from **0.2395 s to 0.2409 s** (0.6%) across the six problems. The FDTD
stepping implementation is unchanged. This identifies a practical improvement
that does not require adopting FIT.

This is a matched **uniform-grid 2D TM** pilot. It does not test the earlier TE
coupled tensor, nonuniform meshing, or the newer upstream native CUDA engine.
TM was chosen because both engines expose the same native full-PEC Ez node
lattice; homogeneous calibration is mandatory before measuring interfaces.

Artifacts:

- [Executed comparison notebook](fit_fdtd_comparison.ipynb): FDTD versus FIT,
  material-sampling controls, accuracy/runtime plots, and benchmark limits.
- [Raw matched measurements](fit_matched_cavity.json), including 210 runs,
  calibration, FEM references, mode-evolution checks, and target selections.
- [Error/runtime plot](fit_matched_cavity.png) and [vector plot](fit_matched_cavity.svg).
- [Benchmark script](../scripts/benchmark_matched_cavity.py) and
  [plot generator](../scripts/plot_matched_cavity.py).
- The main [JSON tracker](fit_benchmark_tracker.json) and
  [CSV tracker](fit_benchmark_tracker.csv) include these paired observations.

### Physical problem and verification

Each case is a 4 µm square TM PEC cavity with relative permittivities 2/12,
interface angles 0, 30, or 60 degrees and two offsets per angle. Every method
runs exactly 2 ps from the same analytic mixture of three sine products in Ez,
with zero initial magnetic field. Seven uniform meshes are tested: 12, 20, 28,
40, 56, 80, and 112 cells/axis. The timestep is selected separately per mesh
below CFL and rounded down so an integer number of steps reaches exactly 2 ps.
Within a mesh, every method uses the same timestep. Timings use synchronized
float32 public API runs and five warm repetitions reset to identical fields.
The actual cell and timestep counts feed the throughput tracker.

Accuracy is the maximum relative frequency error over the first three cavity
modes, including leapfrog temporal dispersion. An independent interface-fitted
P1 FEM solves `-laplacian Ez = lambda epsilon Ez` with Dirichlet walls at 96 and
144 cells/axis. Maximum frequency drift between those references is **0.0330%**;
this is an empirical reference check, not a rigorous uncertainty bound.

The benchmark assembles the sparse nodal eigenproblem from each simulation's
actual material coefficients and applies the exact leapfrog frequency mapping.
For every method/mesh/case it also initializes each discrete eigenmode and
compares the public engine's field after 2 ps with the predicted discrete mode
evolution. The maximum normalized field discrepancy over all 630 checks is
**2.71e-4**. Both engines pass a homogeneous analytic cavity calibration.
This avoids FFT-bin errors while checking that the diagnostic represents the
actual stepping code. The eigenanalysis, FEM reference, and diagnostic runs
are excluded from simulation timing; monitor-based frequency extraction cost
has not been measured.

### Results at the same mesh and physical duration

At 20 cells/axis, worst error over all 18 frequencies and sum of the six median
warm runtimes:

| Method | Worst frequency error | Warm wall time (s) |
| --- | ---: | ---: |
| Current non-FIT FDTD | 4.287% | 0.2395 |
| Non-FIT FDTD with exact dual-node fractions | 0.678% | 0.2409 |
| FIT, staircase with native averaging | 3.188% | 0.1845 |
| FIT, exact primal-cell fractions | 0.623% | 0.1838 |
| FIT, exact dual-node fractions | 0.678% | 0.1850 |

The non-FIT control supplies exact dual-node permittivities through the existing
native raster-to-Ez ownership contract. This is a benchmark adapter, not a new
production material API. Its interior coefficients match FIT's dual-fraction
coefficients to float32 accuracy. Their frequency errors agree, demonstrating
that this TM gain comes from material sampling rather than an inherently more
accurate FIT timestep. The default FDTD comparison includes the current polygon
rasterization and ownership policy, not a deliberately disabled smoother.

### Runtime at the same exploratory error target

For an estimated **1% maximum mode error per case**, select the fastest measured
refinement independently for each geometry, then sum the six median warm times:

| Method | Selected cells/axis across six cases | Warm total (s) | Speedup vs current FDTD |
| --- | --- | ---: | ---: |
| Current non-FIT FDTD | 80, 80, 80, 112, 80, 112 | 1.1927 | 1.00× |
| Non-FIT FDTD, exact dual fractions | 20 in all cases | 0.2409 | 4.95× |
| FIT, staircase | 28, 40, 12, 20, 12, 20 | 0.2087 | 5.72× |
| FIT, exact primal fractions | 20 in all cases | 0.1838 | 6.49× |
| FIT, exact dual fractions | 20 in all cases | 0.1850 | 6.45× |

These are measured warm simulation speedups at an estimated reference error,
not end-to-end production claims. The thresholds 1%, 0.5%, and 0.2% are
exploratory and recorded explicitly. Near-threshold choices should be revisited
with a tighter reference. The original FDTD baseline does not reach 0.5% across
all six cases on this sweep; the report marks unreached targets instead of
extrapolating. The plot uses a common resolution across cases at each point,
whereas this table allows each physical case to choose its own refinement.

Preparation and first-execution timings are recorded separately. The JSON field
`cold_simulation_wall_seconds` is preparation plus first execution in an already
initialized process. JAX cache sharing, run order, and the resumed second batch
affect these samples; they are **not isolated cold-start measurements** and
should not support a cold-start speedup claim. Warm distributions, reference
diagnostics, and limitations are retained in the raw report. Peak memory,
frequency-extraction overhead, nonuniform grids and representative 3D/PML/port
workloads remain part of the overall acceptance gate.

Reproduce or extend the pilot (CUDA access is required):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_matched_cavity.py --platform cuda
# Reuse finished cases/references and add missing refinements or methods:
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_matched_cavity.py --platform cuda --resume
MPLCONFIGDIR=/tmp/beamz-matched-mpl .venv/bin/python scripts/plot_matched_cavity.py
.venv/bin/python scripts/fit_benchmark_tracker.py
```

Use `--output` to preserve the saved report. `--quick --platform cpu` runs one
interface at two resolutions with the same engine calibration and checks.

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

## Follow-up: explicit inverse exploration (2026-09-30)

**Partial success:** the diagonal of the assembled tensor material improves
average accuracy at essentially scalar stepping cost in 2D. A four-term fixed
inverse retains almost all of the coupled operator's spectral accuracy at a
roughly two-to-threefold 2D timestep cost. Neither result establishes the full
production acceptance gate above, particularly for 3D or oblique interfaces.

The standalone experiment is
[`scripts/explore_fit_explicit.py`](../scripts/explore_fit_explicit.py), with
raw observations in [`fit_explicit_exploration.json`](fit_explicit_exploration.json).
It leaves the simulation API and default constitutive solver unchanged.

### Accuracy on the original reference cases

The experiment reuses the saved independent FEM frequencies for the same six
angle/translation cases and three modes. All new eigenproblems use float64.
Values below are mean relative frequency error, in percent:

| Method | 12 cells/axis | 20 cells/axis | 28 cells/axis |
| --- | ---: | ---: | ---: |
| Staircase (original) | 0.8872 | 0.6217 | 0.2732 |
| Scalar fractions (original) | 1.0797 | 0.5530 | 0.3427 |
| Tensor diagonal, one term | 0.5806 | 0.2640 | 0.1923 |
| Two-term inverse | 0.6248 | 0.3092 | 0.2070 |
| Four-term inverse | 0.5177 | 0.2397 | 0.1560 |
| Six-term inverse | 0.5155 | 0.2382 | 0.1549 |
| Coupled CG (original) | 0.5154 | 0.2382 | 0.1549 |

At 28 cells/axis, the diagonal tensor reduces mean error by 44% relative to
scalar fractions and 30% relative to staircasing. However, angle-resolved errors
matter: at 30 degrees it gives 0.2495%, versus 0.2449% for staircasing and 0.1934%
for CG. At zero degrees its error is 0.0778%, equal to CG. Thus the diagonal
approximation is a useful cheap candidate, not a replacement that preserves the
full oblique-interface improvement.

The four-term error at 28 cells/axis is only 0.00112 percentage points above
CG's mean error. Two terms are less accurate than the tensor diagonal on this
study; increasing polynomial order need not monotonically reduce error against
the continuum, even as it converges to the discrete coupled inverse.

### Formulation and stability

Write the normalized assembled material as `M = D + O`, with its exact diagonal
`D`. The experimental inverse is

`K_m = sum_{k=0}^{m-1} (-D^-1 O)^k D^-1`.

In symmetric coordinates, with `B = D^-1/2 O D^-1/2`, this is
`D^-1/2 sum((-B)^k) D^-1/2`. For even `m` and spectral radius `rho(B) < 1`,
`K_m` is symmetric positive definite and `K_m <= M^-1` in quadratic-form order:
the omitted part is `D^-1/2 B^m (I+B)^-1 D^-1/2`, which is positive semidefinite.
Consequently the original conservative CFL bound remains sufficient. The script
certifies a bound on `rho(B)` from local normalized tensor eigenvalues; the local
quadratic-form inequalities survive corner assembly and PEC restriction. It
rejects unsupported term counts or a bound greater than or equal to one.

One term uses the original diagonal timestep with the tensor diagonal replacing
scalar permittivity. Its material diagonal still satisfies the original minimum
permittivity bound. Higher orders evolve displacement with the same discrete
curl and use a fixed explicit inverse. Their electric energy is `d^T K_m d / 2`;
conservation must be judged for this approximate material, not by assuming the
original material relation is solved exactly. There are no CG tolerance or
convergence claims for these approximations.

The 2D off-diagonal application factors the original four corner contributions
into two cell averages and their transposes. Tests compare it with the original
assembled operator. The 3D experiment retains the general corner implementation,
which is a substantial remaining performance limitation.

### Timing protocol and scope

Timing uses float32 CUDA, seven synchronized repetitions of 256 timesteps,
resetting to the same state for each repetition. All methods use the same
physical timestep on a given mesh. Smooth/random magnetic initial fields start
with zero electric displacement so that every approximate constitutive relation
is initially satisfied. Zero fields are tested separately. The report records
geometry/operator construction, first compiled execution, every warm timing,
and energy/charge checks over 2,048 timesteps.

Smooth-field median wall time for 256 steps on the NVIDIA GeForce RTX 5070
Laptop GPU, JAX 0.9.0 (milliseconds):

| Mesh | Scalar | Tensor diagonal | Four terms | Coupled CG |
| --- | ---: | ---: | ---: | ---: |
| 48 × 64 | 1.222 | 1.247 | 2.414 | 56.418 |
| 192 × 256 | 1.780 | 1.730 | 3.882 | 74.252 |
| 12 × 16 × 20 | 2.156 | 2.844 | 19.653 | 101.544 |
| 24 × 32 × 40 | 3.378 | 3.680 | 27.665 | 145.216 |

Across smooth and random 2D fields, the tensor diagonal costs 0.97–1.12 times
scalar, and four terms cost 1.97–2.87 times scalar. Four terms cost 7.33–9.12
times scalar in these 3D runs; this is not a near-scalar 3D solution.

Over all four meshes and both excited initializations, maximum relative
conserved-energy drift was `4.49e-5` for the tensor diagonal and `2.43e-7`
across the polynomial variants. Maximum bulk-charge error, normalized by the
largest final displacement component, was below `7.35e-6` for the diagonal and
`4.73e-6` for the polynomial variants. Zero fields stayed zero. These are finite
2,048-step checks, not long-duration photonic device validation.

These timings measure a directly JIT-compiled timestep loop, without the public
`run()` result packaging or host residual check. The original measurements used
32-step public runs and different initial fields. Therefore their eightfold CG
slowdown and the new ratios should not be treated as the same benchmark. No
simultaneous GPU workloads were used for this comparison.

Accuracy is still a low-mode 2D cavity study; 3D timing and conservation do not
establish 3D interface accuracy. There is no equal-observable-error refinement
study, waveguide transmission, PML, high-contrast sweep, peak-memory measurement,
or production photonics validation in this experiment. The prototype retains
the original corner coefficients for comparison and does not establish memory
savings. Compilation and preparation also matter for short simulations.

Validation: eight new algebra/argument tests passed on CPU and CUDA. The tests
check the polynomial against the independently assembled material, symmetry,
positive eigenvalues, the coupling bound, and the inverse ordering used in the
CFL argument. Batched float32 reference assembly explicitly requests highest
matmul precision to avoid CUDA TF32 rounding obscuring these algebra checks;
the float64 spectra and timed float32 stepping settings were not changed.

Reproduce without overwriting the original reports:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/explore_fit_explicit.py --platform cuda
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_fit_explicit_exploration.py -q
```
