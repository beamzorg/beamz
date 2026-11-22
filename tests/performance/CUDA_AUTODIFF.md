# CUDA FDTD and reverse-mode performance audit

The current implementation uses native component 0.27.0 / ABI 27 on base
`6836a6fe`. Matched measurements use an RTX 3090, driver 610.43.03, Python 3.11.15,
and JAX/JAXLIB 0.9.0. Both backends run on the same GPU with FP32 fields,
coefficients, and CPML memory. Native fast math is disabled.

| Workload | GPU JAX median | CUDA median | JAX / CUDA |
| --- | ---: | ---: | ---: |
| FDTD, CPML + heterogeneous guide + source + DFT, 64×128×192, 512 steps | 187.88 ms | 107.23 ms | 1.75× |
| Bend inverse objective, 100 nm | 50.66 ms | 31.31 ms | 1.62× |
| Bend objective **and full reverse-mode gradient**, 100 nm | 257.24 ms | 179.63 ms | **1.43×** |
| Bend inverse objective, 50 nm | 428.32 ms | 217.48 ms | 1.97× |
| Bend objective **and full reverse-mode gradient**, 50 nm | 1768.76 ms | 1331.84 ms | **1.33×** |

These are complete objective/gradient calls, including forward propagation,
checkpoint replay, native backward propagation, material mapping, source and
monitor transposes, and objective differentiation. No manually constructed
adjoint source is inserted. The [waveguide-bend notebook](../../examples/notebooks/cuda_autodiff_waveguide_bend.ipynb)
uses 22×50×50 material cells, 1299 steps and 900 parameters at 100 nm;
44×100×100 cells, 2597 steps and 3600 parameters at 50 nm. Both backends use the
same physics, guided-mode objective and checkpoint interval 32.

After three warmups, synchronized device-resident calls alternate backend order.
There are nine samples at 100 nm and for ordinary FDTD, and seven at 50 nm.
Compilation, placement and mode solving are excluded. Bootstrap 95% intervals
for gradient speedups are 1.429–1.434× and 1.327–1.329× respectively. Objective
intervals are 1.607–1.633× and 1.957–1.971×; ordinary FDTD is 1.738–1.757×.
These intervals describe variation within these runs, not across GPUs or workloads.

Raw samples, compilation times, temporary storage estimates, source checksums
and binary checksums are retained in [main_inverse_100nm.json](cuda_autodiff_results/main_inverse_100nm.json),
[main_inverse_50nm.json](cuda_autodiff_results/main_inverse_50nm.json) and
[main_fdtd_medium.json](cuda_autodiff_results/main_fdtd_medium.json).
Their runtime/kernel and binary checksums match the measured implementation.

## What changed since the parity result

The previous ABI-24 implementation was approximately 1.00× GPU JAX for the
100 nm objective and gradient: 255.58 ms CUDA versus 255.49 ms JAX. Its complete
records remain in [abi24_inverse_100nm.json](cuda_autodiff_results/abi24_inverse_100nm.json),
[abi24_fdtd_medium.json](cuda_autodiff_results/abi24_fdtd_medium.json) and
[abi24_validation.json](cuda_autodiff_results/abi24_validation.json).

The new work addresses launch, copy and accumulation overhead around the
already fused discrete Yee/CPML transpose:

- Inactive phase residuals use empty slots. The native handler reconstructs
  unread logical field/CPML descriptors from cotangents, avoiding full zero arrays.
- Eligible checkpoint chunks run through the existing native forward graph.
  Backward replay retains only the trajectory needed by active coefficient VJPs.
- A complete material/state reverse step applies E then H transposes with
  declared field aliases and a separate intermediate H bank. Electric-material
  gradients accumulate inside CUDA rather than adding arrays after every step.
  Full-grid accumulation has one writer per cell; scalar/profile gradients use
  atomic reductions. Arbitrary fixed electric and magnetic decay is supported.
- CPML cotangents reuse two aliased banks. A device iteration scalar selects
  opposite read/write banks, preserving stencil dependencies without copying
  all twelve recurrence buffers on every reverse timestep.
- With fixed coefficients and only initial-state derivatives, the Jacobian is
  linear and needs no primal field/CPML replay. Trainable conductivity, source
  amplitudes, clocks, and other recurrence/metric coefficients retain the
  general native phase VJP path.

A diagnostic trace reduced device-copy counts from 51,292 to 25,409, with their
summed duration falling from 60.86 to 34.21 ms. Other GPU-operation counts fell
from 51,686 to 21,850. Native adjoint duration increased from 44.02 to 50.83 ms;
the net gain comes from removing surrounding overhead. These traced totals
are separate from synchronized warm wall-clock measurements. The
[profile comparison](cuda_autodiff_results/reverse_profile_comparison.json)
retains both profiles' summaries and implementation provenance.

Ordinary FDTD retains current main's forward kernels and scheduling. CUDA's
1.75× ordinary-FDTD advantage remains intact. Differentiable observation clocks
use an initial origin plus integer step count. Padding and paired temporal
schedules are disabled inside differentiable scans.

## Correctness and notebook

Initial gradient relative L2 disagreement with GPU JAX is 2.03×10⁻⁷ at 100 nm and
5.17×10⁻⁸ at 50 nm. Three centered finite-difference steps independently agree
with the analytic directional derivative within 0.1% and 0.4%, respectively.
Ordinary FDTD compares every field, CPML, DFT and clock state leaf; the largest
relative leaf error is 8.05×10⁻⁵.

The notebook executed all 25 cells and 160 updates with CUDA. It reaches 88.333%
guided transmission on the 100 nm training mesh and 85.989% after thresholding.
At 25 nm / 400 fs, the corresponding values are 80.032% and 78.604%. At 50 nm,
forward and reverse excitation agree (80.704% continuous, 79.898% binary).
Material transpose asymmetry is 3.03×10⁻⁸. The optimized CUDA/JAX gradient
relative disagreement is approximately 0.0006%; directional finite differences
at steps 0.2 and 0.1 agree within 0.44%. These results supersede older notebook
scores. [Verification rows](cuda_autodiff_results/main_notebook_verification.json)
and [optimized parameters](cuda_autodiff_results/main_optimized_parameters.npz)
are saved for review. Saved optimization and field figures were visually inspected;
the executed notebook includes those figures. HTML browser inspection remains
incomplete because the T3 preview could not connect to the local server.

CUDA supports first-order reverse mode on one GPU with FP32 3D fields. Native
JVPs, higher derivatives, coupled tensor updates, axis permutations,
dispersive binding, and sharded CUDA autodiff remain unsupported. Rebuild the
component for ABI 27 before running this API. See the
[optimization guide](../../beamz/optimization/README.md) for the public API.

Compiler-reported gradient temporary storage is 112.17 MB CUDA / 110.71 MB JAX
at 100 nm and 1481.09 MB CUDA / 1464.24 MB JAX at 50 nm. These are executable
workspace estimates, not total process peak VRAM. Checkpoint boundary states
still dominate storage at finer meshes.

## Validation

All 37 GPU autodiff cases passed, followed by three additional fixed-material
state-gradient cases covering the replay-free branch and even-length chunks.
These cover bare/CPML/mixed-PEC grids, scalar/profile/grid material gradients,
fixed nonunit electric/magnetic losses, field and CPML state, sources, DFTs,
clock/metric/recurrence derivatives, checkpoint tails, and general VJP fallbacks.
The latest CPU selection passed 103 backend/ABI/runtime/differentiation checks,
plus ten public-method and cache-reuse checks after repairing their CI contracts.
Earlier broader CPU selections are retained in the ABI-24 validation record.

The aliased material/state path passes nine JAX integration cases under memcheck
with zero errors, using JAX 0.10.2 to avoid 0.9.0's CUDA symbol-probe diagnostics.
Performance measurements use 0.9.0 throughout. The standalone harness passes
108 cases under initcheck with zero errors and racecheck with zero hazards.
It covers both phases, scalar/profile/grid coefficients, bare/CPML grids,
PEC intersections, accumulator preservation and both CPML bank parities.
It leaves destination banks uninitialized and independently checks diagonal
field and CPML columns. See [main_validation.json](cuda_autodiff_results/main_validation.json).

Ruff formatting/lint, changed-module Pyright, dead-code checks, native CMake
compilation and wheel/sdist build with distribution validation passed.
`make audit` stops at five existing GDSFactory attribute typing errors in
unchanged `beamz/design/gds.py`; its full-suite/coverage stage did not run locally.
An earlier ordinary-CUDA selection passed 11 and failed four cases identically
on current main: two FP32 near-cancellation tolerances and two outdated lossy
schedule assertions. That baseline used unchanged main Python and identical
main forward kernels, adapting only its ABI number to the ABI-24 binary.
These are existing limitations, not a claim of a clean full-suite result.

## Reproduction

Build the native component, then run from the repository root with CUDA-enabled
JAX. Serialize GPU benchmarks and disable XLA preallocation; no global XLA flag
override is needed.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/benchmark_cuda_inverse_design.py --dx-nm 100 --samples 9 \
  --finite-difference --output results/inverse_100nm.json
python scripts/benchmark_cuda_inverse_design.py --dx-nm 50 --samples 7 \
  --finite-difference --output results/inverse_50nm.json
python scripts/benchmark_cuda_fdtd.py --shape 64 128 192 --steps 512 \
  --samples 9 --standard-only --output results/fdtd_medium.json
python scripts/generate_cuda_abi.py --check
python -m pytest tests/hardware/test_cuda_autodiff.py
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  tests/hardware/test_cuda_autodiff.py -k native_material_chunk
compute-sanitizer --tool initcheck --error-exitcode 99 \
  python scripts/check_cuda_adjoint_memory.py --banked --cudart /path/to/libcudart.so
compute-sanitizer --tool racecheck --error-exitcode 99 \
  python scripts/check_cuda_adjoint_memory.py --banked --cudart /path/to/libcudart.so
```

The ordinary workload uses six-cell CPML, a heterogeneous guide, an electric
source and a three-frequency DFT plane. Add `--profile results/profile` to either
runner for a JAX/CUPTI trace. The inverse runner also writes compiled HLO.
`--beta 16 --parameters-npz tests/performance/cuda_autodiff_results/main_optimized_parameters.npz`
checks the saved optimized design. The standalone harness mirrors the private
Linux C++ layout and must be updated when that ABI changes.

## Historical ABI-14 measurements

The earlier ABI-14 implementation on base `e4c2f3d4` measured 2.09× and 1.86×
gradient speedups at 100 and 50 nm against its then-current JAX implementation.
Its JAX medians were 545.62 and 3096.07 ms, substantially slower than the current
reference. Those historical results remain in this directory's original
`inverse_*`, `fdtd_*`, `baseline_*` and `validation.json` records. They are
optimization history and do not establish current performance or a universal
CUDA advantage.
