# Initial dielectric interface validation

Milestone 2 now has an initial planar-interface implementation. Uniform 2D
TE/TM and 3D PEC meshes support two positive isotropic dielectrics, exact cell
fractions and a coupled symmetric positive electric constitutive operator.
This is an accuracy prototype; photonic ports, PML, nonuniform meshes and curved
geometry are not implemented yet.

## Method

Geometry clipping runs on the host before stepping. Normal permittivity uses
the harmonic mean and tangential permittivity the arithmetic mean. Cell-corner
quadrature assembles the global permittivity operator through paired gather and
scatter operations; its free PEC principal system is positive. Maxwell's
leapfrog update evolves electric displacement, with a matrix-free
Jacobi-preconditioned conjugate-gradient solve recovering electric voltages.
GPU stepping stays inside JAX and does not use BCOO/CSR or host callbacks.

The spectral experiment uses a unit-square TE PEC cavity, relative permittivity
2/12, interface angles 0°, 30°, 60° and two translations at each angle. Each
comparison measures the first three nonzero cavity frequencies. A separate
interface-fitted P1 triangular FEM supplies references at 96 and 144 cells per
axis. Their maximum relative frequency difference was 0.0157%. FIT spectra use
float64 CUDA JAX operators and host eigenanalysis. This isolates spatial error
from timestep dispersion and boundary absorption.

Comparators are centre-assigned staircasing followed by the baseline native
scalar averaging, exact-fraction scalar arithmetic averaging, and the coupled
tensor operator. The mean relative frequency error across all eighteen modes
at each resolution was:

| Cells per axis | Staircase | Scalar fraction | Coupled tensor |
| --- | ---: | ---: | ---: |
| 12 | 0.8872% | 1.0797% | 0.5154% |
| 20 | 0.6217% | 0.5530% | 0.2382% |
| 28 | 0.2732% | 0.3427% | 0.1549% |

At the finest tested resolution, tensor mean error is about 2.21 times lower
than scalar fraction error and 1.76 times lower than staircase error. Some
coarse staircase cases perform better through error cancellation. These results
do not establish universal superiority or a convergence order, and do not
compare the newer upstream FDTD tensor smoothing implementation.

## GPU cost

Hardware: NVIDIA RTX 5070 Laptop GPU, JAX/JAXlib 0.9.0, CUDA backend. These are
small float32 workloads: median of three synchronized warm 32-step runs,
without field recording. Compilation is recorded separately in the raw report.

| Workload | Scalar fraction | Coupled tensor | Ratio |
| --- | ---: | ---: | ---: |
| 2D TE, 48 × 64 | 0.989 ms | 8.233 ms | 8.33× |
| 3D, 12 × 16 × 20 | 1.832 ms | 14.369 ms | 7.84× |

After 128 steps, relative conserved-energy drift was approximately `1.08e-7`
and `7.03e-8` respectively; maximum constitutive residuals were `4.57e-7` and
`4.89e-7` against a requested `5e-7`. Larger domains, material contrast and
preconditioners need further study before selecting this operator for production.
The [performance exploration](fit_performance_exploration.md) records the
decision to investigate this cost before extending the backend, and explains
why upstream's CUDA extension does not account for this measured comparison.

Tests cover geometry, effective continuity laws, assembled symmetry/positivity,
equal-material consistency, energy, divergence, charge, current sources,
chunking, solve failures, CUDA placement and two accuracy regressions.
All 54 FIT tests and the 177 existing baseline tests passed on CUDA. The
standalone photonic-scale example also ran on `CudaDevice(id=0)`, with relative
conserved-energy drift of approximately `8.3e-8` after 128 steps.

Reproduce with GPU access available to the process:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m pytest tests/test_fit.py tests/test_fit_interfaces.py
.venv/bin/python scripts/benchmark_fit_interfaces.py --platform cuda
JAX_PLATFORMS=cuda .venv/bin/python examples/2D_basics/6_fit_interface.py
```

Raw results: [convergence and timings](fit_interface_convergence.json),
[existing-engine GPU baseline](fit_baseline_gpu.json).

## Upstream CUDA architecture

The newer upstream uses both JAX and native CUDA: an optional `beamz._cuda`
extension exposes fused Yee/CPML kernels and CUDA-graph stepping through JAX FFI.
See the [upstream CUDA documentation](https://github.com/beamzorg/beamz/blob/main/cuda/README.md).
That extension is absent from this older v0.3.1 checkout. Installing CUDA-enabled
JAX enables this FIT implementation's GPU stencils and CG, but does not install
upstream's custom kernels or improve generic JAX sparse operations automatically.
Upstream also contains newer interface smoothing; comparison and integration
with it remain required before attributing benefits specifically to FIT.
