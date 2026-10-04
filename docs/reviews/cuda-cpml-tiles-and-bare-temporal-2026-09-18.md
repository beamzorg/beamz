# CPML tile generality and bare-domain temporal blocking

This study separates two questions: whether the specialized CPML queues improve
realistic workloads generally, and whether multi-step field reuse can pay off
without CPML, sources, or monitors. Production scheduling remains unchanged.

## What the backend currently does

CUDA graph batching submits many complete timesteps together. It reduces host
launch overhead but does not itself keep fields on-chip between timesteps.

On eligible large SM86 domains, the default fused interior kernel uses a
32×8×8 spatial tile and a two-plane shared-memory ring to compute H then E for
**one timestep**. Two global field banks keep old inputs stable for neighboring
blocks. CPML and monitor work still execute each timestep. Other domain shapes
retain separate phase queues.

The opt-in two-timestep prototype instead stages a 16×8×4 deep-interior tile
plus a two-cell halo and computes H1, E1, H2, E2 in shared memory. It writes both
intermediate and final fields to global memory and uses a third field bank.
CPML and a coupling band advance conventionally on both substeps, with sources
and DFT observations at the correct times. Odd trailing steps are handled
separately. It remains disabled by default because its earlier realistic
reference result was 4.70 GCUPS versus 8.69 for one-step execution. See the
[previous implementation study](cuda-layout-temporal-study-2026-09-18.md).

The CPML queue tiles are spatial thread layouts, not temporal blocking. Their
64×4, 32×8, and 32×4 variants change how absorber faces are covered. When the
combined queue also covers the interior, it remaps threads to a 64-wide core
layout; the 128-thread 32×4 option therefore also changes that core mapping.

## Expanded CPML results

Completed 21 cases / 63 tile configurations across 14 physical shapes,
8.4–25.2 million cells. Every case has exactly 12-cell FP32 CPML, lossless
materials, a solved mode source, and two monitors. This includes all six
64×256×1024 axis permutations, irregular dimensions, widths 63/64/65, several
volumes, full-aperture field monitors, 11-frequency mode monitors, smooth
materials, 1,024 timesteps, and developed-field controls.

Each case uses one fresh process and the same state, coefficients, and source
and monitor definitions for three compiled executables. Only the tile flags
change. Four warmups precede six timing rounds; all six variant orderings occur
once, shuffled with a recorded seed. This reduces timing-order drift. The
native schedules are asserted to use one temporal step and the combined CPML
queue. Storage padding is disabled.

The 32×8 option is **not universally faster**. Across seven short-x cases
(width 63–65), median paired speedups range from **3.9% to 6.7%**, with a median
of **5.1%**. Across the other 14 cases, its median is **−0.15%**, with a range
from **−2.82% to +1.00%**. The 32×4 option is also beneficial on the short-x
cases but has larger regressions elsewhere. These are descriptive case
statistics, not a guarantee for every simulation or source orientation.

Values below are warm executable GCUPS, including CPML/source/DFT work. Gain is
the median of per-round elapsed-time ratios against 64×4, so it need not equal
the ratio of the two separately computed GCUPS medians.

| Case | 64×4 | 32×8 | 32×4 | 32×8 paired gain |
| --- | ---: | ---: | ---: | ---: |
| [aspect_1024_256_64](rtx3090-2026-09-18-cpml-tiles-expanded/aspect_1024_256_64.json) | 6.60 | 6.94 | 6.96 | +5.07% |
| [aspect_1024_64_256](rtx3090-2026-09-18-cpml-tiles-expanded/aspect_1024_64_256.json) | 7.99 | 8.06 | 7.98 | +0.48% |
| [aspect_256_1024_64](rtx3090-2026-09-18-cpml-tiles-expanded/aspect_256_1024_64.json) | 6.54 | 6.84 | 6.86 | +4.46% |
| [aspect_256_64_1024](rtx3090-2026-09-18-cpml-tiles-expanded/aspect_256_64_1024.json) | 8.29 | 8.24 | 8.25 | -0.59% |
| [aspect_64_1024_256](rtx3090-2026-09-18-cpml-tiles-expanded/aspect_64_1024_256.json) | 7.14 | 7.15 | 7.07 | +0.09% |
| [aspect_64_256_1024](rtx3090-2026-09-18-cpml-tiles-expanded/aspect_64_256_1024.json) | 7.88 | 7.85 | 7.81 | -0.40% |
| [developed_short_x](rtx3090-2026-09-18-cpml-tiles-expanded/developed_short_x.json) | 6.57 | 6.91 | 6.83 | +5.10% |
| [eleven_frequencies](rtx3090-2026-09-18-cpml-tiles-expanded/eleven_frequencies.json) | 8.44 | 8.42 | 8.39 | -0.26% |
| [field_1024_256_64](rtx3090-2026-09-18-cpml-tiles-expanded/field_1024_256_64.json) | 5.19 | 5.39 | 5.40 | +3.89% |
| [field_128_256_512](rtx3090-2026-09-18-cpml-tiles-expanded/field_128_256_512.json) | 8.16 | 7.90 | 7.83 | -2.82% |
| [field_64_256_1024](rtx3090-2026-09-18-cpml-tiles-expanded/field_64_256_1024.json) | 7.81 | 7.87 | 7.61 | +0.51% |
| [irregular_1024_256_63](rtx3090-2026-09-18-cpml-tiles-expanded/irregular_1024_256_63.json) | 6.86 | 7.20 | 7.22 | +4.82% |
| [irregular_1024_256_65](rtx3090-2026-09-18-cpml-tiles-expanded/irregular_1024_256_65.json) | 6.46 | 6.90 | 6.96 | +6.74% |
| [irregular_71_479_503](rtx3090-2026-09-18-cpml-tiles-expanded/irregular_71_479_503.json) | 8.02 | 8.18 | 8.11 | -0.04% |
| [irregular_97_289_593](rtx3090-2026-09-18-cpml-tiles-expanded/irregular_97_289_593.json) | 8.12 | 8.21 | 8.12 | +1.00% |
| [long_short_x](rtx3090-2026-09-18-cpml-tiles-expanded/long_short_x.json) | 6.55 | 6.92 | 6.93 | +5.46% |
| [smooth](rtx3090-2026-09-18-cpml-tiles-expanded/smooth.json) | 7.47 | 7.39 | 7.50 | -0.78% |
| [volume_128_256_256](rtx3090-2026-09-18-cpml-tiles-expanded/volume_128_256_256.json) | 8.15 | 8.21 | 8.13 | +0.62% |
| [volume_128_256_512](rtx3090-2026-09-18-cpml-tiles-expanded/volume_128_256_512.json) | 8.34 | 8.32 | 8.36 | -0.35% |
| [volume_128_256_768](rtx3090-2026-09-18-cpml-tiles-expanded/volume_128_256_768.json) | 8.55 | 8.49 | 8.42 | -0.39% |
| [volume_256_256_256](rtx3090-2026-09-18-cpml-tiles-expanded/volume_256_256_256.json) | 8.50 | 8.53 | 8.52 | +0.43% |

[Combined CSV](rtx3090-2026-09-18-cpml-tiles-expanded/summary.csv).
All three variants produced **exactly identical complete output state** in all
21 large cases, including absorber recurrence and DFT accumulators. Independent
JAX parity was also tested on small, nonzero, randomly seeded fields so that
absorber faces not reached by a benchmark's source pulse were still exercised.

The new 36-case JAX suite passed: three tiles × three source normals × fused
and unfused execution × binary and smooth lossless materials. It uses odd
37×49×65 dimensions and rotations, overlapping mode sources near the absorber,
mode or full-field monitors, and 32+33-step continuation. The specialized tiles
must also match 64×4 bit-for-bit on these cases.

At 65 steps with overlapping sources, the old tile and specialized tiles both
showed about two ppm CUDA/JAX differences near cancellation zeros. A diagnostic
comparison across all three tiles and both scheduling paths reproduced this.
For these new tests only, the leaf-scaled absolute tolerance is three ppm;
a separate peak-normalized maximum-error check is bounded by three ppm plus
3e-6 absolute. Per-element relative tolerance remains 3e-5. The old tests'
tolerances were not changed. This is float32 numerical agreement, not a claim
of bitwise agreement with JAX.

The primary study reached 100% sampled utilization, 78°C peak sampled
temperature and 5.48 GiB minimum free GPU memory. Power limit remained 370 W.
The desktop and existing notebook process remained in place. Small absolute
GCUPS differences from previous fresh-process studies should not be interpreted
as version-to-version gains: retaining three compiled variants changes allocator
state, and this is a shared-device measurement.

Four fresh-process repeats used twelve balanced timing rounds with a different
seed. They confirm the narrow-domain benefit, while the larger initial
full-monitor regression does not repeat at the same magnitude:

| Case | 64×4 GCUPS | 32×8 GCUPS | 32×8 paired gain | 32×4 paired gain |
| --- | ---: | ---: | ---: | ---: |
| 1024×256×64, mode | 6.53 | 6.86 | +5.00% | +5.31% |
| 1024×256×65, mode | 6.45 | 6.88 | +6.61% | +8.54% |
| 128×256×512, mode | 8.41 | 8.31 | −0.46% | −0.86% |
| 128×256×512, full-field | 8.27 | 8.25 | −0.25% | −0.71% |

[Raw repeats](rtx3090-2026-09-18-cpml-tiles-repeat/). All three tile outputs
also agree exactly in these repeats. Together the primary study and repeats
contain 25 case runs / 75 tile configurations and 522 timed samples. A separate
initial smoke run is retained but excluded from these counts.

Recommendation: retain 64×4 as the FP32 default. The explicit 32×8 option is
supported by repeated gains on the tested short-x domains; 32×4 sometimes wins
more there but is less consistent elsewhere. Neither should be described as
a universal improvement, and the evidence does not yet justify selecting it
for every domain or extrapolating the narrow-domain rule to untested widths.
The source-orientation correctness tests do not constitute source-orientation
performance measurements: benchmark waveguides propagate along x throughout.

## Bare periodic temporal experiment

The standalone test uses six periodic Yee component arrays of equal logical
extent, homogeneous normalized coefficients, float32, and a stable normalized
timestep of 0.5. Initial fields are nonzero deterministic random values. Periodic
wrapping closes the finite stencil; there are **no CPML or physical boundary
kernels, sources, monitors, material-codebook lookups, or intermediate field
exports**. A finite stencil still needs a mathematical closure, hence the
periodic wrap rather than an undefined edge.

This is a new rolling-plane prototype, not the existing full-3D-tile two-step
backend with a flag switched off. Each H/E stage keeps two planes in shared
memory. Successive stages lag along z and shrink their transverse halo.
The final stage writes H and E once per block of 1, 2, 4, or 8 timesteps.
Two global field banks preserve inputs between tile launches. A separate
H-then-E kernel pair supplies the comparison implementation and large-grid
numerical oracle. One-step rolling controls include both 8- and 16-plane
spatial depths, to distinguish spatial reuse from temporal reuse.

The correctness checks cover 17×23×35 for nine steps, including odd tails and
final-bank selection, and 33×39×67 for sixteen steps. Every variant matched the
independent CPU reference exactly in these tests. Large performance cases also
compare every output field against the separate-phase implementation before
timing. Correctness and sanitizer timings are excluded from performance results.

**Two-step rolling blocking is viable in this bare experiment.** A single fixed
16×8×16 two-step tile beats the best tested one-step control by 27–32% across
all four large domains. Four and eight steps are slower in the tested designs.

| z×y×x | Best tested one-step | Fixed two-step 16×8×16 | Best tested four-step | Eight-step 8×4×32 |
| --- | ---: | ---: | ---: | ---: |
| 128×256×512 | 12.52 | 16.49 | 10.52 | 2.24 |
| 1024×256×64 | 11.10 | 14.09 | 8.98 | 2.14 |
| 71×479×503 | 10.96 | 14.28 | 8.88 | 1.56 |
| 128×256×768 | 12.57 | 16.21 | 10.52 | 2.22 |

All values are GCUPS. The one-step control is the maximum of separate H/E
phases and the two tested one-step rolling tiles. Four-step values similarly
take the best of three tested spatial tiles; the two-step column deliberately
uses the same spatial tile in every row. The best individual two-step result
on the irregular domain was 15.23 GCUPS with 32×8×8. No exhaustive optimum is
claimed for any temporal depth.

Matched-tile controls also show that this is a temporal benefit: changing
32×8×8 from one to two steps improves all four domains by 11–39%. For 32×8×16,
the corresponding gain is 26–50%. Thus the benefit is not solely attributable
to choosing a different spatial tile.

[All 48 bare configurations and resource data](rtx3090-2026-09-18-bulk-temporal/summary.csv).
Every large final field matched the separate-phase implementation exactly.
There are 256 timesteps, three warmups and six shuffled timing rounds per
variant. CUDA events measure graph execution; input reset copies, allocation,
validation, and graph construction are outside the timed interval. These are
bare GPU-kernel throughputs, not public-API or realistic CPML/monitor results.
Inputs are deterministic random float32 fields, with seed 20260918. The domains
contain 16.8–25.2 million useful cells, and halos are excluded from GCUPS.
Peak sampled temperature was 79°C and minimum free GPU memory was 11.77 GiB
during the four large bare runs; the power limit stayed at 370 W.

The independent CPU checks and full two-domain memcheck passed with zero
errors. Bounded racecheck on 9×9×9 for nine steps covered all 12 variants and
reported zero hazards/errors/warnings. The larger racecheck was interrupted
because of instrumentation cost; its partial output is not counted as a pass.

The rolling design stores only the planes needed by successive stages, rather
than the previous prototype's entire 3D halo tile. Example maximum resident
block counts from CUDA's occupancy API, with 256 threads per block:

| Variant | Dynamic shared bytes/block | Registers/thread | Maximum resident blocks/SM |
| --- | ---: | ---: | ---: |
| One-step 32×8×8 | 7,128 | 56 | 4 |
| Two-step 16×8×16 | 13,008 | 55 | 4 |
| Two-step 32×8×16 | 24,528 | 55 | 4 |
| Four-step 32×4×16 | 49,056 | 56 | 2 |
| Eight-step 8×4×32 | 75,840 | 56 | 1 |

The compiler reports zero local memory/stack usage for these kernels. These
occupancy values are resource limits, not measured achieved occupancy.
Higher temporal depth expands the transverse halo, adds redundant work, and
requires more shared state; the tested four/eight-step variants lose enough
parallelism that reduced global writes do not deliver a net win. This does
not rule out other deeper temporal algorithms or tile families.

The z loop currently executes the full selected spatial depth in partial
tiles, masking final writes outside the logical domain. That avoids changing
the physical domain but wastes some computation, especially at depth 32 for
the 71-cell dimension. Shortening the last tile's z loop is a possible further
optimization; allocating inactive cells would not eliminate that work.

## Consequence for the production backend

The positive bare result supports developing the rolling **two-step** approach
further. It does not make `BEAMZ_CUDA_TEMPORAL_STEPS=2` faster today: that flag
still selects the earlier full-tile CPML prototype. The new rolling kernels
are standalone experiments only.

Integration would need the real staggered field layouts and material
coefficients, per-substep source injection, CPML coupling, and every requested
monitor sample. To preserve the bare result's advantage, it should publish
only the intermediate values actually required by the coupling band and
monitors, rather than unconditionally exporting the full intermediate volume.
That integration still requires complete-state validation and realistic timing.
Consistent 9 GCUPS for arbitrary realistic simulations remains unachieved.

## Reproduction

CPML paired study:

```sh
PYTHONPATH=. python scripts/benchmark_cuda_cpml_tiles.py \
  --output /tmp/beamz-cpml-tiles --rounds 6
```

Bare stencil:

```sh
nvcc -std=c++17 -arch=sm_86 -lineinfo \
  cuda/tests/temporal_bulk_benchmark.cu -o /tmp/beamz-bulk
/tmp/beamz-bulk --check
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/beamz-bulk --check
compute-sanitizer --tool racecheck --error-exitcode 1 /tmp/beamz-bulk 9 9 9 9 1
/tmp/beamz-bulk 128 256 512 256 6
```

The local build used CUDA 13.3.73 and `--allow-unsupported-compiler` for the
available host compiler. No fast-math flag is enabled. No H100 measurement or
production integration of the new rolling temporal prototype is claimed.
Nothing was pushed.
