# Raster import equivalence: numerical evidence

The regression fixtures now agree across Gmsh, GDS, and internal Design imports.
This report records tests of the working-tree fixes, using the locally rebuilt
Rust raster extension on macOS ARM64 with Python 3.13.9. Runs used one worker and
limited CPU thread counts. The final Python raster and selected integration
suite passed **738 tests, with no skips**, in 355.39 seconds. All **51 Rust tests**
also passed. Clippy (warnings denied), Rust formatting, Ruff, and whitespace
checks passed.

## Propagation, including a negative control

The [propagation regression](../tests/integration/test_mesh_design_equivalence.py)
writes and reads real `.msh` and `.gds` files. It compares them with a native
Box (3D) or Rectangle (2D), using the same grid and simulation settings:

- A 370 nm dielectric slab in a 1 µm domain, with relative permittivity 4,
  conductivity 0.1, and background relative permittivity 1.
- A 6 × 6 × 6 grid in 3D or 6 × 6 grid in 2D; default diagonal smoothing.
- A Gaussian spatial source with a one-step temporal impulse, PEC boundaries,
  and timestep `0.35 * dx / (c * sqrt(dimensions))`.
- 128 timesteps, comparing every value of all six field arrays at steps
  32, 64, 96, and 128. Every reference checkpoint is finite and nonzero.
- Each case runs with both implicit cladding and a separately tagged adjacent
  cladding region in the Gmsh file.

| Case | Gmsh relative L2 error | GDS relative L2 error | Changed-material relative L2 error |
| --- | ---: | ---: | ---: |
| 3D, implicit cladding | 0 | 0 | 0.518234 |
| 3D, explicit cladding | 0 | 0 | 0.518234 |
| 2D TM, implicit cladding | 0 | 0 | 0.825346 |
| 2D TM, explicit cladding | 0 | 0 | 0.825346 |
| 2D TE, implicit cladding | 0 | 0 | 0.829567 |
| 2D TE, explicit cladding | 0 | 0 | 0.829567 |

Relative L2 means `norm(imported - design) / norm(design)` over the concatenated
checkpoint field arrays. Maximum absolute error was also **zero** for both
import routes in every case. The negative control changes slab permittivity
from 4 to 5 and must exceed 1% relative error. Its measured 52–83% mismatch
shows these comparisons respond to a real material change. These norms compare
representations; they are not electromagnetic energy measurements.

## Independent analytical check

The [analytical regression](../tests/unit/raster/test_geometry_equivalence.py)
checks imported oblique triangular prisms without using another rasterizer as
its reference. Inside a unit square, the material occupies `x + y <= a`, so
its exact area fraction is `f = a² / 2`. The prism extends beyond the grid's
z bounds, leaving one interior planar interface.

For relative permittivities 4 and 9, the exact local laminate formula is:

```text
parallel = 4*f + 9*(1-f)
normal   = 1 / (f/4 + (1-f)/9)
n        = (1, 1, 0) / sqrt(2)
epsilon  = parallel*I + (normal-parallel)*outer(n, n)
```

The test checks full tensors against this expression, diagonal tensors against
its diagonal, and volume smoothing against the arithmetic mean. It covers
three cuts (0.31, 0.73, 1), three coordinate scales (1e-9, 1, 1e6), and all three
smoothing modes, with translated origins. **All 27 cases passed** at relative
tolerance 2e-6 and absolute tolerance 2e-7. None used adaptive sampling.

Additional regressions cover tetrahedral Gmsh physical regions, distant regions,
nonuniform grids, holes, nonconvex polygons, reversed winding, retriangulation,
random convex-polyhedron volume conservation, adjacent oblique materials,
painter order, and stale raster caches.

## Reproduce

Build the current native extension using the instructions in the
[raster README](../beamz/design/raster/README.md#development).
The Python environment needs the test dependencies, including `meshio` and
`gdsfactory`; absent optional dependencies skip the corresponding tests.
No tests were skipped in the recorded propagation or analytical runs.

```sh
env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  nice -n 10 python -m pytest --no-cov -q \
  tests/unit/raster \
  tests/integration/test_mesh_design_equivalence.py \
  tests/integration/test_native_design_rasterization.py \
  --junitxml=raster-proof.xml -o junit_family=legacy
```

The JUnit XML includes measured errors as properties of each propagation test.
For the Rust tests, run sequentially:

```sh
env CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 nice -n 10 \
  cargo test -p fdtd-raster-core -- --test-threads=1
```

These results establish equivalence for the tested geometry and settings. They
do not establish mesh convergence for curved CAD surfaces, long-duration
stability of arbitrary simulations, or correctness of a particular external
mesh that has not been tested. A triangulation that changes the physical shape
can still legitimately produce different coefficients.


## Additional hardening and measured performance

The [robustness regressions](../tests/unit/raster/test_mesh_robustness.py) add
36 cases covering unsupported/mixed elements, malformed tetrahedra, region
priority, coordinate transforms, thin translated features, circular-cylinder
convergence, planar junctions, and disconnected/nested shells.

Two tests exposed further correctness defects before their fixes:

- A perpendicular junction with relative permittivities 1, 4 and 9 should have
  cell-average permittivity 6.3129. Adaptive fast/balanced results were 5.75 and
  5.875. The bounded planar partition now matches 6.3129 to within 2e-6 at all
  qualities. Oblique junctions also match independent triangle-area formulas
  across three coordinate scales.
- Two disjoint, equal-volume shells with opposite winding were rejected as
  zero-volume geometry. Per-shell validation now accepts them; disjoint and
  nested fixtures produce the expected parity volumes with either winding.

Circular-cylinder meshes with 12, 24, 48 and 96 sides match their exact polygon
volumes within 2e-7. Their errors relative to the true circle decrease by about
four when edge count doubles, confirming geometric convergence rather than
claiming a coarse polygon is identical to a curve.

The propagation test now also covers a perpendicular material junction for
**512 timesteps**, with checkpoints at 128, 256, 384 and 512. All six junction
cases (3D/TM/TE, each with implicit/explicit cladding) had zero measured Gmsh and
GDS field error relative to Design. Changed-material controls produced relative
L2 errors of 0.377633 (3D), 0.483639 (TM), and 0.563161 (TE). These supplement the
six original slab cases. They use the same FDTD solver and do not independently
validate its Maxwell discretization.

A bounded sequential benchmark writes actual Gmsh volume files, imports them,
compiles their extracted surfaces, and rasterizes a padded 8×8×8 grid. It checks
the reconstructed unit-cube volume and records peak process memory. The native
mesh-validation duplicate pass was removed, and tetrahedron boundary extraction
was vectorized with explicit topology checks.

| Tetrahedra | Boundary triangles | Import time | Compile time | Raster time | Peak process RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,072 | 768 | 0.0053 s | 0.0021 s | 0.0193 s | 190 MiB |
| 24,576 | 3,072 | 0.0408 s | 0.0082 s | 0.0406 s | 210 MiB |
| 82,944 | 6,912 | 0.1474 s | 0.0200 s | 0.0772 s | 269 MiB |

Absolute reconstructed volume error was 3.70e-9 in every case. On the 3,072-cell
comparison, boundary extraction took 0.00483 s versus 0.16053 s for the previous
algorithm (**33.2× faster**), with identical boundary faces and a valid oriented
surface. These are single-run measurements on this machine, not timing guarantees.
RSS includes the interpreter and dependencies and is the cumulative process
high-water mark across cases. It does not establish million-element capacity
or explain the earlier computer crash.

Raw measurements: [mesh-import-benchmark.json](mesh-import-benchmark.json).
Reproduce with a freshly built native extension and installed meshio:

```sh
env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  nice -n 10 python scripts/benchmark_mesh_import.py \
  --sizes 8 16 24 --output mesh-import-benchmark.json
```

Unsupported/high-order geometry now fails explicitly; it still requires external
tessellation. Partitioning is capped at 12 distinct planes and 256 pieces, with
adaptive integration beyond those limits. Coordinate recentering cannot restore
precision absent from a source file. The original reported mesh is not present
in this checkout, so a device-specific comparison remains unavailable.


## Review regressions: stretched supports and adaptive aliasing

The [partition-accuracy regressions](../tests/unit/raster/test_partition_accuracy.py)
cover both review findings across coordinate scales 1e-9, 1 and 1e6. The stretched
support case also runs under all three cyclic axis permutations. Its thin layer
must contribute to both permittivity and conductivity. Previously, approximate
plane deduplication erased it, producing 1.5 instead of approximately 2.0 for
permittivity. Both the parallel and general partition paths now retain distinct
cut coordinates; geometrically coplanar faces are deduplicated using an exact orientation predicate.

The adaptive estimator previously compared octants with their checkerboard
subset, which agrees automatically for axis-invariant extrusions. It now uses
an independent parent-center sample. A Rust regression checks all three axes
and the nine-query sampling budget; existing depth limits and the fast preset's
work-budget regression remain intact.

For the unit-cell cylinder with radius 0.4 and permittivity 4, the analytical
cell coefficient is 2.507964474. The old balanced result was 3.25 with zero
estimated error. New tests require absolute coefficient error below 0.04 for
balanced and 0.01 for reference, and a nonzero reported sampling-error estimate.
These are fixture-specific numerical tolerances, not general accuracy guarantees.
Raster cache schema 8 invalidates coefficients produced by the earlier policies.


Measured rerun of the original review cases:

| Case | Before | After | Analytical target |
| --- | ---: | ---: | ---: |
| Thin layer on stretched cell | 1.5 | 2.0 | 1.999999995 |
| Cylinder, balanced | 3.25 | 2.5 | 2.507964474 |
| Cylinder, reference | 2.5234375 | 2.50390625 | 2.507964474 |

Both cylinder runs now report nonzero sampling error. The 96-sided mesh remains
at 2.5068882, reflecting its polygonal approximation and exact polyhedral volume
integration. Adaptive coefficients remain approximate and depend on quality.
Before the cross-format expansion, the rerun passed 449 Python tests without skips
and 51 Rust tests; Clippy,
Rust formatting, Ruff, and whitespace checks passed.


## Independent randomized and three-dimensional oracles

The expanded tests check each cell-centered and staggered Yee coefficient for
permittivity, permeability, and conductivity against separately computed volumes:

- **48 randomized prism cases:** eight reproducible seeds, mesh/polygon geometry,
  three coordinate scales, nonuniform grids, cyclic axis rotations, shuffled and
  reversed faces, and overlapping regions with priority/ID ties. Shapely computes
  material ownership by polygon intersection and subtraction. Every coefficient
  must agree within relative tolerance 3e-6 and absolute tolerance 3e-7, without
  adaptive fallback in these bounded fixtures.
- **Six tetrahedral-file cases:** ASCII and binary Gmsh files at three scales,
  27 physical regions, shuffled cells/tags, and reversed tetrahedron winding.
  Independent box-intersection integrals check every constitutive support,
  including supports extending into background outside the mesh.
- **12 arbitrary 3D overlap cases:** rotated tetrahedra at three scales. The
  reference enumerates feasible intersections of three halfspace planes and
  measures their convex hull using SciPy/Qhull, independently of BeamZ's face
  clipping. A separate test checks that oracle against unit-cube and simplex
  volume formulas.
- **Three curved-file cases:** full and domain-clipped circles checked by
  one-dimensional quadrature of circular chords. The tests cover actual Design,
  GDS and Gmsh paths. Design.Circle uses an explicit 96-point polygon; equality
  checks compare matching polygons, while continuous-circle comparisons use
  approximation tolerances. Refining meshes from 24 to 96 sides must reduce
  integrated geometric error by more than eightfold.
- **Nine additional thin-layer cases:** layers 100 times thinner than the earlier
  fixture, across all three axis permutations and scales. A Rust test separately
  verifies that an exact coplanarity check distinguishes a one-ULP displacement.

These tests exposed an additional numerical problem: equivalent triangle planes
could receive different rounded equations, and repeated clipping near an existing
face could fail volume conservation, triggering adaptive fallback. The fix retains
three original points per plane and uses a robust orientation predicate to recognize
exact coplanarity. This avoids re-clipping duplicate planes without introducing a
proximity tolerance that merges distinct interfaces. Raster cache schema 9 rejects
results from the earlier implementation.

Tests are in [test_independent_raster_oracles.py](../tests/unit/raster/test_independent_raster_oracles.py),
[test_3d_overlap_oracles.py](../tests/unit/raster/test_3d_overlap_oracles.py), and
[test_curved_import_oracles.py](../tests/unit/raster/test_curved_import_oracles.py).

## Cross-format files and component entry points

[test_cross_format_equivalence.py](../tests/unit/raster/test_cross_format_equivalence.py)
adds 173 cases that write and read actual files:

| File family | Tested encodings / elements |
| --- | --- |
| STL | ASCII and binary triangle surfaces |
| OBJ, OFF | Triangle surfaces |
| PLY | ASCII and binary triangle surfaces |
| VTK | ASCII/binary surfaces and binary tetrahedra |
| VTU | Triangle surfaces and tetrahedra |
| Gmsh 2.2 | ASCII and binary tetrahedra; tagged surfaces |
| Medit, Abaqus | Tetrahedra |
| GDS | Normalized component imports and transformed hierarchical multilayer layouts |

The 135 mesh-file cases cover boxes, sheared solids, and disconnected solids at
three scales. Vertex and cell order are shuffled and winding is reversed. They
check bounding coordinates, surface area, enclosed volume, and all cell/Yee
permittivity, permeability, and conductivity arrays under all three smoothing
modes. Boxes are additionally compared with analytical Box geometry. Boundary
measure tolerances are 1e-12 for these exactly representable file coordinates;
coefficient tolerances are relative 2e-6 and absolute 2e-7.

Twelve component cases compare active-PDK cell names, callable factories,
component objects, GDS files, direct layer-stack imports, STL, and internal
Design geometry, with translated layouts, both polygon-unification settings,
and all three quality settings. The high-level importer normalizes layout
coordinates and adds padding; the direct layer-stack path retains layout
coordinates. The tests explicitly align those coordinate frames.

Eighteen hierarchical cases compare a rotated/translated two-layer component,
its GDS round trip, tagged Gmsh/VTK/VTU surfaces, and analytical boxes. They verify
material overlap priorities independently of layer-stack iteration order and
exercise all three smoothing modes. VTK/VTU fixtures explicitly carry
`gmsh:physical` tags; arbitrary format-specific metadata is not automatically
interpreted as material regions.

Eight additional cases check decimal-coordinate STL precision, rejection of
real hexahedral VTK/VTU/Abaqus files, and explicit material requirements for
untagged OBJ/PLY/STL. Binary STL stores float32 coordinates, so its exact
reference is the quantized geometry; requested-geometry agreement is also
checked within the coefficient tolerance. Tests narrowly filter meshio's
known ASCII-STL uint32 header-probe overflow warning under NumPy 2.

The component tests exposed a smoothing discontinuity: center/size-derived box
coordinates differed from polygon vertices by a few ULPs at a Yee support
boundary, causing an incidental boundary face to turn a single interface into
a corner. The initial correction used an interface-normal query inset of eight
machine epsilons times each axis's own support width, leaving integration bounds
and partition planes unchanged. Cache schema 10 invalidated earlier coefficients.
The follow-up correction below extends this policy to translated and multi-object
geometry.

Equivalent geometry means equivalent boundaries and constitutive coefficients
within the stated tolerances, not identical vertex numbering or triangulation.
This matrix validates the listed readers and fixtures, not every format meshio
can open or arbitrary CAD geometry. STEP solids still need external tessellation.

## Object-count and translated-boundary corrections

The follow-up review found two gaps in the first boundary-roundoff correction:
partitioned multi-object geometry bypassed the evidence policy, and the
width-only inset rounded to zero at large absolute coordinates. Both paths now
use boundary-aware classification. Planar multi-object classification certifies
a common normal from the inset support, while integration still retains every
original partition plane. Each axis's evidence inset includes its endpoint
magnitudes as well as its width; an inset that would collapse an axis is not
applied. Unsupported curved multi-object geometry retains conservative fallback.
Cache schema 11 invalidates results computed before these corrections.

[test_interface_boundary_roundoff.py](../tests/unit/raster/test_interface_boundary_roundoff.py)
adds 114 regression cases:

- 108 combinations of three axes, three scales, positive/negative/zero offsets,
  Box/triangle-mesh representation, and diagonal/full smoothing. Each checks
  single, duplicated, and split geometry against an aligned analytical slab at
  every cell/Yee coefficient. Normal epsilon is approximately 1.6 and normal mu
  is approximately 4/3; tolerance is relative 2e-6 and absolute 2e-7. Scalar,
  diagonal, and full tensor storage are expanded before comparison.
- Six controls verify that a resolved corner still produces volume averaging
  for single and duplicated objects at all three offsets.

The original review's duplicate-object case changed epsilon from 1.6 to 2.5;
its large-offset case also produced 2.5. The new cases retain approximately 1.6
for every tested representation. They require zero adaptive fallback. These
checks complement the existing high-contrast thin-layer volume tests; they do
not assert accuracy for geometry whose dimensions approach coordinate precision.
