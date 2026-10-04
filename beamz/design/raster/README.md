# BeamZ rasterization

This package turns analytic geometry, GDSFactory layer stacks, and closed
triangle meshes into static real tensor material arrays. Permittivity and
permeability may be isotropic, diagonal, or symmetric positive-definite 3x3
tensors; conductivity may be symmetric positive-semidefinite.

## API

```python
import beamz
import beamz.design.raster as raster

scene = raster.Scene(
    materials=(
        beamz.Material(),
        beamz.Material(permittivity=((12.1, 0, 0), (0, 11.8, 0), (0, 0, 10.9))),
    ),
    objects=(
        raster.Object(
            raster.Box((0, 0, 0), (2e-6, 0.5e-6, 0.22e-6)),
            material_id=1,
        ),
    ),
)
grid = raster.Grid.uniform(
    (0, -0.5e-6, -0.1e-6),
    (2e-6, 1e-6, 0.4e-6),
    (80, 60, 8),
)
result = raster.rasterize(scene, grid)
```

The public package exports:

- `Scene`, `Object`, and scalar/diagonal/symmetric-tensor `Material`
- `Grid`, including explicit nonuniform edges and `Grid.uniform()`
- boxes, spheres, cylinders, extruded polygons, tapered polygons, and meshes
- `RasterOptions`, `RasterResult`, and `CompiledScene`
- `compile_scene()`, `rasterize()`, and `inspect_mesh()`

`RasterOptions` has three choices: `quality` (`fast`, `balanced`, or
`reference`), `smoothing` (`volume`, `farjadpour_diagonal`, or
`farjadpour_full`), and `components` (`all`, `two_dimensional_tm`, or
`two_dimensional_te`). Standalone and design rasterization both default to
`farjadpour_diagonal`. This changes the previous standalone default of
`farjadpour_full`; set `smoothing="farjadpour_full"` explicitly to retain that
policy. Intrinsic off-diagonal material coefficients require `farjadpour_full`
or `volume`; diagonal smoothing rejects them instead of discarding them.
Adaptive tolerances and threading remain internal. Farjadpour modes use the generalized
local-interface tensor transform for permittivity and permeability;
conductivity remains volume averaged.

`RasterResult` contains immutable grid edges, cell-centered `tensors`, summary
diagnostics, and constitutive `yee_tensors` integrated independently over the
Ex/Ey/Ez and Hx/Hy/Hz dual volumes. Full smoothing also integrates epsilon at
the grid nodes shared by the off-diagonal electric terms. Tensor arrays use one
leading component for isotropic materials, three `(xx, yy, zz)` components for
diagonal materials, and six `(xx, yy, zz, xy, xz, yz)` components for full
symmetric materials. It does not produce dense material-ID, boundary-mask, or
error arrays.

Farjadpour smoothing is applied only when the surface patches crossing one Yee
support form a reliable, sign-invariant lamination axis. Boxes and straight
extruded polygons are partitioned at their z boundaries, resolved by priority
then object ID, and unioned by material before integration. Their disjoint
volumes are integrated exactly (up to polygon-clipping roundoff), including
holes and different extrusion heights. Only exposed material boundaries
contribute normals: duplicate objects and hidden core/slab seams do not disable
smoothing. Equal material-table entries are treated as one physical material.

Polygon Boolean operations use the `geo` integer overlay lattice (approximately
29 bits per XY half-extent). Input axis coordinates within a scale-derived
clipping-roundoff tolerance are restored after each operation to keep aligned
faces aligned. This is a clipping precision limit, independent of the adaptive quality preset. If distinct
input axis coordinates are closer than the restoration tolerance can distinguish,
the rasterizer retains the original geometry and uses the planar partition or
adaptive path instead of allowing the Boolean union to erase thin layers.
Regression tolerances of `2e-6` on float32 constitutive outputs cover clipping and output roundoff for the tested
geometries, rather than promising a universal fraction-error bound.

Corners, multiple nonparallel exposed interfaces, unresolved geometry, and
unsupported curved/tapered overlaps still fall back to tensor volume averaging. Identical fully occluded primitives are removed before adaptive
integration; an unrelated curved object does not disable exact extrusion
integration elsewhere. Adaptive disagreement remains an estimate, not a strict
error bound. Summary diagnostics count sampled supports across all requested
components and cells, not physical-cell percentages, and do not allocate dense
diagnostic fields. Scene hashes retain the original input representation;
cached results from the older ownership algorithm are invalidated.

The original-geometry fallback shares a boundary roundoff policy between
single-object and planar multi-object interface classification. Its
normal-evidence query is inset by eight machine epsilons
times the larger of each axis's support width and endpoint magnitudes. Including
coordinate magnitude keeps the inset representable after translation. An inset
that would collapse an axis is not applied. This avoids spurious corners from
rounding at support edges; material-volume integration and partition planes
retain their original coordinates.

Compile once when rasterizing one scene on several grids:

```python
compiled = raster.compile_scene(scene)
coarse = compiled.rasterize(coarse_grid)
fine = compiled.rasterize(fine_grid)
```

Passing `cache_directory=` enables atomic, schema-versioned NPZ caching with
corrupt-cache recovery.

## Standalone versus simulation use

Standalone rasterization accepts uniform and nonuniform rectilinear grids.
BeamZ's FDTD engine accepts the same realized rectilinear grid and applies the
local Yee edge, face, and volume metrics in its curl updates and CFL limit. It
extracts the target diagonal from each electric support tensor. For
`farjadpour_full`, it compiles the inverse diagonal at each E support and the
inverse cross terms at shared grid nodes, then uses the paper's
average-multiply-average coupling. Full coupling currently requires zero
electric conductivity and unit permeability; use CPML rather than a conductive
sponge PML with that mode.

Imported scenes use the same simulation workflow as BeamZ designs; the
solver-specific conversion stays internal:

```python
import beamz
from beamz.design.raster import Grid
from beamz.design.raster.importers import from_mesh

scene = from_mesh("device.msh", material=beamz.Material(permittivity=12.0))
raster_grid = Grid.uniform(minimum, maximum, shape)
simulation = beamz.Simulation(
    scene=scene,
    raster_grid=raster_grid,
    run_time=1e-12,
)
```

The simulation domain and physical coordinates are taken directly from the
grid's edge arrays.

For standalone inspection, `scene.rasterize(raster_grid)` returns the general
`RasterResult`. `MaterialGrid.from_raster_result()` remains an advanced explicit
boundary for callers that need to retain or transform that result themselves.

`Design.rasterize()` returns a `MaterialGrid` directly, always uses the Rust
engine, and selects the reduced TMz or TEz work set requested by the simulation.
Pre-sampled
spatial coefficients enter `Simulation` directly as `MaterialGrid`; there is no
second Python geometry rasterizer or engine-selection switch.

Normal simulations use diagonal Farjadpour smoothing. Full lossless coupling can
be requested explicitly:

```python
import beamz

simulation = beamz.Simulation(
    design=design,
    raster_options=raster.RasterOptions(smoothing="farjadpour_full"),
    run_time=1e-12,
)
```

The compiler consumes those native Yee arrays directly. Full-tensor
conductivity, full permittivity combined with conductivity, and non-unit
permeability fail with a capability-specific message. Conductive diagonal
materials propagate in FDTD but are rejected for `ModeSource` until the mode
bridge represents their frequency-dependent complex permittivity. The current
mode-source and mode-monitor bridge rejects explicitly supplied nonuniform
rectilinear grids. `GridSpec.auto()` selects a compatible uniform grid when a
mode source, mode monitor, or Gaussian beam is present. `GaussianSource`,
`CustomSource`, and field/flux monitors work on graded grids. The current 2D and public
convenience mode solvers use the cell-tensor approximation for diagonal
Farjadpour grids and reject full off-diagonal coupling; the 3D launch planner
retains its Yee-aware refinement path.

## Importers

```python
from beamz.design.raster.importers import (
    from_gdsfactory,
    from_mesh,
    from_mesh_arrays,
    repair_mesh,
)
```

- `from_gdsfactory()` reads PDK layer stacks, derived layer regions, sidewalls,
  material maps, and callable or tabulated `z_to_bias` profiles.
- `from_mesh()` reads meshio-supported surfaces such as STL and extracts closed
  boundaries from Gmsh tetrahedral physical regions.
- `from_mesh_arrays()` accepts raw vertices and triangles.
- `repair_mesh()` uses optional trimesh operations and returns an auditable
  immutable report.

All importers return reusable `Scene` objects. GDSFactory, meshio, and trimesh
remain optional dependencies.

PDK material names have no implicit production constants. Pass `material_map`
with the intended values. `use_builtin_materials=True` is an explicit opt-in to
approximate nondispersive values near 1.55 µm.

Meshes must be closed, consistently oriented, manifold, nondegenerate, and free
of self-intersections. `inspect_mesh()` and scene compilation share the same
native scale-aware validity predicate.

Connected triangle meshes use exact polyhedral overlap integration for a
single candidate object, as boxes and extruded polygons do. Mesh resolution
still determines how closely a triangulated curved surface matches the
original CAD geometry. Multiple objects also use exact integration when all
surfaces crossing a sampling volume lie on parallel planes. This includes
adjacent core/cladding regions and overlapping planar layers; each interval's
material is selected using object priority and ID. Explicit cladding therefore
matches the equivalent background material at a shared planar interface.
Scene JSON preserves floating-point coordinates exactly, so loading a scene
does not move grid-aligned faces by a rounding step.
Nonparallel planar junctions also receive exact volume fractions when the local
arrangement has at most 12 distinct planes and 256 convex pieces. They retain
volume averaging rather than inventing one interface normal. This path also
handles simple disconnected/nested shells with parity containment, independent
of shell winding. Each shell is validated separately, so opposite signed volumes
do not cancel the validity check. More complex arrangements and curved analytic
interfaces retain adaptive integration; their results depend on `quality`.
The complexity cap bounds partition work; it is not an accuracy guarantee for
adaptive results. Check diagnostics and convergence when that fallback is used.
Exact partitions retain distinct geometric planes and cut coordinates. A robust
orientation predicate recognizes coplanar triangle faces even when their rounded
plane equations differ; it does not use a distance tolerance that could erase
thin layers on stretched grids. Distinct near-coincident planes can still add
partition work and reach the existing cap.
Adaptive integration compares octant occupancy with an independent parent-center
sample. An octant checkerboard subset is not a valid convergence check for
axis-invariant extrusions. The estimator remains a sampling estimate, not a
rigorous bound on geometric or electromagnetic error.

Each imported physical region has its own compact vertex array and bounds.
Faces that only touch a sampling-volume boundary do not disable interface
smoothing. Cached results from the earlier rasterization policy are invalidated.

When comparing import routes, use the same physical XYZ grid edges, materials,
`smoothing`, and dimensional `components` (including polarization in 2D).
Gmsh coordinates become `(points - coordinate_origin) * unit_scale`, with
`coordinate_origin=(0, 0, 0)` and `unit_scale=1.0` by default. The origin is in
file units and is subtracted before scaling to allow explicit alignment and
recentering. It cannot recover precision already lost in the source file;
use `1e-9` for nanometres or `1e-6` for micrometres. The GDSFactory scene adapter
defaults to micrometres. The higher-level GDS design importer translates to a
local design origin and reports `world_origin`; account for that translation
when comparing with a mesh. Scalar design rasterization rounds the domain up
to whole cells and extends structures ending on its high boundary through that
padding. Direct scene rasterization uses exactly the supplied grid and geometry.
For equivalence comparisons, choose extents divisible by the spacing or explicitly
match the padded geometry. These coordinate policies are not inferred from meshes.

`from_mesh()` accepts first-order `triangle` and `tetra` cells. Unsupported
surface/volume types, including `triangle6`, `tetra10`, quads and hexahedra,
raise `ValueError` with conversion guidance. Higher-order cells must first be
tessellated to the desired geometric accuracy; corner-only linearization is
no longer implicit. Vertex/line annotations are ignored. Surface cells alongside
tetrahedra must refer to tetrahedral faces and are treated as annotations;
standalone surface geometry mixed into a volume file is rejected rather than
lost. Volume materials come from volume physical tags.

Tetrahedral extraction rejects duplicate and degenerate elements, faces shared
by more than two cells, and same-sided cells sharing a face. Native validation
then checks the extracted surface. These checks do not constitute a complete
volumetric mesh-overlap validator.

Use `priorities={"core": 10, "cladding": 0}` to specify ownership of overlapping
regions independently of tag order. Keys may be physical names or integer tags;
names take precedence if both are supplied. Larger priorities win, followed by
object ID on ties. Unspecified priorities preserve increasing physical-tag
order for compatibility; unknown keys and non-integer/out-of-range values are
rejected. Materials still require explicit mappings.

```python
scene = from_mesh(
    "device.msh",
    materials={"core": core_material, "cladding": cladding_material},
    priorities={"core": 10, "cladding": 0},
    coordinate_origin=(1000, 2000, 0),  # source-file coordinates
    unit_scale=1e-9,
)
```

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
env -u CONDA_PREFIX uv run maturin develop
uv run pytest tests/unit/raster \
  tests/integration/test_native_design_rasterization.py
```

Measured import comparisons, analytical checks, negative controls, and rerun
commands are recorded in the [equivalence validation report](../../../docs/raster-equivalence-validation.md).

### Spatial ownership inspection

Development builds can inspect selected cell and Ex/Ey/Ez supports without
changing the public API or allocating diagnostic volumes in ordinary runs.
The ignored Rust test `write_spatial_ownership_diagnostics` accepts JSON with
`scene`, `grid` (`x_edges`, `y_edges`, `z_edges`), and `supports` (component name
and `[x, y, z]` index). Set `BEAMZ_RASTER_INSPECTION_INPUT` and
`BEAMZ_RASTER_INSPECTION_OUTPUT` to the input and JSONL output paths, then run:

```bash
cargo test --release -p fdtd-raster-core write_spatial_ownership_diagnostics -- --ignored
```

Each output record gives support bounds, material fractions, the resolved normal,
smoothing status, fallback reason, and epsilon. The crossing-specific
`scripts/inspect_raster_ownership.py` selects XY/XZ supports, validates fractions
against GEOS, and plots the physical nonuniform support widths. See
[issue #242 evidence](../../../tests/differential/results/issue-242/README.md)
for the selection, denominators, measured S parameters, and reproduction commands.
