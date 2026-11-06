# GDSFactory components

BeamZ can prepare a GDSFactory component directly for local CPU or GPU FDTD.
The workflow lives in `beamz.design.gdsfactory`; BeamZ does not import or depend
on `gds_fdtd`. Install the released package with its optional layout dependency:

```bash
pip install "beamz[gdsfactory]"
```

`prepare()` turns the PDK component geometry into ordinary BeamZ `Design` and
`Port` values. It does not run or compile a simulation, so the domain, PML,
mode source, monitors, grid, and runtime remain visible and editable.

```python
from beamz.design.gdsfactory import Settings, prepare

settings = Settings(
    wavelengths=(1.50e-6, 1.60e-6),
    wavelength_points=21,
    xy_padding=3.0e-6,
    pml_thickness=1.0e-6,
)
setup = prepare(
    "straight",
    settings=settings,
    component_settings={"length": 10.0},
)

assert {port.name for port in setup.ports} == {"o1", "o2"}
simulation = setup.simulation_for("o1")
assert simulation.sources[0].signed_direction == setup.port("o1").signed_direction
```

Use the pre-run inspection tools before choosing execution resources:

```pycon
>>> setup.preview()                 # geometry, PML, source and monitor placement
>>> setup.estimate_resources()["grid_cells"]
123456
```

Run a named S-matrix locally when the mesh and source placement are satisfactory:

```pycon
>>> result = setup.run_sparameters(termination=None)
>>> result.sparameters.s_matrix[("o2", "o1")]
array([...])
>>> result.check_reciprocity()["max_abs_error"]
...
>>> result.check_passivity()["o1"]
array([...])
```

`S[out, in]` is the outgoing modal amplitude at the output port datum divided
by the incident modal amplitude at the input datum. BeamZ stores source/monitor
settings, layer, PML, frequency grid, and port order in `result.provenance`.
The first release supports axis-aligned optical ports, fundamental TE-like
mode 1, and local 3D FDTD. Use `setup.simulation_for(...)` plus
`Simulation.updated_copy(...)` to make non-default numerical choices explicit.

## Physical stack import

Use the same preparation API with explicit process geometry and materials:

```pycon
>>> setup = prepare(component, layer_stack=stack, material_map=materials,
...                 settings=Settings(xy_padding=2e-6, z_padding=1e-6))
```

To inspect or reuse physical geometry, call `import_component(component,
layer_stack=stack, material_map=materials)` first, then `prepare(geometry,
settings=settings)`. Both forms apply padding and port extensions in prepare.
Cached masks and process values remain unchanged; repeated preparation does not
accumulate padding. Reimport to use layout edits; use native Simulation for an
edited Design. `geometry.report` describes levels and port associations, and
`geometry.source_origin` translates local coordinates to the original stack.

Optional `fill_levels=("box", "clad")` supports older PDK background definitions;
newer background metadata/exclusions work automatically. Use `port_levels={"o1":
"upper"}` for ambiguous vertical ports. Explicit metre-valued bounds are an
alternative to padding and require `Settings(xy_padding=0, z_padding=0)`.

Stacks require material_map. Continuous sidewall reference heights, scalar bias,
height profiles and downward extrusion are supported. Tuple bias and conflicting
nonzero sidewall/profile inputs fail explicitly. Callable profiles use a sampled
1 nm error estimate at the default scale. Process values are nominal. No-stack
defaults match between entry points: n_core=3.47 and 1 µm cladding above/below.

## Export geometry

`export_gds` accepts metre-valued `precision` and an optional mapping from
`Material` values to GDS `(layer, datatype)` pairs. The mapping selects the
materials to export; other materials, including clearing polygons, are omitted.
Without a mapping, material layers are assigned automatically as before.
Export the prepared component directly:

```python
from beamz.design import export_gds

gds_path = export_gds(setup.design, "component.gds", precision=1e-9)
assert gds_path.is_file()
```

For optimized geometry, use the `InverseDesignResult` returned by an optimization
run. This is a different result type from the S-parameter result above. Given
that optimization result as `optimized_result`:

```pycon
>>> from beamz import Material
>>> layout = optimized_result.export_design(mode="contour", sample_spacing=10e-9)
>>> export_gds(layout, "optimized.gds", precision=1e-9,
...            layer_map={Material(4.0): (1, 0)})
PosixPath('optimized.gds')
```

Use the permittivity of your actual core material in the mapping. For an exact
thresholded cell mask, use `mode="pixel_exact"` without `sample_spacing`.
Contours and cell footprints share the same physical cell-edge coordinates.
The GDS database is local to the export, so finer precision does not depend on
or alter gdsfactory's active grid. Excessively fine precision that would overflow
GDS coordinates is rejected. Holes are preserved during serialization.

See the [optimization guide](https://github.com/beamzorg/beamz/blob/main/beamz/optimization/README.md#export-topology-to-gds)
for fixed port masks and the reproducible export/reimport simulation example.
