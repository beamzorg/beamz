# GDSFactory import examples

Five synthetic process stacks exercise the shared native/raster importer:

| Case | Features |
| --- | --- |
| `rib_etch` | Derived shallow etch, 300 nm rib, 90 nm slab, 10° walls |
| `stacked_heater` | Si, SiN, oxide, lossy heater proxy, positive/negative taper |
| `sidewall_references` | One mask at four heights; width_to_z of 0, 0.37, 0.5 and 1 |
| `bias_profiles` | Constant growth, tabulated bulge, quadratic callable |
| `hierarchy_holes` | Rotated/translated child, boolean hole, 100 nm sizing |

Run from the repository root:

```bash
uv sync --extra gdsfactory --extra dev --extra test
uv run python -m examples.gdsfactory_imports.run
```

Choose `--case rib_etch` (or another name) and `--output PATH` as needed. The
default output is `.cache/gdsfactory-imports/`: GDS masks, XY/YZ permittivity PNGs,
and a versioned report checking 27 material probes through the public importer.
Failed probes raise an error. Tests cover native/raster equivalence, background
fills, downward extrusion, repeated preparation and modal-source execution.

See the [API guide](../../docs/gdsfactory.md) for one-call preparation and optional
cached geometry import.

Builder coordinates are µm; BeamZ coordinates are metres. LogicalLayer sizing
uses database units, converted explicitly in the hierarchy case. GDS stores masks,
not materials or sidewalls: reapply the same stack after serialization. Plots use
fast volume averaging; probes sample 1 nm cubes at reference quality. These are
geometry examples, not optical convergence or foundry calibration results.

Materials are illustrative constants. The heater proxy has positive permittivity
and conductivity and is not an optical metal model. The callable example uses a
level-like adapter because GDSFactory 8 LayerLevel accepts tabulated profiles only.
Validated with GDSFactory 8.18.2 and 9.51.0. To use version 9 on Python 3.12:

```bash
uv run --python 3.12 --extra test --with 'gdsfactory==9.51.0' \
  python -m examples.gdsfactory_imports.run --output .cache/gdsfactory-imports-v9
```
