# Material library

BeamZ bundles named dispersive fits that can be used directly in structures and
simulations. All data ship with BeamZ and work offline.

```python
import beamz as bz
from beamz import material_library

silica = material_library["SiO2"]["Palik_LowLoss"]
aluminum = material_library["Al"].medium  # default: Rakic1995
structure = bz.Box(center=(0, 0, 0), size=(1e-6, 1e-6, 40e-9), material=aluminum)
```

Library lookups return ordinary immutable `PoleResidue` materials. Their
`frequency_range` is in Hz; `eps_model(frequency)` evaluates complex relative
permittivity. Use the fit within its stated frequency interval.

| Material | Variants | Default |
| --- | --- | --- |
| `SiO2` (silica) | `Palik_LowLoss` | `Palik_LowLoss` |
| `SiN` (silicon nitride) | `Horiba` | `Horiba` |
| `aSi` (amorphous silicon) | `Horiba` | `Horiba` |
| `Al` (aluminum) | `Rakic1995`, `Rakic1995_CMOS` | `Rakic1995` |
| `CMOS_RGB` (synthetic filters) | `red`, `green`, `blue` | `green` |

The initial collection contains eight fits.
The RGB filters are hypothetical example materials, not measured filter spectra.

## Inspect variants and optical constants

```python
print(tuple(material_library))
print(tuple(material_library["Al"].variants))
variant = material_library["CMOS_RGB"].variants["red"]
print(variant.description, variant.source, variant.license)
print(variant.medium.frequency_range)
wavelength_m, n, k = variant.nk_data.T
```

`nk_data` returns a fresh NumPy array with columns wavelength [m], n, k for the
synthetic filters, or `None` for fits without bundled samples. It includes the
filter fitting baseline `k_offset = 0.01`. Changing the returned array does not
change library data. Library and variant mappings are read-only; construct your
own `PoleResidue` or use `updated_copy` to customize a material.

## Aluminum versions and reproducibility

The `Rakic1995` default is a five-pole aluminum fit.
`Rakic1995_CMOS` preserves the older four-real-pole fit embedded in the published
CMOS example and is used explicitly by the BeamZ CMOS notebook. Choose the
default for new simulations; choose the snapshot variant when reproducing that
example. Changing variants changes the simulated material and requires new results.

Coefficients, source identifiers, and upstream license notices are bundled under
`beamz/material_library/data`. Independent source coefficients are retained separately
as a regression fixture in `tests/fixtures/materials/cmos_reference.json`.
