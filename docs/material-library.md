# Material library

BeamZ bundles named dispersive materials that work offline and can be used
directly in structures and simulations.

```python
import beamz as bz
from beamz import material_library

silica = material_library["SiO2"]["Malitson1965"]
aluminum = material_library["Al"].medium  # default: Rakic1995
structure = bz.Box(center=(0, 0, 0), size=(1e-6, 1e-6, 40e-9), material=aluminum)
```

Lookups return immutable `PoleResidue` materials. `frequency_range` is in Hz;
`eps_model(frequency)` evaluates complex relative permittivity. Every bundled
model currently declares **400–700 nm**, even when its source covers a wider band.

| Material | Variant | Construction |
| --- | --- | --- |
| `SiO2` (fused silica) | `Malitson1965` | Exact Sellmeier conversion |
| `SiN` (silicon nitride) | `Philipp1973` | Exact Sellmeier conversion |
| `aSi` (amorphous silicon, 60 nm film) | `Pierce1972` | Three-oscillator passive fit |
| `Al` (aluminum) | `Rakic1995` | Four-oscillator passive fit |
| `SiN`, `aSi` | `Horiba2006` | Exact conversion of published Lorentz formulas |
| `CMOS_RGB` | `red`, `green`, `blue` | Passive vector fits to analytic filter targets |

The first four materials use CC0 data from the
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database).
The `Horiba2006` variants independently implement the scalar parameters and Lorentz
equation in HORIBA Jobin Yvon Technical Note 08 (2006), equation 9 and page 4.
They bundle neither the publication nor a transcribed measured-data table. Their
Apache-2.0 label covers the implementation and calculated samples, not the
publication. The three filters are hypothetical materials, not measured products.

## Inspect data and fit quality

```python
variant = material_library["aSi"].variants["Pierce1972"]
print(variant.description, variant.source, variant.license)
print(variant.conditions, variant.references)
print(variant.fit)
wavelength_m, n, k = variant.nk_data.T
```

`nk_data` returns a fresh array with wavelength [m], n, k columns. For tabulated
physical sources, it retains original samples inside the use band plus
interpolated endpoints. For formulas, it samples the analytic definition. For synthetic filters,
it contains the intended n,k targets; evaluate the medium to obtain the fitted
response. Changing the array does not change library data.
Library and variant mappings are read-only.

Silica and nitride require no numerical fit. For silicon and aluminum the
catalog reports RMS n,k errors on a 151-point interpolation grid and maximum
absolute errors at retained source samples/endpoints. Fit error and FDTD
resolution error are separate. Check the band, sample conditions, and error
report before choosing a model. In particular, different forms and deposition
conditions of silicon or nitride are not interchangeable.

## Synthetic filter targets

The filter targets have n = 1.45, passband k = 0.01, and stopband k = 0.46.
Raised-cosine transitions span 470–500 nm and/or 600–630 nm. Red passes above
630 nm, green from 500–600 nm, and blue below 470 nm, within the 400–700 nm band.

The builder uses BeamZ's independent scalar vector fitter: at most nine stored
poles (each complex pole represents a conjugate pair), epsilon_inf = 1, up to
200 relocation iterations, and weights (0.1, 1.9) on real/imaginary permittivity.
It tries standard and relaxed relocation with real/complex initial poles and
linear/log spacing. Passive residue fitting follows pole relocation. The
training set has 200 samples; a separate 3001-point grid measures n,k and
absorption-only transmission errors through 1 µm.

The weighted permittivity RMS target is 0.02. `tolerance_met` reports whether
it was achieved; returning a model does not mean the target was met. This
objective allows refractive-index dispersion. Index and transmission errors
are reported separately, including `index_target_met` and
`transmission_target_met`; neither is an additional optimization objective.
Fresnel reflections and interference also affect full-film transmission.

## Fit a material

```python
wavelengths_m, n, k = material_library["aSi"].variants["Pierce1972"].nk_data.T
model, report = bz.fit_nk_vector(
    wavelengths_m, n, k, max_poles=9, epsilon_inf=1.0,
    weights=(0.1, 1.9), num_iters=200, tolerance_rms=0.02,
)
print(report["weighted_rms_epsilon"], report["tolerance_met"])
```

`fit_nk_vector` uses NumPy/SciPy and independently implements
[Gustavsen and Semlyen's vector fitting](https://doi.org/10.1109/61.772353)
and [Gustavsen's relaxed pole relocation](https://doi.org/10.1109/TPWRD.2005.860281).
It does not import or vendor another fitter. Weights are normalized to mean one;
the reported RMS is `sqrt(mean((wr*delta_Re_eps)**2 + (wi*delta_Im_eps)**2))`.
Stable poles enforce causality. Constrained residue optimization enforces sampled
nonnegative loss. A broad independent grid and refinement of local loss minima
check passivity outside the fit band; this is not a global passivity proof.
Different initializations or numerical libraries can produce different fits.
The existing `fit_nk` function remains available for positive Lorentz fits with
an n,k objective.

## Rebuild or extend the catalog

Install the development dependencies and run
`python scripts/build_material_library.py`. This runs on the CPU and uses only
bundled, pinned source snapshots. The catalog retains source URLs, SHA-256
checksums, citations, conditions, fit settings, and diagnostics. Raw YAML files,
CC0 terms, and a data README ship in `beamz/material_library/data`.

To add a variant, select and pin a suitable source, normalize its units, choose
a use band, and convert its analytic formula or fit passive causal oscillators.
Validate optical constants against the source and exercise the model in an
analytical slab test before adding it to the curated catalog. The generator
supports Sellmeier formula 1, the named Lorentz formulas, and tabulated n,k;
it rejects other formats. It does not automatically import the entire upstream database.

These are scalar bulk models for single-device JAX execution. A larger source
catalog alone does not add dispersive anisotropy, surface-conductivity models,
or temperature-dependent constitutive equations. Those require separate solver
and API work. Changing any model requires fresh simulation results; notebook
cache keys include the material coefficients.
