# Bundled optical data

The four `.yml` files are unmodified snapshots of the public-domain
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
at revision `c5c2f188e848453def5970e347399d653df2ffc2`, distributed under
CC0 1.0 (see `LICENSE-CC0-1.0.txt`). The catalog records each source URL,
SHA-256 checksum, original references, and available sample conditions.
Please cite the original experimental papers and M. N. Polyanskiy,
*Scientific Data* **11**, 94 (2024), https://doi.org/10.1038/s41597-023-02898-2.

`python scripts/build_material_library.py` rebuilds the catalog offline using
the development dependencies. All models declare a 400–700 nm use band.
Silica and silicon nitride use exact conversions of the source Sellmeier
formulas. Amorphous silicon (a 60 nm film) and aluminum use independently
computed positive Lorentz fits, with three and four oscillators respectively.
The fitting grid has 151 linearly interpolated n,k samples; these are not new
measurements. CSVs retain the source samples within the use band plus
interpolated endpoints. Formula CSVs are sampled analytic values. All CSV
wavelengths are in **metres**. Source YAML wavelengths are in micrometres.

The catalog reports RMS n,k errors on the interpolation grid and maximum
absolute errors at the retained source samples/endpoints. These metrics
measure optical-constant fitting, not simulation convergence. The underlying
measurements cover wider bands; that does not extend the declared fit band.
Material preparation and measurement conditions matter when choosing a variant.

The RGB filters are synthetic design targets, **not measured data**. Targets
are generated analytically with n = 1.45 and k = 0.01 in the passband, k = 0.46
in the stopband, and raised-cosine transitions over 30 nm. The red transition
is 600–630 nm, the blue transition is 470–500 nm, and green passes 500–600 nm
with those two transitions on either side. CSVs contain these targets, not the
fitted response. No copied filter tables or fitted coefficients are used.

The builder fits the synthetic targets with BeamZ's independent vector fitter,
using at most nine stored poles, epsilon_inf=1, weights=(0.1,1.9) on real and
imaginary permittivity, and up to 200 relocation iterations. The requested
weighted RMS tolerance is 0.02. Actual errors and whether the target was met
are retained in the catalog. Passivity is checked on a broad frequency grid
and by refinement of local loss minima. This is a numerical check, not a global
mathematical certificate. No device-efficiency data enter the fitting objective.
Transmission and index errors are reported separately, not optimized directly.

The `Horiba2006` variants are independent implementations of equation 9 and the
page 4 scalar parameter table in HORIBA Jobin Yvon, *Lorentz Dispersion Model*,
Technical Note 08 (September 2006). The catalog retains the primary source URL,
parameters and citation. Only the mathematical model and calculated samples are
bundled; the publication and measured tables are not. These variants require no
numerical fitting. Their Apache-2.0 label applies to this implementation and
its calculated samples, not to the source publication.

The generator, synthetic targets, synthetic fits and independently implemented
Lorentz models use the project's Apache-2.0 license. The four source YAML files
retain CC0. Glass uses the bundled CC0 Malitson source; no archival glass table
or fit derived from one is included.
