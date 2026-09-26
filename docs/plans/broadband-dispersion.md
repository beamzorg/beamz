# Broadband dispersive sensor implementation

Target: reproduce the CMOS RGB notebook with one broadband device run on the local RTX 3090, rather than 31 wavelength-specific FDTD runs.

1. Bring PR #233 periodic boundaries and notebooks onto this checkout.
2. Implement scalar causal pole-residue media, Drude/Lorentz constructors, spectral evaluation, serialization, passive optical-data fitting, and raster ownership. Use volume averaging consistently for dispersive interfaces; do not apply nondispersive harmonic smoothing to pole residues.
3. Implement a coupled trapezoidal auxiliary-polarization update in JAX; retain polarization in continuation state. Reject unsupported CUDA/custom multi-device paths explicitly. Retain the static-material fast path.
4. Import traceable reference material coefficients and RGB data, and convert the notebook to broadband excitation and multifrequency monitors.
5. Validate material response, slab optics, periodic execution, continuation, and existing contracts. Run on the RTX 3090 and save reproducible plots, timing, and numerical convergence results. Distinguish completed runs from accuracy claims.

Numerics: for each conjugate pole pair, q' = a q + c E, P/epsilon0 = 2 Re(q). The coupled trapezoidal update solves E and q together, including Ohmic conductivity. This supports negative real optical permittivity without a negative static epsilon. Frequency units are Hz for evaluation; poles and residues use radians/second. Passive fits and positive epsilon-infinity are required.

## Execution status

Steps 1–5 implemented. Local RTX 3090 device and reference runs completed at 50, 25, and 20 nm. Focused validation: 273 tests passed. See `docs/reviews/rtx3090-dispersion/README.md` for evidence and remaining blue-wavelength mesh sensitivity. Material and source numerical validation is separate from full-sensor spatial convergence.

The default 25 nm notebook was executed end to end on the local RTX 3090: six code cells, zero errors, stored output figures. Slab artifacts were refreshed at 10, 5, and 2.5 nm using the final source implementation.

## Reference-parity improvements (2026-09-26)

1. Import the exact RGB pole/residue coefficients from the current public Tidy3D page's embedded simulation; preserve a source snapshot, checksums, and the old fits for comparison.
2. Plot all three electric components as total |E|² with common incident normalization and color scales.
3. Support normal full-aperture plane waves on rectilinear meshes; verify directionality and power for all six directions. Correct centered mesh-override coordinates and fractional monitor-aperture integration.
4. Validate exact filter films and a silica/SiN/aSi stack against transfer-matrix optics, including interface-monitor placement and absorber sensitivity. Preserve the scalar volume-averaged ADE interface baseline until a more advanced interface scheme has independent validation.
5. Run corrected uniform and graded sensor grids on the local RTX 3090, compare saved spectra and temporal changes, and execute the revised notebook. Report numerical differences from the published figure as approximate until original flux arrays are available.

Reference-parity implementation and validation are complete. Exact coefficients
for all seven reference media are imported; rectilinear source injection,
monitor quadrature, periodic interface weights, and total-field plots are
corrected. RTX 3090 runs at 20 and 15 nm lateral spacing with 5 nm depth
refinement completed with finite fields. Their full-spectrum differences are
at most 0.11/0.28/0.05 percentage points for R/G/B, while the remaining Tidy3D
plot mismatch is larger. Planar tests identify dispersive-interface flux
colocation as a remaining limitation. See `docs/reviews/cmos-reference-parity`
for quantitative checks, saved spectra, and follow-up priorities. Full numerical
agreement with Tidy3D has not been established.
