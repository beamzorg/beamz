# Contour-path dielectric averaging

Select `RasterOptions(smoothing="contour_path")` explicitly. Simulation defaults
remain `farjadpour_diagonal`; standalone raster defaults remain `farjadpour_full`.
This experimental option implements a restricted contour-path closure, not
Tidy3D's proprietary 3D implementation.

## Constitutive rule

Following Mohammadi, Nadgaran & Agio (2005),
[doi:10.1364/OPEX.13.010367](https://doi.org/10.1364/OPEX.13.010367), equations
(7)–(8), E_i is averaged along its primal Yee edge L and D_i over the transverse
dual surface S. Both pass through the actual Yee location p. With interface
normal n, the scalar boundary projection is `g_m = 1 - n_i² + n_i² epsilon_p / epsilon_m`:

```
epsilon_i = <epsilon_m g_m>_S / <g_m>_L
```

For parallel interfaces, `q = n_i²`, surface fractions s_m and line fractions l_m:

```
epsilon_i = [(1-q) sum(s_m epsilon_m) + q epsilon_p]
            / [(1-q) + q epsilon_p sum(l_m / epsilon_m)]
```

This gives arithmetic surface averaging for tangential E and harmonic line
averaging for normal E. Diagonal Farjadpour instead uses **volume** fractions in
`A(1-q)+Hq`; full Farjadpour also retains cross-component coupling.

In 2D TE, S reduces to a transverse line. Each half-path starts at p and retains
its own intersection normals, including polygon holes. Successive parallel
crossings telescope to the projection above. Successive nonparallel crossings
on one half-path, or conflicting vertex normals, fall back to volume averaging.
2D TM uses exact area averaging because E_z is tangential to the sidewalls.

## Supported geometry and fallbacks

Only positive isotropic, lossless, nonmagnetic dielectrics represented by boxes
or vertical polygon extrusions are accepted. Unsupported materials/geometries
raise errors. This mode adds no dispersive updates; upstream frozen-index
approximations remain frozen-index. Curves must be polygonized.

In 3D, S is an actual area: XY sections use polygon clipping; XZ/YZ sections use
polygon line sections times clipped height. Parallel exposed interfaces use the
formula above, with alignment tolerance `1 - 1e-12`. Locally z-invariant supports
use the 2D closure with separate path normals. Nonparallel cap/sidewall corners
fall back; general curved 3D surfaces are not supported.

Integration uses the priority-resolved partition. Same-material seams, including
hidden height cuts, do not add interfaces. Graded grids use physical Yee
locations, not dual-support midpoints. Domain supports are clipped; exact
interface ties follow half-open ownership. Cell maps remain volume averages;
propagation retains component-specific Yee coefficients even on uniform grids.
H coefficients are unchanged. `smoothed_samples` counts accepted electric
supports; `fallback_multiple_orientations` counts rejected orientations.

## Validation and comparison

Unit tests cover independent path ratios, all-axis arithmetic/harmonic limits,
angle/offset/scale sweeps, graded grids, ownership, holes, thin layers, hidden
height seams, unsupported media and caching. Integration tests check 3D updates.
Convergence tests compare all four methods against analytical Fresnel fields and
check contour-path cylinder convergence against an independent TE Mie oracle.
The cylinder test isolates the frequency-domain spatial operator, not a
time-domain scattering spectrum.

Measured 15-PPW crossing through power was 95.8656% (contour), 95.8587%
(diagonal) and 95.5831% (volume). At 10 PPW, coupler cross power was 51.4332%,
51.4467% and 50.7428%, respectively. These small contour/diagonal changes do not
explain the external-reference discrepancy. Dispersion, monitor placement and
normalization remain separate controls. The 15-PPW coupler exceeded available
memory; full-tensor comparisons used analytical cases, not these device runs.

Contour path improved the tested cylinder errors; full Farjadpour performed
better in some rotated-interface cases. No universal accuracy ranking, arbitrary-
corner second-order convergence, unconditional stability, or Tidy3D equivalence
is claimed. The existing diagonal update and default remain unchanged.
