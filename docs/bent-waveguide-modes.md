# Circular-bend eigenmodes

`beamz.devices.modes.solve_grid` can evaluate a uniformly bent cross-section:

```python
straight = solve_grid(**grid_inputs)
bent = solve_grid(**grid_inputs, bend_radius=10.0, bend_axis="x")
print(straight.n_eff.values, bent.n_eff.values)
```

See the runnable [straight/bent example](https://github.com/beamzorg/beamz/blob/main/examples/bent_waveguide_modes.py).
The optional arguments belong to the direct raster solver. Mode sources,
monitors, ports, and FDTD stepping retain their existing behavior.

## Coordinates and materials

- `bend_radius=None` (default) executes the original straight solve, including
  its fields and diagnostics. Infinity is rejected; use `None` for straight.
- Radius, transverse edges, and wavelength use **micrometres**; frequency uses
  hertz. This direct raster API differs from BeamZ geometry's SI units.
- `bend_axis="x"` (default) selects the first transverse array coordinate;
  `"y"` selects the second. The selected coordinate and local propagation
  direction span the bend plane. This parameter selects the radial direction,
  **not** the axis about which the guide rotates.
- Write the selected coordinate as `u`. The reference line is **u=0**. The
  curvature center is **u=-R**. Positive R makes increasing u point outward;
  negative R reverses this. Every domain edge, including PML, must satisfy
  `h = 1 + u/R > 0`; touching or crossing the curvature axis is invalid.
- There is no automatic geometry centering. To reference a line at coordinate
  `u0`, subtract `u0` from the selected edge array before solving.
- `n_complex` is the propagation constant divided by the vacuum wavenumber,
  per unit reference arc length. The angular propagation magnitude is
  `k0 * abs(R) * n_complex`. Moving the reference line changes `n_complex`;
  the angular propagation of the same physical mode remains unchanged.
  `direction` retains its existing forward/backward field convention and does
  not change which side contains the bend center.
- Material components are in the solver's local transverse/transverse/tangent
  frame. `normal_axis` retains the existing global field-label mapping; it does
  not select the radial direction. Scalar, diagonal, and full tensor grids are
  accepted. Material tensors must be constant along the arc **in this rotating
  frame**. A general laboratory-fixed anisotropic crystal does not satisfy
  that assumption. Isotropic media and uniaxial media with their optic axis
  perpendicular to the bend plane do.

## Formulation

For reference arc length `s`, the circular-coordinate metric is
`dl² = dx² + dy² + h² ds²`. With `A = diag(1, 1, 1/h)`, Maxwell's coordinate
transformation gives

```text
epsilon_equivalent = h A epsilon A
mu_equivalent      = h A mu A
```

For diagonal inputs, both tensors acquire factors `(h, h, 1/h)`.
Transforming permeability is necessary even for nonmagnetic physical media.
This is a coordinate transformation, without a weak-guidance or large-radius
approximation [1]. The private material helper feeds the existing diagonal or
full-tensor sparse operator; the finite-difference discretization is unchanged.
The metric is sampled at the supplied raster's cell centers, matching the
existing material and result coordinates. Mesh refinement remains necessary.

Coordinate longitudinal fields satisfy `E'_s=h E_s`, `H'_s=h H_s`.
The returned longitudinal fields are divided by h, then the existing global
component mapping is applied. Transverse fields and cross-section power
normalization are unchanged. Bent results record `bend_radius` and `bend_axis`
in `solver_info`; straight diagnostics remain unchanged.

## Numerical validation

The independent Bessel-function benchmark of Hiremath and Hammer [2] uses a
1 µm slab, core/cladding indices 1.7/1.6, wavelength 1.3 µm, and a core-center
radius. The second transverse dimension is a singleton, which the existing
solver treats as invariant. The tests select TE by its electric-field component.
The published convention is `beta/k - i alpha/k`; BeamZ reports positive
`Im(n_complex)` for this decaying mode.

At 25 nm spacing, domain `u=[-10,20]` µm and 4 µm PML on each radial end:

| R (µm) | Published Re(n) | BeamZ Re(n) | Published loss index | BeamZ loss index |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 1.66303 | 1.66305504 | 3.309e-4 | 3.29974649e-4 |
| 100 | 1.66096 | 1.66098057 | 1.987e-6 | 1.97663693e-6 |
| 150 | 1.66061 | 1.66062971 | 1.020e-8 | 1.01253624e-8 |
| 200 | 1.66049 | 1.66051112 | 5.066e-11 | 5.02336070e-11 |

At R=50 µm, refining 50 → 25 → 12.5 nm gives phase indices
1.66310664 → 1.66305504 → 1.66304102 (observed order 1.88), and loss indices
3.27789002e-4 → 3.29974649e-4 → 3.30616151e-4.
Moving the outer edge/PML width from 15/3 through 20/4 to 25/5 µm at 25 nm
changes phase by less than 1e-6 and relative loss by less than 0.1%.
The tests gate phase error at 3e-5 and relative loss error at 2% separately.

Increasing radius also recovers the **existing** analytical slab test
(core/cladding 2.04/1.444, width 0.6 µm, wavelength 1.55 µm, 25 nm grid)
and the strip fixture in `tests/unit/modes/test_api.py`:

| Case | Straight | R=100 µm | R=200 µm | R=400 µm |
| --- | ---: | ---: | ---: | ---: |
| Slab TE0 | 1.87495837 | 1.87499766 | 1.87497139 | 1.87496323 |
| Strip fixture | 2.51573826 | 2.51642782 | 2.51610679 | 2.51592844 |

Both differ from straight by less than 1e-8 at R=1e8 µm. The slab's independent
straight analytical value is 1.87464404, satisfying the existing 0.2% gate.
The fixed-grid radius-convergence orders over 100–400 µm are about 1.51 and
0.93 respectively; discretization asymmetry can leave a term linear in 1/R.

## Limits and boundaries

This computes local eigenmodes of a constant-radius guide, not junction loss,
finite-angle transmission, varying-curvature bends, or closed-ring resonances.
Use a straight mode's index as `target_neff` and inspect fields when following
a branch: radiation and finite-window modes can also lie near the search target.

Existing PEC/PMC walls and PML remain available. A radial symmetry wall generally
cannot represent half of a bent guide. Without PML, radiated power reflects
from the truncated domain; a real eigenvalue does not establish zero bend loss.
The existing derivative-stretch PML is applied to the transformed material
grid. Its radial metric is sampled on the real grid, not analytically continued
into complex radius. The reproduced weak-curvature benchmarks are validated;
tight bends and very small loss require their own mesh, domain, and PML study.

## References

1. D. M. Shyroki, [“Exact equivalent-profile formulation for bent optical waveguides”](https://arxiv.org/abs/physics/0605002), equations (7), (12), (13) (2006).
2. K. R. Hiremath and M. Hammer, [“Bend modes: A collection of 2-D example results”](https://www.siio.eu/NaisWP3/Bend/), University of Twente (2001–2005, updated 2021). Independent semi-analytic simulation results and downloadable numerical data.
