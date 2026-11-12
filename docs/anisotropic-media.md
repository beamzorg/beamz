# Anisotropic, Hyperbolic, and Gyrotropic Media

`bz.AnisotropicMedium` describes a material with full symmetric permittivity and
permeability tensors. Resonances can make either tensor dispersive:

- `bz.TensorPole` adds a Lorentz or Drude resonance with a tensor strength.
- `bz.GyrotropicPole` adds a non-reciprocal resonance, for example a magnetized
  plasma.

Together they cover grid-aligned and tilted crystals, magnetic anisotropy,
hyperbolic media whose principal values have opposite signs, and magneto-optic
or plasma media with non-symmetric tensors.

```python
import numpy as np
import beamz as bz

omega = 2 * np.pi * bz.LIGHT_SPEED / (1.5 * bz.um)

crystal = bz.AnisotropicMedium(
    permittivity=2.0,                      # eps_inf: scalar, 3, 6 packed, or 3x3
    permeability=1.0,                      # mu_inf: same forms
    electric_poles=[
        bz.TensorPole.drude((2 * omega, 0.0, 0.0), damping=0.01 * omega),
    ],
)
tilted = crystal.rotated((0.0, 0.0, np.radians(30)))  # Rz(30 deg) @ crystal @ Rz^T
print(tilted.permittivity_tensor(omega / (2 * np.pi)).real)

plasma = bz.AnisotropicMedium(
    electric_poles=[
        bz.GyrotropicPole.magnetized_plasma(
            plasma_frequency=omega, cyclotron=(0.0, 0.0, 0.4 * omega), damping=1e12
        )
    ]
)
```

## Model

With the `exp(-i omega t)` convention, a medium's permittivity is

```text
eps(omega) = eps_inf + sum_p W_p / (omega_p^2 - omega^2 - i gamma_p omega)
```

The permeability follows the same form through `magnetic_poles`.

**Tensor poles.** Each strength `W_p` is a symmetric positive-semidefinite
tensor, so the medium is passive.

- `TensorPole.lorentz(delta_epsilon, resonance=..., damping=...)` sets
  `W = delta_epsilon * resonance^2`.
- `TensorPole.drude(plasma_frequency, damping=...)` has zero resonance and
  `W = plasma_frequency^2`, per axis.

**Gyrotropic poles.** The current precesses about a gyration vector:
`p'' + gamma p' + omega_p^2 p = W F + gyration x p'`. The susceptibility is then
Hermitian when lossless but not symmetric, so the medium is non-reciprocal.
`GyrotropicPole.magnetized_plasma` uses the cyclotron vector `e B / m`.
`GyrotropicPole.lorentz` builds a magneto-optic oscillator.

**Rotation.** `rotated(R)` transforms every tensor as `R T R^T` and every
gyration vector as `R g`. Rotating per-axis poles produces frequency-dependent
off-diagonal terms (oriented poles), which a grid-aligned diagonal model cannot
represent.

A static negative permittivity is unconditionally unstable in FDTD, so
`eps_inf` and `mu_inf` must be positive definite. The negative branch of a
hyperbolic medium has to come from a pole.

## Non-symmetric tensors and conductivity tensors

`permittivity`, `permeability`, `conductivity`, and `magnetic_conductivity`
each accept a scalar, three diagonal values, six packed symmetric values
`(xx, yy, zz, xy, xz, yz)`, nine row-major values, or a 3x3 matrix. A full
matrix may be non-symmetric:

- **Conductivity tensors.** `conductivity` (S/m) and `magnetic_conductivity`
  (Ohm/m) may be any real 3x3 tensor whose symmetric part is positive
  semidefinite. The symmetric part is loss; the antisymmetric part is
  lossless and non-reciprocal, like a Hall response. Rotating a medium now
  rotates its conductivities too.
- **Non-symmetric `eps_inf` or `mu_inf`.** A real non-symmetric tensor is
  non-Hermitian, so it exchanges energy with the field (gain or loss), and
  BeamZ warns. Its symmetric part must be positive definite.

`AnisotropicMedium.from_complex_permittivity(eps, frequency=... or
wavelength=..., permeability=mu)` mirrors fdtdx's constructor of the same
name. It takes a complex scalar, three values, or a 3x3 tensor for each. The
real parts become the tensors, and the imaginary parts become
`sigma = omega0 eps0 Im(eps)` and `sigma_m = omega0 mu0 Im(mu)`, element by
element. `complex_permittivity(f)` and `complex_permeability(f)` return
`eps_inf + chi(f) + i sigma / (omega eps0)`, so they reproduce the input
exactly at the reference frequency. A Hermitian tensor (imaginary
off-diagonals) becomes an antisymmetric conductivity. That gives a lossless
gyrotropic response at `omega0`, scaling as `omega0 / omega` away from it,
which is the causal response of a conductor. For a response matched across a
band, use a `GyrotropicPole`.

```python
medium = bz.AnisotropicMedium.from_complex_permittivity(
    ((2.25, 0.3j, 0.0), (-0.3j, 2.25, 0.0), (0.0, 0.0, 2.25)),
    wavelength=1.55e-6,
)
```

**Update.** Around these media, a cropped box with a two-cell halo takes the
full semi-implicit update

```text
(T + B) F_new = (T - B) F_old + inc,   B = sigma dt / (2 eps0)  (mu0 for H)
F_new = F_old + (T + B)^-1 (inc - 2 B F_old)
```

`inc` is the vacuum kernel's increment, including CPML terms; sources inject
afterwards, as in any other medium. The E kernel runs with vacuum
coefficients inside the box (the H kernel always does), and the box then
applies the full tensors. Each Yee support inverts
its own full `T + B`. Diagonals stay on the sites, and the symmetric and
antisymmetric cross terms sit on the shared support, as in the symmetric
full-tensor update. `B F_old` uses the same collocation. For an antisymmetric
`B` this is a Cayley transform: a uniform field rotates at constant
`E^T eps E`. Plane waves see `T + i cot(omega dt / 2) B`, which tends to
`eps' + i sigma / (omega eps0)` as `omega dt -> 0`.

**Rasterization.** The raster keeps the symmetric `eps_inf` and `mu_inf` and the
diagonal electric conductivity, with the usual smoothing. The antisymmetric
tensor parts, the off-diagonal electric conductivity, and the whole magnetic
conductivity are volume-averaged by each medium's cell fraction on every
support.

**Support.** These media run on JAX (2D and 3D) and on single-device
`backend="cuda_streamed"`, whose native inverse kernel carries the
antisymmetric cross terms. Tensor poles in the same medium use the full
inverse. The following are rejected:

- mode sources and mode monitors that overlap these media;
- 2D simulations with `xz` or `yz` terms;
- `TopologyProblem` designs in the same simulation.

Other off-diagonal permittivity must stay at least three cells away.

## Fitted pole-residue (CCPR) poles

A Lorentz or Drude pole couples only to the field. A general complex-conjugate
pole-residue (CCPR) pole also couples to its time derivative through a second
symmetric tensor `B` (rad/s):

```text
chi_p(omega) = (A_p - i omega B_p) / (omega_p^2 - omega^2 - i gamma_p omega)
p'' + gamma_p p' + omega_p^2 p = A_p E + B_p E'
```

This is the form of any fitted pole-residue pair, so vector-fitted or
critical-point models of metals can be anisotropic and oriented.

- `TensorPole.ccpr(pole, residue, tensor=..., orientation=...)` builds the pair
  `r / (s - q) + r* / (s - q*)`, with `s = -i omega`, as in `bz.PoleResidue`
  and fdtdx's `CCPRPole`. It uses `omega_p = |q|`, `gamma_p = -2 Re q`,
  `A = -2 Re(r q*) T` and `B = 2 Re(r) T`. The weighting tensor `T` is a
  scalar, three axis weights, a symmetric tensor, or `u u^T` for a unit
  `orientation` vector `u`.
- `TensorPole.critical_point(amplitude, phase, resonance=..., damping=...)`
  takes the critical-point parameters reported for metal fits.
- `AnisotropicMedium.from_pole_residue(x, y, z)` turns scalar `bz.PoleResidue`
  fits along the principal axes into one medium with
  `eps_inf = diag(x, y, z)`. Pass only `x` for an isotropic medium. Rotate the
  result with `rotated()`; both `A` and `B` transform as `R T R^T`.
- `beamz.design.anisotropic.pole_residue_tensor_poles(model, tensor=...)`
  returns the poles alone. Pairs of real poles merge into one second-order pole,
  so BeamZ's two-pole Drude term becomes an exact Drude pole again. A leftover
  real pole becomes a Debye term (`A = 0`, `B = 2 Re c`).

```python
gold_like = bz.PoleResidue(
    1.5, [(-5e14 - 3e15j, 2e14 + 1e15j), (-5e14, 2.5e14)], frequency_range=(1e14, 8e14)
)
other = bz.PoleResidue(
    3.0, [(-8e14 - 2e15j, 5e14 + 1.5e15j), (-3e14, 6e14)], frequency_range=(1e14, 8e14)
)
crystal = bz.AnisotropicMedium.from_pole_residue(gold_like, other, gold_like)
tilted = crystal.rotated((0.0, 0.0, np.radians(30)))
```

One fitted pole need not be passive on its own. CCPR poles therefore skip the
per-pole semidefinite check. Instead, the medium checks on a dense frequency
grid that the anti-Hermitian part of `eps(omega)` and `mu(omega)` stays
positive semidefinite, as `PoleResidue` does.

**Discretization.** Every term of the pole update is centred at step `n`, so
the scheme is second order in `dt`:

```text
p^{n+1} = c1 p^n + c2 p^{n-1} + (dt^2 / d) A E^n + (dt / (2 d)) B (E^{n+1} - E^{n-1})
d = 1 + gamma dt / 2
```

This deliberately differs from fdtdx, whose CCPR update uses the first-order
forward difference `(dt / d) B (E^{n+1} - E^n)`. That drive lags by
`omega dt / 2`, which puts a typical fit's permittivity off by about 0.1 to 1%
at 10 to 20 points per wavelength.

The `B` term splits into an explicit half `k (E^n - E^{n-1})` and an implicit
half `k (E^{n+1} - E^n)`, with `k = (dt / (2 d)) B`. Each CCPR region stores
the previous field change `E^n - E^{n-1}` per site as part of its pole state
(zero at the start, as for a field at rest). The explicit half joins the pole
advance. For the implicit half, Ampere's law solves with the static tensor
`eps' = eps_inf + K`. Here `K = sum_p (dt / (2 d_p)) B_p` carries the medium's
volume fraction and the same Yee collocation as the pole drive:

```text
(eps' + sigma dt / (2 eps0)) (E^{n+1} - E^n) = curl term - delta
delta = sum_p (c1 p + c2 p_prev + (dt^2 / d) A E^n + k_p (E^n - E^{n-1}) - p^n)
```

Each pole then gains `k_p (E^{n+1} - E^n)`, and the stored change becomes
`E^{n+1} - E^n`. `K` is symmetric, so `eps'` uses the existing constitutive
supports:

- **Diagonal kernel.** When `eps_inf` has no off-diagonal terms, the kernel
  stays unchanged. The correction recovers the curl term from the kernel's own
  update and solves with the pointwise inverse of `eps' + loss`, even when a
  tilted `B` makes `K` off-diagonal. Magnetic CCPR poles use the same path.
- **Full-tensor kernel.** When `eps_inf` is off-diagonal, the kernel's inverse
  tensors include `K` from the start.

Both give `E^{n+1} - E^n = Inv(eps' + loss)(curl term - delta)`. Gyrotropic
poles are unaffected.

In the conductivity limit (`omega0 = gamma = A = 0`), `p'' = B E'` is a current
`J = eps0 B E`. The centred drive makes the discrete current
`B (E^{n+1} + E^n) / 2`, which is Crank-Nicolson conductivity. It matches the
grid's semi-implicit `sigma = eps0 B` to rounding. The forward difference
would give backward Euler instead.

Validation:

- Under a prescribed harmonic flux, the error against the continuous `eps(omega)`
  drops by 4.0x when `dt` halves. For the first-order drive it drops by 2.0 to
  2.4x.
- In slab tests, the cross-polarization error against Jones calculus roughly
  halves compared with the first-order drive. The gap to the scalar
  `PoleResidue` run shrinks about 3x.

**Timestep.** `eps'` must stay positive definite; only an active medium can
violate this, and compilation rejects it. The von Neumann analysis of the
[timestep check](#timestep) models the centred scheme, including the stored
change, the implicit half of `B` and the pointwise inverse of `eps'`. For CCPR
media too, the predicted limit matches seeded simulations: they are stable at
0.995 of it and divergent at 1.005. For the stiff test crystal, the limit is
1.4% below that of the first-order drive.
`beamz.simulation.anisotropic.max_stable_timestep` reports it in advance.
`fold_fast_poles` leaves CCPR poles unchanged.

**Support.**

- JAX and streamed CUDA. Natively, the pole launch applies the explicit half
  of the `B` term, and an extra launch adds the implicit half. Magnetic CCPR
  poles also run in both.
- Mode ports fold `Re chi(f)`, including the `-i omega B` term.
- Not supported: magnetic CCPR poles with `B != 0` in a medium with
  off-diagonal static permeability, and CCPR poles with `B != 0` anywhere in a
  topology-design program. Both are rejected with an explicit error.

## Material library

`beamz.material_library.anisotropic_library` bundles four crystals. Each one is
given in its crystal frame; use `rotated()` to place it in a device.

| Key | Variant | Frame | Source |
| --- | --- | --- | --- |
| `Al2O3` (sapphire) | `Malitson1972` | uniaxial, optic axis z | Exact Sellmeier conversion, refractiveindex.info (CC0) |
| `LiNbO3` (congruent) | `Zelmon1997` | uniaxial, optic axis z | Exact Sellmeier conversion, refractiveindex.info (CC0) |
| `hBN` | `Caldwell2014` | c axis z, mid-IR | Published TO-LO model, Nat. Commun. 5, 5221 |
| `MoO3` (alpha) | `AlvarezPerez2020` | x=[100], y=[001], z=[010] | Published factorized model, Adv. Mater. 32, 1908176 |

Two of these need a note:

- **hBN.** Caldwell et al.'s supplementary table prints its optical-axis labels
  swapped relative to the paper's own main text. BeamZ uses the physical
  assignment: in-plane 1360/1614 1/cm, c axis 760/825 1/cm.
- **alpha-MoO3.** The x axis has three coupled oscillators with unequal damping.
  A sum of Lorentz poles reproduces that factorized form to about 1% (median),
  so each variant's `fit` records its deviation.

`scripts/build_anisotropic_library.py` rebuilds the catalog from the bundled
sources.

## Discretization

Static `eps_inf` tensors use BeamZ's full-tensor electric update. The other
terms run as exact corrections around the standard kernels:

- **Permeability.** The H kernel advances in vacuum. Inside a cropped box around
  the magnetic media, the increment is then mapped through `mu^-1`.
- **Tensor poles.** A normalized polarization `p = P / eps0` follows an explicit
  central-difference recurrence driven by the previous field. The field is then
  corrected by `-eps_inf^-1 (p_new - p_old)`. Magnetic poles mirror this on H.
- **Gyrotropic poles.** Each component's polarization lives on its own Yee
  site, so the strongly negative diagonal response of a plasma stays exact at
  every wavevector. Only the precession couples components. It is integrated
  with Crank-Nicolson and energy-neutral coupling.

Cross terms live on a shared support: primal nodes for E, cell centres for H.
They are reached with reciprocal averaging, which is the volume-weighted
adjoint of its reverse. At material boundaries, a site's diagonal pole weight
is the shared weight averaged back to that site. This makes every discrete pole
operator symmetric positive semidefinite, which keeps lossless hyperbolic
(Drude) media stable at tilted, finite interfaces.

Periodic and Bloch seams average across the seam with the seam's phase, for
both these operators and the full-tensor permittivity kernel. Conductive and
sponge losses enter the tensor updates semi-implicitly as `(T + sigma dt / 2)^-1`.

## Timestep

The pole recurrences are explicit, so they limit the timestep. Compilation
checks it with a von Neumann analysis of the implemented scheme in the bulk
medium. Every plane wave on the grid evolves independently, so the scheme is
stable when the per-step amplification matrix of each wave vector has spectral
radius at most one. The analysis samples the Brillouin zone and models the
scheme as implemented:

- off-diagonal operators act through the shared-support averaging, which scales
  them by `cos(k h / 2)` along each axis;
- the field update applies the pointwise inverse of `eps_inf`;
- gyrotropic poles keep their Crank-Nicolson precession;
- in 2D, only the simulated polarization counts.

It depends on the medium and `dt`, not on the grid size, and is cached.
Compilation rejects a timestep the analysis finds unstable and reports the
largest stable `dt`. Use `beamz.simulation.anisotropic.max_stable_timestep` to
compute it in advance. In seeded simulations the predicted limit is sharp:
runs stay bounded at 0.98 of it and diverge at 1.02.

The analytic bound `S^2 + (Omega dt / 2)^2 < 1`, with `S` the in-medium
Courant number and `Omega` the fastest coupled pole frequency at `k = 0`, is
not sufficient for multi-pole anisotropic media. A stiff pole tilted against
an anisotropic `eps_inf` couples to the grid dispersion at finite `k`: one such
crystal diverges at 0.98 of that bound, and the analysis puts its limit about
11% lower. BeamZ therefore no longer uses it.

Sellmeier crystals with ultraviolet poles, such as LiNbO3 and sapphire, need a
Courant number noticeably below the vacuum limit.

Ultraviolet poles sit far above optical bands, yet they set this limit.
`medium.fold_fast_poles(max_resonance)` replaces every pole resonating above
`max_resonance` (rad/s) with a static `eps_inf` shift plus a slower pole at
`max_resonance`:

```text
W' = W (w1 / w0)^4,   delta eps_inf = (W / w0^2) (1 - w1^2 / w0^2)
```

The folded medium matches the original susceptibility and its `omega^2` slope
at low frequency and stays passive. It allows a larger timestep. Choose
`max_resonance` well above the simulated band; compilation suggests it when a
timestep is rejected.

## Boundaries

- **PML.** A PML stretches coordinates. That amplifies the backward waves of
  hyperbolic (indefinite) media, which shows up as late-time exponential growth
  inside the absorber. BeamZ warns when such a medium overlaps a PML.
- **Absorber.** `bz.Absorber`, a graded-loss sponge, absorbs backward waves. It
  works with full permittivity tensors, permeability tensors, and all poles. Use
  it when a hyperbolic or gyrotropic medium has to reach the domain edge.

## Ports, monitors, and gradients

- **Mode sources and monitors in pole media.** Ports inside tensor-pole media
  solve their modes at each frequency with the poles folded in:
  `eps_inf + Re chi(f)`, and likewise `mu(f)` for magnetic poles. Broadband
  sources fold at every profile frequency; mode monitors fold at every monitor
  frequency. The folding data is kept in result archives.
- **Magnetic media at ports.** 3D ports reproduce diagonal and full symmetric
  permeability tensors with the propagation's discretization (diagonals on the
  H sites, cross terms at cell centres). 2D ports support diagonal permeability.
- **Not yet supported at ports.** Gyrotropic media, and 2D TM ports that overlap
  in-plane (`xy`) magnetic coupling.
- **Gradients.** `Simulation.differentiable()` and `TopologyProblem` (default
  autodiff backend) differentiate through every correction on JAX. Fixed tensor
  media must not overlap a design region, including their two-cell halo. The
  spectral-adjoint backend is rejected.
- **Anisotropic design materials.** `TopologySpec(material_min=...,
  material_max=...)` takes `Material` or `AnisotropicMedium` endpoints, for
  example a tilted LiNbO3 crystal against air. The high-frequency tensor blends
  linearly with density, and each endpoint's poles fill a cell with volume
  fraction `rho` or `1 - rho`. Design materials must be nonmagnetic and
  lossless apart from their poles.
- **Multi-material design regions.** `TopologySpec(materials=(m0, m1, m2,
  ...))` picks one of K >= 2 materials per cell; see below.

## Multi-material design regions

A design region can choose among K >= 2 materials. Each may be an isotropic
`Material`, a static tensor, or a pole-dispersive `AnisotropicMedium`:

```python
import beamz as bz
from beamz.material_library import anisotropic_library
from beamz.optimization import TopologyProblem, TopologySpec

lnbo3 = anisotropic_library["LiNbO3"]["Zelmon1997"].rotated((0.0, 0.0, 0.4))
topology = TopologySpec(
    design=design,
    region_mask=mask,
    resolution=dx,
    filter_radius=0.1e-6,
    materials=(bz.Material(1.0), bz.Material(2.1), lnbo3, bz.Material(12.1)),
)
problem = TopologyProblem(simulation, topology, objective)
density = topology.initial_state().density  # shape (3, *mask.shape)
value, gradient = problem.value_and_grad(density, beta=4.0)
```

**Parametrization.** The density has `K - 1` nested channels `c_1, ...,
c_{K-1}`, so its shape is `topology.density_shape == (K - 1, *mask.shape)`.
Channel `c_k` answers one question for the cells already past material
`k - 1`: is the cell material `k - 1`, or material `k` or a later one? The
fraction holding material `k` or a later one is `Q_k = c_1 ... c_k`, and
material `k` alone has weight `w_k = Q_k - Q_{k+1}`, so

    eps_inf = m_0 + sum_k Q_k (m_k - m_{k-1}) = sum_k w_k m_k.

Every channel goes through the same filter, projection and morphology as a
two-material density. The weights are nonnegative and sum to one. Binary
channels select exactly one material, so beta continuation drives every cell
to a single material. With K = 2 there is one channel, and
`materials=(a, b)` is identical to `material_min=a, material_max=b`.

**Physics.** The high-frequency tensors, including their off-diagonals, blend
linearly. Every material's poles fill the cell with volume fraction `w_k`,
which generalizes `rho` and `1 - rho`. Identical pole media listed twice share
one pole region. At runtime the inverse tensors and pole strengths are
rebuilt from the traced density, so the default autodiff backend
differentiates through every path. The spectral-adjoint backend is rejected.

**Ordering.** Order matters. `m_0` is the void: `fixed_void`, the region
background on export, and `eps_min`. `m_{K-1}` is the solid: `fixed_solid`,
the filter context of fixed structures outside the region, and `eps_max`.
List the cladding first and the waveguide core last; `eps_min < eps_max` is
required. Only `c_1` has to change for a cell to go from `m_0` to `m_2`, so
the transition need not pass through `m_1`. However, a channel stops
receiving gradient where an earlier channel is exactly zero, so start from
interior values. The default start (every channel at 0.5) weights earlier
materials more heavily: 1/2, 1/4, and so on.

**Scope.** Multi-material regions are available through `TopologySpec` and
`TopologyProblem`. `TopologyDesignRegion` and `InverseDesign` keep their
scalar `eps_bounds` and two materials. `TopologySpec.apply_gradient`, which
maps a scalar permittivity gradient, rejects K > 2; use
`apply_parameter_gradient`.

**Export.** `TopologyProblem.export_design` lays the region down in `m_0`.
It then paints `m_1, ..., m_{K-1}` in order where `Q_k` exceeds the
threshold, so binarized designs export exactly with their real (anisotropic)
materials.

**Comparison with fdtdx.** fdtdx sorts materials by permittivity and maps one
latent per cell onto that ordered axis. With more than two materials, its
`ClosestIndex` snaps every cell to the nearest material and uses a
straight-through gradient, and continuous blends exist only for two
materials. The BeamZ channels stay continuous and differentiable for any K,
and impose no ordering by permittivity. A single ordered index puts an `m_1`
layer into every grey `m_0`/`m_2` boundary; nested channels do not have to.

## Requirements and limitations

- These media run on JAX (CPU or GPU) and on single-device
  `backend="cuda_streamed"`. Streamed CUDA advances the vacuum fields with its
  native phases and applies the pole and inverse-tensor corrections with the
  native `beamz_cuda_tensor_media` kernels. Fused multi-step schedules
  (`cuda_mstep`, `schedule="mstep"`) and sharded runs are rejected, as are
  CUDA gradients through these media. Gyrotropic poles, Bloch (complex) fields,
  and boxes spanning a periodic seam keep their corrections in JAX inside the
  streamed run.
- Tensor-media executables compile without XLA command buffers, because XLA
  0.9 crashes replaying CUDA graphs of some of these programs.
- Off-diagonal `eps_inf` needs `RasterOptions(smoothing="farjadpour_full")`.
  Off-diagonal permeability or pole tensors also work with `"volume"`.
- 2D simulations require tensors without `xz` or `yz` coupling, and gyration
  along z.

## Examples

`examples/anisotropic/hyperbolic_asymmetric_refraction.py` launches two mirrored
45-degree beams onto a calcite slab driven in its Reststrahlen band. The optic
axis is tilted 25 degrees above the surface. One beam refracts positively and
the other negatively. Both match the analytic group-velocity directions
(+23 and +53 degrees).

![Asymmetric refraction](assets/anisotropic/hyperbolic_asymmetric_refraction_field.png)

`examples/anisotropic/hyperbolic_isofrequency.py` drives a point source in a
tilted hyperbolic crystal. It uses the same tensor once as a permittivity (TE)
and once as a permeability (TM). The Fourier transforms of the two fields land
on the same analytic hyperbola, as electromagnetic duality requires.

![Iso-frequency contours](assets/anisotropic/hyperbolic_isofrequency.png)

`examples/anisotropic/gyrotropic_oneway_surface_wave.py` launches a surface
magnetoplasmon on a magnetized plasma inside its one-way band. It runs one way
for +B and the other way for -B, and flipping B reverses it.

![One-way surface wave](assets/anisotropic/gyrotropic_oneway_surface_wave.png)
