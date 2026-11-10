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

The explicit pole recurrence is stable when

```text
S^2 + (Omega dt / 2)^2 < 1
```

In this condition:

- `S` is the Courant number inside the medium, normalized to its CFL limit.
- `Omega` is the fastest coupled normal-mode frequency of the medium's poles.

Compilation rejects a timestep that violates this and reports the largest
stable `dt`. Sellmeier crystals with ultraviolet poles, such as LiNbO3 and
sapphire, need a Courant number noticeably below the vacuum limit. Use
`beamz.simulation.anisotropic.max_stable_timestep` to compute the bound in
advance.

## Boundaries

- **PML.** A PML stretches coordinates. That amplifies the backward waves of
  hyperbolic (indefinite) media, which shows up as late-time exponential growth
  inside the absorber. BeamZ warns when such a medium overlaps a PML.
- **Absorber.** `bz.Absorber`, a graded-loss sponge, absorbs backward waves. It
  works with full permittivity tensors, permeability tensors, and all poles. Use
  it when a hyperbolic or gyrotropic medium has to reach the domain edge.

## Ports, monitors, and gradients

- **Mode sources.** A mode source inside a medium with electric tensor poles is
  solved at its frequency, using `eps_inf + Re chi(f)`. Broadband sources do this
  at every profile frequency. Ports elsewhere are unaffected by the media.
- **Mode monitors.** Projection uses one static material snapshot, so mode
  monitors must stay outside tensor-pole media.
- **Not yet supported at ports.** No mode port may overlap non-unit or
  magnetic-pole permeability, or gyrotropic media.
- **Gradients.** `Simulation.differentiable()` and `TopologyProblem` (default
  autodiff backend) differentiate through every correction on JAX. Design
  regions must not overlap tensor-permeability or tensor-pole media, including
  their two-cell halo. The spectral-adjoint backend is rejected.

## Requirements and limitations

- These media run on the JAX backend, on CPU or GPU through a CUDA-enabled
  jaxlib. Native CUDA (`cuda_streamed`, `cuda_mstep`) and sharded runs are
  rejected.
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
