# Differentiable near-to-far-field projection

Project 3D `FieldMonitor` DFTs into a homogeneous, lossless, nonmagnetic output
medium. Record both tangential E/H components: for a z-normal plane these are
`("Ex", "Ey", "Hx", "Hy")`. Existing Yee interpolation, magnetic half-timestep
alignment and cell-area quadrature are applied automatically.

```python
from beamz.analysis import FarFieldProjection, FarFieldSurface, near_field_box

# One existing planar monitor; the normal points toward the output half-space.
projection = FarFieldProjection(
    surfaces=(FarFieldSurface("aperture", normal_sign=1),),
    directions=((0, 0, 1), (0.6, 0, 0.8)),  # unit Cartesian directions
    refractive_index=1.0,
)
far = projection(results)
# far.electric / far.magnetic: (frequency, direction, xyz)
# far.polarization: complex (E_theta, E_phi); far.intensity: W/sr

# Alternatively add these six tangential monitors before running the simulation.
monitors, surfaces = near_field_box(
    center=(2e-6, 2e-6, 2e-6), size=(2e-6, 2e-6, 2e-6), freqs=[frequency],
)
```

Fields use `Re(E exp(-i omega t))`; angular amplitudes omit `exp(i k r)/r`.
Directions use the global Cartesian frame; polarization uses theta from +z and
phi from +x toward +y (phi=0 at the poles). Coherent targets must share the
projection's angular samples and origin.

For integration, supply `solid_angle_weights` in steradians to the projection
and to `far.power(weights)`. A full-sphere quadrature sums to `4*pi`; for uniform
theta/phi grids include `sin(theta)`, whereas Gauss–Legendre in `cos(theta)`
already includes that measure. Refine angular sampling near collection-cone edges.

## Inverse design and CUDA

```python
import jax
import beamz.optimization as opt

# Here projection includes directions and solid_angle_weights.
cone = opt.CollectionConePower(projection, numerical_aperture=0.7, axis=(0, 0, 1))
beam = opt.TargetBeamOverlap(projection, target_electric)
objective = cone + 0.2 * beam - 0.1 * opt.ModePower("input", direction="-")
design = opt.InverseDesign(
    simulation=simulation, design_region=region,
    backend="cuda", gradient_backend="autodiff", checkpoint_interval=32,
)
value, gradient = jax.value_and_grad(design.make_objective_fn(objective))(
    design.design_region.initial_parameters
)
```

`DirectionalPower(projection)` measures mean intensity / input power (1/sr).
Collection-cone and coherent target-beam objectives measure coupled power /
input power. All objectives are maximized and averaged over selected frequencies.
NA is `n sin(theta_max)`, with `0 < NA <= n`. Targets have shape `(directions, 3)`
or `(frequencies, directions, 3)`; their overall amplitude and phase cancel.

`backend="jax"` supports autodiff and spectral adjoint. Native `backend="cuda"`
requires autodiff, a compatible CUDA extension/JAX installation, a single GPU,
a uniform 3D grid, scalar lossless topology materials and FP32 CPML state.
Projection stays in JAX on the GPU; CUDA supports first-order reverse mode.
`TopologyProblem` accepts the same backend selection and composable objectives.

Run the small six-face example (14,400 cells, 40 fs, one value/gradient):

```bash
python -m examples.optimization.radiation_cuda --backend cuda
# Use --backend jax for the portable reference.
python -m pytest tests/hardware/test_cuda_radiation.py
```

## Scope and limitations

- Keep surfaces in the stated output medium, outside PML and the design region.
  A finite plane truncates the beam; a closed box captures full-space radiation.
  Material/index matching and surface/PML/design intersections are not yet checked.
- Recorded results currently normalize to source 0's spectrum and nominal power;
  use ModeSource/GaussianBeamSource. Raw `near_to_far(...)` accepts already aligned
  SI phasors without source normalization, including manually prepared emitter data.
- 2D radiation, layered/lossy/anisotropic output media and periodic lattice sums
  are excluded. For 2D modal designs with diagnostic field monitors, explicitly
  select ModeMonitor output names to avoid the current default-selection issue.
- Dipole power and surface-invariance checks currently use analytical fields;
  FDTD tests cover gradients and ordinary-result parity. CUDA hardware tests skip
  without a compatible GPU. Converge mesh, runtime, aperture and angular sampling.
