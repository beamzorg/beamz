"""Tensor permeability and tensor-pole (hyperbolic) media against analytic oracles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.const import LIGHT_SPEED
from beamz.design.raster import RasterOptions
from beamz.simulation.anisotropic import apply_tensor_operator
from beamz.simulation.dispersion import (
    update_anisotropic_electric,
    update_anisotropic_magnetic,
)

OMEGA = 2 * np.pi * 5e14
TILT = (0.0, 0.0, np.radians(30.0))


def _tilted(**kwargs):
    return bz.AnisotropicMedium(**kwargs).rotated(TILT)


def test_rotation_and_isotropic_limit_match_scalar_models():
    crystal = bz.AnisotropicMedium(
        (2.0, 3.0, 4.0),
        electric_poles=[
            bz.TensorPole.lorentz((1.0, 0.0, 0.0), resonance=6e15, damping=3e14)
        ],
    )
    quarter = crystal.rotated((0.0, 0.0, np.pi / 2))
    np.testing.assert_allclose(quarter.epsilon_r, (3.0, 2.0, 4.0, 0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(
        quarter.electric_poles[0].weight, (0.0, 6e15**2, 0, 0, 0, 0), atol=1e-3
    )
    rotation = bz.design.anisotropic.rotation_matrix(TILT)
    frequency = OMEGA / (2 * np.pi)
    np.testing.assert_allclose(
        crystal.rotated(TILT).permittivity_tensor(frequency),
        rotation @ crystal.permittivity_tensor(frequency) @ rotation.T,
        rtol=1e-12,
    )

    band = (4e14, 8e14)
    frequencies = np.linspace(*band, 7)
    lorentz = bz.AnisotropicMedium(
        2.0, electric_poles=[bz.TensorPole.lorentz(3.0, resonance=6e15, damping=3e14)]
    )
    reference = bz.PoleResidue.lorentz(
        2.0, strength=3.0, resonance=6e15, damping=3e14, frequency_range=band
    )
    np.testing.assert_allclose(
        lorentz.permittivity_tensor(frequencies)[:, 0, 0],
        reference.eps_model(frequencies),
        rtol=1e-12,
    )
    drude = bz.AnisotropicMedium(
        electric_poles=[bz.TensorPole.drude(1.2e16, damping=2e14)]
    )
    reference = bz.PoleResidue.drude(
        1.0, plasma_frequency=1.2e16, damping=2e14, frequency_range=band
    )
    np.testing.assert_allclose(
        drude.permittivity_tensor(frequencies)[:, 2, 2],
        reference.eps_model(frequencies),
        rtol=1e-12,
    )
    assert crystal.rotated((0, 0, 0.3)).cache_spec() != crystal.cache_spec()


def test_rejects_gain_and_static_hyperbolic_tensors():
    with pytest.raises(ValueError, match="semidefinite"):
        bz.TensorPole(0.0, 1e13, ((1.0, 2.0, 0), (2.0, 1.0, 0), (0, 0, 1.0)))
    with pytest.raises(ValueError, match="damping"):
        bz.TensorPole.lorentz(1.0, resonance=1e15, damping=-1.0)
    # A static negative principal value is unconditionally unstable in FDTD.
    with pytest.raises(ValueError, match="positive definite"):
        bz.AnisotropicMedium((-2.0, 3.0, 3.0))


@pytest.mark.parametrize("family", ["E", "H"])
@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_2d_rejects_out_of_plane_tensor_poles(family, polarization, axis):
    rotation = (np.pi / 4, 0, 0) if axis == "x" else (0, np.pi / 4, 0)
    pole = bz.TensorPole.drude((1e15, 1e15, 0), damping=0.0).rotated(rotation)
    medium = bz.AnisotropicMedium(
        2.0,
        **{"electric_poles" if family == "E" else "magnetic_poles": [pole]},
    )
    with pytest.raises(ValueError, match="tensor poles without xz or yz coupling"):
        _program(medium, polarization, width=1.6e-6, cells=16)


def _program(
    medium,
    polarization,
    *,
    width,
    cells,
    background=None,
    smoothing="farjadpour_full",
    dt=None,
    **kwargs,
):
    design = bz.Design(width=width, height=width, material=background or medium)
    if background is not None:
        design += bz.Rectangle(
            position=(width / 4, width / 4),
            width=width / 2,
            height=width / 2,
            material=medium,
        )
    dt = 0.4 * (width / cells) / LIGHT_SPEED if dt is None else dt
    simulation = bz.Simulation(
        design=design,
        resolution=width / cells,
        time=np.arange(kwargs.pop("steps", 8)) * dt,
        polarization=polarization,
        raster_options=RasterOptions(smoothing=smoothing),
        **kwargs,
    )
    return simulation, simulation.compile(backend="jax"), dt


def _site_volumes(box):
    """Yee control volumes of the box sites: half cells on node-aligned ends."""

    from beamz.lattice import component_axis_offsets_3d

    volumes = []
    axes = ("z", "y", "x")[-len(box.shapes[0]) :]
    for component, shape in zip(box.components, box.shapes, strict=True):
        weight = np.ones(shape)
        offsets = component_axis_offsets_3d(component)
        for index, axis in enumerate(axes):
            if offsets[axis] == 0.0:
                edge = [slice(None)] * len(shape)
                for end in (0, -1):
                    edge[index] = end
                    weight[tuple(edge)] *= 0.5
        volumes.append(weight.ravel())
    return np.concatenate(volumes)


@pytest.mark.parametrize("family", ["E", "H"])
def test_tilted_pole_operator_is_symmetric_positive_semidefinite(family):
    """Edge weighting must keep the polarization energy nonnegative.

    A tilted Drude pole has no restoring force, so any negative eigenvalue of
    the discrete coupling grows without bound; this is the boundary
    instability of naive off-diagonal averaging around a finite slab.
    """

    pole = bz.TensorPole.drude((2e15, 0.0, 0.0), damping=0.0)
    medium = _tilted(
        permittivity=2.0,
        **({"electric_poles": [pole]} if family == "E" else {"magnetic_poles": [pole]}),
    )
    _, program, _ = _program(
        medium,
        "te" if family == "E" else "tm",
        width=1.2e-6,
        cells=12,
        background=bz.Material(1.0),
        smoothing="volume",
        boundaries=[bz.PML(edges="all", thickness=0.1e-6, formulation="cpml")],
    )
    plan = program.dispersion.anisotropic
    region = (plan.electric_regions if family == "E" else plan.magnetic_regions)[0]
    box = region.box
    sizes = [int(np.prod(shape)) for shape in box.shapes]
    columns = []
    for column in range(sum(sizes)):
        unit = np.zeros(sum(sizes))
        unit[column] = 1.0
        parts = np.split(unit, np.cumsum(sizes)[:-1])
        vector = tuple(
            jnp.asarray(p.reshape(shape), dtype=jnp.float32)
            for p, shape in zip(parts, box.shapes, strict=True)
        )
        out = apply_tensor_operator(
            vector,
            tuple(d[0] for d in region.diagonal),
            region.offdiagonal[0],
            box,
        )
        columns.append(np.concatenate([np.asarray(o).ravel() for o in out]))
    operator = np.stack(columns, axis=1)
    weighted = _site_volumes(box)[:, None] * operator
    scale = np.abs(weighted).max()
    np.testing.assert_allclose(weighted, weighted.T, atol=1e-6 * scale)
    assert np.linalg.eigvalsh(0.5 * (weighted + weighted.T)).min() > -1e-6 * scale
    assert np.abs(region.offdiagonal).max() > 0


def _harmonic_amplitudes(trace, times):
    matrix = np.c_[np.cos(OMEGA * times), np.sin(OMEGA * times), np.ones(times.size)]
    amplitude = np.linalg.lstsq(matrix, trace, rcond=None)[0]
    return amplitude[0] + 1j * amplitude[1]


@pytest.mark.parametrize("family", ["E", "H"])
@pytest.mark.parametrize("gyration", [0.0, 1e14])
def test_single_component_gyrotropic_response_matches_drude(family, gyration):
    """Precession about z leaves the out-of-plane response unchanged."""

    def run(pole):
        medium = bz.AnisotropicMedium(
            **{"electric_poles" if family == "E" else "magnetic_poles": [pole]}
        )
        simulation, _, _ = _program(
            medium,
            "tm" if family == "E" else "te",
            width=1.6e-6,
            cells=16,
            steps=33,
            dt=0.1e-7 / LIGHT_SPEED,
            boundaries=[bz.Periodic(axes=("x", "y"))],
        )
        state = simulation.initial_state()
        component = "ez" if family == "E" else "hz"
        rng = np.random.default_rng(0)
        state = state._replace(
            **{component: jnp.asarray(rng.normal(size=getattr(state, component).shape))}
        )
        return simulation.advance(
            state=state, num_steps=32, backend="jax", performance=False
        ).state

    actual = run(
        bz.GyrotropicPole.magnetized_plasma(1e15, (0, 0, gyration), damping=1e13)
    )
    expected = run(bz.TensorPole.drude(1e15, damping=1e13))
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        reference = np.asarray(getattr(expected, component))
        np.testing.assert_allclose(
            getattr(actual, component),
            reference,
            rtol=1e-5,
            atol=1e-5 * max(1.0, float(np.abs(reference).max(initial=0.0))),
        )


@pytest.mark.parametrize("kind", ["tensor", "gyrotropic"])
@pytest.mark.parametrize("family", ["E", "H"])
def test_prescribed_flux_yields_inverse_tensor_response(family, kind):
    """Uniform D (or B) driven harmonically settles to eps(w)^-1 D (mu(w)^-1 B).

    Both the off-diagonal pole coupling and the tensor high-frequency term are
    exercised: the response along y to a drive along x is purely off-diagonal.
    Gyrotropic poles add the non-reciprocal (antisymmetric, imaginary) part.
    """

    static = (2.0, 3.0, 1.5)
    if kind == "tensor":
        pole = bz.TensorPole.lorentz((1.5, 0.4, 0.0), resonance=6e15, damping=6e14)
    else:
        pole = bz.GyrotropicPole.lorentz(
            1.5, resonance=6e15, damping=6e14, gyration=(0.0, 0.0, 2e15)
        )
    medium = _tilted(
        permittivity=static if family == "E" else 1.0,
        permeability=static if family == "H" else 1.0,
        **({"electric_poles": [pole]} if family == "E" else {"magnetic_poles": [pole]}),
    )
    dt = 1e-17
    simulation, program, _ = _program(
        medium, "te" if family == "E" else "tm", width=0.2e-6, cells=4, dt=dt
    )
    plan = program.dispersion
    x, y = ("ex", "ey") if family == "E" else ("hx", "hy")
    tensor = medium.permittivity_tensor if family == "E" else medium.permeability_tensor
    inverse_static = (
        np.linalg.inv(np.real(tensor(1e30))[:2, :2]) if family == "E" else None
    )
    update = (
        update_anisotropic_electric if family == "E" else update_anisotropic_magnetic
    )
    steps = 8000

    def step(state, index):
        old = state
        delta = jnp.cos(OMEGA * dt * (index + 1)) - jnp.cos(OMEGA * dt * index)
        if family == "E":
            # The E kernel already maps the flux increment through eps_inf^-1.
            increment = (inverse_static[0, 0] * delta, inverse_static[1, 0] * delta)
        else:
            # The H kernel advances in vacuum; mu_inf^-1 is applied by the plan.
            increment = (delta, 0.0)
        free = state._replace(
            **{
                x: getattr(state, x) + increment[0],
                y: getattr(state, y) + increment[1],
            }
        )
        state = update(old, free, plan)
        return state, (jnp.mean(getattr(state, x)), jnp.mean(getattr(state, y)))

    state = simulation.initial_state()
    _, (trace_x, trace_y) = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(steps)))(
        state
    )
    times = np.arange(1, steps + 1)[-3000:] * dt
    measured = np.array(
        [
            _harmonic_amplitudes(np.asarray(trace_x)[-3000:], times),
            _harmonic_amplitudes(np.asarray(trace_y)[-3000:], times),
        ]
    )
    expected = np.linalg.inv(tensor(OMEGA / (2 * np.pi))[:2, :2])[:, 0]
    # The cross term is small but must still agree to the same relative precision.
    np.testing.assert_allclose(measured, expected, rtol=1e-3, atol=0.0)
    assert abs(expected[1]) > 0.01


def _forward_packet_speed(permeability, polarization):
    """Speed of the forward half of a split y-invariant pulse in a homogeneous medium."""

    cells, width = 320, 160e-6
    spacing = width / cells
    medium = bz.AnisotropicMedium(permittivity=2.0, permeability=permeability)
    dt = 0.4 * spacing / LIGHT_SPEED
    steps = 400
    simulation = bz.Simulation(
        design=bz.Design(width=width, height=4 * spacing, material=medium),
        resolution=spacing,
        time=np.arange(steps + 1) * dt,
        polarization=polarization,
        boundaries=[bz.Periodic(axes=("x", "y"))],
        raster_options=RasterOptions(smoothing="volume"),
    )
    assert simulation.compile(backend="jax").dispersion.anisotropic.magnetic is not None
    name = "ez" if polarization == "tm" else "hz"
    state = simulation.initial_state()
    field = getattr(state, name)
    offset = 0.0 if polarization == "tm" else 0.5
    x = (np.arange(field.shape[1]) + offset) * spacing
    center = 0.5 * width
    pulse = np.exp(-(((x - center) / 8e-6) ** 2))
    state = state._replace(
        **{name: jnp.asarray(np.tile(pulse, (field.shape[0], 1)), dtype=field.dtype)}
    )
    final = simulation.advance(
        state=state, num_steps=steps, backend="jax", performance=False
    ).state
    energy = np.sum(np.asarray(getattr(final, name)) ** 2, axis=0)
    ahead = x > center
    travelled = np.sum(x[ahead] * energy[ahead]) / np.sum(energy[ahead]) - center
    return travelled / (steps * dt)


@pytest.mark.parametrize(
    ("permeability", "polarization", "mu_effective"),
    [
        # TM along x sees B_x = 0, hence mu_eff = det(mu_xy) / mu_xx.
        (((2.0, 0.6, 0.0), (0.6, 1.5, 0.0), (0.0, 0.0, 1.0)), "tm", (3.0 - 0.36) / 2.0),
        ((1.0, 1.0, 2.5), "te", 2.5),
    ],
)
def test_permeability_tensor_sets_packet_speed(
    permeability, polarization, mu_effective
):
    measured = _forward_packet_speed(permeability, polarization)
    np.testing.assert_allclose(
        measured, LIGHT_SPEED / np.sqrt(2.0 * mu_effective), rtol=5e-3
    )


@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("permeability", [0.25, (0.25, 0.25, 0.25, 0.1, 0, 0)])
def test_static_permeability_enforces_material_cfl(permeability, polarization):
    medium = bz.AnisotropicMedium(permeability=permeability)
    spacing = 1e-7
    options = dict(
        width=16 * spacing,
        cells=16,
        steps=201,
        boundaries=[bz.Periodic(axes=("x", "y"))],
    )
    with pytest.raises(ValueError, match="Static tensor media.*CFL"):
        _program(medium, polarization, dt=0.5 * spacing / LIGHT_SPEED, **options)
    simulation, _, _ = _program(
        medium, polarization, dt=0.2 * spacing / LIGHT_SPEED, **options
    )
    state = simulation.initial_state()
    component = "ez" if polarization == "tm" else "ex"
    rng = np.random.default_rng(0)
    state = state._replace(
        **{component: jnp.asarray(rng.normal(size=getattr(state, component).shape))}
    )
    final = simulation.advance(
        state=state, num_steps=200, backend="jax", performance=False
    ).state
    assert float(jnp.abs(getattr(final, component)).max()) < 20.0


def test_lossless_tilted_drude_slab_is_stable_and_resumable():
    """A finite hyperbolic slab with tilted axes stays bounded for many periods."""

    wavelength = 1.5e-6
    omega = 2 * np.pi * LIGHT_SPEED / wavelength
    medium = _tilted(
        permittivity=2.0,
        electric_poles=[bz.TensorPole.drude((2.0 * omega, 0.0, 0.0), damping=0.0)],
    )
    assert np.real(medium.permittivity_tensor(omega / (2 * np.pi))).diagonal()[0] < 0
    width = 4 * wavelength
    spacing = wavelength / 16
    design = bz.Design(width=width, height=width, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(width / 4, width / 4),
        width=width / 2,
        height=width / 2,
        material=medium,
    )
    dt = 0.5 * spacing / LIGHT_SPEED / np.sqrt(2)
    time = np.arange(3001) * dt
    signal = bz.ramped_cosine(
        time,
        amplitude=1.0,
        frequency=omega / (2 * np.pi),
        ramp_duration=3 * wavelength / LIGHT_SPEED,
        t_max=12 * wavelength / LIGHT_SPEED,
    )
    simulation = bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=(width / 2, 0.85 * width), width=wavelength / 6, signal=signal
            )
        ],
        boundaries=[
            bz.PML(edges="all", thickness=0.5 * wavelength, formulation="cpml")
        ],
        time=time,
        resolution=spacing,
        polarization="te",
        raster_options=RasterOptions(smoothing="volume"),
    )
    mid = simulation.advance(num_steps=1500, backend="jax", performance=False).state
    full = simulation.advance(num_steps=3000, backend="jax", performance=False).state
    resumed = simulation.advance(
        state=mid, num_steps=1500, backend="jax", performance=False
    ).state
    peak_mid = max(float(jnp.abs(getattr(mid, c)).max()) for c in ("ex", "ey"))
    peak_end = max(float(jnp.abs(getattr(full, c)).max()) for c in ("ex", "ey"))
    assert np.isfinite(peak_end) and peak_end < 2.0 * peak_mid
    for actual, reference in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-9)


def test_lossy_full_tensor_matches_semi_implicit_recurrence():
    """A uniform field in a conductive tilted crystal decays as (eps+s)^-1 (eps-s)."""

    from beamz.const import EPS_0

    epsilon = np.array([[3.0, 0.8, 0.0], [0.8, 2.0, 0.0], [0.0, 0.0, 2.5]])
    sigma, dt, steps = 2e4, 2e-17, 50
    simulation = bz.Simulation(
        design=bz.Design(
            width=0.4e-6,
            height=0.4e-6,
            material=bz.Material(permittivity=epsilon, conductivity=sigma),
        ),
        resolution=0.05e-6,
        time=np.arange(steps + 1) * dt,
        polarization="te",
        boundaries=[bz.Periodic(axes=("x", "y"))],
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )
    state = simulation.initial_state()
    state = state._replace(ex=jnp.ones_like(state.ex), ey=jnp.zeros_like(state.ey))
    final = simulation.advance(
        state=state, num_steps=steps, backend="jax", performance=False
    ).state
    loss = sigma * dt / (2 * EPS_0) * np.eye(2)
    step = np.linalg.solve(epsilon[:2, :2] + loss, epsilon[:2, :2] - loss)
    expected = np.linalg.matrix_power(step, steps) @ np.array([1.0, 0.0])
    measured = np.array([float(jnp.mean(final.ex)), float(jnp.mean(final.ey))])
    np.testing.assert_allclose(measured, expected, rtol=1e-5)
    assert expected[1] > 0.1  # loss couples through the off-diagonal tensor


def _edge_to_edge_slab(boundary, medium, polarization, steps):
    wavelength = 1.5e-6
    frequency = LIGHT_SPEED / wavelength
    width = 4 * wavelength
    spacing = wavelength / 16
    design = bz.Design(width=width, height=width, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.0, width / 4), width=width, height=width / 2, material=medium
    )
    dt = 0.5 * spacing / LIGHT_SPEED / np.sqrt(2)
    time = np.arange(steps + 1) * dt
    signal = bz.ramped_cosine(
        time,
        amplitude=1.0,
        frequency=frequency,
        ramp_duration=3 * wavelength / LIGHT_SPEED,
        t_max=10 * wavelength / LIGHT_SPEED,
    )
    return bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=(width / 2, 0.9 * width), width=wavelength / 6, signal=signal
            )
        ],
        boundaries=[boundary],
        time=time,
        resolution=spacing,
        polarization=polarization,
        raster_options=RasterOptions(smoothing="volume"),
    )


@pytest.mark.filterwarnings("ignore:Absorber material varies:RuntimeWarning")
@pytest.mark.parametrize("family", ["E", "H"])
def test_hyperbolic_slab_through_absorber_stays_bounded(family):
    """Loss-based sponges absorb backward waves that a PML would amplify."""

    omega = 2 * np.pi * LIGHT_SPEED / 1.5e-6
    pole = [bz.TensorPole.drude((2.0 * omega, 0.0, 0.0), damping=0.01 * omega)]
    medium = _tilted(
        permittivity=2.0 if family == "E" else 1.0,
        permeability=1.0 if family == "E" else (2.0, 1.2, 1.0),
        **({"electric_poles": pole} if family == "E" else {"magnetic_poles": pole}),
    )
    polarization = "te" if family == "E" else "tm"
    with pytest.warns(UserWarning, match="backward waves"):
        _edge_to_edge_slab(
            bz.PML(edges="all", thickness=0.75e-6, formulation="cpml"),
            medium,
            polarization,
            4,
        ).compile(backend="jax")
    simulation = _edge_to_edge_slab(
        bz.Absorber(edges="all", thickness=1.5e-6), medium, polarization, 4000
    )
    state = simulation.initial_state()
    names = ("ex", "ey", "hz") if family == "E" else ("ez", "hx", "hy")
    peaks = []
    for _ in range(4):
        state = simulation.advance(
            state=state, num_steps=1000, backend="jax", performance=False
        ).state
        peaks.append(max(float(jnp.abs(getattr(state, n)).max()) for n in names))
    assert np.all(np.isfinite(peaks))
    # The source stops after ~1300 steps; the sponge then drains the slab.
    assert peaks[-1] < 0.5 * max(peaks)


def _guide(core, frequency):
    spacing = 50e-9
    design = bz.Design(width=6e-6, height=3e-6, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.0, 1.3e-6), width=6e-6, height=0.4e-6, material=core
    )
    dt = 0.8 * spacing / (LIGHT_SPEED * np.sqrt(2))
    return bz.Simulation(
        design=design,
        resolution=spacing,
        time=np.arange(1400) * dt,
        sources=[
            bz.ModeSource(
                center=(1.2e-6, 1.5e-6, 0.0),
                size=(0.0, 2.0e-6, 1.0e-6),
                direction="+",
                mode_spec=bz.ModeSpec(polarization="tm"),
                source_time=bz.GaussianPulse(freq0=frequency, fwidth=frequency / 4),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(4.5e-6, 1.5e-6, 0.0),
                size=(0.0, 3e-6, 0.0),
                freqs=[frequency],
                fields=("Ez",),
                name="line",
            )
        ],
        boundaries=[bz.PML(thickness=0.5e-6, formulation="cpml")],
        normalize_source=None,
    )


def test_mode_source_in_tensor_pole_core_matches_static_equivalent():
    """A port inside a dispersive core is solved with eps(f0): at f0 the
    launched and transmitted field equals that of the static eps(f0) guide."""

    from beamz.design.material_views import static_material_grid

    frequency = LIGHT_SPEED / 1e-6
    omega = 2 * np.pi * frequency
    core = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.TensorPole.lorentz(16.0 / 9.0, resonance=3 * omega, damping=0.0)
        ],
    )
    np.testing.assert_allclose(core.permittivity_tensor(frequency)[2, 2], 4.0)
    dispersive = _guide(core, frequency)
    static = _guide(bz.Material(4.0), frequency)
    view = static_material_grid(dispersive._material_grid(), frequency)
    np.testing.assert_allclose(
        np.asarray(view.permittivity), np.asarray(static._material_grid().permittivity)
    )
    fields = [
        np.asarray(
            sim.run(backend="jax", performance=False).monitor("line").dft_fields["Ez"]
        ).ravel()
        for sim in (dispersive, static)
    ]
    error = np.linalg.norm(fields[0] - fields[1]) / np.linalg.norm(fields[1])
    assert error < 0.02, error


def _modal_to_flux_ratio(core, source_frequency, frequencies):
    plane = {"center": (4.5e-6, 1.5e-6, 0.0), "size": (0.0, 2.0e-6, 1.0e-6)}
    simulation = _guide(core, source_frequency).updated_copy(
        monitors=[
            bz.ModeMonitor(
                **plane,
                mode_spec=bz.ModeSpec(polarization="tm"),
                freqs=frequencies,
                name="port",
            ),
            bz.FluxMonitor(**plane, freqs=frequencies, name="flux"),
        ]
    )
    result = simulation.run(backend="jax", performance=False)
    modal = result.mode("port").amps.sel(direction="+", mode_index=0).values
    flux = np.asarray(result.monitor("flux").get_dft_flux()).ravel()
    return np.abs(modal) ** 2 / np.abs(flux)


def test_mode_monitor_in_tensor_pole_core_projects_per_frequency():
    """A mode monitor inside a dispersive core uses eps_inf + Re chi(f) per f.

    At each monitor frequency the modal-to-flux power ratio must equal that of
    a static core with permittivity eps(f) (the shared few-percent excess is
    this coarse grid's projection bias). One static snapshot for all
    frequencies would not.
    """

    frequency = LIGHT_SPEED / 1e-6
    omega = 2 * np.pi * frequency
    core = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.TensorPole.lorentz(16.0 / 9.0, resonance=3 * omega, damping=0.0)
        ],
    )
    frequencies = frequency * np.array([0.9, 1.1])
    dispersive = _modal_to_flux_ratio(core, frequency, frequencies)
    for index, f in enumerate(frequencies):
        static_core = bz.Material(float(core.permittivity_tensor(f)[2, 2].real))
        static = _modal_to_flux_ratio(static_core, frequency, frequencies)[index]
        np.testing.assert_allclose(dispersive[index], static, rtol=1e-3)


def test_mode_ports_reject_gyrotropic_media_at_compile_time():
    frequency = LIGHT_SPEED / 1e-6
    core = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.GyrotropicPole.lorentz(
                1.0,
                resonance=6 * np.pi * frequency,
                damping=0.0,
                gyration=(0.0, 0.0, 1e14),
            )
        ],
    )
    simulation = _guide(core, frequency).updated_copy(
        sources=[],
        monitors=[
            bz.ModeMonitor(
                center=(4.5e-6, 1.5e-6, 0.0),
                size=(0.0, 2.0e-6, 1.0e-6),
                mode_spec=bz.ModeSpec(polarization="tm"),
                freqs=[frequency],
                name="port",
            )
        ],
    )
    with pytest.raises(ValueError, match="gyrotropic or magnetic-pole"):
        simulation.compile(backend="jax")


def _graded_edges(extent, count, stretch=0.3):
    unit = np.linspace(0.0, 1.0, count + 1)
    return extent * (unit + stretch * np.sin(2.0 * np.pi * unit) / (2.0 * np.pi))


def _graded_material_grid(medium, background, polarization, extent, cells):
    design = bz.Design(width=extent, height=extent, material=background or medium)
    if background is not None:
        design += bz.Rectangle(
            position=(extent / 4, extent / 4),
            width=extent / 2,
            height=extent / 2,
            material=medium,
        )
    grid = bz.RectilinearGrid(
        _graded_edges(extent, cells),
        _graded_edges(extent, cells, stretch=-0.25),
        np.array([0.0, 1.0]),
    )
    return design.rasterize(grid, polarization=polarization, smoothing="volume")


def _box_volumes(box, widths):
    """Yee control volumes on a graded box: dual widths on node-aligned axes."""

    from beamz.lattice import component_axis_offsets_3d

    axes = ("z", "y", "x")[-len(box.shapes[0]) :]
    volumes = []
    for component, shape in zip(box.components, box.shapes, strict=True):
        weight = np.ones(shape)
        offsets = component_axis_offsets_3d(component)
        for index, axis in enumerate(axes):
            cell = np.asarray(widths["xyz".index(axis)], dtype=float)
            if offsets[axis] == 0.0:
                dual = np.r_[
                    0.5 * cell[0], 0.5 * (cell[:-1] + cell[1:]), 0.5 * cell[-1]
                ]
            else:
                dual = cell
            reshape = [1] * len(shape)
            reshape[index] = dual.size
            weight = weight * dual.reshape(reshape)
        volumes.append(weight.ravel())
    return np.concatenate(volumes)


@pytest.mark.parametrize("family", ["E", "H"])
def test_graded_grid_pole_operator_is_volume_symmetric_and_semidefinite(family):
    pole = bz.TensorPole.drude((2e15, 0.0, 0.0), damping=0.0)
    medium = _tilted(
        permittivity=2.0,
        **({"electric_poles": [pole]} if family == "E" else {"magnetic_poles": [pole]}),
    )
    polarization = "te" if family == "E" else "tm"
    material_grid = _graded_material_grid(
        medium, bz.Material(1.0), polarization, 1.2e-6, 14
    )
    assert material_grid.metric_kind != "isotropic_uniform"
    simulation = bz.Simulation(
        material_grid=material_grid,
        time=np.arange(4) * 1e-18,
        polarization=polarization,
        boundaries=[bz.PML(edges="all", thickness=0.1e-6, formulation="cpml")],
    )
    plan = simulation.compile(backend="jax").dispersion.anisotropic
    region = (plan.electric_regions if family == "E" else plan.magnetic_regions)[0]
    box = region.box
    assert box.widths is not None
    sizes = [int(np.prod(shape)) for shape in box.shapes]
    columns = []
    for column in range(sum(sizes)):
        unit = np.zeros(sum(sizes))
        unit[column] = 1.0
        parts = np.split(unit, np.cumsum(sizes)[:-1])
        out = apply_tensor_operator(
            tuple(
                jnp.asarray(p.reshape(s), dtype=jnp.float32)
                for p, s in zip(parts, box.shapes, strict=True)
            ),
            tuple(d[0] for d in region.diagonal),
            region.offdiagonal[0],
            box,
        )
        columns.append(np.concatenate([np.asarray(o).ravel() for o in out]))
    weighted = _box_volumes(box, box.widths)[:, None] * np.stack(columns, axis=1)
    scale = np.abs(weighted).max()
    np.testing.assert_allclose(weighted, weighted.T, atol=2e-6 * scale)
    assert np.linalg.eigvalsh(0.5 * (weighted + weighted.T)).min() > -2e-6 * scale


def test_permeability_tensor_packet_speed_on_graded_grid():
    """Off-diagonal mu collocation keeps its accuracy on a graded mesh."""

    mu = ((2.0, 0.6, 0.0), (0.6, 1.5, 0.0), (0.0, 0.0, 1.0))
    medium = bz.AnisotropicMedium(permittivity=2.0, permeability=mu)
    width, cells = 160e-6, 360
    x_edges = _graded_edges(width, cells, stretch=0.4)
    spacing = width / cells
    grid = bz.RectilinearGrid(x_edges, np.arange(5) * spacing, np.array([0.0, 1.0]))
    design = bz.Design(width=width, height=4 * spacing, material=medium)
    material_grid = design.rasterize(grid, polarization="tm", smoothing="volume")
    dt = grid.cfl_time_step(0.4, active_axes=("x", "y"))
    steps = int(round(140e-15 / dt))
    simulation = bz.Simulation(
        material_grid=material_grid,
        time=np.arange(steps + 1) * dt,
        polarization="tm",
        boundaries=[bz.Periodic(axes=("x", "y"))],
    )
    state = simulation.initial_state()
    center = 0.5 * width
    pulse = np.exp(-(((x_edges - center) / 8e-6) ** 2))
    state = state._replace(
        ez=jnp.asarray(np.tile(pulse, (state.ez.shape[0], 1)), dtype=state.ez.dtype)
    )
    final = simulation.advance(
        state=state, num_steps=steps, backend="jax", performance=False
    ).state
    energy = np.sum(np.asarray(final.ez) ** 2, axis=0)
    dual = np.r_[x_edges[1] - x_edges[0], 0.5 * (x_edges[2:] - x_edges[:-2]), 0]
    ahead = x_edges > center
    weights = (energy * dual)[ahead]
    travelled = np.sum(x_edges[ahead] * weights) / np.sum(weights) - center
    mu_effective = (2.0 * 1.5 - 0.36) / 2.0
    np.testing.assert_allclose(
        travelled / (steps * dt), LIGHT_SPEED / np.sqrt(2.0 * mu_effective), rtol=5e-3
    )


def test_pole_timestep_bound_is_enforced_and_sufficient():
    """S^2 + (Omega dt / 2)^2 < 1 guards the explicit recurrence: just below
    the bound a lossless multi-pole tilted crystal stays bounded from random
    initial fields; just above it, compilation is refused."""

    from beamz.simulation.anisotropic import max_stable_timestep

    def lorentz(strength, wavelength_um):
        resonance = 2 * np.pi * LIGHT_SPEED / (wavelength_um * 1e-6)
        return strength, resonance

    terms = [((s, 0, s), w) for s, w in (lorentz(2.67, 0.133), lorentz(12.6, 21.8))]
    terms += [((0, s, 0), w) for s, w in (lorentz(2.98, 0.143), lorentz(8.95, 20.4))]
    medium = bz.AnisotropicMedium(
        1.0,
        electric_poles=[
            bz.TensorPole.lorentz(strength, resonance=w, damping=0.0)
            for strength, w in terms
        ],
    ).rotated((0.0, 0.0, 0.5))
    spacing = 30e-9
    limit = max_stable_timestep(medium, "E", (spacing, spacing))

    def simulation(dt):
        return bz.Simulation(
            design=bz.Design(width=24 * spacing, height=24 * spacing, material=medium),
            resolution=spacing,
            time=np.arange(6001) * dt,
            polarization="te",
            boundaries=[bz.Periodic(axes=("x", "y"))],
            raster_options=RasterOptions(smoothing="volume"),
        )

    with pytest.raises(ValueError, match="unstable"):
        simulation(1.02 * limit).compile(backend="jax")
    stable = simulation(0.95 * limit)
    state = stable.initial_state()
    rng = np.random.default_rng(0)
    state = state._replace(
        ex=jnp.asarray(rng.normal(size=state.ex.shape), dtype=jnp.float32),
        ey=jnp.asarray(rng.normal(size=state.ey.shape), dtype=jnp.float32),
    )
    final = stable.advance(
        state=state, num_steps=6000, backend="jax", performance=False
    ).state
    assert float(jnp.abs(final.ex).max()) < 20.0


def _ring_radius(medium, wavelength, polarization="te"):
    """Peak radius (units of k0) of the CW point-source k-space ring in a medium.

    The windowed estimator is biased by a few percent, identically for any
    medium, so results are compared between media rather than to theory.
    """

    frequency = LIGHT_SPEED / wavelength
    omega = 2 * np.pi * frequency
    spacing = wavelength / 20
    cells = 240
    width = cells * spacing
    margin = 20 * spacing
    design = bz.Design(width=width, height=width, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(margin, margin),
        width=width - 2 * margin,
        height=width - 2 * margin,
        material=medium,
    )
    dt = 0.5 * spacing / (LIGHT_SPEED * np.sqrt(2))
    time = np.arange(0.0, 70 * wavelength / LIGHT_SPEED, dt)
    ramp = np.clip(time / (10 * wavelength / LIGHT_SPEED), 0, 1)
    component = "Hz" if polarization == "te" else "Ez"
    simulation = bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=(width / 2, width / 2),
                width=spacing,
                signal=ramp * np.sin(omega * time),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(width / 2, width / 2, 0.0),
                size=(width, width, 0.0),
                freqs=[frequency],
                fields=(component,),
                name="field",
            )
        ],
        boundaries=[bz.PML(edges="all", thickness=10 * spacing, formulation="cpml")],
        time=time,
        resolution=spacing,
        polarization=polarization,
    )
    result = simulation.run(backend="jax", performance=False)
    field = np.asarray(result.monitor("field").dft_fields[component]).reshape(-1)
    side = int(round(np.sqrt(field.size)))
    inner = field.reshape(side, side)[30:-30, 30:-30]
    inner = inner * np.outer(np.hanning(inner.shape[0]), np.hanning(inner.shape[1]))
    size = 1024
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(inner, s=(size, size))))
    k = np.fft.fftshift(np.fft.fftfreq(size, spacing)) * wavelength
    kx, ky = np.meshgrid(k, k)
    radius = np.hypot(kx, ky)
    bins = np.linspace(0.3, 1.5, 241)
    profile = [
        spectrum[(radius >= a) & (radius < b)].mean()
        for a, b in zip(bins[:-1], bins[1:], strict=True)
    ]
    return 0.5 * (bins[:-1] + bins[1:])[int(np.argmax(profile))], bins[1] - bins[0]


def test_magnetized_plasma_voigt_wave_has_extraordinary_index():
    """TE waves across a z-magnetized plasma see eps_v = (exx^2 + exy^2) / exx.

    The gyrotropic run must reproduce a static dielectric of permittivity
    eps_v under the same measurement, and eps_v must differ clearly from the
    unmagnetized plasma.
    """

    wavelength = 1.5e-6
    frequency = LIGHT_SPEED / wavelength
    omega = 2 * np.pi * frequency
    plasma = bz.AnisotropicMedium(
        electric_poles=[
            bz.GyrotropicPole.magnetized_plasma(
                0.5 * omega, (0.0, 0.0, 0.6 * omega), damping=0.004 * omega
            )
        ]
    )
    eps = plasma.permittivity_tensor(frequency)
    voigt = ((eps[0, 0] ** 2 + eps[0, 1] ** 2) / eps[0, 0]).real
    unmagnetized = 1.0 - 0.25  # 1 - (omega_p / omega)^2
    assert abs(np.sqrt(voigt) - np.sqrt(unmagnetized)) > 0.1
    gyrotropic, bin_width = _ring_radius(plasma, wavelength)
    reference, _ = _ring_radius(bz.Material(float(voigt)), wavelength)
    assert abs(gyrotropic - reference) <= bin_width + 1e-12


def test_gyrotropic_pole_without_gyration_reproduces_drude_tensor_pole():
    """The diagonal plasma response stays local (exact at every wavevector).

    Averaging the diagonal response through a shared support makes a strongly
    negative plasma spuriously hyperbolic at large k (fake resonance cones);
    the gyrotropic update must reduce to the site-local Drude update.
    """

    wavelength = 1.5e-6
    omega = 2 * np.pi * LIGHT_SPEED / wavelength
    fields = []
    for pole in (
        bz.GyrotropicPole.magnetized_plasma(
            1.3 * omega, (0, 0, 0), damping=0.01 * omega
        ),
        bz.TensorPole.drude(1.3 * omega, damping=0.01 * omega),
    ):
        simulation = _edge_to_edge_slab(
            bz.PML(edges="all", thickness=0.4e-6, formulation="cpml"),
            bz.AnisotropicMedium(electric_poles=[pole]),
            "te",
            1500,
        )
        state = simulation.advance(
            num_steps=700, backend="jax", performance=False
        ).state
        fields.append(np.asarray(state.hz))
    error = np.linalg.norm(fields[0] - fields[1]) / np.linalg.norm(fields[1])
    assert error < 1e-4, error


@pytest.mark.parametrize("family", ["H", "E", "static_eps"])
def test_tensor_media_keep_bloch_plane_waves_pure_across_the_seam(family):
    """Cross terms average across Bloch seams with their phases.

    A Bloch plane wave exp(i (k x + q y)) in a homogeneous tilted medium
    filling the periodic cell must stay a single spatial harmonic in every
    component; one-sided (copy) or phase-free seam averaging scatters it.
    """

    spacing, cells = 0.1e-6, 24
    width = cells * spacing
    k, q = 2 * np.pi * 0.3 / width, 2 * np.pi * 0.17 / width
    if family == "H":
        medium = bz.AnisotropicMedium(2.0, (2.0, 1.4, 1.0)).rotated(TILT)
        polarization = "tm"
    elif family == "E":
        omega = 2 * np.pi * LIGHT_SPEED / 1.5e-6
        medium = _tilted(
            permittivity=2.0,
            electric_poles=[
                bz.TensorPole.lorentz((1.0, 0.0, 0.0), resonance=2 * omega, damping=0.0)
            ],
        )
        polarization = "te"
    else:
        # The full-tensor permittivity kernel's own cross terms.
        medium = bz.Material(
            permittivity=((3.0, 0.8, 0.0), (0.8, 2.0, 0.0), (0.0, 0.0, 2.5))
        )
        polarization = "te"
    name = "ez" if polarization == "tm" else "hz"
    simulation = bz.Simulation(
        design=bz.Design(width=width, height=width, material=medium),
        resolution=spacing,
        time=np.arange(301) * 0.4 * spacing / LIGHT_SPEED,
        polarization=polarization,
        boundaries=[bz.Bloch(axes=("x", "y"), wavevector=(k, q, 0.0))],
        raster_options=RasterOptions(
            smoothing="farjadpour_full" if family == "static_eps" else "volume"
        ),
    )
    state = simulation.initial_state()
    field = getattr(state, name)
    y, x = np.meshgrid(
        np.arange(field.shape[0]), np.arange(field.shape[1]), indexing="ij"
    )
    offset = 0.0 if name == "ez" else 0.5
    wave = np.exp(1j * ((x + offset) * k + (y + offset) * q) * spacing)
    state = state._replace(**{name: jnp.asarray(wave, dtype=field.dtype)})
    final = simulation.advance(
        state=state, num_steps=300, backend="jax", performance=False
    ).state
    names = ("ez", "hx", "hy") if polarization == "tm" else ("hz", "ex", "ey")
    for component in names:
        values = np.asarray(getattr(final, component))[:cells, :cells]
        rows, columns = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        # Every component of a Bloch eigenfield is the same single harmonic.
        basis = np.exp(1j * (columns * k + rows * q) * spacing)
        amplitude = np.vdot(basis, values) / np.vdot(basis, basis)
        residual = np.linalg.norm(values - amplitude * basis) / np.linalg.norm(values)
        assert residual < 1e-4, (component, residual)


@pytest.mark.filterwarnings("ignore:Absorber material varies:RuntimeWarning")
def test_differentiable_simulation_through_tensor_poles_is_consistent():
    """Reverse mode through tensor-pole corrections equals forward mode."""

    wavelength = 1.5e-6
    omega = 2 * np.pi * LIGHT_SPEED / wavelength
    medium = _tilted(
        permittivity=2.0,
        electric_poles=[
            bz.TensorPole.lorentz((1.5, 0.5, 0.0), resonance=2 * omega, damping=0.0)
        ],
    )
    simulation = _edge_to_edge_slab(
        bz.Absorber(edges="all", thickness=0.5e-6), medium, "te", 400
    )
    solver = simulation.differentiable(backend="jax", checkpoint_interval=50)
    grid = simulation.compile(backend="jax").grid
    permittivity = tuple(
        jnp.asarray(
            np.broadcast_to(
                np.asarray(getattr(grid, "eps_" + a)), getattr(grid, "E" + a).shape
            ),
            jnp.float32,
        )
        for a in "xyz"
    )
    rng = np.random.default_rng(1)
    tangent = []
    for axis, values in zip("xyz", permittivity, strict=True):
        change = np.zeros(values.shape, np.float32)
        if axis != "z":
            change[2:12, 30:40] = rng.normal(
                size=(10, 10)
            )  # vacuum, away from the slab
        tangent.append(jnp.asarray(change))
    tangent = tuple(tangent)

    def objective(eps):
        state = solver(permittivity=eps)
        return jnp.sum(state.hz**2) + jnp.sum(state.ex * 0.3)

    _, forward = jax.jvp(objective, (permittivity,), (tangent,))
    gradient = jax.grad(objective)(permittivity)
    reverse = sum(jnp.sum(g * t) for g, t in zip(gradient, tangent, strict=True))
    np.testing.assert_allclose(float(reverse), float(forward), rtol=1e-4)


def test_dispersive_monitor_materials_survive_result_archives(tmp_path):
    from beamz.simulation.result_io import load_results, save_results

    frequency = LIGHT_SPEED / 1e-6
    omega = 2 * np.pi * frequency
    core = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.TensorPole.lorentz(16.0 / 9.0, resonance=3 * omega, damping=0.0)
        ],
    )
    simulation = _guide(core, frequency).updated_copy(
        monitors=[
            bz.ModeMonitor(
                center=(4.5e-6, 1.5e-6, 0.0),
                size=(0.0, 2.0e-6, 1.0e-6),
                mode_spec=bz.ModeSpec(polarization="tm"),
                freqs=[frequency],
                name="port",
            )
        ]
    )
    result = simulation.run(backend="jax", performance=False)
    path = tmp_path / "result.npz"
    save_results(result, path)
    loaded = load_results(path)
    region = loaded.monitor("port").material_region
    assert region.dispersion_folding is not None
    np.testing.assert_allclose(
        region.at_frequency(frequency).permittivity,
        result.monitor("port").material_region.at_frequency(frequency).permittivity,
    )
    np.testing.assert_allclose(
        loaded.mode("port").amps.values, result.mode("port").amps.values
    )


def test_folding_fast_poles_keeps_low_frequency_response_and_relaxes_timestep():
    """Ultraviolet Sellmeier poles can be slowed without changing the band.

    Folding preserves the static limit exactly and the telecom-band
    permittivity to ~1e-3, and lifts the pole timestep limit so a coarse-grid
    run the original crystal rejects becomes stable.
    """

    from beamz.material_library import anisotropic_library
    from beamz.simulation.anisotropic import max_stable_timestep

    crystal = anisotropic_library["LiNbO3"].medium.rotated((0.0, np.pi / 2, 0.0))
    folded = crystal.fold_fast_poles(2 * np.pi * LIGHT_SPEED / 0.4e-6)
    near_static = np.array([1e9])
    np.testing.assert_allclose(
        folded.permittivity_tensor(near_static),
        crystal.permittivity_tensor(near_static),
        rtol=1e-9,
        atol=1e-12,
    )
    band = LIGHT_SPEED / np.linspace(1.2e-6, 1.7e-6, 101)
    reference = crystal.permittivity_tensor(band)
    error = np.abs(folded.permittivity_tensor(band) - reference).max()
    assert error < 2e-3 * np.abs(reference).max()
    for pole in (*folded.electric_poles,):
        assert (
            np.linalg.eigvalsh(pole.weight_matrix).min()
            > -1e-6 * np.abs(pole.weight_matrix).max()
        )

    spacing = 60e-9
    original_limit = max_stable_timestep(crystal, "E", (spacing, spacing))
    folded_limit = max_stable_timestep(folded, "E", (spacing, spacing))
    dt = 0.95 * min(folded_limit, spacing / (LIGHT_SPEED * np.sqrt(2)))
    assert dt > 1.3 * original_limit

    def simulation(medium):
        return bz.Simulation(
            design=bz.Design(width=24 * spacing, height=24 * spacing, material=medium),
            resolution=spacing,
            time=np.arange(4001) * dt,
            polarization="te",
            boundaries=[bz.Periodic(axes=("x", "y"))],
            raster_options=RasterOptions(smoothing="farjadpour_full"),
        )

    with pytest.raises(ValueError, match="fold_fast_poles"):
        simulation(crystal).compile(backend="jax")
    stable = simulation(folded)
    state = stable.initial_state()
    rng = np.random.default_rng(2)
    state = state._replace(
        ex=jnp.asarray(rng.normal(size=state.ex.shape), dtype=jnp.float32),
        ey=jnp.asarray(rng.normal(size=state.ey.shape), dtype=jnp.float32),
    )
    final = stable.advance(
        state=state, num_steps=4000, backend="jax", performance=False
    ).state
    assert float(jnp.abs(final.ex).max()) < 20.0


def test_tensor_media_lower_to_jax_or_streamed_cuda_only():
    from dataclasses import replace

    from beamz.simulation.backend import CudaBackendUnavailable
    from beamz.simulation.execute import scan_compiler_options

    medium = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.TensorPole.lorentz((1.5, 0.5, 0.2), resonance=2.5e15, damping=1e13)
        ],
    ).rotated((0.2, 0.3, 0.4))
    length = 1.2e-6
    design = bz.Design(width=length, height=length, depth=length)
    design += bz.Box(
        center=(length / 2,) * 3, size=(length / 3,) * 3, material=medium
    )
    sim = bz.Simulation(
        design=design,
        resolution=0.1e-6,
        time=np.arange(4) * 1e-17,
        setup_device="cpu",
    )
    with pytest.raises(CudaBackendUnavailable, match="backend='cuda_streamed'"):
        sim.compile(backend="cuda_mstep")
    with pytest.raises(ValueError, match="single-device"):
        sim.compile(backend="jax", sharding=True)

    program = sim.compile(backend="jax")
    assert "xla_gpu_enable_command_buffer" not in scan_compiler_options(program)
    streamed = replace(program, config=replace(program.config, backend="cuda_streamed"))
    # Per-step native phases must stay out of XLA command buffers.
    assert "CUSTOM_CALL" not in scan_compiler_options(streamed).get(
        "xla_gpu_enable_command_buffer", "CUSTOM_CALL"
    )
