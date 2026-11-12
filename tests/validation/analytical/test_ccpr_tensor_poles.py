"""General complex-conjugate pole-residue (CCPR) tensor poles against analytic oracles.

A CCPR pole adds an ``E'`` coupling ``B`` to the tensor-pole recurrence; see
:mod:`beamz.simulation.ccpr`. The oracles here are the exact discrete-time
response of that recurrence, the continuous susceptibility, scalar
``PoleResidue`` simulations, thin-film optics, and Jones calculus for tilted
crystals. Negative controls drop the ``B`` term or the implicit ``eps'``
augmentation and must fail the same checks.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.const import LIGHT_SPEED
from beamz.design.anisotropic import (
    packed_to_matrix,
    pole_residue_tensor_poles,
    rotation_matrix,
)
from beamz.design.raster import RasterOptions
from beamz.simulation import ccpr, pole_stability
from beamz.simulation.dispersion import (
    update_anisotropic_electric,
    update_anisotropic_magnetic,
)

BAND = (1e14, 8e14)
# A fitted-style model: one CCPR pair with Re(residue) != 0 (b != 0) and a
# Debye relaxation (real pole, pure b term).
MODEL = bz.PoleResidue(
    2.0, [(-5e14 - 3e15j, 2e14 + 1e15j), (-5e14, 2.5e14)], frequency_range=BAND
)


def _other(epsilon_inf=2.0):
    return bz.PoleResidue(
        epsilon_inf,
        [(-8e14 - 2e15j, 5e14 + 1.5e15j), (-3e14, 6e14)],
        frequency_range=BAND,
    )


TILT = (0.0, 0.0, np.radians(30.0))


def _without_edot(poles):
    """The same poles with b = 0 (CCPR poles, so no semidefinite requirement)."""
    return tuple(
        bz.TensorPole(p.resonance, p.damping, p.weight, np.zeros(6)) for p in poles
    )


def _discrete_tensor(static, poles, omega, dt):
    """Exact steady-state response of the implemented recurrence at ``omega``."""
    z = np.exp(-1j * omega * dt)
    value = packed_to_matrix(np.asarray(static)).astype(complex)
    for p in poles:
        den = 1 + 0.5 * p.damping * dt
        response = (
            den * z - (2 - (p.resonance * dt) ** 2) + (1 - 0.5 * p.damping * dt) / z
        )
        value = (
            value + (dt**2 * p.weight_matrix + dt * (z - 1) * p.edot_matrix) / response
        )
    return value


# ---------------------------------------------------------------------------
# Constructors and conversion


def test_ccpr_constructors_match_pole_residue_and_critical_point_forms():
    frequencies = np.geomspace(1e13, 2e15, 41)
    s = -2j * np.pi * frequencies
    q, r = -3e14 - 2.5e15j, 4e14 + 7e14j
    pole = bz.TensorPole.ccpr(q, r)
    np.testing.assert_allclose(pole.resonance, abs(q))
    np.testing.assert_allclose(pole.damping, -2 * q.real)
    np.testing.assert_allclose(pole.weight[0], -2 * (r * np.conj(q)).real)
    np.testing.assert_allclose(pole.edot[0], 2 * r.real)
    np.testing.assert_allclose(
        pole.susceptibility(frequencies)[:, 1, 1],
        r / (s - q) + np.conj(r) / (s - np.conj(q)),
        rtol=1e-12,
    )

    amplitude, phase, resonance, damping = 1.3, -np.pi / 4, 4e15, 5e14
    critical = bz.TensorPole.critical_point(
        amplitude, phase, resonance=resonance, damping=damping
    )
    omega = 2 * np.pi * frequencies
    expected = (
        amplitude
        * resonance
        * (
            np.exp(1j * phase) / (resonance - omega - 1j * damping)
            + np.exp(-1j * phase) / (resonance + omega + 1j * damping)
        )
    )
    np.testing.assert_allclose(
        critical.susceptibility(frequencies)[:, 2, 2], expected, rtol=1e-12
    )

    oriented = bz.TensorPole.ccpr(q, r, orientation=(1.0, 1.0, 0.0))
    u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    np.testing.assert_allclose(
        oriented.susceptibility(frequencies),
        pole.susceptibility(frequencies)[:, :1, :1] * np.outer(u, u),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="Re\\(pole\\)"):
        bz.TensorPole.ccpr(1e14 - 1e15j, r)
    with pytest.raises(ValueError, match="either tensor or orientation"):
        bz.TensorPole.ccpr(q, r, tensor=2.0, orientation=(1, 0, 0))


@pytest.mark.parametrize(
    "model",
    [
        MODEL,
        bz.PoleResidue.drude(
            1.0, plasma_frequency=1.2e16, damping=2e14, frequency_range=BAND
        ),
        bz.material_library["Al"]["Rakic1995"],
        bz.material_library["aSi"]["Pierce1972"],
    ],
    ids=["ccpr_debye", "drude", "aluminum", "silicon"],
)
def test_pole_residue_conversion_reproduces_scalar_models(model):
    frequencies = np.geomspace(1e12, 3e15, 97)
    medium = bz.AnisotropicMedium.from_pole_residue(model)
    tensor = medium.permittivity_tensor(frequencies)
    reference = model.eps_model(frequencies)
    for axis in range(3):
        np.testing.assert_allclose(tensor[:, axis, axis], reference, rtol=1e-10)
    np.testing.assert_allclose(tensor[:, 0, 1], 0.0, atol=1e-12)
    restored = bz.AnisotropicMedium.from_spec(medium.to_spec())
    assert restored.cache_spec() == medium.cache_spec()
    np.testing.assert_allclose(
        restored.permittivity_tensor(frequencies), tensor, rtol=1e-14
    )


def test_drude_pairs_convert_back_to_drude_poles():
    drude = bz.PoleResidue.drude(
        1.0, plasma_frequency=1.2e16, damping=2e14, frequency_range=BAND
    )
    (pole,) = pole_residue_tensor_poles(drude)
    assert pole.resonance == 0.0 and not pole.has_edot
    np.testing.assert_allclose(pole.damping, 2e14)
    np.testing.assert_allclose(pole.weight[:3], 1.2e16**2, rtol=1e-12)


def test_principal_axis_models_rotate_like_tensors():
    frequencies = np.linspace(*BAND, 9)
    other = _other(3.0)
    crystal = bz.AnisotropicMedium.from_pole_residue(MODEL, other, MODEL)
    np.testing.assert_allclose(crystal.epsilon_r, (2.0, 3.0, 2.0, 0, 0, 0))
    # Axes sharing a model share its poles.
    assert len(crystal.electric_poles) == 4
    diagonal = crystal.permittivity_tensor(frequencies)
    np.testing.assert_allclose(diagonal[:, 0, 0], MODEL.eps_model(frequencies))
    np.testing.assert_allclose(diagonal[:, 1, 1], other.eps_model(frequencies))
    np.testing.assert_allclose(diagonal[:, 2, 2], MODEL.eps_model(frequencies))
    rotation = rotation_matrix((0.3, -0.4, 0.7))
    np.testing.assert_allclose(
        crystal.rotated((0.3, -0.4, 0.7)).permittivity_tensor(frequencies),
        rotation @ diagonal @ rotation.T,
        rtol=1e-12,
        atol=1e-12,
    )
    tilted = crystal.rotated((0.3, -0.4, 0.7))
    assert any(np.any(np.asarray(p.edot)[3:] != 0) for p in tilted.electric_poles)


def test_medium_with_ccpr_poles_rejects_gain():
    gain = bz.TensorPole.ccpr(-5e14 - 3e15j, 1e15 - 1e14j)  # Re r > 0, a < 0
    with pytest.raises(ValueError, match="passivity"):
        bz.AnisotropicMedium(2.0, electric_poles=[gain])
    with pytest.raises(ValueError, match="passivity"):
        bz.AnisotropicMedium(magnetic_poles=[gain])
    # One fitted pole need not be passive alone when the sum is (Drude halves).
    halves = (
        bz.TensorPole(0.0, 0.0, 0.0, 1e16),
        bz.TensorPole(1e14, 2e14, -1e30, -1e16),
    )
    assert np.all(
        np.linalg.eigvalsh(
            np.imag(
                bz.AnisotropicMedium(electric_poles=halves).permittivity_tensor(1e14)
            )
        )
        >= 0
    )


# ---------------------------------------------------------------------------
# Constitutive response under a prescribed harmonic flux


OMEGA = 2 * np.pi * 4e14


def _harmonic(trace, times):
    matrix = np.c_[np.cos(OMEGA * times), np.sin(OMEGA * times), np.ones(times.size)]
    amplitude = np.linalg.lstsq(matrix, trace, rcond=None)[0]
    return amplitude[0] + 1j * amplitude[1]


def _medium(path, family, debye=None):
    """Isotropic, tilted (coupled ``B``) or tilted anisotropic-``eps_inf`` (full)."""
    model, other = MODEL, _other(2.0 if path != "full" else 3.0)
    if debye is not None:  # a stronger E'-only term for sharper controls
        model = model.updated_copy(poles=(model.poles[0], (-5e14, debye)))
        other = other.updated_copy(poles=(other.poles[0], (-3e14, debye)))
    if path == "isotropic":
        poles, static = pole_residue_tensor_poles(model), 2.0
    else:
        crystal = bz.AnisotropicMedium.from_pole_residue(model, other, model)
        crystal = crystal.rotated(TILT)
        poles, static = crystal.electric_poles, crystal.epsilon_r
    if family == "E":
        return bz.AnisotropicMedium(static, electric_poles=poles)
    return bz.AnisotropicMedium(1.0, static, magnetic_poles=poles)


def _response(medium, family, control=None):
    """Settled ``(x, y)`` field for a unit harmonic flux along x (2D, 4x4 cells)."""
    dt, width = 1e-17, 0.2e-6
    simulation = bz.Simulation(
        design=bz.Design(width=width, height=width, material=medium),
        resolution=width / 4,
        time=np.arange(8) * dt,
        polarization="te" if family == "E" else "tm",
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )
    plan = simulation.compile(backend="jax").dispersion
    regions = "electric_regions" if family == "E" else "magnetic_regions"
    (region,) = getattr(plan.anisotropic, regions)
    assert isinstance(region, ccpr.CCPRPolarizationRegion)
    poles = medium.electric_poles if family == "E" else medium.magnetic_poles
    static = medium.epsilon_r if family == "E" else medium.mu_r
    full = region.static is None
    kernel = packed_to_matrix(np.asarray(static))[:2, :2]
    if full:  # the full-tensor kernel already includes K
        kernel = (
            kernel
            + sum(dt / (1 + 0.5 * p.damping * dt) * p.edot_matrix for p in poles)[
                :2, :2
            ]
        )
    if control == "drop_edot":
        region = replace(
            region,
            edot_diagonal=tuple(jnp.zeros_like(d) for d in region.edot_diagonal),
            edot_offdiagonal=None,
        )
    elif control == "drop_augmentation":
        twin = bz.AnisotropicMedium(
            static,
            **(
                {"electric_poles": _without_edot(poles)}
                if family == "E"
                else {"magnetic_poles": _without_edot(poles), "permittivity": 1.0}
            ),
        )
        twin_sim = simulation.updated_copy(
            design=bz.Design(width=width, height=width, material=twin)
        )
        (plain,) = getattr(
            twin_sim.compile(backend="jax").dispersion.anisotropic, regions
        )
        region = replace(region, inverse=plain.inverse)
        kernel = packed_to_matrix(np.asarray(static))[:2, :2]
    plan = replace(plan, anisotropic=replace(plan.anisotropic, **{regions: (region,)}))
    x, y = ("ex", "ey") if family == "E" else ("hx", "hy")
    update = (
        update_anisotropic_electric if family == "E" else update_anisotropic_magnetic
    )
    # The E kernel maps a flux increment through its static inverse; the H
    # kernel advances in vacuum and the plan maps mu_inf (here unity).
    increment = np.linalg.inv(kernel)[:, 0] if family == "E" else np.array([1.0, 0.0])
    steps = 8000

    def step(state, index):
        old = state
        delta = jnp.cos(OMEGA * dt * (index + 1)) - jnp.cos(OMEGA * dt * index)
        free = state._replace(
            **{
                x: getattr(state, x) + increment[0] * delta,
                y: getattr(state, y) + increment[1] * delta,
            }
        )
        state = update(old, free, plan)
        return state, (jnp.mean(getattr(state, x)), jnp.mean(getattr(state, y)))

    _, traces = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(steps)))(
        simulation.initial_state()
    )
    times = np.arange(1, steps + 1)[-3000:] * dt
    measured = np.array([_harmonic(np.asarray(t)[-3000:], times) for t in traces])
    discrete = np.linalg.inv(_discrete_tensor(static, poles, OMEGA, dt)[:2, :2])[:, 0]
    tensor = medium.permittivity_tensor if family == "E" else medium.permeability_tensor
    exact = np.linalg.inv(tensor(OMEGA / (2 * np.pi))[:2, :2])[:, 0]
    return measured, discrete, exact, full


@pytest.mark.parametrize(
    "family, path",
    [("E", "isotropic"), ("E", "coupled"), ("E", "full"), ("H", "coupled")],
)
def test_prescribed_flux_matches_discrete_and_continuous_response(family, path):
    """A harmonic D (or B) settles to eps_d(w)^-1 D for the CCPR recurrence.

    ``eps_d`` is the exact discrete-time response of the implemented update,
    so the agreement is limited only by float32 rounding; the continuous
    ``eps(w)`` differs by the first-order ``(w dt / 2) B`` term of the
    forward-difference ``E'`` drive (fdtdx's discretization).
    """
    medium = _medium(path, family)
    measured, discrete, exact, full = _response(medium, family)
    assert full == (path == "full")
    scale = np.linalg.norm(discrete)
    assert np.linalg.norm(measured - discrete) < 1e-3 * scale
    assert np.linalg.norm(measured - exact) < 1e-2 * scale
    if path != "isotropic":  # the tilt couples x to y through every pole
        assert abs(discrete[1]) > 0.05 * scale


@pytest.mark.parametrize("path", ["coupled", "full"])
def test_negative_controls_fail_the_discrete_oracle(path):
    """Dropping the B drive or the implicit eps' = eps_inf + K solve is detected.

    A strong Debye (E'-only) term makes dt B / eps_inf about 0.1, so both
    errors are first order in it, far above the float32 floor of the oracle.
    """
    medium = _medium(path, "E", debye=1e16)
    errors = {}
    for control in (None, "drop_edot", "drop_augmentation"):
        measured, discrete, _, _ = _response(medium, "E", control)
        errors[control] = np.linalg.norm(measured - discrete) / np.linalg.norm(discrete)
    assert errors[None] < 1e-3, errors
    assert errors["drop_edot"] > 0.1, errors
    assert errors["drop_augmentation"] > 0.01, errors


# ---------------------------------------------------------------------------
# Plane-wave slabs


def _analytic_slab(epsilon, wavelength, thickness):
    """Normal-incidence film amplitudes ``(r, t)`` between vacuum half-spaces."""
    n = np.sqrt(epsilon)
    r = (1 - n) / (1 + n)
    phase = np.exp(2j * np.pi * n * thickness / wavelength)
    reflection = r * (1 - phase**2) / (1 - r * r * phase**2)
    transmission = (1 - r * r) * phase / (1 - r * r * phase**2)
    return reflection, transmission


WAVELENGTHS = np.linspace(500e-9, 1500e-9, 7)
THICKNESS = 200e-9


def _slab(material, smoothing="volume", dx=10e-9):
    size = (4 * dx, 4 * dx, 2.4e-6)
    frequencies = LIGHT_SPEED / WAVELENGTHS
    plane = {"size": (size[0], size[1], 0)}
    design = bz.Design(background=bz.Material(1.0))
    if material is not None:
        design += bz.Box(
            center=(0, 0, 0), size=(size[0], size[1], THICKNESS), material=material
        )
    simulation = bz.Simulation(
        design=design,
        size=size,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 0.7e-6),
                **plane,
                source_time=bz.GaussianPulse(
                    float(frequencies.mean()), float(np.ptp(frequencies) * 0.9)
                ),
                direction="-z",
            )
        ],
        monitors=[
            bz.FluxMonitor(
                center=(0, 0, 0.45e-6), **plane, freqs=frequencies, name="input"
            ),
            bz.FluxMonitor(
                center=(0, 0, -0.45e-6), **plane, freqs=frequencies, name="output"
            ),
            bz.FieldMonitor(
                center=(0, 0, -0.45e-6),
                **plane,
                freqs=frequencies,
                fields=("Ex", "Ey"),
                name="field",
            ),
        ],
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.3e-6),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=0.7),
        run_time=90e-15,
        raster_options=RasterOptions(smoothing=smoothing),
    )
    result = simulation.run(backend="jax", progress=False, performance=False)
    fields = result.monitor("field").dft_fields
    jones = np.array(
        [
            np.asarray(fields[c]).reshape(len(frequencies), -1).mean(axis=1)
            for c in ("Ex", "Ey")
        ]
    )
    return (
        np.asarray(result["input"].flux),
        np.asarray(result["output"].flux),
        jones,
    )


@pytest.fixture(scope="module")
def vacuum_slab():
    return _slab(None)


@pytest.mark.simulation
def test_isotropic_ccpr_slab_matches_scalar_pole_residue_and_thin_film(vacuum_slab):
    """Per-axis CCPR poles converted from a fit reproduce the scalar ADE run."""
    reference_in, reference_out, _ = vacuum_slab
    epsilon = MODEL.eps_model(LIGHT_SPEED / WAVELENGTHS)
    reflection, transmission = _analytic_slab(epsilon, WAVELENGTHS, THICKNESS)
    spectra = {}
    for name, material in (
        ("scalar", MODEL),
        ("tensor", bz.AnisotropicMedium.from_pole_residue(MODEL)),
        (
            "without_b",
            bz.AnisotropicMedium(
                2.0, electric_poles=_without_edot(pole_residue_tensor_poles(MODEL))
            ),
        ),
    ):
        incident, output, _ = _slab(material)
        spectra[name] = (1 - incident / reference_in, output / reference_out)
    R, T = spectra["tensor"]
    np.testing.assert_allclose(R, abs(reflection) ** 2, atol=0.004)
    np.testing.assert_allclose(T, abs(transmission) ** 2, atol=0.004)
    np.testing.assert_allclose(R, spectra["scalar"][0], atol=0.003)
    np.testing.assert_allclose(T, spectra["scalar"][1], atol=0.003)
    # Negative control: the same poles without their E' coupling.
    assert np.abs(spectra["without_b"][1] - abs(transmission) ** 2).max() > 0.03


@pytest.mark.simulation
@pytest.mark.parametrize(
    "path, smoothing", [("coupled", "volume"), ("full", "farjadpour_full")]
)
def test_tilted_ccpr_slab_matches_jones_calculus(vacuum_slab, path, smoothing):
    """A crystal tilted by theta about the propagation axis is a linear retarder.

    With principal film transmissions t1, t2 the transmitted Jones vector for
    x-polarized light is ``R(theta) diag(t1, t2) R(theta)^T e_x``, so
    ``Ey / Ex = s c (t1 - t2) / (c^2 t1 + s^2 t2)`` and the power is
    ``c^2 |t1|^2 + s^2 |t2|^2``. The cross-polarized field exists only through
    the off-diagonal (oriented) CCPR tensors, including their b terms.
    """
    _, reference_out, reference = vacuum_slab
    assert np.abs(reference[1]).max() < 1e-6 * np.abs(reference[0]).max()
    other = _other(2.0 if path == "coupled" else 3.0)
    crystal = bz.AnisotropicMedium.from_pole_residue(MODEL, other, MODEL)
    _, output, jones = _slab(crystal.rotated(TILT), smoothing)
    frequencies = LIGHT_SPEED / WAVELENGTHS
    t1 = _analytic_slab(MODEL.eps_model(frequencies), WAVELENGTHS, THICKNESS)[1]
    t2 = _analytic_slab(other.eps_model(frequencies), WAVELENGTHS, THICKNESS)[1]
    c, s = np.cos(TILT[2]), np.sin(TILT[2])
    ratio = s * c * (t1 - t2) / (c * c * t1 + s * s * t2)
    np.testing.assert_allclose(jones[1] / jones[0], ratio, atol=0.01)
    np.testing.assert_allclose(
        output / reference_out, c * c * abs(t1) ** 2 + s * s * abs(t2) ** 2, atol=0.004
    )


# ---------------------------------------------------------------------------
# Mode solves and stability


@pytest.mark.parametrize("family", ["E", "H"])
def test_folded_mode_solver_tensors_include_the_edot_term(family):
    from beamz.design.material_views import static_material_grid

    medium = _medium("coupled", family)
    design = bz.Design(width=0.4e-6, height=0.4e-6, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.1e-6, 0.1e-6), width=0.2e-6, height=0.2e-6, material=medium
    )
    simulation = bz.Simulation(
        design=design,
        resolution=0.05e-6,
        time=np.arange(4) * 1e-17,
        polarization="te" if family == "E" else "tm",
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )
    material_grid = simulation._material_grid()
    frequency = 4e14
    view = static_material_grid(material_grid, frequency)
    tensor = (
        medium.permittivity_tensor(frequency)
        if family == "E"
        else medium.permeability_tensor(frequency)
    ).real
    packed = np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 2],
        ]
    )
    cell = (3, 3)  # fully inside the medium
    if family == "E":
        folded = np.asarray(view.tensors["epsilon"])[(slice(None), *cell)]
    else:
        folded = np.asarray(view.magnetic_tensors["mu_cell"])[(slice(None), *cell)]
    np.testing.assert_allclose(folded, packed, rtol=1e-10)
    lorentz_part = (
        packed_to_matrix(np.asarray(medium.epsilon_r if family == "E" else medium.mu_r))
        + sum(
            replace(p, edot=None).susceptibility(frequency)
            for p in (medium.electric_poles if family == "E" else medium.magnetic_poles)
        )
    ).real
    assert abs(lorentz_part[0, 0] - tensor[0, 0]) > 0.05


@pytest.mark.simulation
def test_mode_monitor_in_ccpr_core_projects_with_folded_edot_term():
    """Modal/flux ratios in a CCPR core match a static core with Re eps(f).

    The overdamped pole puts most of its b term into Re eps (about 0.9 of
    3.57), so folding without it solves the port for a clearly different core.
    """
    from tests.validation.analytical.test_anisotropic_media import _modal_to_flux_ratio

    frequency = LIGHT_SPEED / 1e-6
    omega = 2 * np.pi * frequency
    resonance, damping, strength = 3 * omega, 10 * omega, 1.5 * (3 * omega) ** 2
    # Marginally passive at DC: damping * weight = resonance^2 * edot.
    pole = bz.TensorPole(
        resonance, damping, strength, damping * strength / resonance**2
    )
    core = bz.AnisotropicMedium(2.0, electric_poles=[pole])
    frequencies = frequency * np.array([0.9, 1.1])
    dispersive = _modal_to_flux_ratio(core, frequency, frequencies)
    for index, f in enumerate(frequencies):
        epsilon = core.permittivity_tensor(f)[2, 2]
        without = (2.0 + replace(pole, edot=None).susceptibility(f)[2, 2]).real
        assert abs(without - epsilon.real) > 0.3
        expected, folded_without_b = (
            _modal_to_flux_ratio(bz.Material(float(value)), frequency, frequencies)[
                index
            ]
            for value in (epsilon.real, without)
        )
        np.testing.assert_allclose(dispersive[index], expected, rtol=1e-3)
        # Negative control: a port solved without the b term is clearly off.
        assert abs(dispersive[index] / folded_without_b - 1) > 5e-3


def _stiff_crystal(path):
    """Tilted CCPR crystal whose poles limit dt below the 20 nm vacuum CFL."""
    x = bz.PoleResidue(
        2.0,
        [(-2e15 - 6e16j, 2e15 + 4e16j), (-1e15, 5e14)],
        frequency_range=BAND,
    )
    y = bz.PoleResidue(
        2.0 if path == "coupled" else 3.0,
        [(-3e15 - 4e16j, 3e15 + 5e16j), (-2e15, 1e15)],
        frequency_range=BAND,
    )
    return bz.AnisotropicMedium.from_pole_residue(x, y, x).rotated(TILT)


def _bounded(medium, spacing, dt, steps):
    """Peak |E| of a seeded periodic 2D TE run after each third of ``steps``."""
    width = 16 * spacing
    simulation = bz.Simulation(
        design=bz.Design(width=width, height=width, material=medium),
        resolution=spacing,
        time=np.arange(steps + 1) * dt,
        polarization="te",
        boundaries=[bz.Periodic(axes=("x", "y"))],
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )
    state = simulation.initial_state()
    rng = np.random.default_rng(0)
    state = state._replace(
        ex=jnp.asarray(rng.normal(size=state.ex.shape), jnp.float32),
        ey=jnp.asarray(rng.normal(size=state.ey.shape), jnp.float32),
    )
    peaks = []
    for _ in range(3):
        state = simulation.advance(
            state=state, num_steps=steps // 3, backend="jax", performance=False
        ).state
        peaks.append(float(jnp.abs(state.ex).max()))
    return np.array(peaks)


@pytest.mark.parametrize("path", ["coupled", "full"])
def test_von_neumann_limit_predicts_the_simulated_stability_boundary(path, monkeypatch):
    """The CCPR timestep check is a von Neumann analysis of the implemented
    scheme; a seeded simulation is stable just below its limit and diverges
    just above it. (The Lorentz bound S^2 + (Omega dt / 2)^2 < 1 with Omega
    from the A tensors is not sufficient once B != 0, so it is not used.)"""
    spacing = 20e-9
    medium = _stiff_crystal(path)
    grid = ((spacing, spacing), "te")
    limit = pole_stability.max_stable_timestep(medium, "E", *grid)
    assert limit < 0.9 * spacing / (LIGHT_SPEED * np.sqrt(2))
    above = pole_stability.amplification_radius(medium, "E", 1.02 * limit, *grid)
    below = pole_stability.amplification_radius(medium, "E", 0.99 * limit, *grid)
    assert below < 1 + 1e-6
    steps = 3 * int(np.ceil(15 / np.log(above)))
    stable = _bounded(medium, spacing, 0.98 * limit, steps)
    assert np.isfinite(stable).all() and stable.max() < 2 * stable[0], stable
    # Compilation rejects the unstable step; bypass it to watch the divergence.
    monkeypatch.setattr(pole_stability, "check_stability", lambda *args: None)
    unstable = _bounded(medium, spacing, 1.02 * limit, steps)
    assert not np.isfinite(unstable[-1]) or unstable[-1] > 1e4 * stable[0], unstable


def test_compile_rejects_unstable_ccpr_timesteps_and_singular_updates(monkeypatch):
    spacing, width = 20e-9, 0.32e-6

    def compile_at(medium, dt):
        return bz.Simulation(
            design=bz.Design(width=width, height=width, material=medium),
            resolution=spacing,
            time=np.arange(4) * dt,
            polarization="te",
            raster_options=RasterOptions(smoothing="farjadpour_full"),
        ).compile(backend="jax")

    stiff = _stiff_crystal("full")
    limit = pole_stability.max_stable_timestep(stiff, "E", (spacing, spacing), "te")
    compile_at(stiff, 0.98 * limit)
    with pytest.raises(ValueError, match="Tensor poles are unstable"):
        compile_at(stiff, 1.05 * limit)
    # The implicit update divides by eps_inf + sum dt / (1 + gamma dt / 2) B,
    # which must stay positive definite; only an active medium can violate it.
    monkeypatch.setattr(bz.AnisotropicMedium, "_require_passive", lambda self: None)
    active = bz.AnisotropicMedium(
        1.0, electric_poles=[bz.TensorPole(0.0, 1e15, 0.0, -1e17)]
    )
    with pytest.raises(ValueError, match="positive definite"):
        compile_at(active, 0.5 * spacing / (LIGHT_SPEED * np.sqrt(2)))


def test_tilted_ccpr_block_stays_bounded_and_resumable():
    crystal = bz.AnisotropicMedium.from_pole_residue(MODEL, _other(3.0), MODEL)
    medium = crystal.rotated(TILT)
    width, spacing = 2e-6, 50e-9
    design = bz.Design(width=width, height=width, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(width / 4, width / 4),
        width=width / 2,
        height=width / 2,
        material=medium,
    )
    dt = 0.9 * spacing / LIGHT_SPEED / np.sqrt(2)
    time = np.arange(4001) * dt
    simulation = bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=(width / 2, 0.85 * width),
                width=0.2e-6,
                signal=bz.ramped_cosine(
                    time,
                    amplitude=1.0,
                    frequency=4e14,
                    ramp_duration=10e-15,
                    t_max=30e-15,
                ),
            )
        ],
        boundaries=[bz.PML(edges="all", thickness=0.3e-6, formulation="cpml")],
        time=time,
        resolution=spacing,
        polarization="te",
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )
    peaks, state = [], simulation.initial_state()
    for _ in range(4):
        state = simulation.advance(
            state=state, num_steps=1000, backend="jax", performance=False
        ).state
        peaks.append(max(float(jnp.abs(getattr(state, c)).max()) for c in ("ex", "ey")))
    # The pulse has passed after 1000 steps; what remains must not grow.
    assert np.isfinite(peaks).all() and max(peaks[1:]) < peaks[0], peaks
    # Continuation carries the CCPR pole state exactly (source-free, so only
    # rounding of the separately compiled scans differs).
    simulation = simulation.updated_copy(sources=[])
    start = simulation.initial_state()
    rng = np.random.default_rng(2)
    start = start._replace(
        ex=jnp.asarray(rng.normal(size=start.ex.shape), jnp.float32),
        ey=jnp.asarray(rng.normal(size=start.ey.shape), jnp.float32),
    )

    def advance(state, steps):
        return simulation.advance(
            state=state, num_steps=steps, backend="jax", performance=False
        ).state

    full = advance(start, 600)
    resumed = advance(advance(start, 300), 300)
    for actual, reference in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True
    ):
        scale = float(np.abs(np.asarray(reference)).max(initial=0.0))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-4 * scale)


# ---------------------------------------------------------------------------
# Regression and support boundaries


def test_zero_b_ccpr_poles_reproduce_lorentz_poles_exactly():
    lorentz = bz.TensorPole.lorentz((1.5, 0.5, 0.2), resonance=3e15, damping=3e14)
    as_ccpr = bz.TensorPole(
        lorentz.resonance, lorentz.damping, lorentz.weight, np.zeros(6)
    )

    def run(pole):
        medium = bz.AnisotropicMedium(2.0, electric_poles=[pole]).rotated(TILT)
        simulation = bz.Simulation(
            design=bz.Design(width=0.4e-6, height=0.4e-6, material=medium),
            resolution=0.05e-6,
            time=np.arange(33) * 1e-17,
            polarization="te",
            boundaries=[bz.Periodic(axes=("x", "y"))],
            raster_options=RasterOptions(smoothing="farjadpour_full"),
        )
        state = simulation.initial_state()
        rng = np.random.default_rng(0)
        state = state._replace(
            ex=jnp.asarray(rng.normal(size=state.ex.shape), dtype=jnp.float32)
        )
        return simulation.advance(
            state=state, num_steps=32, backend="jax", performance=False
        ).state

    for actual, expected in zip(
        jax.tree.leaves(run(as_ccpr)), jax.tree.leaves(run(lorentz)), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_unsupported_ccpr_combinations_are_rejected_clearly():
    from beamz.optimization.anisotropic_design import AnisotropicDesignBinding

    crystal = bz.AnisotropicMedium.from_pole_residue(MODEL)
    with pytest.raises(ValueError, match="CCPR"):
        AnisotropicDesignBinding(
            bz.Simulation(
                design=bz.Design(
                    width=0.2e-6, height=0.2e-6, material=bz.Material(1.0)
                ),
                resolution=0.05e-6,
                time=np.arange(4) * 1e-17,
            )._material_grid(),
            np.ones((1, 4, 4), dtype=bool)[0],
            (bz.Material(1.0), crystal),
        )
    hall = bz.AnisotropicMedium(
        crystal.epsilon_r,
        conductivity=((0.0, 1e3, 0.0), (-1e3, 0.0, 0.0), (0.0, 0.0, 0.0)),
        electric_poles=crystal.electric_poles,
    )
    with pytest.raises(ValueError, match="cannot yet overlap"):
        bz.Simulation(
            design=bz.Design(width=0.2e-6, height=0.2e-6, material=hall),
            resolution=0.05e-6,
            time=np.arange(4) * 1e-17,
            polarization="te",
        ).compile(backend="jax")
    magnetic = bz.AnisotropicMedium(
        1.0,
        bz.AnisotropicMedium((1.0, 1.0, 1.0), (1.5, 2.0, 1.0)).rotated(TILT).mu_r,
        magnetic_poles=pole_residue_tensor_poles(MODEL),
    )
    with pytest.raises(ValueError, match="off-diagonal static permeability"):
        bz.Simulation(
            design=bz.Design(width=0.2e-6, height=0.2e-6, material=magnetic),
            resolution=0.05e-6,
            time=np.arange(4) * 1e-17,
            polarization="tm",
            raster_options=RasterOptions(smoothing="farjadpour_full"),
        ).compile(backend="jax")
