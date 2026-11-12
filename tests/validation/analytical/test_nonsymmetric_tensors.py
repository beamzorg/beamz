"""Non-symmetric static tensors and conductivity tensors against analytic oracles.

A full static medium updates as ``(T + B) F_new = (T - B) F_old + inc`` with
``B = sigma dt / (2 eps0)`` (``mu0`` for H). Uniform fields have no curl, so
they follow the exact matrix recurrence; plane waves along z see the
effective tensor ``T + i cot(omega dt / 2) B``, which tends to the physical
``eps' + i sigma / (omega eps0)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
import beamz.simulation.anisotropic as anisotropic
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.design.raster import RasterOptions

WAVELENGTH = 1.5e-6
FREQUENCY = LIGHT_SPEED / WAVELENGTH
EPS = np.diag([3.0, 2.0, 2.5])


def _drop_box_terms(monkeypatch):
    """Negative control: compile without the antisymmetric/off-diagonal terms.

    Media then fall back to the raster's symmetric tensor and diagonal
    electric conductivity, i.e. the behavior before full static tensors.
    """
    from beamz.simulation.compile import clear_program_cache
    from beamz.simulation.execute import clear_execution_cache

    monkeypatch.setattr(
        anisotropic,
        "_static_extras",
        lambda medium, family, dt: (np.zeros((3, 3)), np.zeros((3, 3))),
    )
    clear_program_cache()  # programs compiled earlier still hold the full plan
    clear_execution_cache()


# ---------------------------------------------------------------------------
# Construction


def test_full_tensor_inputs_and_complex_constructor_round_trip():
    epsilon = np.array(
        [[2.0 + 0.05j, 0.3j + 0.1, 0.02j], [-0.3j + 0.1, 2.5, 0.1j], [-0.02j, -0.1j, 3]]
    )
    mu = np.array([[1.2 + 0.01j, 0.05j, 0], [-0.05j, 1.1, 0], [0, 0, 1.3 + 0.02j]])
    medium = bz.AnisotropicMedium.from_complex_permittivity(
        epsilon, permeability=mu, wavelength=WAVELENGTH
    )
    np.testing.assert_allclose(
        medium.complex_permittivity(FREQUENCY), epsilon, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        medium.complex_permeability(FREQUENCY), mu, rtol=0, atol=1e-14
    )
    omega = 2 * np.pi * FREQUENCY
    np.testing.assert_allclose(
        medium.conductivity_matrix, omega * EPS_0 * epsilon.imag, rtol=1e-14
    )
    np.testing.assert_allclose(
        medium.magnetic_conductivity_matrix, omega * MU_0 * mu.imag, rtol=1e-14
    )
    # A Hermitian imaginary part is an antisymmetric (lossless) conductivity.
    sigma = medium.conductivity_matrix
    assert sigma[0, 1] == -sigma[1, 0] != 0.0
    assert medium.requires_static_correction
    same = bz.AnisotropicMedium.from_complex_permittivity(
        epsilon, permeability=mu, frequency=FREQUENCY
    )
    assert same == medium
    assert bz.AnisotropicMedium.from_spec(medium.to_spec()) == medium
    assert medium.cache_spec() != bz.AnisotropicMedium(epsilon.real).cache_spec()

    rotation = bz.design.anisotropic.rotation_matrix((0.3, -0.2, 0.7))
    rotated = medium.rotated((0.3, -0.2, 0.7))
    np.testing.assert_allclose(
        rotated.complex_permittivity(FREQUENCY),
        rotation @ epsilon @ rotation.T,
        atol=1e-12,
    )
    # 9-element row-major and nested 3x3 inputs agree; a non-symmetric real
    # tensor is accepted with a warning about gain or loss.
    matrix = np.array([[2.0, 0.1, 0.0], [-0.1, 2.0, 0.0], [0.0, 0.0, 2.0]])
    with pytest.warns(UserWarning, match="non-Hermitian"):
        nested = bz.AnisotropicMedium(matrix)
    with pytest.warns(UserWarning, match="non-Hermitian"):
        flat = bz.AnisotropicMedium(tuple(matrix.ravel()))
    assert nested == flat
    np.testing.assert_array_equal(nested.permittivity_matrix, matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not bz.AnisotropicMedium(
            EPS, conductivity=(1.0, 2.0, 3.0)
        ).requires_static_correction


def test_rejects_active_conductivity_and_ambiguous_reference():
    with pytest.raises(ValueError, match="passive"):
        bz.AnisotropicMedium(conductivity=((1.0, 3.0, 0), (3.0, 1.0, 0), (0, 0, 1)))
    with pytest.raises(ValueError, match="passive"):
        bz.AnisotropicMedium.from_complex_permittivity(2.0 - 0.1j, frequency=FREQUENCY)
    with pytest.raises(ValueError, match="exactly one"):
        bz.AnisotropicMedium.from_complex_permittivity(2.0)
    with pytest.raises(ValueError, match="exactly one"):
        bz.AnisotropicMedium.from_complex_permittivity(
            2.0, frequency=FREQUENCY, wavelength=WAVELENGTH
        )


# ---------------------------------------------------------------------------
# Uniform fields: the exact semi-implicit recurrence


def _uniform(medium, family, steps, dt, initial, smoothing="volume"):
    polarization = "te" if family == "E" else "tm"
    simulation = bz.Simulation(
        design=bz.Design(width=0.4e-6, height=0.4e-6, material=medium),
        resolution=0.05e-6,
        time=np.arange(steps + 1) * dt,
        polarization=polarization,
        boundaries=[bz.Periodic(axes=("x", "y"))],
        raster_options=RasterOptions(smoothing=smoothing),
    )
    names = ("ex", "ey") if family == "E" else ("hx", "hy")
    state = simulation.initial_state()
    state = state._replace(
        **{
            n: jnp.full_like(getattr(state, n), value)
            for n, value in zip(names, initial, strict=True)
        }
    )
    final = simulation.advance(
        state=state, num_steps=steps, backend="jax", performance=False
    ).state
    values = [np.asarray(getattr(final, n)) for n in names]
    # A uniform field must stay uniform: no spurious gradients at the seams.
    assert max(float(np.ptp(v)) for v in values) < 1e-5 * max(
        float(np.abs(v).max()) for v in values
    )
    return np.array([float(v.mean()) for v in values])


def _recurrence(medium, family, steps, dt, initial):
    if family == "E":
        tensor = medium.permittivity_matrix[:2, :2]
        loss = medium.conductivity_matrix[:2, :2] * dt / (2 * EPS_0)
    else:
        tensor = medium.permeability_matrix[:2, :2]
        loss = medium.magnetic_conductivity_matrix[:2, :2] * dt / (2 * MU_0)
    step = np.linalg.solve(tensor + loss, tensor - loss)
    return np.linalg.matrix_power(step, steps) @ np.asarray(initial, dtype=float)


HALL = bz.AnisotropicMedium(EPS, conductivity=((0, 3e5, 0), (-3e5, 0, 0), (0, 0, 0)))
MAGNETIC = bz.AnisotropicMedium(
    1.0,
    (1.5, 2.0, 1.0),
    magnetic_conductivity=((2e9, 4e9, 0), (-2e9, 1e9, 0), (0, 0, 0)),
)


@pytest.mark.parametrize(
    ("medium", "family"), [(HALL, "E"), (MAGNETIC, "H")], ids=["hall", "magnetic"]
)
def test_uniform_field_follows_the_semi_implicit_recurrence(
    medium, family, monkeypatch
):
    dt, steps = 2e-17, 50
    measured = _uniform(medium, family, steps, dt, (1.0, 0.0))
    expected = _recurrence(medium, family, steps, dt, (1.0, 0.0))
    np.testing.assert_allclose(measured, expected, rtol=2e-5, atol=2e-6)
    assert abs(expected[1]) > 0.2  # the off-diagonal terms drive the y component

    # Negative control: without the new antisymmetric/off-diagonal terms the
    # same run misses the oracle by far more than the tolerance.
    _drop_box_terms(monkeypatch)
    dropped = _uniform(medium, family, steps, dt, (1.0, 0.0))
    assert np.abs(dropped - expected).max() > 0.1


def test_antisymmetric_conductivity_conserves_energy_over_long_runs():
    """A Hall (antisymmetric) conductivity rotates E at constant E^T eps E."""

    dt, steps = 2e-17, 4000
    measured = _uniform(HALL, "E", steps, dt, (1.0, 0.0))
    expected = _recurrence(HALL, "E", steps, dt, (1.0, 0.0))
    energy = measured @ EPS[:2, :2] @ measured
    assert abs(energy / EPS[0, 0] - 1.0) < 2e-4
    np.testing.assert_allclose(measured, expected, atol=2e-3)


def test_hermitian_medium_keeps_propagating_field_energy_over_long_runs():
    """Waves in a lossless gyrotropic (Hermitian eps) medium neither grow nor decay.

    A smooth random field in a periodic cell runs for about 18 optical
    periods; the Yee energy only shows the bounded leapfrog staggering.
    """

    eps = 2.25
    medium = bz.AnisotropicMedium.from_complex_permittivity(
        ((eps, 0.3j, 0), (-0.3j, eps, 0), (0, 0, eps)), frequency=FREQUENCY
    )
    spacing, cells = WAVELENGTH / 20, 40
    dt = 0.5 * spacing / LIGHT_SPEED / np.sqrt(2)
    blocks, steps = 10, 1000
    simulation = bz.Simulation(
        design=bz.Design(
            width=cells * spacing, height=cells * spacing, material=medium
        ),
        resolution=spacing,
        time=np.arange(blocks * steps + 1) * dt,
        polarization="te",
        boundaries=[bz.Periodic(axes=("x", "y"))],
        raster_options=RasterOptions(smoothing="volume"),
    )
    rng = np.random.default_rng(3)

    def smooth(shape, scale=1.0):
        spectrum = np.fft.fft2(rng.normal(size=shape))
        ky, kx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])
        spectrum *= np.exp(-(kx**2 + ky**2) / 0.01)
        return jnp.asarray(scale * np.real(np.fft.ifft2(spectrum)), jnp.float32)

    state = simulation.initial_state()
    state = state._replace(
        ex=smooth(state.ex.shape),
        ey=smooth(state.ey.shape),
        hz=smooth(state.hz.shape, 1 / 377.0),
    )

    def energy(s):
        e = sum(np.asarray(getattr(s, n))[:cells, :cells] ** 2 for n in ("ex", "ey"))
        h = np.asarray(s.hz)[:cells, :cells] ** 2
        return float(EPS_0 * eps * e.sum() + MU_0 * h.sum())

    values = []
    for _ in range(blocks):
        state = simulation.advance(
            state=state, num_steps=steps, backend="jax", performance=False
        ).state
        values.append(energy(state))
    values = np.asarray(values) / values[0]
    assert np.ptp(values) < 0.02, values
    assert abs(values[blocks // 2 :].mean() - values[: blocks // 2].mean()) < 5e-3


def test_rotated_conductivity_decays_at_the_eigenrates_of_eps_inverse_sigma():
    """Uniform E along a generalized eigenvector of (sigma, eps) decays as exp(-lambda t / eps0)."""

    sigma_principal = np.diag([4e4, 1e4, 2e4])
    medium = bz.AnisotropicMedium(EPS, conductivity=sigma_principal).rotated(
        (0.0, 0.0, 0.6)
    )
    assert medium.requires_static_correction  # off-diagonal conductivity
    epsilon = medium.permittivity_matrix[:2, :2]
    sigma = medium.conductivity_matrix[:2, :2]
    rates, vectors = np.linalg.eig(np.linalg.solve(epsilon, sigma))
    assert abs(rates[0] - rates[1]) > 0.3 * rates.max()
    dt, steps = 1e-17, 60
    for rate, vector in zip(rates.real, vectors.T.real, strict=True):
        measured = _uniform(medium, "E", steps, dt, vector, smoothing="farjadpour_full")
        expected = np.exp(-rate * steps * dt / EPS_0)
        amplitude = measured @ vector / (vector @ vector)
        np.testing.assert_allclose(amplitude, expected, rtol=2e-3)
        residual = measured - amplitude * vector
        assert np.abs(residual).max() < 1e-4 * np.abs(vector).max()
    assert expected < 0.8


# ---------------------------------------------------------------------------
# Plane waves along z: circular birefringence of a Hermitian tensor


def _line_fields(medium, family, *, dx=WAVELENGTH / 30):
    """CW (DFT) transverse fields along z inside a medium-filled half space."""
    length = 10e-6
    size = (4 * dx, 4 * dx, length)
    design = bz.Design(background=bz.Material(1.0))
    design += bz.Box(
        center=(0, 0, -length / 4 + 1e-6),
        size=(size[0], size[1], length / 2 + 2e-6),
        material=medium,
    )
    names = ("Ex", "Ey") if family == "E" else ("Hx", "Hy")
    simulation = bz.Simulation(
        design=design,
        size=size,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 3.5e-6),
                size=(size[0], size[1], 0),
                source_time=bz.GaussianPulse(FREQUENCY, 0.1 * FREQUENCY),
                direction="-z",
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(0, 0, -1.25e-6),
                size=(0, 0, 4.5e-6),
                freqs=[FREQUENCY],
                fields=names,
                name="line",
            )
        ],
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=1e-6),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=0.5),
        run_time=300e-15,
    )
    result = simulation.run(backend="jax", progress=False)
    fields = [
        np.asarray(result.monitor("line").dft_fields[n]).reshape(-1) for n in names
    ]
    return fields, float(simulation.dt), dx


def _effective_index_squared(values, dt, dx):
    """n^2 of the Yee plane wave in ``values`` (forward plus backward waves).

    ``A e^{ikz} + B e^{-ikz}`` satisfies ``c[j+1] + c[j-1] = 2 cos(k dx) c[j]``;
    the Yee relation ``sin(k dx / 2) / dx = n sin(w dt / 2) / (c dt)`` then gives
    the complex index without fitting.
    """
    middle = values[1:-1]
    cosine = 0.5 * np.vdot(middle, values[2:] + values[:-2]) / np.vdot(middle, middle)
    sine_squared = 0.5 * (1.0 - cosine)
    theta = 2 * np.pi * FREQUENCY * dt
    return sine_squared / (dx / (LIGHT_SPEED * dt) * np.sin(theta / 2)) ** 2


def _circular_oracle(tensor, loss_rate, dt):
    """Eigenvalues for (1, +-i) of ``tensor + i cot(w dt / 2) B`` and the w dt -> 0 limit."""
    theta = 2 * np.pi * FREQUENCY * dt
    discrete = tensor + 1j * (dt / 2) * loss_rate / np.tan(theta / 2)
    exact = tensor + 1j * loss_rate / (2 * np.pi * FREQUENCY)
    out = []
    for matrix in (discrete, exact):
        values = []
        for vector in (np.array([1.0, 1.0j]), np.array([1.0, -1.0j])):
            image = matrix[:2, :2] @ vector
            values.append(image[0] / vector[0])
            np.testing.assert_allclose(image, values[-1] * vector, atol=1e-12)
        out.append(np.array(values))
    return out


CASES = {
    # Hermitian eps at the reference frequency: antisymmetric conductivity.
    "hermitian_eps": (
        "E",
        lambda: bz.AnisotropicMedium.from_complex_permittivity(
            ((2.25, 0.3j, 0), (-0.3j, 2.25, 0), (0, 0, 2.25)), frequency=FREQUENCY
        ),
    ),
    # Hermitian mu: antisymmetric magnetic conductivity.
    "hermitian_mu": (
        "H",
        lambda: bz.AnisotropicMedium.from_complex_permittivity(
            2.0,
            permeability=((1.2, 0.25j, 0), (-0.25j, 1.2, 0), (0, 0, 1.2)),
            frequency=FREQUENCY,
        ),
    ),
    # Real non-symmetric eps: non-Hermitian, complex circular indices.
    "skew_eps": (
        "E",
        lambda: bz.AnisotropicMedium(((2.25, 0.06, 0), (-0.06, 2.25, 0), (0, 0, 2.25))),
    ),
}


def _circular_indices(case):
    family, build = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        medium = build()
    (x, y), dt, dx = _line_fields(medium, family)
    measured = np.array(
        [
            _effective_index_squared(x - 1j * y, dt, dx),  # (1, +i) amplitude
            _effective_index_squared(x + 1j * y, dt, dx),  # (1, -i) amplitude
        ]
    )
    if family == "E":
        other = np.real(medium.permeability_matrix[0, 0])
        tensor, rate = medium.permittivity_matrix, medium.conductivity_matrix / EPS_0
    else:
        other = np.real(medium.permittivity_matrix[0, 0])
        tensor = medium.permeability_matrix
        rate = medium.magnetic_conductivity_matrix / MU_0
    discrete, exact = _circular_oracle(tensor, rate, dt)
    return measured, other * discrete, other * exact


@pytest.mark.parametrize("case", list(CASES))
def test_plane_waves_split_into_circular_modes_of_the_full_tensor(case):
    measured, discrete, exact = _circular_indices(case)
    # Exact against the discrete dispersion relation of the scheme ...
    np.testing.assert_allclose(measured, discrete, rtol=1.5e-3)
    # ... and, through the Yee relation, close to the physical tensor: they
    # differ only by the (w dt / 2) cot(w dt / 2) factor of the loss term.
    np.testing.assert_allclose(np.sqrt(measured), np.sqrt(exact), rtol=2e-3)
    split = abs(np.sqrt(exact[1]) - np.sqrt(exact[0]))
    assert split > 0.02 * abs(np.sqrt(exact[0]))
    np.testing.assert_allclose(
        abs(np.sqrt(measured[1]) - np.sqrt(measured[0])), split, rtol=2e-2
    )


def test_plane_wave_negative_control_loses_the_circular_splitting(monkeypatch):
    _drop_box_terms(monkeypatch)
    measured, discrete, _ = _circular_indices("hermitian_eps")
    # Both circular modes then see the same isotropic eps.
    np.testing.assert_allclose(measured[0], measured[1], rtol=1e-3)
    assert np.abs(measured / discrete - 1).max() > 0.05


# ---------------------------------------------------------------------------
# Compile-time contracts


def test_static_tensor_media_reject_ports_and_out_of_plane_2d_coupling():
    medium = bz.AnisotropicMedium.from_complex_permittivity(
        ((2.25, 0.3j, 0), (-0.3j, 2.25, 0), (0, 0, 2.25)), frequency=FREQUENCY
    )
    width = 2e-6
    design = bz.Design(width=width, height=width, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.5e-6, 0.5e-6), width=1e-6, height=1e-6, material=medium
    )
    monitor = bz.ModeMonitor(
        center=(1e-6, 1e-6, 0.0),
        size=(0.0, 1.2e-6, 1.0e-6),
        mode_spec=bz.ModeSpec(polarization="te"),
        freqs=[FREQUENCY],
        name="port",
    )
    simulation = bz.Simulation(
        design=design,
        resolution=0.05e-6,
        time=np.arange(4) * 1e-17,
        polarization="te",
        monitors=[monitor],
    )
    with pytest.raises(ValueError, match="non-symmetric"):
        simulation.compile(backend="jax")

    tilted = bz.AnisotropicMedium(
        2.0, conductivity=((0, 0, 1e4), (0, 0, 0), (-1e4, 0, 0))
    )
    simulation = bz.Simulation(
        design=bz.Design(width=width, height=width, material=tilted),
        resolution=0.1e-6,
        time=np.arange(4) * 1e-17,
        polarization="te",
    )
    with pytest.raises(ValueError, match="without xz or yz coupling"):
        simulation.compile(backend="jax")
