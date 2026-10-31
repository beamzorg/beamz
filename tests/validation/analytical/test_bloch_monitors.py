"""Complex acquisition and diffraction power against explicit Fourier sums."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.execute import initial_program_state
from beamz.simulation.observe import update_monitors
from beamz.simulation.results import MonitorResults
from examples.meta_atom_bloch import diffraction_powers


@pytest.mark.parametrize(
    "plane,polarization,component,name,sign",
    [
        ("xy", "tm", "Ez", "ez", 1),
        ("xy", "te", "Ex", "ex", 1),
        ("xz", "tm", "Ey", "ez", -1),
        ("xz", "te", "Ex", "ex", 1),
        ("yz", "tm", "Ex", "ez", 1),
        ("yz", "te", "Ey", "ex", 1),
    ],
)
@pytest.mark.parametrize(
    "transition", ["real_to_bloch", "complex_to_real", "complex_seed"]
)
def test_recorder_continuation_preserves_complex_fields(
    plane, polarization, component, name, sign, transition
):
    kwargs = dict(
        design=bz.Design(width=160e-9, height=160e-9),
        plane_2d=plane,
        resolution=40e-9,
        time=np.arange(12) * 1e-17,
        polarization=polarization,
        monitors=[bz.FieldRecorder((component,), interval=1, name="frames")],
    )
    vector = tuple(1e6 if axis == plane[0] else 0 for axis in "xyz")
    real = bz.Simulation(**kwargs, boundaries=[bz.Periodic(axes=tuple(plane))])
    bloch = bz.Simulation(
        **kwargs, boundaries=[bz.Bloch(axes=tuple(plane), wavevector=vector)]
    )
    original = bloch if transition == "complex_to_real" else real
    resumed = bloch if transition == "real_to_bloch" else real
    seed = original.initial_state()
    if transition != "real_to_bloch":
        seed = seed._replace(
            **{
                c: getattr(seed, c).astype(jnp.complex64)
                for c in ("ex", "ey", "ez", "hx", "hy", "hz")
            }
        )
    value = getattr(seed, name)
    seed = seed._replace(
        **{name: value.at[1, 1].set(1 if transition == "real_to_bloch" else 1 + 2j)}
    )
    first = original.advance(
        state=seed, num_steps=3, backend="jax", performance=False
    ).state
    final = resumed.advance(
        state=first, num_steps=3, backend="jax", performance=False
    ).state
    frames = np.asarray(final.recorded_fields[0])
    count = int(final.recorded_counts[0])
    assert count == 6
    assert np.iscomplexobj(frames)
    np.testing.assert_array_equal(frames[:3], first.recorded_fields[0][:3])
    np.testing.assert_allclose(
        frames[count - 1], sign * getattr(final, name), atol=1e-7
    )
    assert np.max(np.abs(frames[3:count].imag)) > 1e-7
    np.testing.assert_array_equal(
        final.recorded_times[0][:3], first.recorded_times[0][:3]
    )
    np.testing.assert_array_equal(final.recorded_steps[0][:count], np.arange(1, 7))


def test_complex_dft_and_recorder_preserve_both_quadratures():
    frequency, dt = 5e14, 1e-17
    sim = bz.Simulation(
        size=(160e-9, 160e-9, 160e-9),
        resolution=40e-9,
        time=np.arange(3) * dt,
        boundaries=[bz.Bloch(wavevector=(1e6, 0, 0))],
        monitors=[
            bz.FieldMonitor(
                center=(0, 0, 0),
                size=(160e-9, 160e-9, 0),
                freqs=[frequency],
                fields=("Ey", "Hx"),
                name="dft",
            ),
            bz.FieldRecorder(components=("Ey",), interval=1, name="frames"),
        ],
    )
    program = sim.compile()
    state = initial_program_state(program, t=0, current_step=0)
    amplitude = 2 + 3j
    state = state._replace(
        ey=jnp.full_like(state.ey, amplitude),
        hx=jnp.full_like(state.hx, -amplitude / 377),
    )
    observed = update_monitors(
        program,
        state,
        jnp.asarray(0),
        jnp.asarray(dt),
        jnp.asarray(dt),
    )
    result = MonitorResults.from_compiled_state(
        sim.monitors[0], program.monitors[0], observed, program.config
    )
    # One complex sample contributes its full value times exp(+i omega t).
    from beamz.analysis import monitor_dft_component

    np.testing.assert_allclose(
        monitor_dft_component(result, "Ey"),
        amplitude * np.exp(1j * 2 * np.pi * frequency * dt),
        atol=1e-6,
    )
    recorder = MonitorResults.from_compiled_state(
        sim.monitors[1], program.monitors[1], observed, program.config
    )
    np.testing.assert_allclose(recorder.fields["Ey"], amplitude)
    assert np.iscomplexobj(recorder.fields["Ey"])


def test_diffraction_orders_separate_two_known_plane_waves():
    frequency = 5e14
    grid = bz.RectilinearGrid.from_spacing((24, 24, 2), 50e-9)
    x, y = np.meshgrid(grid.centers("x"), grid.centers("y"), indexing="xy")
    k0 = 2 * np.pi * frequency / bz.LIGHT_SPEED
    kx = 0.35e6
    amplitudes = {(0, 0): 1 + 0.2j, (1, 0): 0.3 - 0.4j}
    ey, hx = np.zeros_like(x, dtype=complex), np.zeros_like(x, dtype=complex)
    expected = {}
    area = (1.2e-6) ** 2
    for (m, n), amplitude in amplitudes.items():
        k = kx + m * 2 * np.pi / 1.2e-6
        cosine = np.sqrt(1 - (k / k0) ** 2)
        wave = amplitude * np.exp(1j * k * x)
        ey += wave
        hx -= cosine / np.sqrt(bz.MU_0 / bz.EPS_0) * wave
        expected[(m, n)] = (
            0.5 * area * cosine / np.sqrt(bz.MU_0 / bz.EPS_0) * abs(amplitude) ** 2
        )
    fields = {
        "Ey": ey.reshape(1, -1),
        "Hx": hx.reshape(1, -1),
        "Ex": np.zeros((1, x.size)),
        "Hy": np.zeros((1, x.size)),
    }
    monitor = SimpleNamespace(
        dft_fields=fields,
        dft_frequencies=np.array([frequency]),
        dft_weight_sum=np.ones(1),
        dft_amplitude_scale=1.0,
        integration_weights=np.full(x.size, 50e-9**2),
    )
    orders = diffraction_powers(monitor, grid, (kx, 0, 0), frequency)
    for order in orders:
        np.testing.assert_allclose(
            order["power_w"],
            expected.get((order["m"], order["n"]), 0),
            atol=1e-27,
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        sum(o["power_w"] for o in orders), sum(expected.values()), rtol=1e-12
    )
