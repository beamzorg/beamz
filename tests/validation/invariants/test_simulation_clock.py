"""Long runs must not accumulate float32 clock error in DFT phases."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz import Simulation
from beamz.simulation.execute import build_scan, runtime_inputs


@pytest.mark.parametrize("loop", ["scan", "fori_loop", "checkpoint"])
@pytest.mark.parametrize("start_time", [0.0, 7e-15])
def test_long_run_clock_uses_integer_step_count(loop, start_time):
    steps, dt = 50000, 4e-17
    sim = Simulation(
        domain=(150e-9, 125e-9),
        resolution=25e-9,
        time=start_time + np.arange(steps) * dt,
    )
    program = sim.compile(backend="jax")
    program = replace(
        program,
        config=replace(
            program.config, loop_kind="scan" if loop == "checkpoint" else loop
        ),
    )
    state = runtime_inputs(program, sim.initial_state(), monitor_steps=steps)
    scan = build_scan(program, checkpoint_interval=32 if loop == "checkpoint" else None)
    out = scan(state, program.coefficients)
    expected = float(state.t) + steps * float(np.float32(program.config.dt))
    np.testing.assert_allclose(float(out.t), expected, rtol=2e-7, atol=0)
    assert int(out.current_step) == steps


def test_continuation_preserves_an_explicit_clock_origin():
    sim = Simulation(
        domain=(150e-9, 125e-9), resolution=25e-9, time=np.arange(50000) * 4e-17
    )
    state = sim.initial_state()._replace(t=np.float32(7e-15))
    full = sim.advance(state=state, performance=False, backend="jax").state
    first = sim.advance(
        state=state, num_steps=17000, performance=False, backend="jax"
    ).state
    continued = sim.advance(state=first, performance=False, backend="jax").state
    np.testing.assert_allclose(float(full.t), float(continued.t), rtol=2e-7, atol=0)
    expected = 7e-15 + int(full.current_step) * float(np.float32(sim.dt))
    np.testing.assert_allclose(float(full.t), expected, rtol=2e-7, atol=0)


@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_checkpointed_dispersive_scan_preserves_state_clock_and_gradient(polarization):
    medium = bz.PoleResidue.lorentz(
        1.0,
        strength=0.5,
        resonance=6e15,
        damping=8e14,
        frequency_range=(4e14, 8e14),
    )
    component = "Ez" if polarization == "tm" else "Ex"
    sim = bz.Simulation(
        design=bz.Design(width=200e-9, height=200e-9, background=medium),
        resolution=50e-9,
        time=7e-15 + np.arange(67) * 1e-17,
        polarization=polarization,
        boundaries=[bz.Periodic()],
        monitors=[
            bz.FieldRecorder((component,), interval=1),
            bz.FieldMonitor(
                center=(100e-9, 100e-9, 0),
                size=(0, 200e-9, 0),
                freqs=[5e14],
                fields=(component,),
            ),
        ],
    )
    program = sim.compile(backend="jax")
    initial = sim.initial_state()
    field_name = component.lower()
    initial = initial._replace(
        **{field_name: getattr(initial, field_name).at[1, 1].set(1)},
        t=initial.t + jnp.asarray(2e-17, dtype=jnp.float32),
    )
    initial = runtime_inputs(program, initial, monitor_steps=67)
    assert initial.polarization

    def run(scan, state, factor):
        # Override the runtime ADE coefficients to catch a checkpoint path
        # accidentally using the original, statically bound material plan.
        coefficients = tuple(
            (ratio * factor, inverse, response)
            for _, ratio, inverse, response in program.dispersion.coefficients
        )
        return scan(state, program.coefficients, coefficients)

    ordinary = build_scan(program)
    checkpointed = build_scan(program, checkpoint_interval=8)
    factor = jnp.asarray(1.02, dtype=jnp.float32)
    full = run(ordinary, initial, factor)
    actual = run(checkpointed, initial, factor)
    for value, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves(full), strict=True
    ):
        np.testing.assert_allclose(value, expected, rtol=2e-5, atol=2e-6)
    assert int(actual.recorded_counts[0]) == 67
    np.testing.assert_allclose(
        float(actual.t), float(initial.t) + 67 * sim.dt, rtol=2e-7
    )

    first_program = replace(program, config=replace(program.config, num_steps=23))
    second_program = replace(program, config=replace(program.config, num_steps=44))
    first = run(build_scan(first_program, checkpoint_interval=8), initial, factor)
    resumed = run(build_scan(second_program, checkpoint_interval=8), first, factor)
    # An explicitly offset float32 clock rounds once at the continuation
    # boundary. Compare complex DFT amplitudes together so phase roundoff is
    # measured relative to the phasor, including nearly zero real quadratures.
    np.testing.assert_allclose(
        resumed.dft_vec_re + 1j * resumed.dft_vec_im,
        full.dft_vec_re + 1j * full.dft_vec_im,
        rtol=2e-5,
        atol=2e-6,
    )
    for name in resumed._fields:
        if name in {"dft_vec_re", "dft_vec_im"}:
            continue
        for value, expected in zip(
            jax.tree.leaves(getattr(resumed, name)),
            jax.tree.leaves(getattr(full, name)),
            strict=True,
        ):
            np.testing.assert_allclose(value, expected, rtol=2e-5, atol=2e-6)

    def loss(scan, factor):
        state = run(scan, initial, factor)
        return jnp.sum(getattr(state, field_name) ** 2)

    expected_gradient = jax.grad(lambda x: loss(ordinary, x))(factor)
    gradient = jax.grad(lambda x: loss(checkpointed, x))(factor)
    assert abs(float(expected_gradient)) > 1e-5
    np.testing.assert_allclose(gradient, expected_gradient, rtol=5e-4, atol=1e-6)
