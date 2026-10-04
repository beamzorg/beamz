"""A constant field has an exact finite-window DFT, including after continuation."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.execute import build_scan, initial_program_state


@pytest.mark.parametrize("loop_kind", ["scan", "fori_loop"])
@pytest.mark.parametrize("start_time", [0.0, 3.7e-15])
def test_long_run_dft_tracks_integer_source_clock(loop_kind, start_time):
    steps = 70000
    dt = 2.242730135262796e-18
    f = 7.324e14
    sim = bz.Simulation(
        size=(2e-9, 2e-9, 2e-9),
        resolution=1e-9,
        time=start_time + np.arange(steps + 1) * dt,
        boundaries=[bz.Periodic(axes=("x", "y", "z"))],
        monitors=[
            bz.FieldMonitor(
                center=(0, 0, 0),
                size=(2e-9, 2e-9, 0),
                freqs=[f],
                fields=("Ex",),
                name="probe",
            )
        ],
    )
    program = sim.compile(backend="jax", num_steps=steps)
    program = replace(program, config=replace(program.config, loop_kind=loop_kind))
    state = initial_program_state(program, t=start_time, current_step=0)
    state = state._replace(ex=jnp.ones_like(state.ex))
    final = build_scan(program)(state, program.coefficients)
    dt32 = np.float32(program.config.dt)
    t0 = np.float32(start_time)
    expected_end = np.float32(t0 + dt32 * np.float32(steps))
    assert abs(float(final.t) - float(expected_end)) <= 1.1 * np.spacing(expected_end)
    # Read the raw field DFT directly: no source or analysis normalization involved.
    theta = 2 * np.pi * f * float(dt32)
    expected = (
        np.exp(1j * 2 * np.pi * f * float(t0))
        * np.exp(1j * theta)
        * (-np.expm1(1j * steps * theta))
        / (-np.expm1(1j * theta))
    )
    values = np.asarray(final.dft_vec_re) + 1j * np.asarray(final.dft_vec_im)
    count = program.monitors[0].dft_point_count
    np.testing.assert_allclose(
        values[:count] / steps, expected / steps, atol=4e-7, rtol=2e-4
    )
    half_program = replace(
        program, config=replace(program.config, num_steps=steps // 2)
    )
    advance = build_scan(half_program)
    first = advance(state, program.coefficients)
    second = advance(first, program.coefficients)
    np.testing.assert_allclose(
        np.asarray(second.dft_vec_re) / steps,
        np.asarray(final.dft_vec_re) / steps,
        atol=4e-7,
    )
    np.testing.assert_allclose(
        np.asarray(second.dft_vec_im) / steps,
        np.asarray(final.dft_vec_im) / steps,
        atol=4e-7,
    )
