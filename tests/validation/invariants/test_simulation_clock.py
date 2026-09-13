"""Long runs must not accumulate float32 clock error in DFT phases."""

from dataclasses import replace

import numpy as np
import pytest

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
