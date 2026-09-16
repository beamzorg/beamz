"""Long-run monitor clocks must stay on the source's integer-indexed time grid."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import Design, FieldMonitor, Material, Simulation
from beamz.simulation.execute import initial_program_state


@pytest.mark.parametrize("origin", [0.0, 2.3e-13])
@pytest.mark.parametrize("chunked", [False, True])
def test_long_run_clock_preserves_optical_phase(origin, chunked):
    dt = 1.3233502301891554e-16
    steps = 60000
    dx = 80e-9
    frequency = 299792458 / 1.55e-6
    simulation = Simulation(
        design=Design(
            width=6 * dx, height=6 * dx, depth=6 * dx, background=Material(1)
        ),
        resolution=dx,
        time=origin + np.arange(steps) * dt,
        monitors=[
            FieldMonitor(
                center=(3 * dx, 3 * dx, 3 * dx),
                size=(0, dx, dx),
                freqs=(frequency,),
                fields=("Hx",),
                name="constant",
            )
        ],
    )
    program = simulation.compile(backend="jax")
    state = initial_program_state(
        program, t=origin, current_step=0, monitor_steps=steps
    )
    state = state._replace(hx=jnp.ones_like(state.hx))
    sizes = [10000] * 6 if chunked else [steps]
    for size in sizes:
        run = simulation.advance(
            state=state, num_steps=size, backend="jax", performance=False
        )
        state = run.state
    assert np.max(np.abs(np.asarray(state.hx)[..., 1:-1] - 1)) < 1e-6
    expected_time = origin + steps * simulation.dt
    phase_error = 2 * np.pi * frequency * abs(float(state.t) - expected_time)
    assert phase_error < 0.002, f"Optical DFT phase error is {phase_error:.6g} rad"
