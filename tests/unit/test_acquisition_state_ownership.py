"""Continuation retains produced acquisitions only when arena layouts agree."""

import jax.numpy as jnp
import numpy as np

from beamz import FieldMonitor, Simulation, um
from beamz.simulation.execute import initial_program_state


def test_changed_dft_layout_resets_acquisitions_without_resetting_fields():
    monitor = FieldMonitor(
        center=(0, 0, 0),
        size=(0, 0.4 * um, 0),
        freqs=[2e14],
        fields=("Ez", "Hy"),
        name="fields",
    )
    simulation = Simulation(
        domain=(0.8 * um, 0.6 * um),
        resolution=0.1 * um,
        time=np.arange(3) * 1e-17,
        monitors=[monitor],
    )
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0)
    state = state._replace(
        ez=jnp.ones_like(state.ez),
        dft_vec_re=jnp.full_like(state.dft_vec_re, 7),
        dft_weight_sum=jnp.full_like(state.dft_weight_sum, 2),
    )
    continued = initial_program_state(
        program, t=1e-17, current_step=1, continuation=state
    )
    np.testing.assert_array_equal(continued.dft_vec_re, state.dft_vec_re)
    np.testing.assert_array_equal(continued.dft_weight_sum, state.dft_weight_sum)

    changed = simulation.updated_copy(
        monitors=[monitor.updated_copy(freqs=[2e14, 3e14])]
    ).compile(backend="jax")
    restarted = initial_program_state(
        changed, t=1e-17, current_step=1, continuation=state
    )
    assert restarted.dft_weight_sum.shape == (2,)
    np.testing.assert_array_equal(restarted.dft_vec_re, 0)
    np.testing.assert_array_equal(restarted.dft_weight_sum, 0)
    np.testing.assert_array_equal(restarted.ez, state.ez)
    assert int(restarted.current_step) == 1
