"""Fresh state copying preserves an explicit setup device after its context ends."""

import os
import subprocess
import sys


def test_fresh_field_copies_preserve_setup_device_and_ownership():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import jax
import numpy as np
from beamz.simulation.model import SimulationState
from beamz.simulation.execute import initial_program_state
from tests.performance.h100_workloads import H100Workload
setup_device = jax.devices()[1]
with jax.default_device(setup_device):
    sim = H100Workload(name='initial_placement', shape_zyx=(9,10,11),
        timesteps=2, resolution=80e-9, pml_cells=2, cpml=True).build()
    program = sim.compile(backend='jax')
for state in (SimulationState.initial(program.grid, t=0),
              initial_program_state(program, t=0, current_step=0)):
    for component in ('ex','ey','ez','hx','hy','hz'):
        value = getattr(state, component)
        assert value.devices() == {setup_device}, (component, value.sharding)
        original = getattr(program.grid, component[0].upper()+component[1])
        expected = np.asarray(original).copy()
        result = jax.jit(lambda x: x + 1, donate_argnums=(0,))(value)
        result.block_until_ready()
        np.testing.assert_array_equal(original, expected)
        np.testing.assert_array_equal(result, expected + 1)
""",
        ],
        env=dict(
            os.environ,
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
            JAX_PLATFORMS="cpu",
        ),
        check=True,
        timeout=120,
    )
