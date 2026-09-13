"""An orthogonal output port must measure outgoing power and its derivative."""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianPulse,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Rectangle,
    Simulation,
)
from beamz.optimization import ModePower, TopologyProblem, TopologySpec


@pytest.mark.optimization
@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("gradient_backend", ["autodiff", "adjoint"])
def test_bend_output_direction_and_external_density_gradient(
    polarization, gradient_backend
):
    dx, frequency = 100e-9, LIGHT_SPEED / 1.55e-6
    core = Material(permittivity=4)
    design = Design(width=6e-6, height=6e-6, material=Material(permittivity=1))
    design += Rectangle(position=(0, 2.6e-6), width=2e-6, height=0.8e-6, material=core)
    design += Rectangle(
        position=(2.6e-6, 4e-6), width=0.8e-6, height=2e-6, material=core
    )
    modes = ModeSpec(num_modes=1, polarization=polarization)
    source = ModeSource(
        center=(1e-6, 3e-6, 0),
        size=(0, 2.8e-6, 1e-6),
        direction="+",
        mode_spec=modes,
        source_time=GaussianPulse(freq0=frequency, fwidth=0.15 * frequency, offset=2),
    )
    incoming = ModeMonitor(
        center=(1.5e-6, 3e-6, 0),
        size=(0, 2.8e-6, 1e-6),
        freqs=[frequency],
        mode_spec=modes,
        name="input",
    )
    outgoing = incoming.updated_copy(
        center=(3e-6, 4.8e-6, 0), size=(2.8e-6, 0, 1e-6), name="top"
    )
    simulation = Simulation(
        design=design,
        resolution=dx,
        run_time=320e-15 if gradient_backend == "adjoint" else 220e-15,
        sources=[source],
        monitors=[incoming, outgoing],
        polarization=polarization,
        boundaries=[PML(thickness=0.6e-6, formulation="cpml")],
    )
    mask = np.zeros(simulation.compile(backend="jax").grid.material_grid.shape, bool)
    mask[20:40, 20:40] = True
    topology = TopologySpec(
        design=design,
        region_mask=mask,
        resolution=dx,
        eps_min=1,
        eps_max=4,
        projection_type="identity",
    )
    problem = TopologyProblem(
        simulation,
        topology,
        ModePower("top", reference_monitor="input"),
        gradient_backend=gradient_backend,
    )
    density = np.zeros(mask.shape)
    # Seed a connected bend to avoid a weak resonant-cavity signal masquerading
    # as a successful direction/derivative check.
    density[mask] = 0.1
    density[26:34, 20:34] = 0.9
    density[26:40, 26:34] = 0.9
    value, gradient = problem.value_and_grad(density)
    data = problem.material_simulation(density).run(backend="jax", performance=False)
    input_power = abs(data.mode("input").amps.sel(direction="+").item()) ** 2
    forward = abs(data.mode("top").amps.sel(direction="+").item()) ** 2
    backward = abs(data.mode("top").amps.sel(direction="-").item()) ** 2
    assert forward / input_power > 1e-3
    assert forward > 10 * backward
    np.testing.assert_allclose(value, forward / input_power, rtol=4e-5)
    np.testing.assert_array_equal(gradient[~mask], 0)
    direction = np.random.default_rng(7).normal(size=mask.shape) * mask
    direction /= np.linalg.norm(direction)
    h = 0.01
    finite_difference = (
        problem.value(density + h * direction) - problem.value(density - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(
        finite_difference, np.sum(gradient * direction), rtol=0.01, atol=3e-6
    )
