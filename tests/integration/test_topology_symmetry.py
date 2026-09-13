"""Reflection must survive material mapping, injection, measurement, and VJPs."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import FieldMonitor
from beamz.optimization import TopologyProblem
from beamz.simulation.differentiable import topology_yee_permittivity
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization


@pytest.fixture(scope="module")
def problem():
    return make_problem(target_mode=0, run_time=320e-15)


def relative_mirror_error(values):
    values = np.asarray(values)
    return np.linalg.norm(values - values[::-1]) / max(np.linalg.norm(values), 1e-30)


def test_material_map_preserves_reflection_and_bounds(problem):
    rho = np.random.default_rng(7).uniform(0.2, 0.8, problem.topology.region_mask.shape)
    rho = 0.5 * (rho + rho[::-1])
    eps = problem._permittivity(jnp.asarray(rho), 1.0)
    values = topology_yee_permittivity(problem.program, eps, problem._mask)
    for value in values:
        assert relative_mirror_error(value) < 2e-7
        assert np.min(value) >= 1 and np.max(value) <= 4
    # Unselected supports retain the original analytic raster exactly.
    untouched = topology_yee_permittivity(
        problem.program, eps, jnp.zeros_like(problem._mask)
    )
    for value, axis in zip(untouched, "xyz", strict=True):
        np.testing.assert_array_equal(
            value, getattr(problem.program.grid, f"eps_{axis}")
        )


def test_tm_source_coefficients_use_mirror_closed_node_support(problem):
    for source in problem.program.sources:
        coefficients = np.zeros(problem.program.grid.component_shapes[source.component])
        coefficients[source.index] = np.asarray(source.coeff).squeeze()
        assert relative_mirror_error(coefficients) < 2e-7


@pytest.mark.parametrize("backend", ["autodiff", "adjoint"])
def test_objective_gradient_is_reflection_equivariant(problem, backend):
    p = TopologyProblem(
        problem.simulation,
        problem.topology,
        problem.objective,
        gradient_backend=backend,
    )
    rho = np.random.default_rng(42).uniform(0.4, 0.6, p.topology.region_mask.shape)
    value, gradient = p.value_and_grad(rho)
    mirrored_value, mirrored_gradient = p.value_and_grad(rho[::-1].copy())
    np.testing.assert_allclose(value, mirrored_value, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, mirrored_gradient[::-1], rtol=2e-3, atol=2e-5)
    symmetric = 0.5 * (rho + rho[::-1])
    _, g = p.value_and_grad(symmetric)
    assert relative_mirror_error(g) < 2e-4
    # Include a boundary-pixel perturbation: the spectral adjoint must retain
    # every staggered field sample touched by the interpolation stencil.
    if backend == "adjoint":
        _, reference = problem.value_and_grad(symmetric)
        assert np.linalg.norm(g - reference) / np.linalg.norm(reference) < 0.005


def test_symmetric_forward_field_and_mode_basis(problem):
    rho = np.full(problem.topology.region_mask.shape, 0.5)
    monitor = FieldMonitor(
        center=(3e-6, 2e-6, 0),
        size=(6e-6, 4e-6, 0),
        freqs=problem.simulation.monitors[0].freqs,
        fields=("Ez",),
        name="plane",
    )
    # Offset apertures sample reflected portions of the output cross-section.
    # This catches a shared half-cell basis shift that a centered port hides.
    # Include decayed modal tails: a tightly truncated asymmetric cross-section
    # also has finite mode-domain boundary error, independently of Yee placement.
    output = problem.simulation.monitors[1]
    paired = [
        output.updated_copy(center=(5e-6, y, 0), size=(0, 2.4e-6, 1e-6), name=name)
        for y, name in [(1.6e-6, "lower"), (2.4e-6, "upper")]
    ]
    sim = problem.material_simulation(rho).updated_copy(
        monitors=[*problem.simulation.monitors, *paired, monitor]
    )
    data = sim.run(backend="jax", performance=False)
    ez = (
        data["plane"]
        .get_dft_component("Ez")
        .reshape(problem.topology.region_mask.shape)
    )
    assert relative_mirror_error(ez) < 2e-5
    from beamz.analysis.data import analysis_inputs
    from beamz.analysis.mode_projection import _monitor_profile_slice

    for name in ["input", "output"]:
        eps, _, _, _ = _monitor_profile_slice(
            analysis_inputs(data)[name], data[name].monitor, "x", 6
        )
        np.testing.assert_allclose(eps, eps[::-1], rtol=0, atol=1e-10)
    profiles = [
        _monitor_profile_slice(analysis_inputs(data)[m.name], m, "x", 6)[0]
        for m in paired
    ]
    np.testing.assert_allclose(profiles[0], profiles[1][::-1], rtol=0, atol=1e-10)
    powers = [
        abs(data.mode(m.name).amps.sel(direction="+", mode_index=0).item()) ** 2
        for m in paired
    ]
    np.testing.assert_allclose(*powers, rtol=2e-5)
