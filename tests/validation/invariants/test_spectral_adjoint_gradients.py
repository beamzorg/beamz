"""Independent checks of pulsed adjoint FDTD against full time-loop AD and FD."""

import numpy as np
import pytest

from beamz.optimization import SoftMinModePower, TopologyProblem
from tests.topology_case import make_problem, make_wdm_problem

pytestmark = pytest.mark.optimization


def perturbed_density(problem):
    mask = problem.topology.region_mask
    density = problem.topology.initial_state().density.copy()
    density[mask] += 0.04 * np.random.default_rng(10).normal(size=mask.sum())
    return density


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("reference", [False, True])
def test_adjoint_full_gradient_and_finite_difference(polarization, reference):
    ad = make_problem(
        polarization=polarization, reference=reference, target_mode=0, run_time=320e-15
    )
    adjoint = TopologyProblem(
        ad.simulation, ad.topology, ad.objective, gradient_backend="adjoint"
    )
    density = perturbed_density(ad)
    value, expected = ad.value_and_grad(density, beta=2)
    actual_value, actual = adjoint.value_and_grad(density, beta=2)
    np.testing.assert_allclose(actual_value, value, rtol=2e-6)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.003
    np.testing.assert_array_equal(actual[~ad.topology.region_mask], 0)
    # A gradient-aligned direction avoids a nearly cancelling random projection.
    direction = expected / np.linalg.norm(expected)
    for h in (0.02, 0.01):
        finite_difference = (
            ad.value(density + h * direction, beta=2)
            - ad.value(density - h * direction, beta=2)
        ) / (2 * h)
        np.testing.assert_allclose(
            np.sum(actual * direction), finite_difference, rtol=0.003
        )
    diagnostics = adjoint.gradient_diagnostics
    assert diagnostics["adjoint_solves"] == 1
    assert diagnostics["terminal_field_ratio"] < 1e-4
    assert max(diagnostics["adjoint_terminal_field_ratios"]) < 1e-4


@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_adjoint_broadband_weighted_softmin_and_reference_sources(polarization):
    base = make_wdm_problem(run_time=400e-15, polarization=polarization)
    objective = base.objective + 0.3 * SoftMinModePower(
        "upper", reference_monitor="input", temperature=0.05
    )
    ad = TopologyProblem(base.simulation, base.topology, objective)
    adjoint = TopologyProblem(
        base.simulation, base.topology, objective, gradient_backend="adjoint"
    )
    density = perturbed_density(ad)
    value, expected = ad.value_and_grad(density, beta=2)
    actual_value, actual = adjoint.value_and_grad(density, beta=2)
    np.testing.assert_allclose(actual_value, value, rtol=3e-6)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.005
    assert adjoint.gradient_diagnostics["adjoint_solves"] == 6
    direction = expected / np.linalg.norm(expected)
    h = 0.01
    finite_difference = (
        ad.value(density + h * direction, beta=2)
        - ad.value(density - h * direction, beta=2)
    ) / (2 * h)
    np.testing.assert_allclose(
        np.sum(actual * direction), finite_difference, rtol=0.005
    )


def test_adjoint_run_duration_convergence_and_fixed_dft_storage():
    gradients, stored = [], []
    for duration in (320e-15, 480e-15):
        base = make_problem(target_mode=0, run_time=duration)
        adjoint = TopologyProblem(
            base.simulation, base.topology, base.objective, gradient_backend="adjoint"
        )
        _, gradient = adjoint.value_and_grad(perturbed_density(base))
        gradients.append(gradient)
        stored.append(adjoint.gradient_diagnostics["stored_design_dft_values"])
    # The 16 x 20 cell patch affects a 17 x 21 Ez node patch, including edges.
    assert stored[0] == stored[1] == 17 * 21
    assert (
        np.linalg.norm(gradients[1] - gradients[0]) / np.linalg.norm(gradients[1])
        < 0.003
    )
