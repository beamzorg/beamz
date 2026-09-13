"""Full electromagnetic derivatives, including CPML, Yee sampling and modal DFTs."""

import numpy as np
import pytest

from beamz.optimization import TopologyProblem
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization


def initial_density(problem):
    density = problem.topology.initial_state().density.copy()
    mask = problem.topology.region_mask
    density[mask] += 0.05 * np.random.default_rng(10).normal(size=mask.sum())
    # Explicitly excite the odd target mode. Symmetry-preserving discretization
    # correctly suppresses it for a uniform design; small filtered noise alone
    # gives a weak signal and cancellation-dominated finite differences.
    density += 0.25 * np.linspace(-1, 1, density.shape[0])[:, None] * mask
    return density


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("reference", [False, True])
def test_full_solver_gradient_matches_finite_difference(
    polarization, reference, validation_metrics
):
    problem = make_problem(polarization=polarization, reference=reference)
    density = initial_density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    assert value > 1e-4  # A nontrivial signal, not a vanishing-pulse false positive.
    np.testing.assert_array_equal(gradient[~problem.topology.region_mask], 0)
    for seed in (1, 7):
        direction = np.random.default_rng(seed).normal(size=density.shape)
        direction *= problem.topology.region_mask
        direction /= np.linalg.norm(direction)
        # Keep a random component, but avoid nearly cancelling projections whose
        # small objective differences are dominated by float32 roundoff at h=.003.
        sign = 1.0 if np.sum(gradient * direction) >= 0 else -1.0
        direction += sign * 0.1 * gradient / np.linalg.norm(gradient)
        direction /= np.linalg.norm(direction)
        adjoint = float(np.sum(gradient * direction))
        assert abs(adjoint) > 1e-5
        for h in (0.03, 0.01, 0.003):
            difference = (
                problem.value(density + h * direction, beta=2)
                - problem.value(density - h * direction, beta=2)
            ) / (2 * h)
            validation_metrics.check(
                f"{polarization} modal gradient seed={seed} h={h}",
                measured=difference,
                reference=adjoint,
                tolerance="gradient_float32",
                metadata={"h": h, "seed": seed, "reference_monitor": reference},
            )
            # These centered differences are above cancellation noise. A 1% gate
            # accommodates float32 roundoff across supported JAX CPU/GPU targets.
            np.testing.assert_allclose(difference, adjoint, rtol=0.01, atol=3e-6)
    ordinary = problem.material_simulation(density, beta=2).run(
        backend="jax", performance=False
    )
    expected = (
        abs(ordinary.mode("output").amps.sel(direction="+", mode_index=1).item()) ** 2
    )
    if reference:
        expected /= (
            abs(ordinary.mode("input").amps.sel(direction="+", mode_index=0).item())
            ** 2
        )
    np.testing.assert_allclose(value, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("interval", [1, 32, 1000])
def test_checkpointed_adjoint_matches_uncheckpointed_scan(interval):
    reference = make_problem(checkpoint_interval=None)
    checkpointed = TopologyProblem(
        reference.simulation,
        reference.topology,
        reference.objective,
        checkpoint_interval=interval,
    )
    density = initial_density(reference)
    reference_value, reference_gradient = reference.value_and_grad(density, beta=2)
    value, gradient = checkpointed.value_and_grad(density, beta=2)
    np.testing.assert_allclose(value, reference_value, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=5e-4, atol=1e-7)


@pytest.mark.parametrize("projection", ["ssp", "identity"])
def test_projection_full_solver_gradient_matches_finite_difference(projection):
    from dataclasses import replace

    base = make_problem()
    problem = TopologyProblem(
        base.simulation,
        replace(
            base.topology,
            projection_type=projection,
            filter_radius=0
            if projection == "identity"
            else base.topology.filter_radius,
        ),
        base.objective,
    )
    density = initial_density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    direction = (
        np.random.default_rng(1).normal(size=density.shape)
        * problem.topology.region_mask
    )
    direction /= np.linalg.norm(direction)
    h = 0.01
    finite_difference = (
        problem.value(density + h * direction, beta=2)
        - problem.value(density - h * direction, beta=2)
    ) / (2 * h)
    assert value > 1e-4
    np.testing.assert_allclose(
        finite_difference, np.sum(gradient * direction), rtol=0.01, atol=3e-6
    )
