"""Full electromagnetic gradients and checkpoint identity for worst-band goals."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp

from beamz.optimization import InverseDesign, ModePower, SoftMinModePower
from tests.topology_case import make_optimizer, make_wdm_problem


@pytest.mark.optimization
@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_softmin_full_solver_derivative_and_checkpoint(polarization, tmp_path):
    base = make_wdm_problem(polarization=polarization)
    objective = SoftMinModePower("upper", reference_monitor="input", temperature=0.04)
    problem = InverseDesign(
        base.simulation, base.design_region, objective, gradient_backend="autodiff"
    )
    mask = problem.design_region.region_mask
    density = problem.design_region.initial_parameters.copy()
    density[mask] += 0.04 * np.random.default_rng(10).normal(size=mask.sum())
    value, gradient = problem.value_and_grad(density, beta=2)
    spectrum = problem.spectra(density, beta=2)[0]
    expected = -objective.temperature * logsumexp(-spectrum / objective.temperature)
    np.testing.assert_allclose(value, expected, rtol=3e-5, atol=2e-7)
    np.testing.assert_array_equal(gradient[~mask], 0)
    direction = np.random.default_rng(7).normal(size=mask.shape) * mask
    direction /= np.linalg.norm(direction)
    for h in (0.03, 0.01):
        finite_difference = (
            problem.value(density + h * direction, beta=2)
            - problem.value(density - h * direction, beta=2)
        ) / (2 * h)
        np.testing.assert_allclose(
            finite_difference, np.sum(gradient * direction), rtol=0.01, atol=3e-6
        )

    path = tmp_path / "worst-frequency.npz"
    make_optimizer(problem, 5, preset="wdm").run(steps=5, stop_after=0, checkpoint=path)
    assert make_optimizer(problem, 5, preset="wdm").load_result(
        path
    ).fingerprint == make_optimizer(problem, 5, preset="wdm")._fingerprint(
        problem.objective
    )
    for changed in (
        replace(objective, temperature=0.08),
        ModePower("upper", reference_monitor="input"),
    ):
        other = InverseDesign(
            base.simulation, base.design_region, changed, gradient_backend="autodiff"
        )
        with pytest.raises(ValueError, match="different optimization problem"):
            make_optimizer(other, 5, preset="wdm").load_result(path)
