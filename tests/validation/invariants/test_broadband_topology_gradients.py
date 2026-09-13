"""Frequency/port indexing and full electromagnetic gradients for combined goals."""

from dataclasses import replace

import numpy as np
import pytest

from beamz import LIGHT_SPEED, FieldMonitor
from beamz.optimization import ModePower, TopologyProblem, WeightedObjective
from tests.topology_case import make_wdm_problem

pytestmark = pytest.mark.optimization


def _density(problem):
    values = problem.topology.initial_state().density.copy()
    mask = problem.topology.region_mask
    values[mask] += 0.05 * np.random.default_rng(4).normal(size=mask.sum())
    return values


def _ordinary_spectrum(results, term):
    target = results.mode(term.monitor).amps.sel(
        direction=term.direction, mode_index=term.mode_index
    )
    frequencies = (
        target.f.values if term.frequencies is None else np.asarray(term.frequencies)
    )
    indices = [int(np.argmin(abs(target.f.values - f))) for f in frequencies]
    power = abs(target.values[indices]) ** 2
    if term.reference_monitor is None:
        return power / results.sources[0].power
    reference = results.mode(term.reference_monitor).amps.sel(
        direction=term.reference_direction, mode_index=term.reference_mode_index
    )
    indices = [int(np.argmin(abs(reference.f.values - f))) for f in frequencies]
    return power / abs(reference.values[indices]) ** 2


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("reference", [True, False])
def test_broadband_multiport_gradients_and_ordinary_analysis(
    polarization, reference, validation_metrics
):
    base = make_wdm_problem(polarization=polarization)
    # Unequal frequency counts/order and an unrelated DFT allocation test both
    # ragged arena offsets and frequency matching independently of array indices.
    monitors = list(base.simulation.monitors)
    monitors[0] = monitors[0].updated_copy(
        freqs=np.r_[monitors[0].freqs[::-1], LIGHT_SPEED / 1.42e-6]
    )
    monitors[1] = monitors[1].updated_copy(freqs=monitors[1].freqs[::-1])
    auxiliary = FieldMonitor(
        center=(3e-6, 2.5e-6, 0),
        size=(0, 1e-6, 1e-6),
        freqs=[LIGHT_SPEED / 1.42e-6],
        name="auxiliary",
    )
    objective = base.objective
    if not reference:
        objective = WeightedObjective(
            tuple(replace(t, reference_monitor=None) for t in objective.terms),
            objective.weights,
        )
    problem = TopologyProblem(
        base.simulation.updated_copy(monitors=[auxiliary, *monitors]),
        base.topology,
        objective,
    )
    density = _density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    spectra = problem.spectra(density, beta=2)
    ordinary = problem.material_simulation(density, beta=2).run(
        backend="jax", performance=False
    )
    for spectrum, term in zip(spectra, objective.terms, strict=True):
        np.testing.assert_allclose(
            spectrum, _ordinary_spectrum(ordinary, term), rtol=4e-5, atol=2e-7
        )
    expected = sum(
        w * np.mean(s) for w, s in zip(objective.weights, spectra, strict=True)
    )
    np.testing.assert_allclose(value, expected, rtol=2e-5, atol=1e-7)
    np.testing.assert_array_equal(gradient[~problem.topology.region_mask], 0)
    direction = (
        np.random.default_rng(7).normal(size=density.shape)
        * problem.topology.region_mask
    )
    direction /= np.linalg.norm(direction)
    adjoint = float(np.sum(gradient * direction))
    assert abs(adjoint) > 1e-6
    for h in (0.03, 0.01):
        finite_difference = (
            problem.value(density + h * direction, beta=2)
            - problem.value(density - h * direction, beta=2)
        ) / (2 * h)
        validation_metrics.check(
            "broadband multiport gradient",
            measured=finite_difference,
            reference=adjoint,
            tolerance="gradient_float32",
            metadata={"h": h, "polarization": polarization, "reference": reference},
        )
        np.testing.assert_allclose(finite_difference, adjoint, rtol=0.01, atol=3e-6)


def test_default_broadband_mean_and_reference_mode_selection():
    base = make_wdm_problem()
    # A reflected-reference diagnostic is unusual physically, but verifies the
    # explicit reference direction rather than silently selecting the + mode.
    term = ModePower("upper", reference_monitor="input", reference_direction="-")
    problem = TopologyProblem(base.simulation, base.topology, term)
    density = _density(problem)
    ordinary = problem.material_simulation(density, beta=2).run(
        backend="jax", performance=False
    )
    expected = _ordinary_spectrum(ordinary, term)
    np.testing.assert_allclose(
        problem.spectra(density, beta=2)[0], expected, rtol=4e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        problem.value(density, beta=2), expected.mean(), rtol=4e-5
    )
