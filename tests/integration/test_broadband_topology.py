"""Broadband objective contracts, checkpoint identity, and optimization lifecycle."""

from dataclasses import replace

import numpy as np
import pytest

from beamz.optimization import TopologyProblem, WeightedObjective
from tests.topology_case import make_wdm_problem

pytestmark = pytest.mark.optimization


@pytest.fixture(scope="module")
def problem():
    return make_wdm_problem()


def test_multiport_optimization_resume_and_spectra(problem, tmp_path):
    density = problem.topology.initial_state().density.copy()
    mask = problem.topology.region_mask
    density[mask] += 0.05 * np.random.default_rng(4).normal(size=mask.sum())
    full = problem.run(5, initial_density=density)
    partial = problem.run(
        5, initial_density=density, stop_after=2, checkpoint=tmp_path / "wdm.npz"
    )
    fresh = TopologyProblem(problem.simulation, problem.topology, problem.objective)
    resumed = fresh.run(5, resume=fresh.load(tmp_path / "wdm.npz"))
    assert partial.completed_steps == 2
    np.testing.assert_array_equal(resumed.final_params, full.final_params)
    np.testing.assert_array_equal(
        [h.objective for h in resumed.history], [h.objective for h in full.history]
    )
    assert full.objective > full.initial_objective + 0.03
    spectra = problem.spectra(full.final_params, beta=full.beta)
    assert [len(s) for s in spectra] == [3, 3, 3, 3, 6]
    np.testing.assert_allclose(
        full.objective,
        sum(
            w * s.mean()
            for w, s in zip(problem.objective.weights, spectra, strict=True)
        ),
        rtol=2e-5,
        atol=1e-7,
    )


def test_checkpoint_records_weights_frequency_selection_and_reference(
    problem, tmp_path
):
    problem.run(5, stop_after=0, checkpoint=tmp_path / "identity.npz")
    variants = [
        problem.objective * 0.5,
        WeightedObjective(
            (
                replace(
                    problem.objective.terms[0],
                    frequencies=problem.objective.terms[0].frequencies[:1],
                ),
                *problem.objective.terms[1:],
            ),
            problem.objective.weights,
        ),
        WeightedObjective(
            (
                replace(problem.objective.terms[0], reference_direction="-"),
                *problem.objective.terms[1:],
            ),
            problem.objective.weights,
        ),
    ]
    for objective in variants:
        other = TopologyProblem(problem.simulation, problem.topology, objective)
        with pytest.raises(ValueError, match="different optimization problem"):
            other.load(tmp_path / "identity.npz")


def test_frequency_matching_rejects_missing_target_or_reference(problem):
    terms = problem.objective.terms
    missing = replace(terms[0], frequencies=[1e14])
    with pytest.raises(ValueError, match="record frequency"):
        TopologyProblem(problem.simulation, problem.topology, missing)
    monitors = list(problem.simulation.monitors)
    monitors[0] = monitors[0].updated_copy(freqs=monitors[0].freqs[1:])
    with pytest.raises(ValueError, match="input.*record frequency"):
        TopologyProblem(
            problem.simulation.updated_copy(monitors=monitors),
            problem.topology,
            terms[0],
        )
