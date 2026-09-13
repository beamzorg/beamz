"""Broadband spatial-source grouping against independent frequency solves and AD."""

import numpy as np
import pytest

from beamz.optimization import TopologyProblem
from tests.topology_case import make_problem
from tests.validation.invariants.test_spectral_adjoint_gradients import (
    perturbed_density,
)

pytestmark = pytest.mark.optimization


@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_broadband_basis_preserves_full_gradient(polarization):
    base = make_problem(
        polarization=polarization, reference=False, target_mode=0, run_time=400e-15
    )
    center = base.simulation.monitors[0].freqs[0]
    simulation = base.simulation.updated_copy(
        monitors=[
            m.updated_copy(freqs=center * np.linspace(0.98, 1.02, 9))
            for m in base.simulation.monitors
        ]
    )
    density = perturbed_density(base)
    gradients = {}
    for name, backend, grouping in [
        ("ad", "autodiff", "frequency"),
        ("frequency", "adjoint", "frequency"),
        ("grouped", "adjoint", "auto"),
    ]:
        problem = TopologyProblem(
            simulation,
            base.topology,
            base.objective,
            gradient_backend=backend,
            adjoint_source_grouping=grouping,
        )
        value, grad = problem.value_and_grad(density, beta=2)
        gradients[name] = np.asarray(grad)
        assert np.isfinite(value)
    assert problem.gradient_diagnostics["adjoint_solves"] < 9
    assert problem.gradient_diagnostics["source_reconstruction_error"] <= 1e-7
    assert (
        np.linalg.norm(gradients["grouped"] - gradients["frequency"])
        / np.linalg.norm(gradients["frequency"])
        < 1e-4
    )
    assert (
        np.linalg.norm(gradients["grouped"] - gradients["ad"])
        / np.linalg.norm(gradients["ad"])
        < 0.003
    )


@pytest.mark.parametrize(
    "frequencies",
    [np.array([1e14, 3e14]), np.r_[np.linspace(1e14, 1.01e14, 9), 1.19e14]],
)
def test_wide_bands_keep_individually_centered_pulses(frequencies):
    from types import SimpleNamespace

    from beamz.simulation.spectral_adjoint import SpectralAdjoint

    backend = object.__new__(SpectralAdjoint)
    backend.problem = SimpleNamespace(adjoint_source_grouping="auto")
    backend.frequencies = frequencies
    backend.last_diagnostics = {}
    # These sources have rank one, but grouping their very wide band would
    # normalize by negligible pulse DFTs. They must retain independent solves.
    groups = list(
        backend._source_groups(
            iter(np.array([1j, 2]) * (i + 1) for i in range(len(frequencies)))
        )
    )
    assert len(groups) == len(frequencies)
    assert backend.last_diagnostics["source_grouping"] == "frequency"
    for fi, source, weights in groups:
        np.testing.assert_allclose(
            source * weights[0], [1j * (fi[0] + 1), 2 * (fi[0] + 1)]
        )


def test_vector_3d_broadband_basis_matches_autodiff():
    import jax
    import jax.numpy as jnp

    from tests.topology_3d_case import make_3d_design

    base = make_3d_design()
    monitor = base.simulation.monitors[0]
    simulation = base.simulation.updated_copy(
        monitors=[
            monitor.updated_copy(freqs=monitor.freqs[0] * np.linspace(0.99, 1.01, 9))
        ]
    )
    ad = base.updated_copy(simulation=simulation)
    adj = ad.updated_copy(gradient_backend="adjoint", adjoint_source_grouping="auto")
    params = ad.design_region.initial_parameters + np.random.default_rng(5).uniform(
        -0.05, 0.05, ad.design_region.params_shape
    )

    def objective(data):
        amp = data["output"].amps.sel(direction="+", mode_index=0).values
        return jnp.mean(abs(amp) ** 2 + 0.01 * amp.real)

    va, ga = jax.value_and_grad(ad.make_objective_fn(objective))(params)
    vb, gb = jax.value_and_grad(adj.make_objective_fn(objective))(params)
    np.testing.assert_allclose(va, vb, rtol=2e-5)
    assert np.linalg.norm(ga - gb) / np.linalg.norm(ga) < 0.003
    assert adj.gradient_diagnostics["adjoint_solves"] < 9
    assert adj.gradient_diagnostics["source_reconstruction_error"] <= 1e-7
