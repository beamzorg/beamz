"""Backend selection, decay checks, zero objectives, and optimizer lifecycle."""

import numpy as np
import pytest

from beamz.optimization import TopologyProblem
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization


@pytest.fixture(scope="module")
def base():
    return make_problem(target_mode=0, run_time=320e-15)


def test_adjoint_optimizer_resume_and_backend_identity(base, tmp_path):
    adjoint = TopologyProblem(
        base.simulation, base.topology, base.objective, gradient_backend="adjoint"
    )
    density = base.topology.initial_state().density
    full = adjoint.run(3, initial_density=density)
    path = tmp_path / "adjoint.npz"
    partial = adjoint.run(3, initial_density=density, stop_after=1, checkpoint=path)
    fresh = TopologyProblem(
        base.simulation, base.topology, base.objective, gradient_backend="adjoint"
    )
    resumed = fresh.run(3, resume=fresh.load(path))
    assert partial.completed_steps == 1
    np.testing.assert_array_equal(full.state.density, resumed.state.density)
    assert full.objective > full.initial_objective
    with pytest.raises(ValueError, match="different optimization problem"):
        base.load(path)
    tighter = TopologyProblem(
        base.simulation,
        base.topology,
        base.objective,
        gradient_backend="adjoint",
        adjoint_decay_tolerance=1e-5,
    )
    with pytest.raises(ValueError, match="different optimization problem"):
        tighter.load(path)
    assert base.gradient_diagnostics is None


def test_zero_objective_skips_adjoint_runs(base):
    adjoint = TopologyProblem(
        base.simulation, base.topology, 0 * base.objective, gradient_backend="adjoint"
    )
    value, gradient = adjoint.value_and_grad(base.topology.initial_state().density)
    assert value == 0
    np.testing.assert_array_equal(gradient, 0)
    assert adjoint.gradient_diagnostics["adjoint_solves"] == 0


def test_rejects_undecayed_forward_and_adjoint(base):
    short = base.simulation.updated_copy(run_time=80e-15)
    adjoint = TopologyProblem(
        short, base.topology, base.objective, gradient_backend="adjoint"
    )
    with pytest.raises(ValueError, match="forward pulse has not decayed"):
        adjoint.value_and_grad(base.topology.initial_state().density)
    # Here the forward fields decay farther than the adjoint dual fields. Verify
    # that the backward run has an independent convergence guard.
    strict = TopologyProblem(
        base.simulation,
        base.topology,
        base.objective,
        gradient_backend="adjoint",
        adjoint_decay_tolerance=1e-6,
    )
    with pytest.raises(ValueError, match="adjoint pulse has not decayed"):
        strict.value_and_grad(base.topology.initial_state().density)


def test_rejects_downsampled_monitors(base):
    simulation = base.simulation.updated_copy(
        monitors=[m.updated_copy(interval=2) for m in base.simulation.monitors]
    )
    with pytest.raises(ValueError, match="rectangular DFT"):
        TopologyProblem(
            simulation, base.topology, base.objective, gradient_backend="adjoint"
        )


@pytest.mark.parametrize("backend", ["unknown", "spectral_adjoint", None])
def test_rejects_unknown_backend(base, backend):
    with pytest.raises(ValueError, match="gradient_backend"):
        TopologyProblem(
            base.simulation, base.topology, base.objective, gradient_backend=backend
        )


@pytest.mark.parametrize(
    "tolerance", [0, 1, -1, float("nan"), float("inf"), True, "1e-4"]
)
def test_rejects_invalid_decay_tolerance(base, tolerance):
    with pytest.raises(ValueError, match="adjoint_decay_tolerance"):
        TopologyProblem(
            base.simulation,
            base.topology,
            base.objective,
            adjoint_decay_tolerance=tolerance,
        )
