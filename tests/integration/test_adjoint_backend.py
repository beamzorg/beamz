"""Backend selection, decay checks, zero objectives, and optimizer lifecycle."""

import numpy as np
import pytest

from beamz import PML, Bloch, Periodic
from beamz.optimization import InverseDesign
from tests.topology_case import make_optimizer, make_problem

pytestmark = pytest.mark.optimization


@pytest.fixture(scope="module")
def base():
    return make_problem(target_mode=0, run_time=320e-15)


def test_adjoint_optimizer_resume_and_backend_identity(base, tmp_path):
    adjoint = InverseDesign(
        base.simulation, base.design_region, base.objective, gradient_backend="adjoint"
    )
    density = base.design_region.initial_parameters
    full = make_optimizer(adjoint, 3).run(steps=3, initial_params=density)
    path = tmp_path / "adjoint.npz"
    partial = make_optimizer(adjoint, 3).run(
        steps=3, initial_params=density, stop_after=1, checkpoint=path
    )
    fresh = InverseDesign(
        base.simulation, base.design_region, base.objective, gradient_backend="adjoint"
    )
    resumed = make_optimizer(fresh, 3).run(
        steps=3, resume=make_optimizer(fresh, 3).load_result(path)
    )
    assert partial.completed_steps == 1
    np.testing.assert_array_equal(full.final_params, resumed.final_params)
    assert full.objective > full.initial_objective
    with pytest.raises(ValueError, match="different optimization problem"):
        make_optimizer(base, 3).load_result(path)
    tighter = InverseDesign(
        base.simulation,
        base.design_region,
        base.objective,
        gradient_backend="adjoint",
        adjoint_decay_tolerance=1e-5,
    )
    with pytest.raises(ValueError, match="different optimization problem"):
        make_optimizer(tighter, 3).load_result(path)
    assert base.gradient_diagnostics is None


def test_zero_objective_skips_adjoint_runs(base):
    adjoint = InverseDesign(
        base.simulation,
        base.design_region,
        0 * base.objective,
        gradient_backend="adjoint",
    )
    value, gradient = adjoint.value_and_grad(base.design_region.initial_parameters)
    assert value == 0
    np.testing.assert_array_equal(gradient, 0)
    assert adjoint.gradient_diagnostics["adjoint_solves"] == 0


def test_rejects_undecayed_forward_and_adjoint(base):
    short = base.simulation.updated_copy(run_time=80e-15)
    adjoint = InverseDesign(
        short, base.design_region, base.objective, gradient_backend="adjoint"
    )
    with pytest.raises(ValueError, match="forward pulse has not decayed"):
        adjoint.value_and_grad(base.design_region.initial_parameters)
    # Here the forward fields decay farther than the adjoint dual fields. Verify
    # that the backward run has an independent convergence guard.
    strict = InverseDesign(
        base.simulation,
        base.design_region,
        base.objective,
        gradient_backend="adjoint",
        adjoint_decay_tolerance=1e-6,
    )
    with pytest.raises(ValueError, match="adjoint pulse has not decayed"):
        strict.value_and_grad(base.design_region.initial_parameters)


def test_rejects_downsampled_monitors(base):
    simulation = base.simulation.updated_copy(
        monitors=[m.updated_copy(interval=2) for m in base.simulation.monitors]
    )
    with pytest.raises(ValueError, match="rectangular DFT"):
        InverseDesign(
            simulation, base.design_region, base.objective, gradient_backend="adjoint"
        )


@pytest.mark.parametrize("boundary", [Periodic(axes="y"), Bloch(axes="y")])
def test_spectral_adjoint_rejects_periodic_seams(base, boundary):
    simulation = base.simulation.updated_copy(
        boundaries=[
            PML(thickness=0.6e-6, edges=("left", "right"), formulation="cpml"),
            boundary,
        ]
    )
    with pytest.raises(ValueError, match="adjoint.*periodic"):
        InverseDesign(
            simulation, base.design_region, base.objective, gradient_backend="adjoint"
        )
    # The discrete autodiff path differentiates the ordinary seam update.
    problem = InverseDesign(
        simulation, base.design_region, base.objective, gradient_backend="autodiff"
    )
    assert problem.program.boundary.periodic_axes


@pytest.mark.parametrize("backend", ["unknown", "spectral_adjoint", None])
def test_rejects_unknown_backend(base, backend):
    with pytest.raises(ValueError, match="gradient_backend"):
        InverseDesign(
            base.simulation,
            base.design_region,
            base.objective,
            gradient_backend=backend,
        )


@pytest.mark.parametrize(
    "tolerance", [0, 1, -1, float("nan"), float("inf"), True, "1e-4"]
)
def test_rejects_invalid_decay_tolerance(base, tolerance):
    with pytest.raises(ValueError, match="adjoint_decay_tolerance"):
        InverseDesign(
            base.simulation,
            base.design_region,
            base.objective,
            adjoint_decay_tolerance=tolerance,
            gradient_backend="autodiff",
        )
