"""The public inverse-design entry point lowers to the native CUDA VJP."""

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.backend import cuda_backend_status
from tests.topology_3d_case import make_3d_design


@pytest.mark.hardware
def test_public_3d_inverse_design_cuda_gradient():
    status = cuda_backend_status()
    if not status.available:
        pytest.skip(status.reason or "CUDA unavailable")

    design = make_3d_design(run_time=120e-15)
    simulation = design.simulation.updated_copy(
        resolution=40e-9, boundaries=(bz.PML(formulation="cpml"),)
    )
    jax_design = design.updated_copy(simulation=simulation)
    cuda_design = jax_design.updated_copy(backend="cuda")
    density = design.design_region.initial_parameters

    def objective(data):
        amplitude = data["output"].amps.sel(direction="+", mode_index=0).values[0]
        return (amplitude.conj() * amplitude).real

    run_jax = jax.jit(jax.value_and_grad(jax_design.make_objective_fn(objective)))
    run_cuda = jax.jit(jax.value_and_grad(cuda_design.make_objective_fn(objective)))
    expected_value, expected_gradient = run_jax(density)
    executable = run_cuda.lower(density).compile()
    assert "beamz_cuda_material_adjoint" in executable.as_text()
    value, gradient = executable(density)
    assert float(value) > 0 and np.linalg.norm(gradient) > 1e-7
    np.testing.assert_allclose(value, expected_value, rtol=2e-5, atol=1e-7)
    assert (
        np.linalg.norm(gradient - expected_gradient) / np.linalg.norm(expected_gradient)
        < 3e-4
    )
