"""Check host checkpointing against the same differentiable FDTD trajectory."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.backend import cuda_backend_status
from tests.integration.test_differentiable_simulation import (
    make_simulation,
    seeded_state,
)


@pytest.mark.parametrize("backend", ("jax", "cuda"))
def test_host_checkpointed_material_gradient_matches_monolithic(backend):
    if backend == "cuda" and not cuda_backend_status().available:
        pytest.skip("CUDA extension unavailable")
    sim, dielectric = make_simulation(
        cpml=True,
        mixed=True,
        metric_kind="rectilinear",
        sources=True,
        monitors=True,
    )
    solver = sim.differentiable(backend=backend, num_steps=13)
    initial = seeded_state(solver)
    parameters = jnp.array([0.05, 0.1], dtype=jnp.float32)

    def materials(params):
        return tuple(
            epsilon + params[0] + (axis + 1) * params[1]
            for axis, epsilon in enumerate(dielectric)
        )

    def objective(state):
        fields = solver.monitor_fields(state, "output")
        power = jnp.sum(jnp.abs(fields) ** 2)
        return power, fields

    reference = jax.value_and_grad(
        lambda params: objective(solver(materials(params), state=initial)),
        has_aux=True,
    )
    host = solver.host_checkpointed_value_and_grad(
        materials,
        objective,
        segment_steps=3,
        segments_per_group=2,
        has_aux=True,
    )
    expected = reference(parameters)
    actual = host(parameters, initial_state=initial)
    assert float(jnp.linalg.norm(expected[1])) > 0
    for lhs, rhs in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        scale = float(jnp.max(jnp.abs(rhs), initial=0.0))
        np.testing.assert_allclose(lhs, rhs, rtol=4e-4, atol=max(1e-14, 5e-6 * scale))


def test_host_checkpointed_gradient_rejects_nonpositive_segment_size():
    sim, _ = make_simulation()
    solver = sim.differentiable(backend="jax")
    with pytest.raises(ValueError, match="segment_steps"):
        solver.host_checkpointed_value_and_grad(
            lambda p: p, lambda s: s.t, segment_steps=0
        )
