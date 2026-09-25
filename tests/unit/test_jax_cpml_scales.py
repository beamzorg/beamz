"""Precomputed JAX material scales must retain lossy physics and gradients."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.execute import build_scan, compiled_source_batches, forward_step
from beamz.simulation.kernels import select_update_kernel
from tests.unit.test_cuda_runtime_contract import _program_and_state


@pytest.mark.parametrize("conductivity", [0.0, 1e3, 1e6])
def test_cpml_runtime_scales_match_direct_update_and_material_gradient(conductivity):
    program, state, context = _program_and_state(cpml=True)
    context = replace(
        context,
        config=program.config,
        source_batches=compiled_source_batches(program.sources),
    )
    kernel = select_update_kernel(context)
    rng = np.random.default_rng(915)
    state = state._replace(
        **{
            name: jnp.asarray(
                rng.normal(size=getattr(state, name).shape), dtype=jnp.float32
            )
            * (1e-3 if name.startswith("e") else 1e-6)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    coefficients = program.coefficients._replace(
        **{
            **{
                f"e_conductivity_{c}": jnp.full_like(
                    getattr(program.coefficients, f"e_permittivity_{c}"), conductivity
                )
                for c in "xyz"
            },
            **{
                f"h_sigma_m_{c}": jnp.asarray(conductivity * 1e3, dtype=jnp.float32)
                for c in "xyz"
            },
        }
    )
    scan = build_scan(program)

    def direct(epsilon):
        coeffs = coefficients._replace(e_permittivity_x=epsilon)
        return jax.lax.fori_loop(
            0,
            program.config.num_steps,
            lambda i, carry: forward_step(
                carry,
                ctx=context,
                coeffs=coeffs,
                program=program,
                update_kernel=kernel,
                observation_time=state.t + context.dt_scalar * (i + 1),
            ),
            state,
        )

    def optimized(epsilon):
        return scan(state, coefficients._replace(e_permittivity_x=epsilon))

    epsilon = coefficients.e_permittivity_x
    for ref, actual in zip(
        jax.tree.leaves(direct(epsilon)),
        jax.tree.leaves(optimized(epsilon)),
        strict=True,
    ):
        ref, actual = np.asarray(ref), np.asarray(actual)
        scale = float(np.max(np.abs(ref), initial=0))
        np.testing.assert_allclose(
            actual, ref, rtol=3e-5, atol=max(1e-12, 3e-6 * scale)
        )

    def loss(run, epsilon):
        return jnp.sum(run(epsilon).ex ** 2)

    ref_grad = jax.grad(lambda eps: loss(direct, eps))(epsilon)
    actual_grad = jax.grad(lambda eps: loss(optimized, eps))(epsilon)
    np.testing.assert_allclose(actual_grad, ref_grad, rtol=1e-4, atol=1e-10)
