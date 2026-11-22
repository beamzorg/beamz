"""The packed CPML recurrence corrects only active absorbers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.kernels import correct_cpml_term
from beamz.simulation.model import CpmlPackedSlabSpec, CpmlTerm


@pytest.mark.parametrize("steps", [1, 31])
def test_cpml_fp32_recurrence(steps):
    derivative = np.array([[0.12345, 0.4321, -0.28123, 0.78123]], dtype=np.float32)
    term = CpmlTerm(
        "ex",
        1,
        -1.0,
        jnp.array([[0.0012345, -0.0023456]], dtype=jnp.float32),
        jnp.array([[0.99321, 0.98765]], dtype=jnp.float32),
        jnp.array([[0.81327, 0.73456]], dtype=jnp.float32),
        CpmlPackedSlabSpec(1, 1, 1, (1, 2)),
    )
    psi = jnp.array([[1.0078125, -0.50390625]], dtype=jnp.float32)
    expected_psi = np.asarray(psi)
    packed_derivative = derivative[:, [0, 3]]
    expected_corrected = None
    for _ in range(steps):
        next_psi = (
            np.asarray(term.b) * expected_psi + np.asarray(term.a) * packed_derivative
        )
        expected_corrected = derivative.copy()
        expected_corrected[:, [0, 3]] = (
            packed_derivative * np.asarray(term.inv_kappa) + next_psi
        )
        expected_corrected *= term.sign
        expected_psi = next_psi

    @jax.jit
    def advance(psi):
        def step(carry, _):
            corrected, new_psi = correct_cpml_term(jnp.asarray(derivative), carry, term)
            return new_psi, corrected

        return jax.lax.scan(step, psi, None, length=steps)

    actual_psi, corrected = advance(psi)
    assert actual_psi.dtype == jnp.float32
    assert corrected.dtype == jnp.float32
    np.testing.assert_allclose(corrected[-1], expected_corrected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(actual_psi), expected_psi, rtol=2e-6, atol=1e-7
    )
