"""The optical phase must not depend on XLA's timestep reassociation."""

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.observe import _dft_phase_angle


def test_dft_phase_rounds_frequency_and_clock_before_multiplication():
    frequencies = np.linspace(1.8e14, 2.0e14, 101, dtype=np.float32)

    @jax.jit
    def phases(origin, dt):
        def step(_, index):
            time = origin + dt * (index + 1)
            return None, (_dft_phase_angle(frequencies, time, jnp.float32), time)

        return jax.lax.scan(step, None, jnp.arange(4096))[1]

    actual, clocks = jax.device_get(
        phases(np.float32(1.3e-13), np.float32(1.4636332e-16))
    )
    # Evaluate each specified rounding in float64 and cast explicitly; NumPy
    # cannot silently reassociate this with the loop's clock calculation.
    omega = (np.float64(np.float32(2 * np.pi)) * frequencies.astype(np.float64)).astype(
        np.float32
    )
    expected = (clocks.astype(np.float64)[:, None] * omega.astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_array_equal(actual, expected)


def test_dft_phase_retains_time_derivative():
    frequencies = jnp.asarray([1.8e14, 2.0e14], dtype=jnp.float32)
    derivative = jax.jit(
        jax.grad(lambda t: jnp.sum(_dft_phase_angle(frequencies, t, jnp.float32)))
    )(jnp.float32(1e-13))
    np.testing.assert_allclose(
        derivative, np.sum(np.float32(2 * np.pi) * np.asarray(frequencies)), rtol=1e-7
    )
