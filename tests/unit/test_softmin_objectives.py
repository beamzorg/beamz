"""Worst-frequency objectives and their composition must preserve the reducer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from beamz.optimization import ModePower, SoftMinModePower


@pytest.fixture
def bind_spectrum_to_state(monkeypatch):
    monkeypatch.setattr(
        ModePower, "bind_spectrum", lambda *_: lambda state: jnp.asarray(state)
    )


@pytest.mark.usefixtures("bind_spectrum_to_state")
def test_softmin_bound_and_gradient_focus_on_weak_frequencies():
    term = SoftMinModePower("output", temperature=0.03)
    reduce = term.bind(None, None)
    powers = jnp.array([0.2, 0.8, 0.9])
    score, weights = jax.value_and_grad(reduce)(powers)
    assert powers.min() - 0.03 * np.log(3) <= score <= powers.min()
    assert weights[0] > 0.999
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(reduce(jnp.array([0.4])), 0.4, atol=1e-7)
    np.testing.assert_allclose(
        reduce(jnp.full(3, 0.4)), 0.4 - 0.03 * np.log(3), atol=1e-7
    )


@pytest.mark.usefixtures("bind_spectrum_to_state")
def test_weighted_composition_keeps_softmin_instead_of_averaging():
    powers = jnp.array([0.1, 0.4, 0.9])
    objective = 0.8 * SoftMinModePower("output", temperature=0.05) - 0.2 * ModePower(
        "reflection"
    )
    actual = objective.bind(None, None)(powers)
    expected = 0.8 * (-0.05 * logsumexp(-np.asarray(powers) / 0.05))
    expected -= 0.2 * float(powers.mean())
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan, True, "0.03"])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        SoftMinModePower("output", temperature=temperature)
