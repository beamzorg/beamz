"""Exact JAX coefficient representation, storage accounting and pool gradients."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.compile import _add_memory_arrays
from beamz.simulation.compile.material_blocks import (
    MaterialBlocks,
    compact_material,
    material_values,
    reciprocal_material,
)
from tests.jax_compat import enable_x64


def sample(dtype, encoding):
    value = np.ones((15, 17, 35), dtype)
    value[4:10] = 6
    if encoding == "blocks":
        value.ravel()[100:450] = np.arange(350, dtype=dtype) / 1000 + 2
    return value


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("encoding", ["bytes", "blocks"])
def test_exact_values_and_actual_storage(dtype, encoding):
    value = sample(dtype, encoding)
    # Signed zero and one-ULP differences cannot be discarded.
    value.ravel()[:3] = [0.0, -0.0, np.nextafter(dtype(1), dtype(2))]
    compressed = compact_material(value)
    assert isinstance(compressed, MaterialBlocks)
    assert compressed.headers.dtype == (np.uint8 if encoding == "bytes" else np.int32)
    with enable_x64(True):
        actual = np.asarray(jax.jit(material_values)(compressed))
    bits = np.uint32 if dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(actual.view(bits), value.view(bits))
    assert compressed.nbytes <= 0.75 * value.nbytes
    entries = []
    _add_memory_arrays(entries, "epsilon", compressed, "coefficients")
    assert sum(e["bytes"] for e in entries) == compressed.nbytes
    assert len(entries) == 2


def test_uniform_scalar_and_incompressible_fallback():
    value = np.full((8, 9, 10), 3, np.float32)
    scalar = compact_material(value)
    assert scalar.shape == () and scalar.dtype == value.dtype and scalar == 3
    varying = np.arange(value.size, dtype=np.float32).reshape(value.shape)
    assert compact_material(varying) is varying
    varying[0, 0, 0] = np.nan
    assert compact_material(varying) is varying
    assert compact_material(np.empty((0,), np.float32)).size == 0


@pytest.mark.parametrize("encoding", ["bytes", "blocks"])
def test_pool_gradient_and_reciprocal_match_independent_index_sum(encoding):
    value = sample(np.float64, encoding)
    c = compact_material(value)
    assert isinstance(c, MaterialBlocks)
    heads = c.headers
    if heads.dtype == np.uint8:
        index = heads
    else:
        linear = np.arange(value.size)
        h = heads[linear // 32]
        index = np.where(h < 0, ~h, h + linear % 32)
    weights = np.random.default_rng(42).normal(size=value.shape)
    expected = np.zeros_like(c.values)
    np.add.at(expected, index, -weights.ravel() / value.ravel() ** 2)
    with enable_x64(True):

        def loss(pool):
            inverse = reciprocal_material(replace(c, values=pool))
            return jnp.sum(material_values(inverse) * weights)

        result, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(c.values))
    np.testing.assert_allclose(result, np.sum(weights / value), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-12)
