"""Exact storage checks independent of the native coefficient reader."""

import numpy as np
import pytest

from beamz.simulation.cuda.storage import pack_material_blocks, unpack_material_blocks


def decode(pool, headers, shape):
    headers = np.asarray(headers).ravel()
    index = np.arange(np.prod(shape))
    offsets = headers[index // 32]
    addresses = np.where(offsets < 0, ~offsets, offsets + index % 32)
    return np.asarray(pool)[addresses].reshape(shape)


@pytest.mark.parametrize("shape", [(7, 9, 35), (2, 3, 128), (1, 1, 1)])
def test_bit_exact_staggered_components_and_mixed_blocks(shape):
    rng = np.random.default_rng(361)
    values = []
    for component in range(3):
        dimensions = tuple(n + (a == component) for a, n in enumerate(shape))
        v = np.full(dimensions, 1.0 + component, np.float32)
        flat = v.ravel()
        flat[37:57] = rng.uniform(1, 3, size=flat[37:57].size)
        # Preserve signed zero and differences smaller than an allclose tolerance.
        flat[:3] = [0.0, -0.0, np.nextafter(np.float32(1), np.float32(2))][
            : flat[:3].size
        ]
        values.append(v)
    pools, headers = pack_material_blocks(values, max_bytes=np.inf)
    assert pools[0] is pools[1] is pools[2]
    for v, pool, header in zip(values, pools, headers, strict=True):
        np.testing.assert_array_equal(
            decode(pool, header, v.shape).view(np.uint32), v.view(np.uint32)
        )
        np.testing.assert_array_equal(
            unpack_material_blocks(pool, header, v.shape).view(np.uint32),
            v.view(np.uint32),
        )


def test_constant_interiors_share_pool_and_reduce_storage():
    values = [np.ones((32, 32, 32), np.float32) for _ in range(3)]
    for v in values:
        v[8:24] = 6.0
    pools, headers = pack_material_blocks(values, max_bytes=3 * 32**3)
    assert pools[0].size == 2
    assert all(np.all(np.asarray(h) < 0) for h in headers)
    assert pools[0].nbytes + sum(h.nbytes for h in headers) < 0.04 * sum(
        v.nbytes for v in values
    )


def test_unprofitable_or_nonfinite_encoding_falls_back():
    rng = np.random.default_rng(7)
    values = [rng.uniform(1, 2, (8, 8, 8)).astype(np.float32) for _ in range(3)]
    assert pack_material_blocks(values, max_bytes=sum(v.nbytes for v in values)) is None
    values[0][0, 0, 0] = np.nan
    assert pack_material_blocks(values, max_bytes=np.inf) is None
    assert pack_material_blocks([np.float32(1)] * 3, max_bytes=np.inf) is None
