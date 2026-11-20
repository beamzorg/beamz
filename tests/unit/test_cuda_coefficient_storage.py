"""Exact storage checks independent of the native coefficient reader."""

import numpy as np
import pytest
from beamz.simulation.cuda.material_blocks import (
    pack_material_blocks,
    unpack_material_blocks,
)

from beamz._region_array import RegionArray


def decode(pool, headers, shape):
    headers = np.asarray(headers).ravel()
    index = np.arange(np.prod(shape))
    offsets = headers[index // 32]
    addresses = np.where(offsets < 0, ~offsets, offsets + index % 32)
    return np.asarray(pool)[addresses].reshape(shape)


@pytest.mark.parametrize("budget", [128, 300, 4096])
def test_recipe_packing_preserves_flat_blocks_with_bounded_reads(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    monkeypatch.setattr("beamz._region_array.TILE_BYTES", budget)
    value = np.random.default_rng(41).integers(0, 5, (7, 9, 35)).astype(np.float32)
    value.reshape(-1)[:3] = [0.0, -0.0, np.nextafter(np.float32(1), np.float32(2))]
    reads = []

    def read(region: tuple[slice, ...] = ()) -> np.ndarray:
        part = value[region]
        reads.append(part.nbytes)
        return part

    recipe = RegionArray(value.shape, value.dtype, read)

    def no_array(*args: object, **kwargs: object) -> None:
        raise AssertionError("Packing materialized the entire recipe")

    monkeypatch.setattr(RegionArray, "__array__", no_array)
    pools, headers = pack_material_blocks((recipe,), max_bytes=np.inf)
    np.testing.assert_array_equal(
        decode(pools[0], headers[0], value.shape).view(np.uint32), value.view(np.uint32)
    )
    assert reads and max(reads) <= budget


def test_constant_region_replay_preserves_cross_tile_literals(monkeypatch):
    monkeypatch.setattr("beamz._region_array.TILE_BYTES", 37 * 4)
    values = np.ones((1, 1, 157), np.float32)
    values[..., 71:77] = -0.0
    values[..., 113:127] = np.float32(2.5)
    calls = []

    def read(region):
        calls.append(region[-1].start)
        return values[region]

    recipe = RegionArray(values.shape, values.dtype, read)
    pools, headers = pack_material_blocks((recipe,), max_bytes=np.inf, host=True)
    assert isinstance(headers[0], np.ndarray) and not headers[0].flags.writeable
    assert isinstance(pools[0], np.ndarray) and not pools[0].flags.writeable
    np.testing.assert_array_equal(
        decode(pools[0], headers[0], values.shape).view(np.uint32),
        values.view(np.uint32),
    )
    assert calls.count(0) == 1


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


def test_components_can_be_packed_and_released_independently():
    values = []
    for c in range(3):
        value = np.ones((9, 11, 67), np.float32)
        value[:, 3:7, 18:42] = np.float32(2 + c)
        values.append(value)
    together = pack_material_blocks(values, max_bytes=np.inf)
    separate = [pack_material_blocks((value,), max_bytes=np.inf) for value in values]
    for value, (pools, headers) in zip(values, separate, strict=True):
        assert len(pools) == len(headers) == 1
        np.testing.assert_array_equal(
            unpack_material_blocks(pools[0], headers[0], value.shape).view(np.uint32),
            value.view(np.uint32),
        )
    # The native ABI uploads one table per component. Independent pools avoid
    # triplicating the unrelated components' literal blocks on the GPU.
    assert sum(pools[0].nbytes for pools, _ in separate) < 3 * together[0][0].nbytes
    assert pack_material_blocks((), max_bytes=np.inf) is None


def test_layered_material_keeps_bitwise_page_carries(monkeypatch):
    from beamz import _region_array
    from beamz.simulation.region_materials import sample_material

    monkeypatch.setattr(_region_array, "TILE_BYTES", 128)
    layers = np.array([2.0, -0.0, 0.0, 3.0, 1.0], np.float64)
    material = np.broadcast_to(layers[:, None, None], (5, 3, 13))
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        sampled = sample_material(material, component, (6, 4, 14))
        reference = np.asarray(sample_material(material.copy(), component, (6, 4, 14)))
        np.testing.assert_array_equal(np.asarray(sampled).view("u4"), reference.view("u4"))
    inputs = (material, np.broadcast_to(np.float32(-0.0), (5, 3, 17)))
    pools, headers = pack_material_blocks(inputs, max_bytes=np.inf, host=True)
    for source, pool, codes in zip(inputs, pools, headers, strict=True):
        np.testing.assert_array_equal(
            decode(pool, codes, source.shape).view("u4"), source.astype("f4").view("u4")
        )


@pytest.mark.parametrize("shape", [(9, 7, 33), (8, 4, 64), (7, 1, 65), (23, 1, 63)])
def test_layer_run_headers_match_dense_encoding_for_unaligned_interfaces(shape):
    values = np.resize(np.array([1, 1, -0., 0., 3, 7, 7], np.float32), shape[0])
    layers = np.broadcast_to(values[::-1, None, None], shape)
    dense = layers.copy()
    for size in (np.inf, dense.nbytes // 3):
        expected = pack_material_blocks((dense,), max_bytes=size, host=True)
        actual = pack_material_blocks((layers,), max_bytes=size, host=True)
        assert (actual is None) == (expected is None)
        if expected is not None:
            np.testing.assert_array_equal(actual[1][0], expected[1][0])
            np.testing.assert_array_equal(actual[0][0].view(np.uint32), expected[0][0].view(np.uint32))
