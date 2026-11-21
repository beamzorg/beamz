import pytest


def test_compact_trail_faces_are_bijective_and_include_edges_and_corners():
    # Enumerate the complete physical export surface independently of padding.
    bz, by, bx = 7, 15, 31
    slots = []
    for z in range(bz):
        for y in range(by):
            for x in range(bx):
                if z == bz - 1:
                    slots.append(y * bx + x)
                elif y == by - 1:
                    slots.append(by * bx + z * bx + x)
                elif x == bx - 1:
                    slots.append(by * bx + (bz - 1) * bx + z * (by - 1) + y)
    assert sorted(slots) == list(range(735))
    assert len(slots) == bz * by * bx - (bz - 1) * (by - 1) * (bx - 1)


@pytest.mark.parametrize("strided", [False, True])
def test_bounded_material_check_matches_numpy_at_chunk_boundaries(strided):
    import numpy as np

    from beamz.design.discretization import _allclose_bounded

    values = np.ones((1025, 1024), np.float64)
    if strided:
        values = values.T[::2]
    for value in (1.0, 1.0 + 1e-11, 1.01, np.inf, np.nan):
        values[-1, -1] = value
        assert _allclose_bounded(values, 1.0, rtol=1e-10, atol=1e-12) == np.allclose(
            values, 1.0, rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("constant", [False, True])
def test_material_tensor_comparison_bounds_both_operands(constant, monkeypatch):
    import numpy as np

    from beamz.design.discretization import _allclose_bounded

    shape = (1025, 1024)
    left = (
        np.broadcast_to(np.float32(2), shape)
        if constant
        else np.full(shape, 2, np.float32).T
    )
    right = np.broadcast_to(np.float32(2), left.shape)
    actual = np.allclose
    sizes = []

    def checked(a, b, **kwargs):
        sizes.append(max(np.size(a), np.size(b)))
        return actual(a, b, **kwargs)

    monkeypatch.setattr(np, "allclose", checked)
    assert _allclose_bounded(left, right, rtol=1e-10, atol=1e-12)
    assert max(sizes) <= (1 if constant else 262144)
    assert not _allclose_bounded(left, right + np.float32(0.1), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("value", [0.0, -0.0, 2.25, float("inf"), float("nan")])
def test_constant_material_snapshot_preserves_bits_and_owns_input(value):
    import numpy as np

    from beamz.design.discretization import _material_snapshot

    original = np.full((513, 512), value, np.float64).T
    before = original.copy()
    snapshot = _material_snapshot(original)
    assert not snapshot.flags.writeable
    assert not np.shares_memory(snapshot, original)
    assert snapshot.strides == (0, 0)
    original[:] = 17
    np.testing.assert_array_equal(
        snapshot.copy().view(np.uint64), before.view(np.uint64)
    )


def test_material_snapshot_keeps_signed_zero_and_nan_payload_distinctions():
    import numpy as np

    from beamz.design.discretization import _material_snapshot

    for bits in (0x8000000000000000, 0x7FF8000000000001):
        original = np.zeros((513, 512), np.float64)
        original.view(np.uint64)[-1, -1] = bits
        snapshot = _material_snapshot(original)
        assert snapshot.flags.c_contiguous
        np.testing.assert_array_equal(
            snapshot.view(np.uint64), original.view(np.uint64)
        )
        original[:] = 1
        assert snapshot.view(np.uint64)[-1, -1] == bits


def test_private_raster_snapshot_adopts_whole_arrays_and_detaches_partial_views():
    import numpy as np

    from beamz.design.discretization import _material_snapshot

    tensor = np.arange(3 * 17 * 19, dtype=np.float32).reshape(3, 17, 19)
    diagonal = _material_snapshot(tensor[1], owned=True)
    np.testing.assert_array_equal(diagonal, tensor[1])
    assert not diagonal.flags.writeable and not np.shares_memory(diagonal, tensor)

    retained = _material_snapshot(tensor, owned=True)
    assert retained is tensor and not retained.flags.writeable


@pytest.mark.parametrize(
    "shape,axis", [((9, 13), 0), ((5, 7, 9), 1), ((6, 5, 7, 9), 3)]
)
@pytest.mark.parametrize("dtype", ["f4", ">f8", "c8", "i2"])
def test_broadcast_material_snapshot_owns_only_distinct_sites(shape, axis, dtype):
    import numpy as np

    from beamz.design.discretization import _material_snapshot

    values = np.arange(shape[axis]).astype(dtype)
    compact_shape = [1] * len(shape)
    compact_shape[axis] = shape[axis]
    view = np.broadcast_to(values[::-1].reshape(compact_shape), shape)
    expected = view.copy()
    snapshot = _material_snapshot(view)
    values.fill(0)
    assert not snapshot.flags.writeable
    assert snapshot.dtype == expected.dtype
    assert all(snapshot.strides[d] == 0 for d in range(len(shape)) if d != axis)
    np.testing.assert_array_equal(snapshot, expected)
