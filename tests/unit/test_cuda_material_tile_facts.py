import numpy as np
import pytest

from scripts.cuda_material_tile_facts import uniform_boxes


def pack(codes):
    values = np.asarray(codes, dtype=np.uint8).ravel()
    padded = np.zeros((values.size + 3) // 4 * 4, dtype=np.uint8)
    padded[: values.size] = values
    return padded.view(np.int32)


def test_moving_trajectory_includes_low_region_absent_from_initial_tile():
    codes = np.zeros((12, 12, 12), dtype=np.uint8)
    codes[3, 3, 3] = 1
    fact = uniform_boxes(
        pack(codes),
        np.array([1, 2], np.float32),
        codes.shape,
        [[6, 6, 6], [10, 10, 10]],
        owned=(2, 2, 2),
        depth=4,
    )
    assert fact["admitted"].tolist() == [False, True]


def test_different_codes_with_identical_bits_are_certified():
    codes = np.indices((6, 7, 8)).sum(axis=0).astype(np.uint8) % 2
    fact = uniform_boxes(
        pack(codes),
        np.array([2, 2], np.float32),
        codes.shape,
        [[2, 2, 2]],
        owned=(3, 3, 3),
        depth=2,
    )
    assert fact["admitted"].tolist() == [True]


def test_signed_zero_is_distinct_and_invalid_codes_are_positive_zero():
    codes = np.zeros((6, 6, 6), dtype=np.uint8)
    codes[2, 2, 2] = 1
    table = np.array([0, -0.0], np.float32)
    fact = uniform_boxes(
        pack(codes), table, codes.shape, [[2, 2, 2]], owned=(2, 2, 2), depth=1
    )
    assert not fact["admitted"][0]
    codes[2, 2, 2] = 255
    fact = uniform_boxes(
        pack(codes), table, codes.shape, [[2, 2, 2]], owned=(2, 2, 2), depth=1
    )
    assert fact["admitted"][0] and fact["bits"] == 0


def test_support_clipping_and_empty_tiles_are_safe():
    shape = (5, 6, 7)
    fact = uniform_boxes(
        pack(np.zeros(shape, np.uint8)),
        np.array([3], np.float32),
        shape,
        [[0, 0, 0], [20, 20, 20]],
        depth=4,
    )
    assert fact["admitted"].all()
    assert (fact["lower"][1] == fact["upper"][1]).all()


def test_wrong_packet_size_is_rejected():
    with pytest.raises(ValueError, match="Packed size"):
        uniform_boxes(
            np.zeros(1, np.int32), np.array([1], np.float32), (6, 7, 8), [[1, 1, 1]]
        )
