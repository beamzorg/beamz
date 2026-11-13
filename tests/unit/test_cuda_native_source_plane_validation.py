"""Ensure the bounded source probe actually replaces the original source packet."""
import numpy as np

from scripts.validate_cuda_native_source_plane import make_packet


def packet():
    values = [np.empty(0, np.float32) for _ in range(134)]
    for index, shape in enumerate(((64, 96, 129), (64, 97, 128), (65, 96, 128),
                                   (65, 97, 128), (65, 96, 129), (64, 97, 129))):
        values[index] = np.broadcast_to(np.float32(0), shape)
    for group in range(9):
        values[86 + 3 * group] = np.full((2, 1, 1, 1), 999, np.float32)
        values[87 + 3 * group] = np.full((2, 7), 999, np.float32)
        values[88 + 3 * group] = np.zeros((2, 3), np.int32)
    values[113] = np.zeros((1, 6, 1, 1), np.int32)
    values[114] = np.ones((1, 6, 1, 1), np.float32)
    return values


def test_synthetic_probe_does_not_restore_original_coefficients_or_other_sources():
    original = packet()
    values, attrs = make_packet(original, {}, "direct-source-DFT", 17, True)
    assert attrs["nsteps"] == 17
    assert attrs["coincident_source_group_mask"] == 160
    assert values[101].shape == values[107].shape == (2, 40, 72, 1)
    assert values[102].shape == values[108].shape == (2, 17)
    assert values[103][0].tolist() == [12, 12, 31]
    assert values[109][0].tolist() == [12, 12, 32]
    assert values[101][0, 0, 0, 0] == np.float32(.1732050776)
    assert values[101][0, -1, -1, 0] != values[101][0, 0, 0, 0]
    assert all(values[86 + 3 * group].shape[0] == 0
               for group in range(9) if group not in (5, 7))
    assert original[101].shape == (2, 1, 1, 1)
    assert original[101][0, 0, 0, 0] == 999


def test_count_three_negative_keeps_full_physical_sheet_and_live_third_source():
    values, _ = make_packet(packet(), {}, "count3", 35, True)
    assert values[101].shape == values[107].shape == (3, 40, 72, 1)
    assert values[102].shape == values[108].shape == (3, 35)
    np.testing.assert_array_equal(values[101][2], values[101][0] * np.float32(.625))
