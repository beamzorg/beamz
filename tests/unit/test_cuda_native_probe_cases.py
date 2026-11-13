"""The reliability packet must hit intended Yee sites without mutating input data."""

import numpy as np
import pytest

from scripts.cuda_native_probe_cases import (
    branch_witness,
    point_probe_packet,
    probe_points,
)


def packet(n=40):
    arrays = [np.zeros((), np.float32) for _ in range(134)]
    shapes = [(n, n, n + 1), (n, n + 1, n), (n + 1, n, n),
              (n + 1, n + 1, n), (n + 1, n, n + 1), (n, n + 1, n + 1)]
    arrays[:6] = [np.zeros(shape, np.float32) for shape in shapes]
    arrays[113] = np.full((2, 6, 3, 2), -1, np.int32)
    arrays[114] = np.zeros((2, 6, 3, 2), np.float32)
    return arrays


def test_point_packet_uses_native_component_order_and_two_live_sources():
    original = packet()
    attrs = {"nsteps": 7, "coincident_source_group_mask": 0}
    point = (15, 16, 32)
    values, updated = point_probe_packet(original, attrs, point, 35)
    assert attrs["nsteps"] == 7 and updated["nsteps"] == 35
    assert updated["coincident_source_group_mask"] == 160
    for group in range(9):
        count = 2 if group in (5, 7) else 0
        assert values[86 + 3 * group].shape == (count, 1, 1, 1)
        assert values[87 + 3 * group].shape == (count, 35)
        if count:
            assert np.any(values[86 + 3 * group] * values[87 + 3 * group][:, :1])
            assert np.all(values[88 + 3 * group] == np.array(point))
    for component in range(6):
        native = (component + 3) % 6
        for monitor in range(2):
            flat = values[113][monitor, component, 0, 0]
            assert tuple(np.unravel_index(flat, original[native].shape)) == point
    assert np.count_nonzero(values[114]) == 12
    assert np.all(original[113] == -1) and not np.any(original[114])
    assert values[0] is original[0]  # Seeded state is preserved, not reinitialized.


def test_terminal_z_probe_respects_real_staggered_supports():
    values = packet()
    changed, _ = point_probe_packet(values, {}, (40, 15, 31), 35)
    # At z=N only Ex, Ey and Hz exist; Hx/Hy/Ez remain unsupported exact zero.
    assert [np.any(changed[113][:, c] >= 0) for c in range(6)] == [True, True, False, False, False, True]
    with pytest.raises(ValueError, match="source supports"):
        point_probe_packet(values, {}, (41, 15, 31), 35)


def test_probe_sites_cover_retirement_seams_and_high_cpml():
    sites = probe_points(packet())
    assert {0, 14, 15, 29, 30, 31, 39, 40}.issubset({p[0] for p in sites})
    assert any(y == 15 and x == 31 for _, y, x in sites)
    assert any(y == 16 and x == 32 for _, y, x in sites)


def test_positive_branch_coverage_requires_both_macro_and_certificate():
    assert not branch_witness({"moving_uniform_bulk_tiles": 2}, {"moving_macro_count": 0})["positive_uniform_branch"]
    assert not branch_witness({"moving_uniform_bulk_tiles": 0}, {"moving_macro_count": 2})["positive_uniform_branch"]
    assert branch_witness({"moving_uniform_bulk_tiles": 2}, {"moving_macro_count": 1})["positive_uniform_branch"]
