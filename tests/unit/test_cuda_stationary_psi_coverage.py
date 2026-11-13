"""Independent finite-domain CPML update/first-owned/high-surface counters."""

import itertools

import numpy as np
import pytest

from scripts.analyze_cuda_stationary_psi_coverage import cache_axis, volume


def test_cache_region_class_requires_one_axis_and_no_io():
    for flags in range(64):
        selected = cache_axis(flags)
        has_io = bool(flags & 17)
        pml = (flags & 14) >> 1
        expected = pml.bit_length() - 1 if not has_io and pml.bit_count() == 1 else -1
        assert selected == expected
    with pytest.raises(ValueError):
        cache_axis(64)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_active_volume_and_lazy_io_counts_match_explicit_physical_points(axis):
    shape = (24, 25, 27)
    owned = (3, 4, 5)
    lower = np.array([[-2, 0, 1], [10, 11, 12], [22, 23, 25], [24, 26, 27]])
    upper = lower + owned
    active = volume(lower, upper, shape, axis)
    incoming = active - volume(lower + 1, upper, shape, axis)
    outgoing = active - volume(lower, upper - 1, shape, axis)
    for row, lo in enumerate(lower):
        expected = enters = leaves = 0
        for relative in itertools.product(*(range(b) for b in owned)):
            point = tuple(int(a + b) for a, b in zip(lo, relative, strict=True))
            valid = all(0 <= p < size for p, size in zip(point, shape, strict=True))
            if valid and (point[axis] < 12 or point[axis] >= shape[axis] - 12):
                expected += 1
                enters += any(r == 0 for r in relative)
                leaves += any(r == b - 1 for r, b in zip(relative, owned, strict=True))
        assert active[row] == expected
        assert incoming[row] == enters
        assert outgoing[row] == leaves
