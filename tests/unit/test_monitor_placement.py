from __future__ import annotations

from types import MappingProxyType

import pytest

from beamz.devices._placement import SnappedInterval, SnappedRegion


def test_snapped_interval_snapshots_rectilinear_edges():
    edges = [0.0, 0.2, 1.0]
    interval = SnappedInterval(0, 2, 0.2, edges)  # type: ignore[arg-type]

    edges[1] = 0.5

    assert interval.edges == (0.0, 0.2, 1.0)
    assert interval.size == 1.0


def test_compact_snapped_interval_retains_exact_physical_bounds():
    interval = SnappedInterval(0, 2, 0.2, physical_bounds=(0.0, 1.0))

    assert interval.lower == 0.0
    assert interval.upper == 1.0
    assert interval.center == 0.5
    assert interval.size == 1.0

    with pytest.raises(ValueError, match="nondecreasing"):
        SnappedInterval(0, 2, 0.2, physical_bounds=(1.0, 0.0))


def test_snapped_region_snapshots_and_freezes_intervals():
    intervals = {"X": SnappedInterval(0, 2, 0.2)}
    region = SnappedRegion(2, "Y", 1, 0.3, intervals)

    intervals.clear()

    assert isinstance(region.intervals, MappingProxyType)
    assert region.normal_axis == "y"
    assert region.axis_interval("x") == SnappedInterval(0, 2, 0.2)
    with pytest.raises(TypeError):
        region.intervals["x"] = SnappedInterval(1, 2, 0.2)  # type: ignore[index]


def test_snapped_region_rejects_invalid_interval_values():
    with pytest.raises(TypeError, match="SnappedInterval"):
        SnappedRegion(2, "y", 1, 0.3, {"x": object()})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="axis"):
        SnappedRegion(2, "y", 1, 0.3, {"time": SnappedInterval(0, 1, 0.2)})

    with pytest.raises(ValueError, match="normal axis"):
        SnappedRegion(2, "time", 1, 0.3)


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("sign", [-1, 1])
def test_source_region_preserves_scalar_and_realized_grid_tie_policies(axis, sign):
    import numpy as np

    from beamz.design.grid import RectilinearGrid
    from beamz.devices._placement import (
        snap_mode_source_region,
        snap_mode_source_region_grid,
    )

    # Coordinate 2 is equidistant from centers 1.5 and 2.5: scalar half-even
    # selects cell 2, realized nearest-center selects the first cell, 1.
    arguments = dict(
        center=(2.0, 2.0, 2.0),
        width=2.0,
        height=2.0,
        axis=axis,
        direction_sign=sign,
        is_3d=True,
    )
    scalar = snap_mode_source_region(**arguments, grid_shape=(5, 5, 5), resolution=1.0)
    realized = snap_mode_source_region_grid(
        **arguments, grid=RectilinearGrid(*(np.arange(6.0),) * 3)
    )
    assert scalar.plane_index == 2
    assert realized.plane_index == 1
    assert scalar.companion_index == (1 if sign > 0 else 3)
    assert realized.companion_index == (0 if sign > 0 else 2)
    for region in (scalar, realized):
        for interval in region.intervals.values():
            assert (interval.start, interval.stop) == (1, 3)


def test_scalar_endpoint_tolerance_is_preserved_separately_from_realized_edges():
    import numpy as np

    from beamz.devices._placement import (
        snap_edge_interval,
        snap_rectilinear_edge_interval,
    )

    upper = np.nextafter(3.0, np.inf)
    assert snap_edge_interval(1.0, upper, 1.0, 5).stop == 3
    assert snap_rectilinear_edge_interval(1.0, upper, np.arange(6.0)).stop == 4


def test_realized_source_support_translates_with_grid_origin():
    import numpy as np

    from beamz.design.grid import RectilinearGrid
    from beamz.devices._placement import snap_mode_source_region_grid

    grid = RectilinearGrid(np.array([0, 1, 3, 6.0]), np.arange(5.0), np.arange(4.0))
    offset = (10.0, -4.0, 7.0)
    arguments = dict(width=2.0, height=2.0, axis="x", direction_sign=-1, is_3d=True)
    original = snap_mode_source_region_grid(center=(2, 2, 2), grid=grid, **arguments)
    moved = snap_mode_source_region_grid(
        center=tuple(2 + v for v in offset), grid=grid.translated(offset), **arguments
    )
    assert moved.plane_index == original.plane_index
    assert moved.companion_index == original.companion_index
    assert moved.plane_coord == original.plane_coord + offset[0]
    for i, axis in enumerate("yz", 1):
        a, b = original.axis_interval(axis), moved.axis_interval(axis)
        assert (a.start, a.stop) == (b.start, b.stop)
        assert b.lower == a.lower + offset[i]
        assert b.upper == a.upper + offset[i]
