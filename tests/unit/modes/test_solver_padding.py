"""Padding changes the material domain, independently of injection support."""

import numpy as np
import pytest

from beamz import ModeSpec
from beamz._cache_tokens import cache_token
from beamz.design.grid import RectilinearGrid
from beamz.devices._placement import mode_source_region
from beamz.devices.modes.plane import _local_mode_plane_spec


@pytest.mark.parametrize("value", [-1, (0, -1)])
def test_negative_padding_is_rejected(value):
    with pytest.raises(ValueError, match="nonnegative"):
        ModeSpec(solver_padding_cells=value)


@pytest.mark.parametrize(
    "value", [True, 1.5, np.nan, [1, 2], (1,), (1, 2, 3), (0, False)]
)
def test_padding_requires_integer_counts(value):
    with pytest.raises(TypeError, match="solver_padding_cells"):
        ModeSpec(solver_padding_cells=value)


def test_padding_is_immutable_and_part_of_cache_identity():
    automatic = ModeSpec()
    zero = ModeSpec(solver_padding_cells=0)
    assert zero.solver_padding_cells == (0, 0)
    assert cache_token(automatic) != cache_token(zero)
    assert cache_token(zero) != cache_token(ModeSpec(solver_padding_cells=(0, 1)))
    assert ModeSpec(solver_padding_cells=3) == ModeSpec(solver_padding_cells=(3, 3))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("stretched", [False, True])
@pytest.mark.parametrize("padding", [(0, 0), (2, 4), (100, 100)])
def test_padding_counts_actual_cells_and_clips_at_boundaries(axis, stretched, padding):
    edges = np.linspace(0, 2e-6, 41)
    if stretched:
        edges = (edges / edges[-1]) ** 1.2 * edges[-1]
    grid = RectilinearGrid(edges, edges, edges)
    center = (0.7e-6,) * 3
    region = mode_source_region(
        center=center,
        width=0.5e-6,
        height=0.4e-6,
        axis=axis,
        direction_sign=1,
        is_3d=True,
        grid=grid,
    )
    plane = _local_mode_plane_spec(
        np.ones((40, 40)),
        axis=axis,
        grid_shape=(40, 40, 40),
        center=center,
        width=0.5e-6,
        height=0.4e-6,
        plane_index=region.plane_index,
        offset_index=region.companion_index,
        resolution=0.05e-6,
        snapped_region=region,
        grid=grid,
        aperture_pad_cells=padding,
    )
    for name, pad in zip([a for a in "xyz" if a != axis], padding, strict=True):
        interval = region.axis_interval(name)
        start, stop = max(0, interval.start - pad), min(40, interval.stop + pad)
        position = "zyx".index(name)
        assert plane["origin_zyx"][position] == start
        assert plane["grid_shape"][position] == stop - start
        np.testing.assert_allclose(
            plane["grid"].axis_edges(name), edges[start : stop + 1] - edges[start]
        )
        assert plane["axis_origins"][name] == edges[start]
