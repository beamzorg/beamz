"""Padding changes the material domain, independently of injection support."""

import numpy as np
import pytest

from beamz import ModeSpec
from beamz._cache_tokens import cache_token
from beamz.design.grid import RectilinearGrid
from beamz.devices._placement import snap_mode_source_region_grid
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
    default = ModeSpec()
    zero = ModeSpec(solver_padding_cells=0)
    assert zero.solver_padding_cells == (0, 0)
    assert cache_token(default) == cache_token(zero)
    assert cache_token(zero) != cache_token(ModeSpec(solver_padding_cells=(0, 1)))
    assert ModeSpec(solver_padding_cells=3) == ModeSpec(solver_padding_cells=(3, 3))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("stretched", [False, True])
@pytest.mark.parametrize("padding", [(0, 0), (2, 4), (100, 100)])
@pytest.mark.parametrize("presnapped", [True, False])
def test_padding_counts_actual_cells_and_clips_at_boundaries(
    axis, stretched, padding, presnapped
):
    edges = np.linspace(0, 2e-6, 41)
    if stretched:
        edges = (edges / edges[-1]) ** 1.2 * edges[-1]
    grid = RectilinearGrid(edges, edges, edges)
    center = (0.7e-6,) * 3
    region = snap_mode_source_region_grid(
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
        snapped_region=region if presnapped else None,
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


def test_no_taper_preserves_nondecaying_fields_at_aperture_edges():
    from beamz.devices.modes.discrete import ModePlaneSpec, _build_profiles

    fields = {
        prefix + axis: np.ones((12, 12), complex) for prefix in "EH" for axis in "xyz"
    }
    request = ModePlaneSpec(
        scalar_permittivity=np.ones((12, 12)),
        frequency=2e14,
        resolution=1.0,
        dt=None,
        axis="x",
        direction="+x",
        transverse_axes=("z", "y"),
        grid_shape=(12, 12, 4),
        center=(2, 6, 6),
        width=4,
        height=4,
        plane_index=1,
        offset_index=0,
        aperture_pad_cells=0,
    )
    profiles, _, _ = _build_profiles(fields, request)
    for values in profiles.values():
        assert np.all(np.abs(values) > 0), (
            "Preparation must not taper nonzero edge fields"
        )
        np.testing.assert_allclose(values, values.flat[0])


def test_edge_diagnostic_checks_raw_electric_fields():
    from beamz.devices.modes.discrete import _mode_edge_fraction

    fields = {"E" + axis: np.zeros((20, 20), complex) for axis in "xyz"}
    fields["Ey"][10, 10] = 1
    assert _mode_edge_fraction(fields) == 0
    fields["Ey"][0, 10] = 1
    assert _mode_edge_fraction(fields) == 0.5
    fields["Ey"][:] = 1
    assert _mode_edge_fraction(fields) == 0.2
