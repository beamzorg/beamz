"""Mode apertures are centered on physical coordinates, not storage order."""

from __future__ import annotations

import numpy as np
import pytest

from beamz.devices.modes.discrete import ModePlaneSpec, _build_transverse_profiles

# Storage order is (z, y, x). The physical center below is deliberately
# different on every axis so a zyx index into ``center`` misses the guide.
_SHAPE = (48, 36, 24)
_CENTER = (3.0, 12.0, 30.0)
_EXTENT = 6.0
_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _expected_window(center: float, count: int) -> tuple[int, int]:
    edges = np.arange(count + 1, dtype=float)
    lower = center - 0.5 * _EXTENT
    upper = center + 0.5 * _EXTENT
    start = int(np.searchsorted(edges, lower, side="right") - 1)
    stop = int(np.searchsorted(edges, upper, side="left"))
    return start, stop


@pytest.mark.parametrize(
    ("axis", "component", "row_axis", "col_axis"),
    [
        ("x", "Ex", "z", "y"),
        ("y", "Ey", "z", "x"),
        ("z", "Ez", "y", "x"),
    ],
)
def test_transverse_aperture_uses_physical_xyz_center(
    axis, component, row_axis, col_axis
):
    counts = {"z": _SHAPE[0], "y": _SHAPE[1], "x": _SHAPE[2]}
    fields = {
        name: np.ones((counts[row_axis], counts[col_axis]), dtype=np.complex128)
        for name in _COMPONENTS
    }
    _profiles, indices, _extra = _build_transverse_profiles(
        fields,
        ModePlaneSpec(
            scalar_permittivity=np.ones((4, 4)),
            frequency=1.5e14,
            resolution=1.0,
            dt=None,
            axis=axis,
            direction=f"+{axis}",
            transverse_axes=(row_axis, col_axis),
            grid_shape=_SHAPE,
            center=_CENTER,
            width=_EXTENT,
            height=_EXTENT,
            plane_index=2,
            offset_index=2,
            aperture_pad_cells=0,
            aperture_window_alpha=0.0,
            yee_refinement=False,
        ),
    )
    placed = {"z": 0, "y": 1, "x": 2}
    centers = {"x": _CENTER[0], "y": _CENTER[1], "z": _CENTER[2]}
    row = indices[component][placed[row_axis]]
    col = indices[component][placed[col_axis]]
    assert isinstance(row, slice)
    assert isinstance(col, slice)
    assert (row.start, row.stop) == _expected_window(
        centers[row_axis], counts[row_axis]
    )
    assert (col.start, col.stop) == _expected_window(
        centers[col_axis], counts[col_axis]
    )
