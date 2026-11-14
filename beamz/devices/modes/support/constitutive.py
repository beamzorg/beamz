"""Constitutive support shared by mode assembly and deferred analysis."""

from __future__ import annotations

import numpy as np


def require_uniform_normal_spacing(grid, axis):
    """A Fourier port needs a translation-invariant local longitudinal stencil."""
    if grid is None:
        return
    widths = np.asarray(grid.cell_widths(axis))
    if not np.allclose(widths, widths[0], rtol=1e-10, atol=0):
        raise ValueError(
            "Full-tensor ports require uniform normal-axis spacing over their constitutive neighborhood"
        )


def crop_tensor_support(tensors, *, origin, shape):
    """Crop packed tensors, retaining the upper node on every support axis.

    ``origin`` and ``shape`` describe cells in the supplied material region.
    Callers must retain the complete neighborhood; negative indices must never
    silently wrap to the opposite side of a stored monitor region.
    """
    result = {}
    for name, values in (tensors or {}).items():
        array = np.asarray(values)
        if array.ndim != len(shape) + 1 or array.shape[0] not in (1, 3, 6):
            raise ValueError(
                f"Invalid packed constitutive tensor {name}: {array.shape}"
            )
        if any(int(start) < 0 for start in origin):
            raise ValueError(
                "Modal material context does not contain the required halo"
            )
        if name == "eps_node":
            counts = tuple(int(n) + 1 for n in shape)
        elif name in {"eps_x", "eps_y", "eps_z"}:
            from beamz.lattice import component_axis_offsets_3d

            axes = ("z", "y", "x")[-len(shape) :]
            offsets = component_axis_offsets_3d("E" + name[-1])
            counts = tuple(
                int(n) + int(offsets[a] == 0) for a, n in zip(axes, shape, strict=True)
            )
        else:
            counts = tuple(int(n) for n in shape)
        if any(
            int(start) + count > size
            for start, count, size in zip(origin, counts, array.shape[1:], strict=True)
        ):
            raise ValueError(
                "Modal material context does not contain the required support"
            )
        selection = tuple(
            slice(int(start), int(start) + count)
            for start, count in zip(origin, counts, strict=True)
        )
        result[name] = array[(slice(None), *selection)]
    return result


def has_cross_terms(tensors):
    return any(
        np.asarray(values).shape[0] == 6 and np.any(np.asarray(values)[3:] != 0)
        for values in (tensors or {}).values()
    )


def require_diagonal_support(tensors, *, operation):
    if has_cross_terms(tensors):
        raise ValueError(
            f"{operation} cannot yet reproduce full off-diagonal permittivity "
            "on the port's constitutive support (including neighboring Yee nodes). "
            "Use a locally diagonal port or farjadpour_diagonal."
        )
