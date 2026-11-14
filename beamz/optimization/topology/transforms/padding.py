"""Fold Yee padding into physical density gradients."""

from __future__ import annotations

from typing import Any

import numpy as np


def fold_high_side_yee_padding_to_shape(
    grad: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Fold high-side Yee padding planes back onto a material-grid shape.

    Some full-Yee field components include the high boundary sample explicitly,
    so their field-derived gradients can be one element larger than the
    cell-centered material grid. Material sampling clips those high-side samples
    to the last material cell; this helper applies the corresponding adjoint by
    accumulating the high-side plane into the last target index.
    """

    out = np.asarray(grad, dtype=float)
    target_shape = tuple(int(v) for v in target_shape)
    if out.shape == target_shape:
        return out
    if out.ndim != len(target_shape):
        raise ValueError(
            f"Gradient ndim {out.ndim} does not match target ndim {len(target_shape)}."
        )

    for axis, target in enumerate(target_shape):
        size = out.shape[axis]
        if size == target:
            continue
        if size != target + 1:
            raise ValueError(
                "Cannot align gradient shape "
                f"{out.shape} to target shape {target_shape}: axis {axis} has "
                f"size {size}, expected {target} or {target + 1}."
            )

        core: list[Any] = [slice(None)] * out.ndim
        core[axis] = slice(0, target)
        folded = out[tuple(core)].copy()

        last: list[Any] = [slice(None)] * out.ndim
        last[axis] = target - 1
        high: list[Any] = [slice(None)] * out.ndim
        high[axis] = target
        folded[tuple(last)] += out[tuple(high)]
        out = folded

    return out
