"""Opt-in mirrors on the solver's collocated, cell-center representation.

Never reflect a compact Yee array by index: its support need not be closed
about the requested physical center. Projection precedes Yee interpolation.
"""

from __future__ import annotations

import numpy as np


def mirror_closed(coordinates, center):
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return False
    spacing = np.diff(values)
    if np.any(spacing <= 0):
        return False
    tolerance = 1e-9 * np.min(spacing)
    return bool(np.allclose(values + values[::-1], 2 * center, rtol=0, atol=tolerance))


def material_symmetric(values, axis):
    values = np.asarray(values)
    # Pointwise, complex-valued comparison; a large uniform background cannot
    # hide an unmatched inclusion through an aggregate correlation score.
    return bool(
        np.all(np.isfinite(values))
        and np.allclose(values, np.flip(values, axis), rtol=1e-10, atol=1e-12)
    )


def project_mode_symmetry(
    fields, *, requested, edges, axes, center, materials, supports, rejected_axes=()
):
    """Return projected fields and an explicit processing-stage diagnostic.

    Only exact grid mirrors (to roundoff) are supported. Non-mirror-closed
    graded meshes are skipped, never interpolated or reversed approximately.
    All diagonal constitutive entries and all component supports must agree.
    A single vector parity is selected per mirror: E is polar and H axial.
    """
    applied = []
    skipped = {}
    out = dict(fields)
    if requested:
        for index, (axis, edge, midpoint) in enumerate(
            zip(axes, edges, center, strict=True)
        ):
            if axis in rejected_axes:
                skipped[axis] = (
                    "staggered material support or tensor is not mirror symmetric"
                )
                continue
            if not mirror_closed(edge, midpoint):
                skipped[axis] = (
                    "grid edges are not mirror closed about the requested center"
                )
                continue
            if any(not material_symmetric(value, index) for value in materials):
                skipped[axis] = "material tensors are not mirror symmetric"
                continue
            if any(
                name not in supports
                or len(supports[name]) != len(axes)
                or len(supports[name][index]) != value.shape[index]
                or not mirror_closed(supports[name][index], midpoint)
                for name, value in out.items()
            ):
                skipped[axis] = "component supports are not mirror closed"
                continue
            signs = {
                name: (-1 if name[1].lower() == axis else 1)
                * (-1 if name.startswith("H") else 1)
                for name in out
            }
            # Normalize each vector family in this overlap so SI impedance
            # units do not suppress H when choosing the common mirror parity.
            overlap = 0.0
            for family in ("E", "H"):
                norm = sum(
                    float(np.vdot(v, v).real) for n, v in out.items() if n[0] == family
                )
                if norm > 0:
                    overlap += (
                        sum(
                            float(np.vdot(v, signs[n] * np.flip(v, index)).real)
                            for n, v in out.items()
                            if n[0] == family
                        )
                        / norm
                    )
            parity = 1 if overlap >= 0 else -1
            out = {
                name: 0.5 * (value + parity * signs[name] * np.flip(value, index))
                for name, value in out.items()
            }
            applied.append(axis)
    return out, {
        "force_symmetry_requested": bool(requested),
        "force_symmetry_applied": bool(applied),
        "force_symmetry_axes": tuple(applied),
        "force_symmetry_skipped": skipped,
        "symmetry_stage": "collocated_solver_fields_before_yee_interpolation",
    }
