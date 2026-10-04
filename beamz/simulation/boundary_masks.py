"""Compact separable PEC masks for the JAX execution context."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from beamz._region_array import SeparableMask
from beamz.simulation.model import BoundaryPlan


@dataclass(frozen=True, eq=False)
class AxisMask:
    """Represent a Boolean mask as the union of broadcast axis profiles."""

    profiles: tuple[np.ndarray, ...]


def compact_mask(mask):
    """Use short profiles only when they reproduce every mask entry exactly."""
    if mask is None:
        return None
    if isinstance(mask, SeparableMask):
        profiles = []
        for axis, profile in enumerate(mask.profiles):
            if np.any(profile):
                shape = [1] * mask.ndim
                shape[axis] = profile.size
                profiles.append(profile.reshape(shape))
        return AxisMask(tuple(profiles)) if profiles else None
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return None
    profiles = tuple(
        np.all(
            values,
            axis=tuple(other for other in range(values.ndim) if other != axis),
            keepdims=True,
        )
        for axis in range(values.ndim)
    )
    rebuilt = np.zeros(values.shape, dtype=bool)
    for profile in profiles:
        rebuilt |= profile
    if not np.array_equal(rebuilt, values):
        return mask
    profiles = tuple(profile for profile in profiles if np.any(profile))
    for profile in profiles:
        profile.setflags(write=False)
    return AxisMask(profiles)


def compact_boundary_masks(boundary: BoundaryPlan) -> BoundaryPlan:
    """Keep canonical boundary arrays intact while reducing JIT constants."""
    metallic = boundary.metallic
    updates = {
        field.name: compact_mask(getattr(metallic, field.name))
        for field in fields(metallic)
    }
    return replace(boundary, metallic=replace(metallic, **updates))
