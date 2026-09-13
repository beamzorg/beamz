"""Fixed-plan material binding for the 2D topology optimization backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.lattice import canonical_component_2d
from beamz.simulation.kernels import precompute_e_update_coefficients

# Histories created before the symmetric material/source/projection corrections
# or integer-step observation clock cannot resume this numerical objective.
TOPOLOGY_NUMERICS_VERSION = 3


def _cell_centers_to_yee(values, component, polarization):
    """Interpolate cell centers onto Yee supports without a preferred side."""
    values = jnp.asarray(values)
    canonical = canonical_component_2d(component, "xy", polarization)
    if canonical is None:
        return values[:1, :1]
    node_axes = {"Ex": (0,), "Ey": (1,), "Ez": (0, 1)}[canonical]
    for axis in node_axes:
        padding = [(0, 0), (0, 0)]
        padding[axis] = (1, 1)
        padded = jnp.pad(values, padding, mode="edge")
        lower, upper = [slice(None)] * 2, [slice(None)] * 2
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        values = 0.5 * (padded[tuple(lower)] + padded[tuple(upper)])
    return values


def topology_yee_permittivity(program, permittivity, mask):
    """Paint design cells onto Yee supports, preserving fixed raster interfaces.

    Interpolate neighboring design cells at staggered locations. Blend partially
    covered supports with the fixed rasterizer's Yee values. This convex map
    preserves reflection symmetry and positive material bounds; JAX applies its
    exact transpose when returning gradients to design cells.
    """
    values = []
    density_dtype = jnp.asarray(permittivity).dtype
    selected = jnp.asarray(mask, dtype=density_dtype)
    for component, axis in zip(("Ex", "Ey", "Ez"), "xyz", strict=True):
        fraction = _cell_centers_to_yee(
            selected,
            component,
            program.config.polarization_2d,
        )
        sampled = _cell_centers_to_yee(
            selected * permittivity,
            component,
            program.config.polarization_2d,
        )
        values.append(sampled + (1.0 - fraction) * getattr(program.grid, f"eps_{axis}"))
    return tuple(values)


def topology_coefficients(program, permittivity, mask):
    """Bind scalar design permittivity to an existing 2D JAX execution plan."""
    updates = {}
    for axis, component, eps in zip(
        "xyz",
        ("Ex", "Ey", "Ez"),
        topology_yee_permittivity(program, permittivity, mask),
        strict=True,
    ):
        decay, source = precompute_e_update_coefficients(
            shape=program.grid.component_shapes[component],
            conductivity=getattr(program.grid, f"sig_{axis}"),
            permittivity=eps,
            dt=program.config.dt,
            region=getattr(program.grid, f"region_{axis}"),
        )
        updates[f"e_decay_{axis}"] = decay
        updates[f"e_source_{axis}"] = source
    return program.coefficients._replace(**updates)


def topology_material_grid(program, permittivity, mask):
    """Detach the same material mapping for an independent ordinary simulation."""
    from dataclasses import replace

    original = program.grid.material_grid
    yee = dict(original.yee_materials)
    yee.update(
        {
            f"eps_{axis}": np.asarray(value)
            for axis, value in zip(
                "xyz",
                topology_yee_permittivity(program, permittivity, mask),
                strict=True,
            )
        }
    )
    # Cell tensors are metadata for the old geometry. This route is scalar only;
    # keep all electric metadata consistent with the new design cells.
    return replace(
        original,
        permittivity=np.asarray(permittivity),
        yee_materials=yee,
        smoothing="farjadpour_diagonal",
        tensors={},
        yee_tensors={},
    )
