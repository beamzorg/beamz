"""Fixed-plan material binding for the 2D/3D topology optimization backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.lattice import canonical_component_2d, component_axis_offsets_3d
from beamz.simulation.kernels import precompute_e_update_coefficients

# Histories created before the material/source/projection and observation-clock
# corrections, physical-thickness CPML default, or frequency-aligned reference
# normalization, or polarization-aware mode candidate selection cannot resume this objective.
TOPOLOGY_NUMERICS_VERSION = 6


def _cell_centers_to_yee(values, component, polarization):
    """Interpolate cell centers onto Yee supports without a preferred side."""
    values = jnp.asarray(values)
    if values.ndim == 3:
        offsets = component_axis_offsets_3d(component)
        node_axes = tuple(i for i, axis in enumerate("zyx") if offsets[axis] == 0)
    else:
        canonical = canonical_component_2d(component, "xy", polarization)
        if canonical is None:
            return values[:1, :1]
        node_axes = {"Ex": (0,), "Ey": (1,), "Ez": (0, 1)}[canonical]
    for axis in node_axes:
        padding = [(0, 0)] * values.ndim
        padding[axis] = (1, 1)
        padded = jnp.pad(values, padding, mode="edge")
        lower, upper = [slice(None)] * values.ndim, [slice(None)] * values.ndim
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
        if program.sharding.layout.enabled:
            from beamz.simulation import sharding
            from beamz.simulation.distributed_differentiation import sample_material

            fraction = sample_material(program, selected, component)
            sampled = sample_material(program, selected * permittivity, component)
            fixed = sharding._pad_high_to_shape(
                getattr(program.grid, f"eps_{axis}"),
                program.sharding.layout.padded_shapes[component],
                pad_value=1,
            )
            values.append(sampled + (1.0 - fraction) * fixed)
            continue
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
    """Bind scalar design permittivity to an existing JAX execution plan."""
    updates = {}
    for axis, component, eps in zip(
        "xyz",
        ("Ex", "Ey", "Ez"),
        topology_yee_permittivity(program, permittivity, mask),
        strict=True,
    ):
        if program.config.is_3d:
            updates[f"e_permittivity_{axis}"] = eps
            continue
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
            f"eps_{axis}": np.asarray(value)[
                tuple(
                    slice(0, n)
                    for n in program.sharding.layout.logical_shapes[f"E{axis}"]
                )
            ]
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
        permittivity=np.asarray(permittivity)[
            tuple(slice(0, n) for n in original.shape)
        ],
        yee_materials=yee,
        smoothing="farjadpour_diagonal",
        tensors={},
        yee_tensors={},
    )
