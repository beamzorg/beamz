"""Canonical Yee update kernels and static kernel selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.constitutive import advance_e_tensor_dense
from beamz.constitutive import collocate_yee_component as collocate_yee_component
from beamz.constitutive import fit_array_to_shape as fit_array_to_shape
from beamz.lattice import (
    adjacent_difference as _adjacent_difference,
)
from beamz.lattice import build_h_boundary_views_for_e_3d, component_axis_offsets_3d
from beamz.simulation.boundary_masks import AxisMask
from beamz.simulation.material_blocks import material_values, reciprocal_material
from beamz.simulation.model import (
    BoundaryPlan,
    CpmlPackedSlabSpec,
    CpmlTerm,
    RunConfig,
    SimulationState,
    UpdateCoefficients,
)
from beamz.simulation.tensor_regions import TensorRegion

# Logical shapes describe physics while storage shapes may contain device padding. Every
# phase must preserve this distinction so padded cells never reach results or monitors.


def advance_e_centered_tensor(
    fields,
    curls,
    inverse_diagonals,
    inverse_offdiagonal,
    components,
    dt,
    *,
    cell_widths=None,
):
    """Advance E with diagonal terms on E sites and cross terms on dual centers."""

    inverse_diagonals = tuple(material_values(v) for v in inverse_diagonals)
    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    if isinstance(inverse_offdiagonal, TensorRegion):
        region = inverse_offdiagonal
        axes = ("z", "y", "x")[-fields[0].ndim :]
        bounds = tuple(
            (max(0, start - 1), min(count, stop + 1))
            for (start, stop), count in zip(
                region.bounds, region.node_shape, strict=True
            )
        )
        values = jnp.pad(
            region.values,
            tuple(
                (start - outer_start, outer_stop - stop)
                for (start, stop), (outer_start, outer_stop) in zip(
                    region.bounds, bounds, strict=True
                )
            )
            + ((0, 0),),
        )
        slices = tuple(
            tuple(
                slice(
                    start, stop - int(component_axis_offsets_3d(component)[axis] == 0.5)
                )
                for axis, (start, stop) in zip(axes, bounds, strict=True)
            )
            for component in components
        )
        widths = cell_widths
        if widths is not None:
            widths = list(widths)
            for axis, (start, stop) in zip(axes, bounds, strict=True):
                index = "xyz".index(axis)
                if jnp.ndim(widths[index]) and jnp.size(widths[index]):
                    widths[index] = widths[index][start : stop - 1]
            widths = tuple(widths)
        local_curls = tuple(
            curl[part] for curl, part in zip(curls, slices, strict=True)
        )
        local = _tensor_cross_curls(local_curls, values, components, widths)
        return tuple(
            field
            + scale
            * (
                diagonal * curl
                + jnp.pad(
                    value,
                    tuple(
                        (part.start, size - part.stop)
                        for part, size in zip(parts, field.shape, strict=True)
                    ),
                )
            )
            for field, diagonal, curl, parts, value in zip(
                fields, inverse_diagonals, curls, slices, local, strict=True
            )
        )
    return advance_e_tensor_dense(
        fields,
        curls,
        inverse_diagonals,
        inverse_offdiagonal,
        components,
        dt,
        cell_widths=cell_widths,
    )


def _tensor_cross_curls(curls, inverse_offdiagonal, components, cell_widths):
    """Sum cross terms at their shared support before interpolating once per E."""
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths
        )
        for curl, source in zip(curls, components, strict=True)
    )
    cross_component = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    updated = []
    for row, (curl, target) in enumerate(zip(curls, components, strict=True)):
        centered = sum(
            inverse_offdiagonal[..., cross_component[(min(row, col), max(row, col))]]
            * value
            for col, value in enumerate(centered_curls)
            if col != row
        )
        updated.append(
            collocate_yee_component(
                centered, "Node", target, curl.shape, cell_widths=cell_widths
            )
        )
    return tuple(updated)


def constitutive_cell_widths(metrics):
    """Recover separable primal widths from the compiled curl metrics."""
    return tuple(jnp.reciprocal(getattr(metrics, "e_to_h_" + axis)) for axis in "xyz")


def cpml_coefficients(sigma, kappa, alpha, dt):
    """Precompute one CPML recurrence as ``a``, ``b``, and inverse-kappa."""
    dtype = jnp.result_type(sigma, kappa, alpha, jnp.float32)
    sigma = jnp.asarray(sigma, dtype=dtype)
    kappa = jnp.maximum(jnp.asarray(kappa, dtype=dtype), 1.0)
    alpha = jnp.asarray(alpha, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    decay = (sigma / kappa + alpha) * (
        jnp.asarray(dt, dtype=dtype) / jnp.asarray(EPS_0, dtype=dtype)
    )
    b = jnp.exp(-decay)
    denominator = (sigma + kappa * alpha) * kappa
    # Preserve weak damping when exp(-decay) rounds to one; vacuum has a=0.
    a = jnp.expm1(-decay) * (sigma / jnp.where(denominator > 0, denominator, one))
    return a, b, one / kappa


def advance_h_from_coefficients(field, curl, decay, source):
    return decay * field - source * curl


def advance_e_from_coefficients(field, curl, decay, source):
    return decay * field + source * curl


def apply_zero_mask(field, mask):
    """Apply a compiled PEC mask without changing the field representation."""
    if isinstance(mask, AxisMask):
        combined = jnp.asarray(False)
        for profile in mask.profiles:
            combined = combined | jnp.asarray(profile)
        return jnp.where(combined, 0.0, field)
    return field if mask is None else jnp.where(mask, 0.0, field)


def _scalar_like(value, dtype):
    return jnp.asarray(value, dtype=dtype)


def _scale_by_axis_metric(values, inverse_distance, axis):
    """Multiply one derivative by a scalar or separable 1-D inverse distance."""
    metric = jnp.asarray(inverse_distance, dtype=values.dtype)
    if metric.ndim == 0:
        return values * metric
    shape = [1] * values.ndim
    shape[int(axis)] = metric.shape[0]
    return values * jnp.reshape(metric, shape)


def _metric_difference(values, axis, inverse_distance):
    return _scale_by_axis_metric(jnp.diff(values, axis=axis), inverse_distance, axis)


def _axis_region(ndim, axis, start, stop):
    region = [slice(None)] * ndim
    region[axis] = slice(start, stop)
    return tuple(region)


def _pack_cpml_slab(arr, slab):
    """Gather the active low/high edge regions of one derivative."""
    axis, low, high = int(slab.axis), int(slab.low), int(slab.high)
    if low <= 0 and high <= 0:
        return jnp.zeros(slab.shape, dtype=arr.dtype)
    parts = []
    if low:
        parts.append(arr[_axis_region(arr.ndim, axis, 0, low)])
    logical_stop = (
        int(arr.shape[axis]) if int(slab.logical_stop) < 0 else int(slab.logical_stop)
    )
    if high:
        parts.append(
            arr[_axis_region(arr.ndim, axis, logical_stop - high, logical_stop)]
        )
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=axis)


def _unpack_cpml_slab(base, packed, slab):
    """Rebuild a derivative with its packed low/high CPML corrections."""
    axis, low, high = int(slab.axis), int(slab.low), int(slab.high)
    if not low and not high:
        return base
    logical_stop = (
        int(base.shape[axis]) if int(slab.logical_stop) < 0 else int(slab.logical_stop)
    )
    parts = []
    if low:
        parts.append(packed[_axis_region(packed.ndim, axis, 0, low)])
    middle_stop = logical_stop - high
    if middle_stop > low:
        parts.append(base[_axis_region(base.ndim, axis, low, middle_stop)])
    if high:
        parts.append(packed[_axis_region(packed.ndim, axis, low, low + high)])
    if logical_stop < int(base.shape[axis]):
        parts.append(
            base[_axis_region(base.ndim, axis, logical_stop, int(base.shape[axis]))]
        )
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=axis)


def _active_slab_counts(a, inv_kappa, axis):
    active = (np.abs(np.asarray(a)) > 1e-30) | (
        np.abs(np.asarray(inv_kappa) - 1.0) > 1e-7
    )
    transverse = tuple(index for index in range(active.ndim) if index != axis)
    active = np.any(active, axis=transverse) if transverse else active
    low = next((i for i, value in enumerate(active) if not value), active.size)
    tail = active[low:][::-1]
    high = next((i for i, value in enumerate(tail) if not value), tail.size)
    return int(low), int(high)


def compile_cpml_term(*, component, axis, sign, sigma, kappa, alpha, dt, full_shape):
    """Precompute and pack one dimension-independent CPML recurrence."""
    a, b, inv_kappa = cpml_coefficients(sigma, kappa, alpha, dt)
    low, high = _active_slab_counts(a, inv_kappa, int(axis))
    shape = list(full_shape)
    shape[int(axis)] = low + high
    slab = CpmlPackedSlabSpec(
        int(axis), low, high, tuple(shape), int(full_shape[int(axis)])
    )
    return CpmlTerm(
        str(component),
        int(axis),
        float(sign),
        _pack_cpml_slab(a, slab),
        _pack_cpml_slab(b, slab),
        _pack_cpml_slab(inv_kappa, slab),
        slab,
    )


def correct_cpml_term(derivative, psi, term):
    """Apply CPML in field precision, rounding only the stored auxiliary state."""
    if not term.slab.low and not term.slab.high:
        return term.sign * derivative, psi
    arithmetic_dtype = jnp.result_type(derivative.dtype, psi.dtype, jnp.float32)
    derivative_slab = _pack_cpml_slab(derivative.astype(arithmetic_dtype), term.slab)
    next_psi = (
        term.b.astype(arithmetic_dtype) * psi.astype(arithmetic_dtype)
        + term.a.astype(arithmetic_dtype) * derivative_slab
    )
    corrected = derivative_slab * term.inv_kappa.astype(arithmetic_dtype) + next_psi
    return (
        term.sign * _unpack_cpml_slab(derivative, corrected, term.slab),
        next_psi.astype(psi.dtype),
    )


def fused_update_h_lossy_3d_material(
    ex, ey, ez, hx, hy, hz, sigma_x, sigma_y, sigma_z, dt, resolution
):
    """Advance all magnetic components directly from collocated loss grids."""
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    inv_res = one / jnp.asarray(resolution, dtype=hx.dtype)
    out = []
    differences = (
        (ez[:, 1:, :] - ez[:, :-1, :], ey[1:, :, :] - ey[:-1, :, :]),
        (ex[1:, :, :] - ex[:-1, :, :], ez[:, :, 1:] - ez[:, :, :-1]),
        (ey[:, :, 1:] - ey[:, :, :-1], ex[:, 1:, :] - ex[:, :-1, :]),
    )
    for field, sigma, terms in zip(
        (hx, hy, hz), (sigma_x, sigma_y, sigma_z), differences, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        denom = one + alpha
        curl = (
            fit_array_to_shape(terms[0], field.shape)
            - fit_array_to_shape(terms[1], field.shape)
        ) * inv_res
        out.append((one - alpha) / denom * field - (dt_over_mu / denom) * curl)
    return tuple(out)


def fused_update_h_lossy_3d_material_metric(
    ex, ey, ez, hx, hy, hz, sigma_x, sigma_y, sigma_z, dt, metrics
):
    """Advance 3D H fields using statically planned separable grid metrics."""
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    differences = (
        (
            _metric_difference(ez, 1, metrics.e_to_h_y),
            _metric_difference(ey, 0, metrics.e_to_h_z),
        ),
        (
            _metric_difference(ex, 0, metrics.e_to_h_z),
            _metric_difference(ez, 2, metrics.e_to_h_x),
        ),
        (
            _metric_difference(ey, 2, metrics.e_to_h_x),
            _metric_difference(ex, 1, metrics.e_to_h_y),
        ),
    )
    out = []
    for field, sigma, terms in zip(
        (hx, hy, hz), (sigma_x, sigma_y, sigma_z), differences, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        denom = one + alpha
        curl = fit_array_to_shape(terms[0], field.shape) - fit_array_to_shape(
            terms[1], field.shape
        )
        out.append((one - alpha) / denom * field - (dt_over_mu / denom) * curl)
    return tuple(out)


def _advance_e_diagonal(fields, conductivities, inverse_permittivities, curls, dt):
    """Apply the same lossy constitutive solve to uniform and metric curls."""
    one = jnp.asarray(1.0, dtype=fields[0].dtype)
    dt_over_eps = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    out = []
    for field, conductivity, inv_permittivity, curl in zip(
        fields, conductivities, inverse_permittivities, curls, strict=True
    ):
        inv_permittivity = material_values(inv_permittivity)
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        denom = one + beta
        out.append(
            (one - beta) / denom * field
            + (dt_over_eps * inv_permittivity / denom) * curl
        )
    return tuple(out)


def fused_update_e_lossy_3d_material(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    conductivity_x,
    inv_permittivity_x,
    conductivity_y,
    inv_permittivity_y,
    conductivity_z,
    inv_permittivity_z,
    dt,
    resolution,
    *,
    boundary_views,
    inverse_diagonals=None,
    inverse_offdiagonal=None,
):
    """Advance all electric components directly from collocated material grids."""
    del hx, hy, hz
    derivative_pairs = (
        (("hz_y", 1), ("hy_z", 0)),
        (("hx_z", 0), ("hz_x", 2)),
        (("hy_x", 2), ("hx_y", 1)),
    )
    curls = []
    for field, pair in zip((ex, ey, ez), derivative_pairs, strict=True):
        curl = fit_array_to_shape(
            _adjacent_difference(boundary_views[pair[0][0]], pair[0][1], resolution),
            field.shape,
        ) - fit_array_to_shape(
            _adjacent_difference(boundary_views[pair[1][0]], pair[1][1], resolution),
            field.shape,
        )
        curls.append(curl)
    if inverse_offdiagonal is not None:
        return advance_e_centered_tensor(
            (ex, ey, ez),
            curls,
            inverse_diagonals,
            inverse_offdiagonal,
            ("Ex", "Ey", "Ez"),
            dt,
        )
    return _advance_e_diagonal(
        (ex, ey, ez),
        (conductivity_x, conductivity_y, conductivity_z),
        (inv_permittivity_x, inv_permittivity_y, inv_permittivity_z),
        curls,
        dt,
    )


def fused_update_e_lossy_3d_material_metric(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    conductivity_x,
    inv_permittivity_x,
    conductivity_y,
    inv_permittivity_y,
    conductivity_z,
    inv_permittivity_z,
    dt,
    metrics,
    *,
    boundary_views,
    inverse_diagonals=None,
    inverse_offdiagonal=None,
):
    """Advance 3D E fields using staggered center-to-edge distances."""
    del hx, hy, hz
    derivative_pairs = (
        (("hz_y", 1, metrics.h_to_e_y), ("hy_z", 0, metrics.h_to_e_z)),
        (("hx_z", 0, metrics.h_to_e_z), ("hz_x", 2, metrics.h_to_e_x)),
        (("hy_x", 2, metrics.h_to_e_x), ("hx_y", 1, metrics.h_to_e_y)),
    )
    curls = []
    for field, pair in zip((ex, ey, ez), derivative_pairs, strict=True):
        curls.append(
            fit_array_to_shape(
                _metric_difference(boundary_views[pair[0][0]], pair[0][1], pair[0][2]),
                field.shape,
            )
            - fit_array_to_shape(
                _metric_difference(boundary_views[pair[1][0]], pair[1][1], pair[1][2]),
                field.shape,
            )
        )
    if inverse_offdiagonal is not None:
        return advance_e_centered_tensor(
            (ex, ey, ez),
            curls,
            inverse_diagonals,
            inverse_offdiagonal,
            ("Ex", "Ey", "Ez"),
            dt,
            cell_widths=constitutive_cell_widths(metrics),
        )
    return _advance_e_diagonal(
        (ex, ey, ez),
        (conductivity_x, conductivity_y, conductivity_z),
        (inv_permittivity_x, inv_permittivity_y, inv_permittivity_z),
        curls,
        dt,
    )


def cpml_update_h_from_e_3d(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    resolution,
    *,
    terms,
    psi_terms,
    dt,
    magnetic_conductivities,
    source_scales=None,
):
    """Advance 3D H fields and their six packed CPML memories."""
    derivatives = (
        (ez, 1, hx.shape),
        (ey, 0, hx.shape),
        (ex, 0, hy.shape),
        (ez, 2, hy.shape),
        (ey, 2, hz.shape),
        (ex, 1, hz.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(jnp.diff(field, axis=axis) / resolution, shape),
                psi,
                term,
            )
            for (field, axis, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    if source_scales is not None:
        updated = tuple(
            field - scale * (curl + sigma * field)
            for field, curl, sigma, scale in zip(
                (hx, hy, hz), curls, magnetic_conductivities, source_scales, strict=True
            )
        )
        return *updated, tuple(next_psi)
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    updated = []
    for field, curl, sigma in zip(
        (hx, hy, hz), curls, magnetic_conductivities, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        updated.append(((one - alpha) * field - dt_over_mu * curl) / (one + alpha))
    return *updated, tuple(next_psi)


def cpml_update_h_from_e_3d_metric(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    metrics,
    *,
    terms,
    psi_terms,
    dt,
    magnetic_conductivities,
):
    """Advance 3D H/CPML using physical cell widths for every derivative."""
    derivatives = (
        (ez, 1, metrics.e_to_h_y, hx.shape),
        (ey, 0, metrics.e_to_h_z, hx.shape),
        (ex, 0, metrics.e_to_h_z, hy.shape),
        (ez, 2, metrics.e_to_h_x, hy.shape),
        (ey, 2, metrics.e_to_h_x, hz.shape),
        (ex, 1, metrics.e_to_h_y, hz.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(_metric_difference(field, axis, metric), shape),
                psi,
                term,
            )
            for (field, axis, metric, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    one = jnp.asarray(1.0, dtype=hx.dtype)
    dt_over_mu = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    updated = []
    for field, curl, sigma in zip(
        (hx, hy, hz), curls, magnetic_conductivities, strict=True
    ):
        alpha = sigma * (0.5 * dt_over_mu)
        updated.append(((one - alpha) * field - dt_over_mu * curl) / (one + alpha))
    return *updated, tuple(next_psi)


def cpml_update_e_from_h_3d(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    resolution,
    *,
    terms,
    psi_terms,
    metallic_edges,
    dt,
    conductivities,
    inverse_permittivities,
    inverse_diagonals=None,
    inverse_offdiagonal=None,
    logical_shapes=None,
    periodic_axes=frozenset(),
    periodic_phases=(),
    source_scales=None,
):
    """Advance 3D E fields and their six packed CPML memories."""
    views = build_h_boundary_views_for_e_3d(
        hx,
        hy,
        hz,
        metallic_edges,
        logical_shapes=logical_shapes,
        periodic_axes=periodic_axes,
        periodic_phases=periodic_phases,
    )
    derivatives = (
        ("hz_y", 1, ex.shape),
        ("hy_z", 0, ex.shape),
        ("hx_z", 0, ey.shape),
        ("hz_x", 2, ey.shape),
        ("hy_x", 2, ez.shape),
        ("hx_y", 1, ez.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(
                    _adjacent_difference(views[name], axis, resolution), shape
                ),
                psi,
                term,
            )
            for (name, axis, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    if inverse_offdiagonal is not None:
        updated = advance_e_centered_tensor(
            (ex, ey, ez),
            curls,
            inverse_diagonals,
            inverse_offdiagonal,
            ("Ex", "Ey", "Ez"),
            dt,
        )
        return *updated, tuple(next_psi)
    if source_scales is not None:
        updated = tuple(
            field + material_values(scale) * (curl - sigma * field)
            for field, curl, sigma, scale in zip(
                (ex, ey, ez), curls, conductivities, source_scales, strict=True
            )
        )
        return *updated, tuple(next_psi)
    one = jnp.asarray(1.0, dtype=ex.dtype)
    dt_over_eps = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    updated = []
    for field, curl, conductivity, inv_permittivity in zip(
        (ex, ey, ez),
        curls,
        conductivities,
        inverse_permittivities,
        strict=True,
    ):
        inv_permittivity = material_values(inv_permittivity)
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        updated.append(
            ((one - beta) * field + dt_over_eps * inv_permittivity * curl)
            / (one + beta)
        )
    return *updated, tuple(next_psi)


def cpml_update_e_from_h_3d_metric(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    metrics,
    *,
    terms,
    psi_terms,
    metallic_edges,
    dt,
    conductivities,
    inverse_permittivities,
    inverse_diagonals=None,
    inverse_offdiagonal=None,
    logical_shapes=None,
    periodic_axes=frozenset(),
    periodic_phases=(),
):
    """Advance 3D E/CPML using physical center-to-center distances."""
    views = build_h_boundary_views_for_e_3d(
        hx,
        hy,
        hz,
        metallic_edges,
        logical_shapes=logical_shapes,
        periodic_axes=periodic_axes,
        periodic_phases=periodic_phases,
    )
    derivatives = (
        ("hz_y", 1, metrics.h_to_e_y, ex.shape),
        ("hy_z", 0, metrics.h_to_e_z, ex.shape),
        ("hx_z", 0, metrics.h_to_e_z, ey.shape),
        ("hz_x", 2, metrics.h_to_e_x, ey.shape),
        ("hy_x", 2, metrics.h_to_e_x, ez.shape),
        ("hx_y", 1, metrics.h_to_e_y, ez.shape),
    )
    corrected, next_psi = zip(
        *(
            correct_cpml_term(
                fit_array_to_shape(
                    _metric_difference(views[name], axis, metric), shape
                ),
                psi,
                term,
            )
            for (name, axis, metric, shape), psi, term in zip(
                derivatives, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    if inverse_offdiagonal is not None:
        updated = advance_e_centered_tensor(
            (ex, ey, ez),
            curls,
            inverse_diagonals,
            inverse_offdiagonal,
            ("Ex", "Ey", "Ez"),
            dt,
            cell_widths=constitutive_cell_widths(metrics),
        )
        return *updated, tuple(next_psi)
    one = jnp.asarray(1.0, dtype=ex.dtype)
    dt_over_eps = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    updated = []
    for field, curl, conductivity, inv_permittivity in zip(
        (ex, ey, ez),
        curls,
        conductivities,
        inverse_permittivities,
        strict=True,
    ):
        inv_permittivity = material_values(inv_permittivity)
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        updated.append(
            ((one - beta) * field + dt_over_eps * inv_permittivity * curl)
            / (one + beta)
        )
    return *updated, tuple(next_psi)


def precompute_h_update_coefficients(sigma_m, dt):
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)
    decay = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom
    source = (dt / MU_0) / denom
    return decay.astype(jnp.float32), source.astype(jnp.float32)


def precompute_e_update_coefficients(shape, conductivity, permittivity, dt, region):
    denom = 1.0 + conductivity * dt / (2.0 * EPS_0 * permittivity)
    decay = (
        jnp.ones(shape, dtype=jnp.float32)
        .at[region]
        .set(
            ((1.0 - conductivity * dt / (2.0 * EPS_0 * permittivity)) / denom).astype(
                jnp.float32
            )
        )
    )
    source_value = (dt / (EPS_0 * permittivity)) / denom
    source = (
        jnp.zeros(shape, dtype=jnp.float32)
        .at[region]
        .set(source_value.astype(jnp.float32))
    )
    return decay, source


def tm_xy_curl_e_to_h_2d(ez, resolution, hx_shape, hy_shape, metallic_edges):
    """Differentiate Ez onto the native Hx and Hy supports."""
    del hx_shape, hy_shape, metallic_edges
    resolution = _scalar_like(resolution, ez.dtype)
    return (
        (ez[1:, :] - ez[:-1, :]) / resolution,
        -(ez[:, 1:] - ez[:, :-1]) / resolution,
    )


def _pad_h_axis(field, axis, metallic_edges, periodic_axes, periodic_phases):
    """Extend one H support with its periodic or metallic boundary samples."""
    low = field[_axis_region(field.ndim, axis, 0, 1)]
    high = field[_axis_region(field.ndim, axis, -1, None)]
    if axis in periodic_axes:
        phase = periodic_phases[axis] if periodic_phases else 1.0
        low, high = high / phase, low * phase
    low_edge, high_edge = (("bottom", "top"), ("left", "right"))[axis]
    if low_edge in metallic_edges:
        low = jnp.zeros_like(low)
    if high_edge in metallic_edges:
        high = jnp.zeros_like(high)
    return jnp.concatenate((low, field, high), axis=axis)


def _tm_xy_h_derivatives(
    hx, hy, resolution, metallic_edges, periodic_axes=frozenset(), periodic_phases=()
):
    """Return padded H derivatives on the complete Ez node lattice."""
    resolution = _scalar_like(resolution, hy.dtype)
    padded_hy = _pad_h_axis(hy, 1, metallic_edges, periodic_axes, periodic_phases)
    padded_hx = _pad_h_axis(hx, 0, metallic_edges, periodic_axes, periodic_phases)
    d_hy_dx = (padded_hy[:, 1:] - padded_hy[:, :-1]) / resolution
    d_hx_dy = (padded_hx[1:, :] - padded_hx[:-1, :]) / resolution
    return d_hy_dx, d_hx_dy


def tm_xy_curl_h_to_e_2d(
    hx,
    hy,
    resolution,
    ez_shape,
    metallic_edges=frozenset(),
    periodic_axes=frozenset(),
    periodic_phases=(),
):
    """Differentiate Hx and Hy onto the native Ez support."""
    d_hy_dx, d_hx_dy = _tm_xy_h_derivatives(
        hx, hy, resolution, metallic_edges, periodic_axes, periodic_phases
    )
    curl = d_hy_dx - d_hx_dy
    if curl.shape != ez_shape:
        raise ValueError(f"curl(H) shape {curl.shape} does not match Ez {ez_shape}")
    return curl


def tm_xy_cpml_curl_e_to_h_2d(
    ez,
    resolution,
    *,
    terms,
    psi_h_terms,
):
    """Correct the two Ez derivatives with canonical 2D CPML memory."""
    resolution = _scalar_like(resolution, ez.dtype)
    derivatives = (
        (ez[1:] - ez[:-1]) / resolution,
        (ez[:, 1:] - ez[:, :-1]) / resolution,
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0], corrected[1][0], tuple(item[1] for item in corrected)


def tm_xy_cpml_curl_h_to_e_2d(
    hx,
    hy,
    resolution,
    ez_shape,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
    *,
    terms,
    psi_e_terms,
):
    """Correct the two H derivatives with canonical 2D CPML memory."""
    derivatives = _tm_xy_h_derivatives(
        hx, hy, resolution, metallic_edges, periodic_axes, periodic_phases
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    curl = corrected[0][0] + corrected[1][0]
    if curl.shape != ez_shape:
        raise ValueError(
            f"CPML curl(H) shape {curl.shape} does not match Ez {ez_shape}"
        )
    return curl.astype(hx.dtype), tuple(item[1] for item in corrected)


def te_xy_curl_e_to_h_2d(ex, ey, resolution, hz_shape):
    """Differentiate in-plane E onto the canonical Hz support."""
    resolution = _scalar_like(resolution, ex.dtype)
    curl = (ey[:, 1:] - ey[:, :-1]) / resolution - (ex[1:, :] - ex[:-1, :]) / resolution
    if curl.shape != hz_shape:
        raise ValueError(f"curl(E) shape {curl.shape} does not match Hz {hz_shape}")
    return curl


def _te_xy_h_derivatives(
    hz, resolution, metallic_edges, periodic_axes=frozenset(), periodic_phases=()
):
    """Return padded Hz derivatives on the complete Ex and Ey supports."""
    resolution = _scalar_like(resolution, hz.dtype)
    padded_y = _pad_h_axis(hz, 0, metallic_edges, periodic_axes, periodic_phases)
    padded_x = _pad_h_axis(hz, 1, metallic_edges, periodic_axes, periodic_phases)
    return (
        (padded_y[1:, :] - padded_y[:-1, :]) / resolution,
        (padded_x[:, 1:] - padded_x[:, :-1]) / resolution,
    )


def te_xy_curl_h_to_e_2d(
    hz,
    resolution,
    ex_shape,
    ey_shape,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
):
    """Differentiate Hz onto the canonical Ex and Ey supports."""
    d_hz_dy, d_hz_dx = _te_xy_h_derivatives(
        hz, resolution, metallic_edges, periodic_axes, periodic_phases
    )
    curls = d_hz_dy, -d_hz_dx
    if curls[0].shape != ex_shape or curls[1].shape != ey_shape:
        raise ValueError(
            f"curl(H) shapes {(curls[0].shape, curls[1].shape)} do not match "
            f"Ex/Ey {(ex_shape, ey_shape)}"
        )
    return curls


def te_xy_cpml_curl_e_to_h_2d(ex, ey, resolution, *, terms, psi_h_terms):
    """Correct the two in-plane E derivatives used by the Hz update."""
    resolution = _scalar_like(resolution, ex.dtype)
    derivatives = (
        (ey[:, 1:] - ey[:, :-1]) / resolution,
        (ex[1:, :] - ex[:-1, :]) / resolution,
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0] + corrected[1][0], tuple(item[1] for item in corrected)


def te_xy_cpml_curl_h_to_e_2d(
    hz,
    resolution,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
    *,
    terms,
    psi_e_terms,
):
    """Correct the Hz derivatives used by the Ex and Ey updates."""
    derivatives = _te_xy_h_derivatives(
        hz, resolution, metallic_edges, periodic_axes, periodic_phases
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    return (corrected[0][0], corrected[1][0]), tuple(item[1] for item in corrected)


def tm_xy_curl_e_to_h_2d_metric(ez, metrics):
    """Differentiate Ez onto H using physical x/y cell widths."""
    return (
        _metric_difference(ez, 0, metrics.e_to_h_y),
        -_metric_difference(ez, 1, metrics.e_to_h_x),
    )


def _tm_xy_h_derivatives_metric(
    hx, hy, metrics, metallic_edges, periodic_axes=frozenset(), periodic_phases=()
):
    padded_hy = _pad_h_axis(hy, 1, metallic_edges, periodic_axes, periodic_phases)
    padded_hx = _pad_h_axis(hx, 0, metallic_edges, periodic_axes, periodic_phases)
    return (
        _metric_difference(padded_hy, 1, metrics.h_to_e_x),
        _metric_difference(padded_hx, 0, metrics.h_to_e_y),
    )


def tm_xy_curl_h_to_e_2d_metric(
    hx,
    hy,
    metrics,
    ez_shape,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
):
    """Differentiate H onto Ez using physical center-to-center distances."""
    d_hy_dx, d_hx_dy = _tm_xy_h_derivatives_metric(
        hx, hy, metrics, metallic_edges, periodic_axes, periodic_phases
    )
    curl = d_hy_dx - d_hx_dy
    if curl.shape != ez_shape:
        raise ValueError(f"curl(H) shape {curl.shape} does not match Ez {ez_shape}")
    return curl


def tm_xy_cpml_curl_e_to_h_2d_metric(ez, metrics, *, terms, psi_h_terms):
    derivatives = (
        _metric_difference(ez, 0, metrics.e_to_h_y),
        _metric_difference(ez, 1, metrics.e_to_h_x),
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0], corrected[1][0], tuple(item[1] for item in corrected)


def tm_xy_cpml_curl_h_to_e_2d_metric(
    hx,
    hy,
    metrics,
    ez_shape,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
    *,
    terms,
    psi_e_terms,
):
    derivatives = _tm_xy_h_derivatives_metric(
        hx, hy, metrics, metallic_edges, periodic_axes, periodic_phases
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    curl = corrected[0][0] + corrected[1][0]
    if curl.shape != ez_shape:
        raise ValueError(
            f"CPML curl(H) shape {curl.shape} does not match Ez {ez_shape}"
        )
    return curl.astype(hx.dtype), tuple(item[1] for item in corrected)


def te_xy_curl_e_to_h_2d_metric(ex, ey, metrics, hz_shape):
    curl = _metric_difference(ey, 1, metrics.e_to_h_x) - _metric_difference(
        ex, 0, metrics.e_to_h_y
    )
    if curl.shape != hz_shape:
        raise ValueError(f"curl(E) shape {curl.shape} does not match Hz {hz_shape}")
    return curl


def _te_xy_h_derivatives_metric(
    hz, metrics, metallic_edges, periodic_axes=frozenset(), periodic_phases=()
):
    padded_y = _pad_h_axis(hz, 0, metallic_edges, periodic_axes, periodic_phases)
    padded_x = _pad_h_axis(hz, 1, metallic_edges, periodic_axes, periodic_phases)
    return (
        _metric_difference(padded_y, 0, metrics.h_to_e_y),
        _metric_difference(padded_x, 1, metrics.h_to_e_x),
    )


def te_xy_curl_h_to_e_2d_metric(
    hz,
    metrics,
    ex_shape,
    ey_shape,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
):
    d_hz_dy, d_hz_dx = _te_xy_h_derivatives_metric(
        hz, metrics, metallic_edges, periodic_axes, periodic_phases
    )
    curls = d_hz_dy, -d_hz_dx
    if curls[0].shape != ex_shape or curls[1].shape != ey_shape:
        raise ValueError(
            f"curl(H) shapes {(curls[0].shape, curls[1].shape)} do not match "
            f"Ex/Ey {(ex_shape, ey_shape)}"
        )
    return curls


def te_xy_cpml_curl_e_to_h_2d_metric(ex, ey, metrics, *, terms, psi_h_terms):
    derivatives = (
        _metric_difference(ey, 1, metrics.e_to_h_x),
        _metric_difference(ex, 0, metrics.e_to_h_y),
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_h_terms, terms, strict=True)
    )
    return corrected[0][0] + corrected[1][0], tuple(item[1] for item in corrected)


def te_xy_cpml_curl_h_to_e_2d_metric(
    hz,
    metrics,
    metallic_edges,
    periodic_axes=frozenset(),
    periodic_phases=(),
    *,
    terms,
    psi_e_terms,
):
    derivatives = _te_xy_h_derivatives_metric(
        hz, metrics, metallic_edges, periodic_axes, periodic_phases
    )
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_e_terms, terms, strict=True)
    )
    return (corrected[0][0], corrected[1][0]), tuple(item[1] for item in corrected)


def as_array(value) -> jnp.ndarray:
    # Make dtype conversion explicit to prevent promotion from changing JAX
    # signatures.
    return cast(jnp.ndarray, value)


def astype_like(value, ref: jnp.ndarray) -> jnp.ndarray:
    # Make dtype conversion explicit to prevent promotion from changing JAX
    # signatures.
    return as_array(value).astype(ref.dtype)


def apply_post_source_boundaries(
    values: tuple[jnp.ndarray, ...],
    metallic_masks: tuple[jnp.ndarray, ...],
    *,
    components: tuple[str, ...] = (),
    periodic_axes: frozenset[int] = frozenset(),
    periodic_phases=(),
    material_shape: tuple[int, ...] = (),
    logical_shapes=None,
) -> tuple[jnp.ndarray, ...]:
    # Apply the precomputed boundary policy so kernels never reinterpret configuration
    # objects.
    constrained = tuple(
        jnp.asarray(apply_zero_mask(value, mask))
        for value, mask in zip(values, metallic_masks, strict=True)
    )
    if not periodic_axes:
        return constrained
    if (
        len(components) != len(constrained)
        or not material_shape
        or logical_shapes is None
    ):
        raise ValueError(
            "Periodic constraints require component and logical shape metadata."
        )
    out = []
    for component, value in zip(components, constrained, strict=True):
        logical_shape = logical_shapes[component]
        for axis in sorted(periodic_axes):
            if int(logical_shape[axis]) != int(material_shape[axis]) + 1:
                continue
            low = jnp.take(value, jnp.asarray([0]), axis=axis)
            high = jnp.take(
                value, jnp.asarray([int(logical_shape[axis]) - 1]), axis=axis
            )
            phase = periodic_phases[axis] if periodic_phases else 1.0
            seam = 0.5 * (low + high / phase)
            low_index = [slice(None)] * value.ndim
            high_index = [slice(None)] * value.ndim
            low_index[axis] = slice(0, 1)
            high_index[axis] = slice(
                int(logical_shape[axis]) - 1, int(logical_shape[axis])
            )
            value = value.at[tuple(low_index)].set(seam)
            value = value.at[tuple(high_index)].set(seam * phase)
        out.append(value)
    return tuple(out)


# Each function is one static combination of dimension and boundary physics. Some
# repetition is deliberate: branch-free graphs are smaller and update order stays visible.


def _astype_terms(values, references):
    return tuple(
        value.astype(reference.dtype)
        for value, reference in zip(values, references, strict=True)
    )


def _replace_h(eng, hx, hy, hz, *, cpml_h=None):
    # Place the complete pytree consistently so one state never spans incompatible
    # shardings.
    updates: dict[str, Any] = {
        "hx": astype_like(hx, eng.hx),
        "hy": astype_like(hy, eng.hy),
        "hz": astype_like(hz, eng.hz),
    }
    if cpml_h is not None:
        updates["cpml_psi_h_terms"] = _astype_terms(cpml_h, eng.cpml_psi_h_terms)
    return eng._replace(**updates)


def _replace_e(eng, ex, ey, ez, *, cpml_e=None):
    # Place the complete pytree consistently so one state never spans incompatible
    # shardings.
    updates: dict[str, Any] = {
        "ex": astype_like(ex, eng.ex),
        "ey": astype_like(ey, eng.ey),
        "ez": astype_like(ez, eng.ez),
    }
    if cpml_e is not None:
        updates["cpml_psi_e_terms"] = _astype_terms(cpml_e, eng.cpml_psi_e_terms)
    return eng._replace(**updates)


def update_h_3d_cpml(eng, ctx, coeffs):
    # 1. Advance H and its packed-slab CPML memories together using the coefficients and
    # slab geometry fixed during planning.
    cpml = ctx.boundary.cpml
    hx, hy, hz, psi_h = cpml_update_h_from_e_3d(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.resolution,
        terms=cpml.h_terms,
        psi_terms=eng.cpml_psi_h_terms,
        dt=ctx.dt_scalar,
        source_scales=(
            (coeffs.h_source_x, coeffs.h_source_y, coeffs.h_source_z)
            if coeffs.h_source_x.size
            else None
        ),
        magnetic_conductivities=(
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
        ),
    )
    # 2. Replace only H and its 3D memory, preserving all other fields and auxiliary state
    # in the immutable engine carry.
    return _replace_h(
        eng,
        hx,
        hy,
        hz,
        cpml_h=psi_h,
    )


def update_e_3d_cpml(eng, ctx, coeffs):
    # 1. Derive inverse permittivity at execution time so the compiled plan stores only
    # the authoritative material grids.
    cpml = ctx.boundary.cpml
    inv_x = reciprocal_material(coeffs.e_permittivity_x)
    inv_y = reciprocal_material(coeffs.e_permittivity_y)
    inv_z = reciprocal_material(coeffs.e_permittivity_z)
    inverse_diagonals = (
        (
            coeffs.e_inverse_diagonal_x,
            coeffs.e_inverse_diagonal_y,
            coeffs.e_inverse_diagonal_z,
        )
        if coeffs.e_inverse_offdiagonal.size
        else None
    )
    # 2. Advance E while updating psi only inside the packed boundary slabs selected by
    # the planner.
    ex, ey, ez, psi_e = cpml_update_e_from_h_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        ctx.resolution,
        terms=cpml.e_terms,
        psi_terms=eng.cpml_psi_e_terms,
        metallic_edges=cpml.metallic_edges,
        dt=ctx.dt_scalar,
        source_scales=(
            (coeffs.e_source_x, coeffs.e_source_y, coeffs.e_source_z)
            if coeffs.e_source_x.size
            else None
        ),
        conductivities=(
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
        ),
        inverse_permittivities=(inv_x, inv_y, inv_z),
        inverse_diagonals=inverse_diagonals,
        inverse_offdiagonal=(
            coeffs.e_inverse_offdiagonal if coeffs.e_inverse_offdiagonal.size else None
        ),
        logical_shapes=ctx.boundary.logical_component_shapes,
        periodic_axes=ctx.boundary.periodic_axes,
        periodic_phases=ctx.boundary.periodic_phases,
    )
    # 3. Replace E and packed 3D memory atomically, leaving unrelated carry fields intact.
    return _replace_e(
        eng,
        ex,
        ey,
        ez,
        cpml_e=psi_e,
    )


def update_h_3d_yee(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    hx, hy, hz = fused_update_h_lossy_3d_material(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        coeffs.h_sigma_m_x,
        coeffs.h_sigma_m_y,
        coeffs.h_sigma_m_z,
        ctx.dt_scalar,
        ctx.resolution,
    )
    return _replace_h(eng, hx, hy, hz)


def update_e_3d_yee(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    boundary_views = build_h_boundary_views_for_e_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.boundary.cpml.metallic_edges,
        logical_shapes=ctx.boundary.logical_component_shapes,
        periodic_axes=ctx.boundary.periodic_axes,
        periodic_phases=ctx.boundary.periodic_phases,
    )
    ex, ey, ez = fused_update_e_lossy_3d_material(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        coeffs.e_conductivity_x,
        reciprocal_material(coeffs.e_permittivity_x),
        coeffs.e_conductivity_y,
        reciprocal_material(coeffs.e_permittivity_y),
        coeffs.e_conductivity_z,
        reciprocal_material(coeffs.e_permittivity_z),
        ctx.dt_scalar,
        ctx.resolution,
        boundary_views=boundary_views,
        inverse_diagonals=(
            (
                coeffs.e_inverse_diagonal_x,
                coeffs.e_inverse_diagonal_y,
                coeffs.e_inverse_diagonal_z,
            )
            if coeffs.e_inverse_offdiagonal.size
            else None
        ),
        inverse_offdiagonal=(
            coeffs.e_inverse_offdiagonal if coeffs.e_inverse_offdiagonal.size else None
        ),
    )
    return _replace_e(eng, ex, ey, ez)


def _update_h_tm_from_curls(eng, ctx, coeffs, curl_tm_hx, curl_tm_hy, psi_h=None):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    hx = advance_h_from_coefficients(
        eng.hx, curl_tm_hx, coeffs.h_decay_x, coeffs.h_source_x
    )
    hy = advance_h_from_coefficients(
        eng.hy, curl_tm_hy, coeffs.h_decay_y, coeffs.h_source_y
    )
    return _replace_h(eng, hx, hy, eng.hz, cpml_h=psi_h)


def update_h_2d_tm_xy(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        eng.ez,
        ctx.resolution,
        eng.hx.shape,
        eng.hy.shape,
        ctx.boundary.metallic_edges_2d,
    )
    return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy)


def update_h_2d_tm_xy_cpml(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    cpml = ctx.boundary.cpml
    curl_hx, curl_hy, psi_h = tm_xy_cpml_curl_e_to_h_2d(
        eng.ez,
        ctx.resolution,
        terms=cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy, psi_h=psi_h)


def _update_e_tm_from_curl(eng, ctx, coeffs, curl_tm_ez, psi_e=None):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    ez = advance_e_from_coefficients(
        eng.ez, curl_tm_ez, coeffs.e_decay_z, coeffs.e_source_z
    )
    return _replace_e(
        eng,
        eng.ex,
        eng.ey,
        ez,
        cpml_e=psi_e,
    )


def update_e_2d_tm_xy(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    curl_ez = tm_xy_curl_h_to_e_2d(
        eng.hx,
        eng.hy,
        ctx.resolution,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl_ez)


def update_e_2d_tm_xy_cpml(eng, ctx, coeffs):
    # Return a new SimulationState so the timestep remains a pure JAX transformation.
    cpml = ctx.boundary.cpml
    curl_ez, psi_e = tm_xy_cpml_curl_h_to_e_2d(
        eng.hx,
        eng.hy,
        ctx.resolution,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
        terms=cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl_ez, psi_e=psi_e)


def _update_h_te_from_curl(eng, coeffs, curl_hz, psi_h=None):
    hz = advance_h_from_coefficients(
        eng.hz, curl_hz, coeffs.h_decay_z, coeffs.h_source_z
    )
    return _replace_h(eng, eng.hx, eng.hy, hz, cpml_h=psi_h)


def update_h_2d_te_xy(eng, ctx, coeffs):
    curl_hz = te_xy_curl_e_to_h_2d(eng.ex, eng.ey, ctx.resolution, eng.hz.shape)
    return _update_h_te_from_curl(eng, coeffs, curl_hz)


def update_h_2d_te_xy_cpml(eng, ctx, coeffs):
    curl_hz, psi_h = te_xy_cpml_curl_e_to_h_2d(
        eng.ex,
        eng.ey,
        ctx.resolution,
        terms=ctx.boundary.cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_te_from_curl(eng, coeffs, curl_hz, psi_h=psi_h)


def _update_e_te_from_curls(eng, ctx, coeffs, curls, psi_e=None):
    if coeffs.e_inverse_offdiagonal.size:
        ex, ey = advance_e_centered_tensor(
            (eng.ex, eng.ey),
            curls,
            (coeffs.e_inverse_diagonal_x, coeffs.e_inverse_diagonal_y),
            coeffs.e_inverse_offdiagonal,
            ("Ex", "Ey"),
            ctx.dt_scalar,
            cell_widths=constitutive_cell_widths(ctx.metrics),
        )
    else:
        ex = advance_e_from_coefficients(
            eng.ex, curls[0], coeffs.e_decay_x, coeffs.e_source_x
        )
        ey = advance_e_from_coefficients(
            eng.ey, curls[1], coeffs.e_decay_y, coeffs.e_source_y
        )
    return _replace_e(eng, ex, ey, eng.ez, cpml_e=psi_e)


def update_e_2d_te_xy(eng, ctx, coeffs):
    curls = te_xy_curl_h_to_e_2d(
        eng.hz,
        ctx.resolution,
        eng.ex.shape,
        eng.ey.shape,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
    )
    return _update_e_te_from_curls(eng, ctx, coeffs, curls)


def update_e_2d_te_xy_cpml(eng, ctx, coeffs):
    curls, psi_e = te_xy_cpml_curl_h_to_e_2d(
        eng.hz,
        ctx.resolution,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
        terms=ctx.boundary.cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_te_from_curls(eng, ctx, coeffs, curls, psi_e=psi_e)


def update_h_3d_cpml_metric(eng, ctx, coeffs):
    cpml = ctx.boundary.cpml
    hx, hy, hz, psi_h = cpml_update_h_from_e_3d_metric(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.metrics,
        terms=cpml.h_terms,
        psi_terms=eng.cpml_psi_h_terms,
        dt=ctx.dt_scalar,
        magnetic_conductivities=(
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
        ),
    )
    return _replace_h(eng, hx, hy, hz, cpml_h=psi_h)


def update_e_3d_cpml_metric(eng, ctx, coeffs):
    cpml = ctx.boundary.cpml
    ex, ey, ez, psi_e = cpml_update_e_from_h_3d_metric(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        ctx.metrics,
        terms=cpml.e_terms,
        psi_terms=eng.cpml_psi_e_terms,
        metallic_edges=cpml.metallic_edges,
        dt=ctx.dt_scalar,
        conductivities=(
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
        ),
        inverse_permittivities=(
            reciprocal_material(coeffs.e_permittivity_x),
            reciprocal_material(coeffs.e_permittivity_y),
            reciprocal_material(coeffs.e_permittivity_z),
        ),
        inverse_diagonals=(
            (
                coeffs.e_inverse_diagonal_x,
                coeffs.e_inverse_diagonal_y,
                coeffs.e_inverse_diagonal_z,
            )
            if coeffs.e_inverse_offdiagonal.size
            else None
        ),
        inverse_offdiagonal=(
            coeffs.e_inverse_offdiagonal if coeffs.e_inverse_offdiagonal.size else None
        ),
        logical_shapes=ctx.boundary.logical_component_shapes,
        periodic_axes=ctx.boundary.periodic_axes,
        periodic_phases=ctx.boundary.periodic_phases,
    )
    return _replace_e(eng, ex, ey, ez, cpml_e=psi_e)


def update_h_3d_yee_metric(eng, ctx, coeffs):
    hx, hy, hz = fused_update_h_lossy_3d_material_metric(
        eng.ex,
        eng.ey,
        eng.ez,
        eng.hx,
        eng.hy,
        eng.hz,
        coeffs.h_sigma_m_x,
        coeffs.h_sigma_m_y,
        coeffs.h_sigma_m_z,
        ctx.dt_scalar,
        ctx.metrics,
    )
    return _replace_h(eng, hx, hy, hz)


def update_e_3d_yee_metric(eng, ctx, coeffs):
    boundary_views = build_h_boundary_views_for_e_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        ctx.boundary.cpml.metallic_edges,
        logical_shapes=ctx.boundary.logical_component_shapes,
        periodic_axes=ctx.boundary.periodic_axes,
        periodic_phases=ctx.boundary.periodic_phases,
    )
    ex, ey, ez = fused_update_e_lossy_3d_material_metric(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        coeffs.e_conductivity_x,
        reciprocal_material(coeffs.e_permittivity_x),
        coeffs.e_conductivity_y,
        reciprocal_material(coeffs.e_permittivity_y),
        coeffs.e_conductivity_z,
        reciprocal_material(coeffs.e_permittivity_z),
        ctx.dt_scalar,
        ctx.metrics,
        boundary_views=boundary_views,
        inverse_diagonals=(
            (
                coeffs.e_inverse_diagonal_x,
                coeffs.e_inverse_diagonal_y,
                coeffs.e_inverse_diagonal_z,
            )
            if coeffs.e_inverse_offdiagonal.size
            else None
        ),
        inverse_offdiagonal=(
            coeffs.e_inverse_offdiagonal if coeffs.e_inverse_offdiagonal.size else None
        ),
    )
    return _replace_e(eng, ex, ey, ez)


def update_h_2d_tm_xy_metric(eng, ctx, coeffs):
    curls = tm_xy_curl_e_to_h_2d_metric(eng.ez, ctx.metrics)
    return _update_h_tm_from_curls(eng, ctx, coeffs, *curls)


def update_e_2d_tm_xy_metric(eng, ctx, coeffs):
    curl = tm_xy_curl_h_to_e_2d_metric(
        eng.hx,
        eng.hy,
        ctx.metrics,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl)


def update_h_2d_tm_xy_cpml_metric(eng, ctx, coeffs):
    curl_hx, curl_hy, psi_h = tm_xy_cpml_curl_e_to_h_2d_metric(
        eng.ez,
        ctx.metrics,
        terms=ctx.boundary.cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy, psi_h=psi_h)


def update_e_2d_tm_xy_cpml_metric(eng, ctx, coeffs):
    curl, psi_e = tm_xy_cpml_curl_h_to_e_2d_metric(
        eng.hx,
        eng.hy,
        ctx.metrics,
        eng.ez.shape,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
        terms=ctx.boundary.cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl, psi_e=psi_e)


def update_h_2d_te_xy_metric(eng, ctx, coeffs):
    curl = te_xy_curl_e_to_h_2d_metric(eng.ex, eng.ey, ctx.metrics, eng.hz.shape)
    return _update_h_te_from_curl(eng, coeffs, curl)


def update_e_2d_te_xy_metric(eng, ctx, coeffs):
    curls = te_xy_curl_h_to_e_2d_metric(
        eng.hz,
        ctx.metrics,
        eng.ex.shape,
        eng.ey.shape,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
    )
    return _update_e_te_from_curls(eng, ctx, coeffs, curls)


def update_h_2d_te_xy_cpml_metric(eng, ctx, coeffs):
    curl, psi_h = te_xy_cpml_curl_e_to_h_2d_metric(
        eng.ex,
        eng.ey,
        ctx.metrics,
        terms=ctx.boundary.cpml.h_terms,
        psi_h_terms=eng.cpml_psi_h_terms,
    )
    return _update_h_te_from_curl(eng, coeffs, curl, psi_h=psi_h)


def update_e_2d_te_xy_cpml_metric(eng, ctx, coeffs):
    curls, psi_e = te_xy_cpml_curl_h_to_e_2d_metric(
        eng.hz,
        ctx.metrics,
        ctx.boundary.metallic_edges_2d,
        ctx.boundary.periodic_axes,
        ctx.boundary.periodic_phases,
        terms=ctx.boundary.cpml.e_terms,
        psi_e_terms=eng.cpml_psi_e_terms,
    )
    return _update_e_te_from_curls(eng, ctx, coeffs, curls, psi_e=psi_e)


@dataclass(frozen=True)
class CompiledStepContext:
    """Static data captured by the compiled step builder."""

    # Select one concrete kernel from static plan flags. Timesteps then avoid tracing Python
    # feature branches into every compiled execution.

    config: RunConfig
    boundary: BoundaryPlan
    source_batches: Any
    metrics: Any
    resolution: float
    dt: float
    dt_scalar: jnp.ndarray
    is_3d: bool
    differentiable: bool = False
    sharding_plan: Any = None
    # Capacity scheduling is selected separately from buffer donation.
    low_memory: bool = False


@dataclass(frozen=True)
class StepUpdateKernel:
    """Planner-selected E/H update functions for a compiled program."""

    # Group these values because their positional and shape relationships form one
    # invariant.
    kind: str
    update_h: Callable[
        [SimulationState, CompiledStepContext, UpdateCoefficients], SimulationState
    ]
    update_e: Callable[
        [SimulationState, CompiledStepContext, UpdateCoefficients], SimulationState
    ]


def select_update_kernel(ctx: CompiledStepContext) -> StepUpdateKernel:
    """Select the static update-kernel variant before JAX tracing."""

    if ctx.config.backend != "jax":
        if not ctx.is_3d:
            raise ValueError(
                "CUDA execution currently requires a three-dimensional grid"
            )
        if ctx.config.sharding.enabled:
            from beamz.simulation.cuda.sharding import select_sharded_kernel

            return select_sharded_kernel(ctx)
        from beamz.simulation.cuda import update_e, update_h

        return StepUpdateKernel(ctx.config.backend, update_h, update_e)

    boundary = ctx.boundary
    metric = ctx.config.metric_kind != "isotropic_uniform"
    family = "3d" if ctx.is_3d else ctx.config.polarization_2d
    variants = {
        ("3d", False, False): (
            update_h_3d_yee,
            update_e_3d_yee,
        ),
        ("3d", False, True): (
            update_h_3d_cpml,
            update_e_3d_cpml,
        ),
        ("3d", True, False): (
            update_h_3d_yee_metric,
            update_e_3d_yee_metric,
        ),
        ("3d", True, True): (
            update_h_3d_cpml_metric,
            update_e_3d_cpml_metric,
        ),
        ("tm", False, False): (
            update_h_2d_tm_xy,
            update_e_2d_tm_xy,
        ),
        ("tm", False, True): (
            update_h_2d_tm_xy_cpml,
            update_e_2d_tm_xy_cpml,
        ),
        ("tm", True, False): (
            update_h_2d_tm_xy_metric,
            update_e_2d_tm_xy_metric,
        ),
        ("tm", True, True): (
            update_h_2d_tm_xy_cpml_metric,
            update_e_2d_tm_xy_cpml_metric,
        ),
        ("te", False, False): (
            update_h_2d_te_xy,
            update_e_2d_te_xy,
        ),
        ("te", False, True): (
            update_h_2d_te_xy_cpml,
            update_e_2d_te_xy_cpml,
        ),
        ("te", True, False): (
            update_h_2d_te_xy_metric,
            update_e_2d_te_xy_metric,
        ),
        ("te", True, True): (
            update_h_2d_te_xy_cpml_metric,
            update_e_2d_te_xy_cpml_metric,
        ),
    }
    update_h, update_e = variants[family, metric, boundary.cpml.enabled]
    kind = "yee_3d" if ctx.is_3d else f"physical_{family}_xy"
    if boundary.cpml.enabled:
        kind = "cpml_3d" if ctx.is_3d else kind + "_cpml"
    if metric:
        kind = f"{ctx.config.metric_kind}_{kind}"
    return StepUpdateKernel(kind, update_h, update_e)
