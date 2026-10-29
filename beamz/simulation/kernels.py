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


def _coordinate_difference(values, axis):
    axis = int(axis)
    upper = [slice(None)] * values.ndim
    lower = [slice(None)] * values.ndim
    upper[axis] = slice(1, None)
    lower[axis] = slice(None, -1)
    return values[tuple(upper)] - values[tuple(lower)]


def _lossy_h(fields, sigmas, curls, dt):
    one = jnp.asarray(1.0, dtype=fields[0].dtype)
    dt_over_mu = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        MU_0, dtype=fields[0].dtype
    )
    updated = []
    for field, sigma, curl in zip(fields, sigmas, curls, strict=True):
        alpha = sigma * (0.5 * dt_over_mu)
        denom = one + alpha
        updated.append((one - alpha) / denom * field - (dt_over_mu / denom) * curl)
    return tuple(updated)


def _lossy_e(fields, conductivities, inverse_permittivities, curls, dt):
    one = jnp.asarray(1.0, dtype=fields[0].dtype)
    dt_over_eps = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    updated = []
    for field, conductivity, inv_permittivity, curl in zip(
        fields, conductivities, inverse_permittivities, curls, strict=True
    ):
        inv_permittivity = material_values(inv_permittivity)
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        denom = one + beta
        updated.append(
            (one - beta) / denom * field
            + (dt_over_eps * inv_permittivity / denom) * curl
        )
    return tuple(updated)


def _centered_cpml_h(fields, curls, sigmas, dt, source_scales):
    if source_scales is not None:
        return tuple(
            field - scale * (curl + sigma * field)
            for field, curl, sigma, scale in zip(
                fields, curls, sigmas, source_scales, strict=True
            )
        )
    one = jnp.asarray(1.0, dtype=fields[0].dtype)
    dt_over_mu = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        MU_0, dtype=fields[0].dtype
    )
    updated = []
    for field, curl, sigma in zip(fields, curls, sigmas, strict=True):
        alpha = sigma * (0.5 * dt_over_mu)
        updated.append(((one - alpha) * field - dt_over_mu * curl) / (one + alpha))
    return tuple(updated)


def _centered_cpml_e(
    fields,
    curls,
    conductivities,
    inverse_permittivities,
    dt,
    inverse_diagonals,
    inverse_offdiagonal,
    source_scales,
    spacing,
    metric,
):
    if inverse_offdiagonal is not None:
        return advance_e_centered_tensor(
            fields,
            curls,
            inverse_diagonals,
            inverse_offdiagonal,
            ("Ex", "Ey", "Ez"),
            dt,
            cell_widths=constitutive_cell_widths(spacing) if metric else None,
        )
    if source_scales is not None:
        return tuple(
            field + material_values(scale) * (curl - sigma * field)
            for field, curl, sigma, scale in zip(
                fields, curls, conductivities, source_scales, strict=True
            )
        )
    one = jnp.asarray(1.0, dtype=fields[0].dtype)
    dt_over_eps = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    updated = []
    for field, curl, conductivity, inv_permittivity in zip(
        fields, curls, conductivities, inverse_permittivities, strict=True
    ):
        inv_permittivity = material_values(inv_permittivity)
        beta = conductivity * (0.5 * dt_over_eps) * inv_permittivity
        updated.append(
            ((one - beta) * field + dt_over_eps * inv_permittivity * curl)
            / (one + beta)
        )
    return tuple(updated)


# (source index or view name, axis, metric attribute) for the positive then negative term.
_H_CURL_PAIRS = (
    ((2, 1, "e_to_h_y"), (1, 0, "e_to_h_z")),
    ((0, 0, "e_to_h_z"), (2, 2, "e_to_h_x")),
    ((1, 2, "e_to_h_x"), (0, 1, "e_to_h_y")),
)
_E_CURL_PAIRS = (
    (("hz_y", 1, "h_to_e_y"), ("hy_z", 0, "h_to_e_z")),
    (("hx_z", 0, "h_to_e_z"), ("hz_x", 2, "h_to_e_x")),
    (("hy_x", 2, "h_to_e_x"), ("hx_y", 1, "h_to_e_y")),
)
# Six CPML derivatives, then paired into Ex/Ey/Ez or Hx/Hy/Hz curls.
_H_CPML_TERMS = (
    (2, 1, "e_to_h_y", 0),
    (1, 0, "e_to_h_z", 0),
    (0, 0, "e_to_h_z", 1),
    (2, 2, "e_to_h_x", 1),
    (1, 2, "e_to_h_x", 2),
    (0, 1, "e_to_h_y", 2),
)
_E_CPML_TERMS = (
    ("hz_y", 1, "h_to_e_y", 0),
    ("hy_z", 0, "h_to_e_z", 0),
    ("hx_z", 0, "h_to_e_z", 1),
    ("hz_x", 2, "h_to_e_x", 1),
    ("hy_x", 2, "h_to_e_x", 2),
    ("hx_y", 1, "h_to_e_y", 2),
)


def _h_derivative(field, axis, spacing, metric, metric_name):
    # ``metric`` is a Python literal at each kernel entry, so tracing keeps one operator.
    if metric:
        return _metric_difference(field, axis, getattr(spacing, metric_name))
    return _coordinate_difference(field, axis)


def _h_fused_curls(electric, magnetic, spacing, metric):
    scale = None
    if not metric:
        scale = jnp.asarray(1.0, dtype=magnetic[0].dtype) / jnp.asarray(
            spacing, dtype=magnetic[0].dtype
        )
    curls = []
    for field, pair in zip(magnetic, _H_CURL_PAIRS, strict=True):
        terms = tuple(
            _h_derivative(electric[index], axis, spacing, metric, name)
            for index, axis, name in pair
        )
        curl = fit_array_to_shape(terms[0], field.shape) - fit_array_to_shape(
            terms[1], field.shape
        )
        curls.append(curl if scale is None else curl * scale)
    return tuple(curls)


def _e_derivative(views, name, axis, spacing, metric, metric_name):
    if metric:
        return _metric_difference(views[name], axis, getattr(spacing, metric_name))
    return _adjacent_difference(views[name], axis, spacing)


def _e_fused_curls(fields, views, spacing, metric):
    curls = []
    for field, pair in zip(fields, _E_CURL_PAIRS, strict=True):
        terms = tuple(
            _e_derivative(views, name, axis, spacing, metric, metric_name)
            for name, axis, metric_name in pair
        )
        curls.append(
            fit_array_to_shape(terms[0], field.shape)
            - fit_array_to_shape(terms[1], field.shape)
        )
    return tuple(curls)


def _paired_cpml_curls(raw_shapes, psi_terms, terms):
    corrected, next_psi = zip(
        *(
            correct_cpml_term(fit_array_to_shape(raw, shape), psi, term)
            for (raw, shape), psi, term in zip(
                raw_shapes, psi_terms, terms, strict=True
            )
        ),
        strict=True,
    )
    curls = tuple(corrected[index] + corrected[index + 1] for index in (0, 2, 4))
    return curls, tuple(next_psi)


def _h_cpml_raw(electric, magnetic, spacing, metric):
    raw = []
    for index, axis, metric_name, shape_index in _H_CPML_TERMS:
        field = electric[index]
        if metric:
            value = _metric_difference(field, axis, getattr(spacing, metric_name))
        else:
            value = jnp.diff(field, axis=int(axis)) / spacing
        raw.append((value, magnetic[shape_index].shape))
    return tuple(raw)


def _e_cpml_raw(views, shapes, spacing, metric):
    raw = []
    for name, axis, metric_name, shape_index in _E_CPML_TERMS:
        if metric:
            value = _metric_difference(views[name], axis, getattr(spacing, metric_name))
        else:
            value = _adjacent_difference(views[name], axis, spacing)
        raw.append((value, shapes[shape_index]))
    return tuple(raw)


def fused_update_h_lossy_3d_material(
    ex, ey, ez, hx, hy, hz, sigma_x, sigma_y, sigma_z, dt, resolution
):
    """Advance all magnetic components directly from collocated loss grids."""
    curls = _h_fused_curls((ex, ey, ez), (hx, hy, hz), resolution, False)
    return _lossy_h((hx, hy, hz), (sigma_x, sigma_y, sigma_z), curls, dt)


def fused_update_h_lossy_3d_material_metric(
    ex, ey, ez, hx, hy, hz, sigma_x, sigma_y, sigma_z, dt, metrics
):
    """Advance 3D H fields using statically planned separable grid metrics."""
    curls = _h_fused_curls((ex, ey, ez), (hx, hy, hz), metrics, True)
    return _lossy_h((hx, hy, hz), (sigma_x, sigma_y, sigma_z), curls, dt)


def _fused_e(
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
    spacing,
    *,
    metric,
    boundary_views,
    inverse_diagonals=None,
    inverse_offdiagonal=None,
):
    del hx, hy, hz
    curls = _e_fused_curls((ex, ey, ez), boundary_views, spacing, metric)
    if inverse_offdiagonal is not None:
        return advance_e_centered_tensor(
            (ex, ey, ez),
            curls,
            inverse_diagonals,
            inverse_offdiagonal,
            ("Ex", "Ey", "Ez"),
            dt,
            cell_widths=constitutive_cell_widths(spacing) if metric else None,
        )
    return _lossy_e(
        (ex, ey, ez),
        (conductivity_x, conductivity_y, conductivity_z),
        (inv_permittivity_x, inv_permittivity_y, inv_permittivity_z),
        curls,
        dt,
    )


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
    **kwargs,
):
    """Advance all electric components directly from collocated material grids."""
    return _fused_e(
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
        metric=False,
        **kwargs,
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
    **kwargs,
):
    """Advance 3D E fields using staggered center-to-edge distances."""
    return _fused_e(
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
        metric=True,
        **kwargs,
    )


def _cpml_h_update(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    spacing,
    *,
    metric,
    terms,
    psi_terms,
    dt,
    magnetic_conductivities,
    source_scales=None,
):
    curls, next_psi = _paired_cpml_curls(
        _h_cpml_raw((ex, ey, ez), (hx, hy, hz), spacing, metric), psi_terms, terms
    )
    updated = _centered_cpml_h(
        (hx, hy, hz), curls, magnetic_conductivities, dt, source_scales
    )
    return *updated, next_psi


def _cpml_e_update(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    spacing,
    *,
    metric,
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
    views = build_h_boundary_views_for_e_3d(
        hx,
        hy,
        hz,
        metallic_edges,
        logical_shapes=logical_shapes,
        periodic_axes=periodic_axes,
        periodic_phases=periodic_phases,
    )
    curls, next_psi = _paired_cpml_curls(
        _e_cpml_raw(views, (ex.shape, ey.shape, ez.shape), spacing, metric),
        psi_terms,
        terms,
    )
    updated = _centered_cpml_e(
        (ex, ey, ez),
        curls,
        conductivities,
        inverse_permittivities,
        dt,
        inverse_diagonals,
        inverse_offdiagonal,
        source_scales,
        spacing,
        metric,
    )
    return *updated, next_psi


def cpml_update_h_from_e_3d(ex, ey, ez, hx, hy, hz, resolution, **kwargs):
    """Advance 3D H fields and their six packed CPML memories."""
    return _cpml_h_update(ex, ey, ez, hx, hy, hz, resolution, metric=False, **kwargs)


def cpml_update_h_from_e_3d_metric(ex, ey, ez, hx, hy, hz, metrics, **kwargs):
    """Advance 3D H/CPML using physical cell widths for every derivative."""
    return _cpml_h_update(
        ex, ey, ez, hx, hy, hz, metrics, metric=True, source_scales=None, **kwargs
    )


def cpml_update_e_from_h_3d(hx, hy, hz, ex, ey, ez, resolution, **kwargs):
    """Advance 3D E fields and their six packed CPML memories."""
    return _cpml_e_update(hx, hy, hz, ex, ey, ez, resolution, metric=False, **kwargs)


def cpml_update_e_from_h_3d_metric(hx, hy, hz, ex, ey, ez, metrics, **kwargs):
    """Advance 3D E/CPML using physical center-to-center distances."""
    return _cpml_e_update(
        hx, hy, hz, ex, ey, ez, metrics, metric=True, source_scales=None, **kwargs
    )


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


_TRANSVERSE_EDGES = (("bottom", "top"), ("left", "right"))


def _pad_transverse(field, axis, periodic_axes, periodic_phases, metallic_edges):
    axis = int(axis)
    if axis == 0:
        low, high = field[:1, :], field[-1:, :]
    else:
        low, high = field[:, :1], field[:, -1:]
    if axis in periodic_axes:
        phase = periodic_phases[axis] if periodic_phases else 1.0
        low, high = high / phase, low * phase
    low_name, high_name = _TRANSVERSE_EDGES[axis]
    if low_name in metallic_edges:
        low = jnp.zeros_like(low)
    if high_name in metallic_edges:
        high = jnp.zeros_like(high)
    return jnp.concatenate((low, field, high), axis=axis)


def _scaled_difference(scale):
    def difference(values, axis, _metric_name):
        return _coordinate_difference(values, axis) / scale

    return difference


def _named_metric_difference(metrics):
    def difference(values, axis, metric_name):
        return _metric_difference(values, axis, getattr(metrics, metric_name))

    return difference


def _correct_derivatives(derivatives, psi_terms, terms):
    corrected = tuple(
        correct_cpml_term(derivative, psi, term)
        for derivative, psi, term in zip(derivatives, psi_terms, terms, strict=True)
    )
    return tuple(item[0] for item in corrected), tuple(item[1] for item in corrected)


def _tm_h_derivatives(
    hx, hy, metallic_edges, periodic_axes, periodic_phases, difference
):
    padded_hy = _pad_transverse(hy, 1, periodic_axes, periodic_phases, metallic_edges)
    padded_hx = _pad_transverse(hx, 0, periodic_axes, periodic_phases, metallic_edges)
    return difference(padded_hy, 1, "h_to_e_x"), difference(padded_hx, 0, "h_to_e_y")


def _te_h_derivatives(hz, metallic_edges, periodic_axes, periodic_phases, difference):
    padded_y = _pad_transverse(hz, 0, periodic_axes, periodic_phases, metallic_edges)
    padded_x = _pad_transverse(hz, 1, periodic_axes, periodic_phases, metallic_edges)
    return difference(padded_y, 0, "h_to_e_y"), difference(padded_x, 1, "h_to_e_x")


def _require_curl_shape(curl, expected, message):
    if curl.shape != expected:
        raise ValueError(message)
    return curl


def tm_xy_curl_e_to_h_2d(ez, resolution, hx_shape, hy_shape, metallic_edges):
    """Differentiate Ez onto the native Hx and Hy supports."""
    del hx_shape, hy_shape, metallic_edges
    difference = _scaled_difference(_scalar_like(resolution, ez.dtype))
    return difference(ez, 0, None), -difference(ez, 1, None)


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
    d_hy_dx, d_hx_dy = _tm_h_derivatives(
        hx,
        hy,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _scaled_difference(_scalar_like(resolution, hy.dtype)),
    )
    curl = d_hy_dx - d_hx_dy
    return _require_curl_shape(
        curl, ez_shape, f"curl(H) shape {curl.shape} does not match Ez {ez_shape}"
    )


def tm_xy_cpml_curl_e_to_h_2d(ez, resolution, *, terms, psi_h_terms):
    """Correct the two Ez derivatives with canonical 2D CPML memory."""
    difference = _scaled_difference(_scalar_like(resolution, ez.dtype))
    values, psi = _correct_derivatives(
        (difference(ez, 0, None), difference(ez, 1, None)), psi_h_terms, terms
    )
    return values[0], values[1], psi


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
    derivatives = _tm_h_derivatives(
        hx,
        hy,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _scaled_difference(_scalar_like(resolution, hy.dtype)),
    )
    values, psi = _correct_derivatives(derivatives, psi_e_terms, terms)
    curl = values[0] + values[1]
    _require_curl_shape(
        curl, ez_shape, f"CPML curl(H) shape {curl.shape} does not match Ez {ez_shape}"
    )
    return curl.astype(hx.dtype), psi


def te_xy_curl_e_to_h_2d(ex, ey, resolution, hz_shape):
    """Differentiate in-plane E onto the canonical Hz support."""
    difference = _scaled_difference(_scalar_like(resolution, ex.dtype))
    curl = difference(ey, 1, None) - difference(ex, 0, None)
    return _require_curl_shape(
        curl, hz_shape, f"curl(E) shape {curl.shape} does not match Hz {hz_shape}"
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
    d_hz_dy, d_hz_dx = _te_h_derivatives(
        hz,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _scaled_difference(_scalar_like(resolution, hz.dtype)),
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
    difference = _scaled_difference(_scalar_like(resolution, ex.dtype))
    values, psi = _correct_derivatives(
        (difference(ey, 1, None), difference(ex, 0, None)), psi_h_terms, terms
    )
    return values[0] + values[1], psi


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
    derivatives = _te_h_derivatives(
        hz,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _scaled_difference(_scalar_like(resolution, hz.dtype)),
    )
    values, psi = _correct_derivatives(derivatives, psi_e_terms, terms)
    return (values[0], values[1]), psi


def tm_xy_curl_e_to_h_2d_metric(ez, metrics):
    """Differentiate Ez onto H using physical x/y cell widths."""
    difference = _named_metric_difference(metrics)
    return difference(ez, 0, "e_to_h_y"), -difference(ez, 1, "e_to_h_x")


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
    d_hy_dx, d_hx_dy = _tm_h_derivatives(
        hx,
        hy,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _named_metric_difference(metrics),
    )
    curl = d_hy_dx - d_hx_dy
    return _require_curl_shape(
        curl, ez_shape, f"curl(H) shape {curl.shape} does not match Ez {ez_shape}"
    )


def tm_xy_cpml_curl_e_to_h_2d_metric(ez, metrics, *, terms, psi_h_terms):
    difference = _named_metric_difference(metrics)
    values, psi = _correct_derivatives(
        (difference(ez, 0, "e_to_h_y"), difference(ez, 1, "e_to_h_x")),
        psi_h_terms,
        terms,
    )
    return values[0], values[1], psi


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
    derivatives = _tm_h_derivatives(
        hx,
        hy,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _named_metric_difference(metrics),
    )
    values, psi = _correct_derivatives(derivatives, psi_e_terms, terms)
    curl = values[0] + values[1]
    _require_curl_shape(
        curl, ez_shape, f"CPML curl(H) shape {curl.shape} does not match Ez {ez_shape}"
    )
    return curl.astype(hx.dtype), psi


def te_xy_curl_e_to_h_2d_metric(ex, ey, metrics, hz_shape):
    difference = _named_metric_difference(metrics)
    curl = difference(ey, 1, "e_to_h_x") - difference(ex, 0, "e_to_h_y")
    return _require_curl_shape(
        curl, hz_shape, f"curl(E) shape {curl.shape} does not match Hz {hz_shape}"
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
    d_hz_dy, d_hz_dx = _te_h_derivatives(
        hz,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _named_metric_difference(metrics),
    )
    curls = d_hz_dy, -d_hz_dx
    if curls[0].shape != ex_shape or curls[1].shape != ey_shape:
        raise ValueError(
            f"curl(H) shapes {(curls[0].shape, curls[1].shape)} do not match "
            f"Ex/Ey {(ex_shape, ey_shape)}"
        )
    return curls


def te_xy_cpml_curl_e_to_h_2d_metric(ex, ey, metrics, *, terms, psi_h_terms):
    difference = _named_metric_difference(metrics)
    values, psi = _correct_derivatives(
        (difference(ey, 1, "e_to_h_x"), difference(ex, 0, "e_to_h_y")),
        psi_h_terms,
        terms,
    )
    return values[0] + values[1], psi


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
    derivatives = _te_h_derivatives(
        hz,
        metallic_edges,
        periodic_axes,
        periodic_phases,
        _named_metric_difference(metrics),
    )
    values, psi = _correct_derivatives(derivatives, psi_e_terms, terms)
    return (values[0], values[1]), psi


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


def _astype_terms(values, references):
    return tuple(
        value.astype(reference.dtype)
        for value, reference in zip(values, references, strict=True)
    )


def _replace_phase(eng, prefix, values, cpml):
    names = tuple(prefix + axis for axis in "xyz")
    updates: dict[str, Any] = {
        name: astype_like(value, getattr(eng, name))
        for name, value in zip(names, values, strict=True)
    }
    if cpml is not None:
        key = f"cpml_psi_{prefix}_terms"
        updates[key] = _astype_terms(cpml, getattr(eng, key))
    return eng._replace(**updates)


def _replace_h(eng, hx, hy, hz, *, cpml_h=None):
    return _replace_phase(eng, "h", (hx, hy, hz), cpml_h)


def _replace_e(eng, ex, ey, ez, *, cpml_e=None):
    return _replace_phase(eng, "e", (ex, ey, ez), cpml_e)


def _component_tuple(owner, template):
    return tuple(getattr(owner, template.format(axis=axis)) for axis in "xyz")


def _electric_update_materials(coeffs, *, sources=False):
    off = coeffs.e_inverse_offdiagonal
    materials = {
        "conductivities": _component_tuple(coeffs, "e_conductivity_{axis}"),
        "inverse_permittivities": tuple(
            reciprocal_material(getattr(coeffs, f"e_permittivity_{axis}"))
            for axis in "xyz"
        ),
        "inverse_diagonals": (
            _component_tuple(coeffs, "e_inverse_diagonal_{axis}") if off.size else None
        ),
        "inverse_offdiagonal": off if off.size else None,
    }
    if sources:
        materials["source_scales"] = (
            _component_tuple(coeffs, "e_source_{axis}")
            if coeffs.e_source_x.size
            else None
        )
    return materials


def _boundary_views(eng, boundary):
    return build_h_boundary_views_for_e_3d(
        eng.hx,
        eng.hy,
        eng.hz,
        boundary.cpml.metallic_edges,
        logical_shapes=boundary.logical_component_shapes,
        periodic_axes=boundary.periodic_axes,
        periodic_phases=boundary.periodic_phases,
    )


def update_h_3d_cpml(eng, ctx, coeffs):
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
            _component_tuple(coeffs, "h_source_{axis}")
            if coeffs.h_source_x.size
            else None
        ),
        magnetic_conductivities=_component_tuple(coeffs, "h_sigma_m_{axis}"),
    )
    return _replace_h(eng, hx, hy, hz, cpml_h=psi_h)


def update_e_3d_cpml(eng, ctx, coeffs):
    cpml = ctx.boundary.cpml
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
        logical_shapes=ctx.boundary.logical_component_shapes,
        periodic_axes=ctx.boundary.periodic_axes,
        periodic_phases=ctx.boundary.periodic_phases,
        **_electric_update_materials(coeffs, sources=True),
    )
    return _replace_e(eng, ex, ey, ez, cpml_e=psi_e)


def update_h_3d_yee(eng, ctx, coeffs):
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


def _update_e_3d_yee(eng, ctx, coeffs, update, spacing):
    materials = _electric_update_materials(coeffs)
    conductivity = materials["conductivities"]
    inverse = materials["inverse_permittivities"]
    ex, ey, ez = update(
        eng.hx,
        eng.hy,
        eng.hz,
        eng.ex,
        eng.ey,
        eng.ez,
        conductivity[0],
        inverse[0],
        conductivity[1],
        inverse[1],
        conductivity[2],
        inverse[2],
        ctx.dt_scalar,
        spacing,
        boundary_views=_boundary_views(eng, ctx.boundary),
        inverse_diagonals=materials["inverse_diagonals"],
        inverse_offdiagonal=materials["inverse_offdiagonal"],
    )
    return _replace_e(eng, ex, ey, ez)


def update_e_3d_yee(eng, ctx, coeffs):
    return _update_e_3d_yee(
        eng, ctx, coeffs, fused_update_e_lossy_3d_material, ctx.resolution
    )


def _update_h_tm_from_curls(eng, ctx, coeffs, curl_tm_hx, curl_tm_hy, psi_h=None):
    hx = advance_h_from_coefficients(
        eng.hx, curl_tm_hx, coeffs.h_decay_x, coeffs.h_source_x
    )
    hy = advance_h_from_coefficients(
        eng.hy, curl_tm_hy, coeffs.h_decay_y, coeffs.h_source_y
    )
    return _replace_h(eng, hx, hy, eng.hz, cpml_h=psi_h)


def _update_e_tm_from_curl(eng, ctx, coeffs, curl_tm_ez, psi_e=None):
    ez = advance_e_from_coefficients(
        eng.ez, curl_tm_ez, coeffs.e_decay_z, coeffs.e_source_z
    )
    return _replace_e(eng, eng.ex, eng.ey, ez, cpml_e=psi_e)


def _update_h_te_from_curl(eng, coeffs, curl_hz, psi_h=None):
    hz = advance_h_from_coefficients(
        eng.hz, curl_hz, coeffs.h_decay_z, coeffs.h_source_z
    )
    return _replace_h(eng, eng.hx, eng.hy, hz, cpml_h=psi_h)


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


def _spacing(ctx, metric):
    return ctx.metrics if metric else ctx.resolution


def _run_tm_h(eng, ctx, coeffs, metric, cpml):
    spacing = _spacing(ctx, metric)
    if cpml:
        curl_hx, curl_hy, psi_h = (
            tm_xy_cpml_curl_e_to_h_2d_metric if metric else tm_xy_cpml_curl_e_to_h_2d
        )(
            eng.ez,
            spacing,
            terms=ctx.boundary.cpml.h_terms,
            psi_h_terms=eng.cpml_psi_h_terms,
        )
        return _update_h_tm_from_curls(eng, ctx, coeffs, curl_hx, curl_hy, psi_h=psi_h)
    if metric:
        curls = tm_xy_curl_e_to_h_2d_metric(eng.ez, spacing)
    else:
        curls = tm_xy_curl_e_to_h_2d(
            eng.ez, spacing, eng.hx.shape, eng.hy.shape, ctx.boundary.metallic_edges_2d
        )
    return _update_h_tm_from_curls(eng, ctx, coeffs, *curls)


def _run_tm_e(eng, ctx, coeffs, metric, cpml):
    spacing = _spacing(ctx, metric)
    boundary = ctx.boundary
    args = (
        eng.hx,
        eng.hy,
        spacing,
        eng.ez.shape,
        boundary.metallic_edges_2d,
        boundary.periodic_axes,
        boundary.periodic_phases,
    )
    if cpml:
        curl, psi_e = (
            tm_xy_cpml_curl_h_to_e_2d_metric if metric else tm_xy_cpml_curl_h_to_e_2d
        )(
            *args,
            terms=boundary.cpml.e_terms,
            psi_e_terms=eng.cpml_psi_e_terms,
        )
        return _update_e_tm_from_curl(eng, ctx, coeffs, curl, psi_e=psi_e)
    curl = (tm_xy_curl_h_to_e_2d_metric if metric else tm_xy_curl_h_to_e_2d)(*args)
    return _update_e_tm_from_curl(eng, ctx, coeffs, curl)


def _run_te_h(eng, ctx, coeffs, metric, cpml):
    spacing = _spacing(ctx, metric)
    if cpml:
        curl, psi_h = (
            te_xy_cpml_curl_e_to_h_2d_metric if metric else te_xy_cpml_curl_e_to_h_2d
        )(
            eng.ex,
            eng.ey,
            spacing,
            terms=ctx.boundary.cpml.h_terms,
            psi_h_terms=eng.cpml_psi_h_terms,
        )
        return _update_h_te_from_curl(eng, coeffs, curl, psi_h=psi_h)
    curl = (te_xy_curl_e_to_h_2d_metric if metric else te_xy_curl_e_to_h_2d)(
        eng.ex, eng.ey, spacing, eng.hz.shape
    )
    return _update_h_te_from_curl(eng, coeffs, curl)


def _run_te_e(eng, ctx, coeffs, metric, cpml):
    spacing = _spacing(ctx, metric)
    boundary = ctx.boundary
    if cpml:
        curls, psi_e = (
            te_xy_cpml_curl_h_to_e_2d_metric if metric else te_xy_cpml_curl_h_to_e_2d
        )(
            eng.hz,
            spacing,
            boundary.metallic_edges_2d,
            boundary.periodic_axes,
            boundary.periodic_phases,
            terms=boundary.cpml.e_terms,
            psi_e_terms=eng.cpml_psi_e_terms,
        )
        return _update_e_te_from_curls(eng, ctx, coeffs, curls, psi_e=psi_e)
    curls = (te_xy_curl_h_to_e_2d_metric if metric else te_xy_curl_h_to_e_2d)(
        eng.hz,
        spacing,
        eng.ex.shape,
        eng.ey.shape,
        boundary.metallic_edges_2d,
        boundary.periodic_axes,
        boundary.periodic_phases,
    )
    return _update_e_te_from_curls(eng, ctx, coeffs, curls)


def _make_2d_update(name, runner, metric, cpml):
    # metric and cpml are Python constants, so each kernel traces one branch.
    def update(eng, ctx, coeffs):
        return runner(eng, ctx, coeffs, metric, cpml)

    update.__name__ = name
    update.__qualname__ = name
    return update


update_h_2d_tm_xy = _make_2d_update("update_h_2d_tm_xy", _run_tm_h, False, False)
update_h_2d_tm_xy_cpml = _make_2d_update(
    "update_h_2d_tm_xy_cpml", _run_tm_h, False, True
)
update_e_2d_tm_xy = _make_2d_update("update_e_2d_tm_xy", _run_tm_e, False, False)
update_e_2d_tm_xy_cpml = _make_2d_update(
    "update_e_2d_tm_xy_cpml", _run_tm_e, False, True
)
update_h_2d_te_xy = _make_2d_update("update_h_2d_te_xy", _run_te_h, False, False)
update_h_2d_te_xy_cpml = _make_2d_update(
    "update_h_2d_te_xy_cpml", _run_te_h, False, True
)
update_e_2d_te_xy = _make_2d_update("update_e_2d_te_xy", _run_te_e, False, False)
update_e_2d_te_xy_cpml = _make_2d_update(
    "update_e_2d_te_xy_cpml", _run_te_e, False, True
)


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
        magnetic_conductivities=_component_tuple(coeffs, "h_sigma_m_{axis}"),
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
        logical_shapes=ctx.boundary.logical_component_shapes,
        periodic_axes=ctx.boundary.periodic_axes,
        periodic_phases=ctx.boundary.periodic_phases,
        **_electric_update_materials(coeffs),
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
    return _update_e_3d_yee(
        eng, ctx, coeffs, fused_update_e_lossy_3d_material_metric, ctx.metrics
    )


update_h_2d_tm_xy_metric = _make_2d_update(
    "update_h_2d_tm_xy_metric", _run_tm_h, True, False
)
update_e_2d_tm_xy_metric = _make_2d_update(
    "update_e_2d_tm_xy_metric", _run_tm_e, True, False
)
update_h_2d_tm_xy_cpml_metric = _make_2d_update(
    "update_h_2d_tm_xy_cpml_metric", _run_tm_h, True, True
)
update_e_2d_tm_xy_cpml_metric = _make_2d_update(
    "update_e_2d_tm_xy_cpml_metric", _run_tm_e, True, True
)
update_h_2d_te_xy_metric = _make_2d_update(
    "update_h_2d_te_xy_metric", _run_te_h, True, False
)
update_e_2d_te_xy_metric = _make_2d_update(
    "update_e_2d_te_xy_metric", _run_te_e, True, False
)
update_h_2d_te_xy_cpml_metric = _make_2d_update(
    "update_h_2d_te_xy_cpml_metric", _run_te_h, True, True
)
update_e_2d_te_xy_cpml_metric = _make_2d_update(
    "update_e_2d_te_xy_cpml_metric", _run_te_e, True, True
)


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

    # 1. Describe variants in priority order: stateful boundaries must precede generic
    # dimensional kernels or their auxiliary state would be ignored.
    boundary = ctx.boundary
    metric = ctx.config.metric_kind != "isotropic_uniform"
    variants = (
        (
            ctx.is_3d and metric and boundary.cpml.enabled,
            f"{ctx.config.metric_kind}_cpml_3d",
            update_h_3d_cpml_metric,
            update_e_3d_cpml_metric,
        ),  # fmt: skip
        (
            ctx.is_3d and metric,
            f"{ctx.config.metric_kind}_yee_3d",
            update_h_3d_yee_metric,
            update_e_3d_yee_metric,
        ),  # fmt: skip
        (
            ctx.is_3d and boundary.cpml.enabled,
            "cpml_3d",
            update_h_3d_cpml,
            update_e_3d_cpml,
        ),  # fmt: skip
        (ctx.is_3d, "yee_3d", update_h_3d_yee, update_e_3d_yee),
        (
            (not ctx.is_3d)
            and metric
            and ctx.config.polarization_2d == "tm"
            and boundary.cpml.enabled,
            f"{ctx.config.metric_kind}_physical_tm_xy_cpml",
            update_h_2d_tm_xy_cpml_metric,
            update_e_2d_tm_xy_cpml_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and metric and ctx.config.polarization_2d == "tm",
            f"{ctx.config.metric_kind}_physical_tm_xy",
            update_h_2d_tm_xy_metric,
            update_e_2d_tm_xy_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d)
            and ctx.config.polarization_2d == "tm"
            and boundary.cpml.enabled,
            "physical_tm_xy_cpml",
            update_h_2d_tm_xy_cpml,
            update_e_2d_tm_xy_cpml,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and ctx.config.polarization_2d == "tm",
            "physical_tm_xy",
            update_h_2d_tm_xy,
            update_e_2d_tm_xy,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and metric and boundary.cpml.enabled,
            f"{ctx.config.metric_kind}_physical_te_xy_cpml",
            update_h_2d_te_xy_cpml_metric,
            update_e_2d_te_xy_cpml_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and metric,
            f"{ctx.config.metric_kind}_physical_te_xy",
            update_h_2d_te_xy_metric,
            update_e_2d_te_xy_metric,
        ),  # fmt: skip
        (
            (not ctx.is_3d) and boundary.cpml.enabled,
            "physical_te_xy_cpml",
            update_h_2d_te_xy_cpml,
            update_e_2d_te_xy_cpml,
        ),  # fmt: skip
        (
            not ctx.is_3d,
            "physical_te_xy",
            update_h_2d_te_xy,
            update_e_2d_te_xy,
        ),  # fmt: skip
    )

    # 2. Every supported runtime is canonical 3D, TMxy, or TExy.
    _, kind, update_h, update_e = next(variant for variant in variants if variant[0])
    return StepUpdateKernel(kind, update_h, update_e)
