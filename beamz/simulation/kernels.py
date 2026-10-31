"""Canonical Yee update kernels and static kernel selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, cast

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.constitutive import advance_e_tensor_dense
from beamz.constitutive import collocate_yee_component as collocate_yee_component
from beamz.constitutive import fit_array_to_shape as fit_array_to_shape
from beamz.lattice import (
    CURL_TERMS,
    _pad_with_boundary_ghosts,
    component_axis_offsets_3d,
)
from beamz.lattice import (
    adjacent_difference as _adjacent_difference,
)
from beamz.lattice import (
    build_h_boundary_views_for_e_3d as build_h_boundary_views_for_e_3d,
)
from beamz.lattice import (
    metric_adjacent_difference as _metric_difference,
)
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


def build_phase_update(ctx, *, phase):
    """Specialize component, derivative and boundary choices before tracing.

    Descriptors are ordinary Python tuples: no dynamic component interpreter is
    present in the JAX graph. Packed CPML term order follows target component,
    then positive/negative curl derivative, as in boundary compilation.
    """
    is_h = phase == "h"
    family = "3d" if ctx.is_3d else ctx.config.polarization_2d
    active = {"3d": ("xyz", "xyz"), "tm": ("xy", "z"), "te": ("z", "xy")}
    targets = active[family][0 if is_h else 1]
    other = "e" if is_h else "h"
    metric = ctx.config.metric_kind != "isotropic_uniform"
    cpml = ctx.boundary.cpml.enabled
    # z derivatives vanish in a 2D plane. Array axes are zyx in 3D, yx in 2D.
    descriptors = tuple(
        (target, source, axis if ctx.is_3d else axis - 1, sign)
        for target in targets
        for source, axis, sign in CURL_TERMS["xyz".index(target)]
        if ctx.is_3d or axis != 0
    )
    edges = (
        ctx.boundary.cpml.metallic_edges
        if ctx.is_3d
        else ctx.boundary.metallic_edges_2d
    )
    periodic_axes = ctx.boundary.periodic_axes
    phases = ctx.boundary.periodic_phases
    terms = ctx.boundary.cpml.h_terms if is_h else ctx.boundary.cpml.e_terms
    differences = []
    for _target, _source, axis, _sign in descriptors:
        if metric:
            inverse = getattr(
                ctx.metrics,
                ("e_to_h_" if is_h else "h_to_e_")
                + "zyx"[axis if ctx.is_3d else axis + 1],
            )
            difference = partial(
                _metric_difference, axis=axis, inverse_distance=inverse
            )
        else:
            difference = partial(
                _adjacent_difference, axis=axis, resolution=ctx.resolution
            )
        differences.append(difference)

    def update(state, context, coefficients):
        curls = {
            target: jnp.zeros_like(getattr(state, phase + target)) for target in targets
        }
        memories = getattr(state, f"cpml_psi_{phase}_terms")
        next_memories = []
        for index, ((target, source, axis, sign), difference) in enumerate(
            zip(descriptors, differences, strict=True)
        ):
            field = getattr(state, other + source)
            shape = getattr(state, phase + target).shape
            if not is_h:
                if ctx.is_3d:
                    field = _pad_with_boundary_ghosts(
                        field,
                        axis,
                        edges,
                        logical_size=ctx.boundary.logical_component_shapes[
                            other.upper() + source
                        ][axis],
                        periodic=axis in periodic_axes,
                        periodic_phase=phases[axis] if phases else 1.0,
                    )
                else:
                    field = _pad_h_axis(field, axis, edges, periodic_axes, phases)
            derivative = fit_array_to_shape(difference(field), shape)
            if cpml:
                derivative, memory = correct_cpml_term(
                    derivative, memories[index], terms[index]
                )
                next_memories.append(memory)
            else:
                derivative = sign * derivative
            curls[target] = curls[target] + derivative
        fields = tuple(getattr(state, phase + target) for target in targets)
        curl_values = tuple(curls[target] for target in targets)
        tensor = not is_h and coefficients.e_inverse_offdiagonal.size
        if tensor:
            updated = advance_e_centered_tensor(
                fields,
                curl_values,
                tuple(
                    getattr(coefficients, f"e_inverse_diagonal_{target}")
                    for target in targets
                ),
                coefficients.e_inverse_offdiagonal,
                tuple("E" + target for target in targets),
                context.dt_scalar,
                cell_widths=constitutive_cell_widths(context.metrics),
            )
        elif not ctx.is_3d:
            advance = (
                advance_h_from_coefficients if is_h else advance_e_from_coefficients
            )
            updated = tuple(
                advance(
                    field,
                    curl,
                    getattr(coefficients, f"{phase}_decay_{target}"),
                    getattr(coefficients, f"{phase}_source_{target}"),
                )
                for target, field, curl in zip(
                    targets, fields, curl_values, strict=True
                )
            )
        else:
            updated = []
            unit = MU_0 if is_h else EPS_0
            scale = jnp.asarray(context.dt_scalar, dtype=fields[0].dtype) / jnp.asarray(
                unit, dtype=fields[0].dtype
            )
            for target, field, curl in zip(targets, fields, curl_values, strict=True):
                sigma = getattr(
                    coefficients,
                    f"{phase}_{'sigma_m' if is_h else 'conductivity'}_{target}",
                )
                source = getattr(coefficients, f"{phase}_source_{target}")
                if cpml and not metric and source.size:
                    value = field + material_values(source) * (
                        (-curl if is_h else curl) - sigma * field
                    )
                else:
                    inverse = (
                        1
                        if is_h
                        else material_values(
                            reciprocal_material(
                                getattr(coefficients, f"e_permittivity_{target}")
                            )
                        )
                    )
                    alpha = sigma * (0.5 * scale) * inverse
                    signed = -curl if is_h else curl
                    value = ((1 - alpha) * field + scale * inverse * signed) / (
                        1 + alpha
                    )
                updated.append(value)
        replacements = {
            phase + target: value.astype(field.dtype)
            for target, field, value in zip(targets, fields, updated, strict=True)
        }
        if cpml:
            replacements[f"cpml_psi_{phase}_terms"] = _astype_terms(
                next_memories, memories
            )
        return state._replace(**replacements)

    return update


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
    update_h = build_phase_update(ctx, phase="h")
    update_e = build_phase_update(ctx, phase="e")
    kind = "yee_3d" if ctx.is_3d else f"physical_{family}_xy"
    if boundary.cpml.enabled:
        kind = "cpml_3d" if ctx.is_3d else kind + "_cpml"
    if metric:
        kind = f"{ctx.config.metric_kind}_{kind}"
    return StepUpdateKernel(kind, update_h, update_e)
