"""CPML coefficient and packed-slab updates."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.simulation.model import CpmlPackedSlabSpec, CpmlTerm


def cpml_coefficients(sigma, kappa, alpha, dt):
    """Precompute one CPML recurrence as ``a``, ``b``, and inverse-kappa."""
    xp = np if all(isinstance(v, np.ndarray) for v in (sigma, kappa, alpha)) else jnp
    dtype = jnp.result_type(sigma, kappa, alpha, jnp.float32)
    sigma = xp.asarray(sigma, dtype=dtype)
    kappa = xp.maximum(xp.asarray(kappa, dtype=dtype), 1.0)
    alpha = xp.asarray(alpha, dtype=dtype)
    one = xp.asarray(1.0, dtype=dtype)
    decay = (sigma / kappa + alpha) * (
        xp.asarray(dt, dtype=dtype) / xp.asarray(EPS_0, dtype=dtype)
    )
    b = xp.exp(-decay)
    denominator = (sigma + kappa * alpha) * kappa
    # Preserve weak damping when exp(-decay) rounds to one; vacuum has a=0.
    a = xp.expm1(-decay) * (sigma / xp.where(denominator > 0, denominator, one))
    return a, b, one / kappa


def _axis_region(ndim, axis, start, stop):
    region = [slice(None)] * ndim
    region[axis] = slice(start, stop)
    return tuple(region)


def _pack_cpml_slab(arr, slab):
    """Gather the active low/high edge regions of one derivative."""
    xp = np if isinstance(arr, np.ndarray) else jnp
    axis, low, high = int(slab.axis), int(slab.low), int(slab.high)
    if low <= 0 and high <= 0:
        return xp.zeros(slab.shape, dtype=arr.dtype)
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
    return parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=axis)


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
