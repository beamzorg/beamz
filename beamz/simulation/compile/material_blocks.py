"""Exact, differentiable constant-block material storage for JAX propagation."""

from dataclasses import dataclass, replace
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0

BLOCK_SIZE = 32


def block_pool_indices(headers, size):
    """Map sample indices to a 32-wide constant-block pool."""
    linear = jnp.arange(size, dtype=jnp.int32)
    header = jnp.asarray(headers).reshape(-1)[linear >> 5]
    return jnp.where(header < 0, ~header, header + (linear & 31))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaterialBlocks:
    """A value pool and fixed block map on one logical Yee support.

    Uint8 headers are per-sample IDs into a 256-entry pool. Otherwise negative
    int32 headers reference one constant via ``~header`` and other headers
    address 32 literal values. Values are differentiable; integer topology is
    fixed. Recompile or supply dense arrays to untie shared coefficients.
    """

    values: jax.Array | np.ndarray
    headers: jax.Array | np.ndarray
    shape: tuple[int, ...]

    @property
    def size(self):
        return prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def nbytes(self):
        return self.values.nbytes + self.headers.nbytes

    def tree_flatten(self):
        return (self.values, self.headers), self.shape

    @classmethod
    def tree_unflatten(cls, shape, leaves):
        values, headers = leaves
        return cls(values, headers, shape)

    def dense(self):
        """Gather at use sites so XLA can fuse reads into the field update."""
        if self.headers.dtype == np.uint8:
            return jnp.asarray(self.values)[jnp.asarray(self.headers)].reshape(
                self.shape
            )
        index = block_pool_indices(self.headers, self.size)
        return jnp.asarray(self.values)[index].reshape(self.shape)


def material_values(value):
    return value.dense() if isinstance(value, MaterialBlocks) else value


def reciprocal_material(value):
    if isinstance(value, MaterialBlocks):
        return replace(value, values=jnp.reciprocal(value.values))
    return jnp.reciprocal(value)


def compact_material(value):
    """Keep exact scalars/blocks only when they save at least 25% of storage."""
    array = np.asarray(value)
    if (
        array.ndim != 3
        or not array.size
        or array.dtype.kind != "f"
        or array.dtype.itemsize not in (4, 8)
        or not np.isfinite(array).all()
    ):
        return value
    bits = array.ravel().view(np.uint32 if array.dtype.itemsize == 4 else np.uint64)
    if np.all(bits == bits[0]):
        return array.ravel()[0].copy()
    # A byte map uses one gather instead of two dependent gathers. Padding the
    # table to 256 entries proves every uint8 index is in bounds to XLA.
    table = np.empty(0, dtype=bits.dtype)
    chunk = 1 << 20
    for start in range(0, bits.size, chunk):
        table = np.union1d(table, np.unique(bits[start : start + chunk]))
        if table.size > 256:
            break
    if table.size <= 256 and bits.size + 256 * array.itemsize <= 0.75 * array.nbytes:
        codes = np.empty(bits.size, dtype=np.uint8)
        for start in range(0, bits.size, chunk):
            codes[start : start + chunk] = np.searchsorted(
                table, bits[start : start + chunk]
            )
        pool = np.pad(table, (0, 256 - table.size), mode="edge").view(array.dtype)
        return MaterialBlocks(pool, codes, array.shape)
    pages = np.pad(bits, (0, -bits.size % BLOCK_SIZE), mode="edge").reshape(
        -1, BLOCK_SIZE
    )
    constant = np.all(pages == pages[:, :1], axis=1)
    constants = np.unique(pages[constant, 0])
    mixed_count = np.count_nonzero(~constant)
    pool_size = constants.size + BLOCK_SIZE * mixed_count
    if (
        pool_size * array.itemsize + constant.size * 4 > 0.75 * array.nbytes
        or pool_size >= np.iinfo(np.int32).max
        or array.size >= np.iinfo(np.int32).max
    ):
        return value
    pool = np.concatenate((constants, pages[~constant].ravel())).view(array.dtype)
    headers = np.empty(constant.size, np.int32)
    headers[constant] = ~np.searchsorted(constants, pages[constant, 0]).astype(np.int32)
    headers[~constant] = constants.size + BLOCK_SIZE * np.arange(
        mixed_count, dtype=np.int32
    )
    return MaterialBlocks(pool, headers, array.shape)


@dataclass(frozen=True)
class PolarizationRegion:
    component: str
    slices: tuple[slice, ...]
    a: object
    b: object
    weights: object
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DispersionPlan:
    regions: tuple[PolarizationRegion, ...]
    coefficients: tuple[tuple[str, object, object, object], ...]
    interfaces: tuple = ()
    complex_fields: bool = False


def periodic_support_average(value, cell_shape, periodic_axes, geometry=None):
    """Join the two half-support volumes at each periodic electric seam."""
    value = np.array(value, copy=True)
    if value.ndim != len(cell_shape):
        return value
    for axis in sorted(periodic_axes):
        if value.shape[axis] != cell_shape[axis] + 1:
            continue
        low: list[slice | int] = [slice(None)] * value.ndim
        high = low.copy()
        low[axis], high[axis] = 0, -1
        first, last = 1.0, 1.0
        if geometry is not None:
            physical_axis = ("zyx" if len(cell_shape) == 3 else "yx")[axis]
            widths = geometry.cell_widths(physical_axis)
            first, last = float(widths[0]), float(widths[-1])
        mean = (first * value[tuple(low)] + last * value[tuple(high)]) / (first + last)
        value[tuple(low)] = value[tuple(high)] = mean
    return value


def compile_dispersion(grid, dt, periodic_axes=frozenset(), *, complex_fields=False):
    regions = []
    alpha = {}
    for medium, supports in grid.material_grid.dispersion:
        poles = np.asarray(medium.poles, dtype=complex)
        a = (2 + dt * poles[:, 0]) / (2 - dt * poles[:, 0])
        b = dt * poles[:, 1] / (2 - dt * poles[:, 0])
        response = 2 * b.real.sum()
        # With complex E, conjugate poles have independent states: conjugating q
        # would also conjugate the driving field and discard its second quadrature.
        if complex_fields:
            a = np.concatenate((a, a.conj()))
            b = np.concatenate((b, b.conj()))
        for component, weight in supports.items():
            weight = periodic_support_average(
                weight, grid.material_grid.shape, periodic_axes, grid.geometry
            )
            locations = np.nonzero(weight > 1e-7)
            if not locations[0].size:
                continue
            slices = tuple(slice(int(x.min()), int(x.max()) + 1) for x in locations)
            cropped = np.asarray(weight[slices], dtype=np.float32)
            reshape = (-1,) + (1,) * cropped.ndim
            regions.append(
                PolarizationRegion(
                    component.lower(),
                    slices,
                    jnp.asarray(a.reshape(reshape), dtype=jnp.complex64),
                    jnp.asarray(b.reshape(reshape), dtype=jnp.complex64),
                    jnp.asarray(cropped),
                    (len(a), *cropped.shape),
                )
            )
            if component not in alpha:
                alpha[component] = np.zeros(weight.shape, dtype=np.float64)
            alpha[component][slices] += response * cropped
    interfaces = compile_interfaces(
        grid, dt, periodic_axes, complex_fields=complex_fields
    )
    for interface in interfaces:
        name = interface.component.capitalize()
        if name not in alpha:
            alpha[name] = np.zeros(grid.component_shapes[name], dtype=float)
        np.add.at(
            alpha[name].reshape(-1),
            np.asarray(interface.indices),
            np.asarray(interface.delta),
        )
    coefficients = []
    for component, response in alpha.items():
        axis = component[-1].lower()
        base = np.asarray(getattr(grid, f"eps_{axis}")) + dt / (2 * EPS_0) * np.asarray(
            getattr(grid, f"sig_{axis}")
        )
        denominator = base + response
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError(
                "Dispersive update has nonpositive denominator; reduce timestep or refit medium."
            )
        coefficients.append(
            (
                component.lower(),
                jnp.asarray(base / denominator, dtype=jnp.float32),
                jnp.asarray(1 / denominator, dtype=jnp.float32),
                jnp.asarray(response, dtype=jnp.float32),
            )
        )
    return DispersionPlan(
        tuple(regions), tuple(coefficients), interfaces, complex_fields
    )


def initial_polarization(plan, continuation=None):
    regions = (*plan.regions, *plan.interfaces)
    if continuation is not None and continuation.polarization:
        old = continuation.polarization
        if len(old) != len(regions) or any(
            q.shape != region.shape for q, region in zip(old, regions, strict=True)
        ):
            raise ValueError(
                "Continuation polarization does not match dispersive material layout."
            )
        return old
    return tuple(jnp.zeros(r.shape, dtype=jnp.complex64) for r in regions)


def update_dispersion(old, state, plan):
    if not plan.regions and not plan.interfaces:
        return state
    histories = {
        name: jnp.zeros_like(getattr(state, name)) for name, *_ in plan.coefficients
    }
    bulk_states = state.polarization[: len(plan.regions)]
    interface_states = state.polarization[len(plan.regions) :]
    for region, q in zip(plan.regions, bulk_states, strict=True):
        total = jnp.sum((region.a - 1) * q, axis=0)
        change = total if plan.complex_fields else 2 * jnp.real(total)
        histories[region.component] = (
            histories[region.component].at[region.slices].add(change)
        )
    contexts = []
    for interface, q in zip(plan.interfaces, interface_states, strict=True):
        history, context = interface_history(
            interface,
            q,
            getattr(old, interface.component),
            complex_fields=plan.complex_fields,
        )
        contexts.append(context)
        value = histories[interface.component]
        histories[interface.component] = (
            value.reshape(-1).at[interface.indices].add(history).reshape(value.shape)
        )
    fields = {}
    for name, ratio, inverse, response in plan.coefficients:
        fields[name] = ratio * getattr(state, name) - inverse * (
            histories[name] + response * getattr(old, name)
        )
    polarization = tuple(
        region.a * q
        + region.b
        * region.weights
        * (
            getattr(old, region.component)[region.slices]
            + fields[region.component][region.slices]
        )
        for region, q in zip(plan.regions, bulk_states, strict=True)
    )
    polarization += tuple(
        advance_interface(interface, q, context, fields[interface.component])
        for interface, q, context in zip(
            plan.interfaces, interface_states, contexts, strict=True
        )
    )
    return state._replace(**fields, polarization=polarization)


@dataclass(frozen=True)
class InterfacePlan:
    component: str
    indices: object
    fractions: object
    weight: object
    epsilon: object
    a: object
    b: object
    response: object
    harmonic_inf: object
    harmonic_step: object
    delta: object
    shape: tuple[int, ...]


def _periodic_copies(interface, shape, periodic_axes, geometry):
    indices = np.asarray(interface.indices)
    fractions = np.asarray(interface.fractions)
    weights = np.asarray(interface.normal_squared)
    for axis in sorted(periodic_axes):
        if shape[axis] != geometry.shape_zyx[-len(shape) :][axis] + 1:
            continue
        coordinates = np.array(np.unravel_index(indices, shape))
        first = coordinates[axis] == 0
        last = coordinates[axis] == shape[axis] - 1
        seam = first | last
        if not np.any(seam):
            continue
        physical_axis = ("zyx" if len(shape) == 3 else "yx")[axis]
        widths = geometry.cell_widths(physical_axis)
        factors = np.where(first, widths[0], widths[-1]) / (widths[0] + widths[-1])
        weights = weights * np.where(seam, factors, 1)
        mirrors = coordinates[:, seam].copy()
        mirrors[axis] = shape[axis] - 1 - mirrors[axis]
        indices = np.r_[indices, np.ravel_multi_index(mirrors, shape)]
        fractions = np.concatenate((fractions, fractions[:, seam]), axis=1)
        weights = np.r_[weights, weights[seam]]
    return indices, fractions, weights


def compile_interfaces(grid, dt, periodic_axes, *, complex_fields=False):
    plans = []
    for interface in grid.material_grid.dispersion_interfaces:
        component = interface.component.lower()
        shape = grid.component_shapes[interface.component]
        indices, fractions, weights = _periodic_copies(
            interface, shape, periodic_axes, grid.geometry
        )
        pole_lists: list[Any] = [getattr(m, "poles", ()) for m in interface.materials]
        max_poles = max(map(len, pole_lists))
        a = np.zeros((len(pole_lists), max_poles, 1), dtype=np.complex128)
        b = np.zeros_like(a)
        for j, poles in enumerate(pole_lists):
            for k, (pole, residue) in enumerate(poles):
                a[j, k, 0] = (2 + dt * pole) / (2 - dt * pole)
                b[j, k, 0] = dt * residue / (2 - dt * pole)
        epsilon = np.array([float(m.permittivity) for m in interface.materials])[
            :, None
        ]
        response = 2 * b.real.sum(axis=1)
        if complex_fields:
            a = np.concatenate((a, a.conj()), axis=1)
            b = np.concatenate((b, b.conj()), axis=1)
        if np.any(epsilon + response <= 0):
            raise ValueError(
                "Dispersive interface has a nonpositive constituent step denominator."
            )
        harmonic_inf = 1 / np.sum(fractions / epsilon, axis=0)
        harmonic_step = 1 / np.sum(fractions / (epsilon + response), axis=0)
        delta = weights * (harmonic_step - harmonic_inf)
        plans.append(
            InterfacePlan(
                component=component,
                indices=jnp.asarray(indices, dtype=jnp.int32),
                fractions=jnp.asarray(fractions, dtype=jnp.float32),
                weight=jnp.asarray(weights, dtype=jnp.float32),
                epsilon=jnp.asarray(epsilon, dtype=jnp.float32),
                a=jnp.asarray(a, dtype=jnp.complex64),
                b=jnp.asarray(b, dtype=jnp.complex64),
                response=jnp.asarray(response, dtype=jnp.float32),
                harmonic_inf=jnp.asarray(harmonic_inf, dtype=jnp.float32),
                harmonic_step=jnp.asarray(harmonic_step, dtype=jnp.float32),
                delta=jnp.asarray(delta, dtype=jnp.float32),
                shape=(len(pole_lists), a.shape[1], len(indices)),
            )
        )
    return tuple(plans)


def interface_history(plan, q, old_e, *, complex_fields=False):
    e = old_e.reshape(-1)[plan.indices]
    total = jnp.sum(q, axis=1)
    polarization = total if complex_fields else 2 * jnp.real(total)
    normal_p = plan.harmonic_inf * jnp.sum(
        plan.fractions * polarization / plan.epsilon, axis=0
    )
    normal_d = plan.harmonic_inf * e + normal_p
    local_old = (normal_d - polarization) / plan.epsilon
    total = jnp.sum(plan.a * q, axis=1)
    h = (total if complex_fields else 2 * jnp.real(total)) + plan.response * local_old
    harmonic_h = plan.harmonic_step * jnp.sum(
        plan.fractions * h / (plan.epsilon + plan.response), axis=0
    )
    history = plan.weight * (harmonic_h - normal_p) - plan.delta * e
    return history, (local_old, h, harmonic_h)


def advance_interface(plan, q, context, new_e):
    local_old, h, harmonic_h = context
    e = new_e.reshape(-1)[plan.indices]
    normal_d = plan.harmonic_step * e + harmonic_h
    local_new = (normal_d - h) / (plan.epsilon + plan.response)
    return plan.a * q + plan.b * (local_old + local_new)[:, None, :]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TensorRegion:
    """Shared-node coefficients on their fixed active bounding box.

    Bounds are static compilation metadata; values remain dynamic JAX leaves.
    The kernel adds an immutable zero halo for exact cropped averaging.
    """

    values: jax.Array | np.ndarray
    bounds: tuple[tuple[int, int], ...]
    node_shape: tuple[int, ...]

    @property
    def size(self):
        return self.values.size

    def tree_flatten(self):
        return (self.values,), (self.bounds, self.node_shape)

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        return cls(leaves[0], *metadata)


def compact_tensor_region(values):
    """Compress only when the halo-inclusive bounding box is substantially smaller.

    Exact zeros determine support: small physical cross coefficients are retained.
    Dense media retain the existing array representation and update path.
    """
    values = np.asarray(values)
    active = np.any(values != 0, axis=-1)
    if not np.any(active):
        return values
    bounds = []
    for axis in range(active.ndim):
        occupied = np.flatnonzero(
            np.any(active, axis=tuple(i for i in range(active.ndim) if i != axis))
        )
        bounds.append((int(occupied[0]), int(occupied[-1]) + 1))
    volume = np.prod(
        [
            min(count, stop + 1) - max(0, start - 1)
            for (start, stop), count in zip(bounds, active.shape, strict=True)
        ]
    )
    if volume > 0.5 * active.size:
        return values
    region = tuple(slice(start, stop) for start, stop in bounds)
    return TensorRegion(values[region].copy(), tuple(bounds), active.shape)


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
    from beamz.simulation.model import CpmlPackedSlabSpec, CpmlTerm

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
