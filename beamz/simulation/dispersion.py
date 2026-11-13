"""Coupled trapezoidal ADE updates for scalar pole-residue media."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.simulation.interface_dispersion import (
    advance_interface,
    compile_interfaces,
    interface_history,
)


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
    anisotropic: Any = None

    @property
    def scalar_state_count(self) -> int:
        return len(self.regions) + len(self.interfaces)


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
    from beamz.simulation.anisotropic import compile_anisotropic

    return DispersionPlan(
        tuple(regions),
        tuple(coefficients),
        interfaces,
        complex_fields,
        compile_anisotropic(grid, dt, periodic_axes),
    )


def initial_polarization(plan, continuation=None):
    """Allocate auxiliary media state: scalar ADE states, then tensor poles."""
    regions = (*plan.regions, *plan.interfaces)
    fresh = tuple(jnp.zeros(r.shape, dtype=jnp.complex64) for r in regions)
    if plan.anisotropic is not None:
        from beamz.simulation.anisotropic import initial_states

        fresh += initial_states(
            plan.anisotropic, jnp.complex64 if plan.complex_fields else jnp.float32
        )
    if continuation is not None and continuation.polarization:
        old = continuation.polarization
        if len(old) != len(fresh) or any(
            q.shape != f.shape for q, f in zip(old, fresh, strict=True)
        ):
            raise ValueError(
                "Continuation polarization does not match dispersive material layout."
            )
        return old
    return fresh


def _update_tensor_states(update, old, state, plan, native):
    """Run ``update`` on the tensor-pole states that follow the scalar ones."""
    head = state.polarization[: plan.scalar_state_count]
    tail = state.polarization[plan.scalar_state_count :]
    state, tail = update(old, state, plan.anisotropic, tail, native)
    return state._replace(polarization=head + tuple(tail))


def update_anisotropic_magnetic(old, state, plan, native=False):
    """Apply tensor permeability and magnetic poles after the H kernel."""
    from beamz.simulation.anisotropic import update_magnetic

    return _update_tensor_states(update_magnetic, old, state, plan, native)


def update_anisotropic_electric(old, state, plan, native=False):
    """Apply electric tensor poles after the E kernel and its sources."""
    from beamz.simulation.anisotropic import update_electric

    return _update_tensor_states(update_electric, old, state, plan, native)


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
    # Tensor-pole states (if any) follow the scalar states and advance elsewhere.
    polarization += tuple(state.polarization[plan.scalar_state_count :])
    return state._replace(**fields, polarization=polarization)
