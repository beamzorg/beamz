"""Coupled trapezoidal ADE updates for scalar pole-residue media."""

from __future__ import annotations

from dataclasses import dataclass

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


def compile_dispersion(grid, dt, periodic_axes=frozenset()):
    regions = []
    alpha = {}
    for medium, supports in grid.material_grid.dispersion:
        poles = np.asarray(medium.poles, dtype=complex)
        a = (2 + dt * poles[:, 0]) / (2 - dt * poles[:, 0])
        b = dt * poles[:, 1] / (2 - dt * poles[:, 0])
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
                    (len(poles), *cropped.shape),
                )
            )
            if component not in alpha:
                alpha[component] = np.zeros(weight.shape, dtype=np.float64)
            alpha[component][slices] += 2 * b.real.sum() * cropped
    interfaces = compile_interfaces(grid, dt, periodic_axes)
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
    return DispersionPlan(tuple(regions), tuple(coefficients), interfaces)


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
        change = 2 * jnp.real(jnp.sum((region.a - 1) * q, axis=0))
        histories[region.component] = (
            histories[region.component].at[region.slices].add(change)
        )
    contexts = []
    for interface, q in zip(plan.interfaces, interface_states, strict=True):
        history, context = interface_history(
            interface, q, getattr(old, interface.component)
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
