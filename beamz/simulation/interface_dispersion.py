"""Constituent-field realization of the harmonic dispersive normal response.

Normal constituents share D; their fraction-weighted electric fields sum to E.
Eliminating constituent E at each trapezoidal step avoids fitted effective poles.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


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


def compile_interfaces(grid, dt, periodic_axes):
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
                shape=(len(pole_lists), max_poles, len(indices)),
            )
        )
    return tuple(plans)


def interface_history(plan, q, old_e):
    e = old_e.reshape(-1)[plan.indices]
    polarization = 2 * jnp.real(jnp.sum(q, axis=1))
    normal_p = plan.harmonic_inf * jnp.sum(
        plan.fractions * polarization / plan.epsilon, axis=0
    )
    normal_d = plan.harmonic_inf * e + normal_p
    local_old = (normal_d - polarization) / plan.epsilon
    h = 2 * jnp.real(jnp.sum(plan.a * q, axis=1)) + plan.response * local_old
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
