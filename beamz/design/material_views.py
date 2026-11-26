"""Frequency-domain views of a material grid for modal solves.

Mode solvers need static coefficients. For grids with tensor poles, these
views fold ``Re chi(frequency)`` into every constitutive support so a CW mode
solved at that frequency sees the medium the time stepper propagates.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from beamz.constitutive import collocate_yee_component, expand_packed
from beamz.lattice import component_shapes


def _axis_index(component: str) -> int:
    return "xyz".index(component[-1].lower())


def _cell_susceptibility(material_grid, frequency) -> np.ndarray:
    """Volume-weighted real tensor susceptibility of all tensor poles, per cell."""
    from beamz.design.anisotropic import AnisotropicMedium, GyrotropicPole

    total = np.zeros((6, *material_grid.shape))
    for medium, fraction in material_grid.tensor_dispersion:
        assert isinstance(medium, AnisotropicMedium)
        if medium.magnetic_poles:
            raise ValueError(
                "Mode solves cannot yet include magnetic tensor poles; keep mode "
                "ports outside media with dispersive permeability."
            )
        if any(isinstance(p, GyrotropicPole) for p in medium.electric_poles):
            raise ValueError(
                "Mode solves cannot yet include gyrotropic (non-reciprocal) media; "
                "keep mode ports outside them."
            )
        chi = np.zeros((3, 3), dtype=complex)
        for pole in medium.electric_poles:
            chi = chi + np.asarray(pole.susceptibility(frequency))
        chi = np.real(chi)
        packed = np.array(
            [chi[0, 0], chi[1, 1], chi[2, 2], chi[0, 1], chi[0, 2], chi[1, 2]]
        )
        total += packed.reshape((6,) + (1,) * fraction.ndim) * np.asarray(fraction)
    return total


def static_material_grid(material_grid, frequency: float):
    """Fold tensor poles into static coefficients at one frequency.

    The returned grid carries ``eps_inf + Re chi(frequency)`` on every
    constitutive support (cells, Yee sites, and the shared nodes), which is the
    material a CW mode solve at that frequency must see. Grids without tensor
    poles are returned unchanged.
    """

    if not material_grid.tensor_dispersion:
        return material_grid
    chi = _cell_susceptibility(material_grid, frequency)
    polarization = material_grid.polarization
    if len(material_grid.shape) == 3:
        components = ("Ex", "Ey", "Ez")
    elif polarization == "tm":
        components = ("Ez",)
    else:
        components = ("Ex", "Ey")

    def at(values, target, shape):
        return np.asarray(
            collocate_yee_component(jnp.asarray(values), "Cell", target, shape),
            dtype=np.float64,
        )

    base_cell = expand_packed(material_grid.tensors["epsilon"])
    cell_tensor = base_cell + chi
    tensors = dict(material_grid.tensors)
    tensors["epsilon"] = cell_tensor
    scalar = np.mean(cell_tensor[[_axis_index(c) for c in components]], axis=0)

    yee_materials = dict(material_grid.yee_materials)
    for component in components:
        name = "eps_" + component[-1].lower()
        if name in yee_materials:
            shape = np.asarray(yee_materials[name]).shape
            yee_materials[name] = np.asarray(yee_materials[name]) + at(
                chi[_axis_index(component)], component, shape
            )

    coupled = bool(np.any(np.abs(chi[3:]) > 1e-12)) and len(components) > 1
    yee_tensors = dict(material_grid.yee_tensors)
    if coupled or yee_tensors:
        shapes = component_shapes(tuple(material_grid.shape), polarization or "tm")
        node_shape = tuple(n + 1 for n in material_grid.shape)
        for component in components:
            name = "eps_" + component[-1].lower()
            shape = shapes[component]
            if name in yee_tensors:
                base = expand_packed(yee_tensors[name])
            else:
                base = np.stack([at(v, component, shape) for v in base_cell])
                if name in material_grid.yee_materials:
                    # Keep the smoothed self-diagonal the propagation update uses.
                    base[_axis_index(component)] = material_grid.yee_materials[name]
            yee_tensors[name] = base + np.stack([at(v, component, shape) for v in chi])
        if len(components) > 1:
            base = (
                expand_packed(yee_tensors["eps_node"])
                if "eps_node" in material_grid.yee_tensors
                else np.stack([at(v, "Node", node_shape) for v in base_cell])
            )
            yee_tensors["eps_node"] = base + np.stack(
                [at(v, "Node", node_shape) for v in chi]
            )
    return replace(
        material_grid,
        permittivity=scalar,
        tensors=tensors,
        yee_materials=yee_materials,
        yee_tensors=yee_tensors,
        tensor_dispersion=(),
    )


class FrequencyFields:
    """Compiled-lattice view whose materials are frozen at one frequency."""

    def __init__(self, fields, frequency: float):
        self._fields = fields
        self.material_grid = static_material_grid(fields.material_grid, frequency)
        self.permittivity = jnp.asarray(self.material_grid.permittivity)

    def __getattr__(self, name):
        return getattr(self._fields, name)


def fields_at_frequency(fields, frequency: float, port=None):
    """Return ``fields`` with tensor poles frozen at ``frequency`` for mode solves.

    With ``port`` (any object with ``center`` and ``size``), the view is only
    built when the port plane reaches tensor-pole media; elsewhere the static
    coefficients already describe the port and are returned unchanged.
    """
    material_grid = getattr(fields, "material_grid", None)
    if material_grid is None or not material_grid.tensor_dispersion:
        return fields
    if port is not None:
        cells = port_cells(material_grid, port.center, port.size)
        if not any(
            np.any(cells & (np.asarray(fraction) > 1e-7))
            for _medium, fraction in material_grid.tensor_dispersion
        ):
            return fields
    return FrequencyFields(fields, frequency)


def port_cells(material_grid, center, size) -> np.ndarray:
    """Cells within one cell of a port plane's footprint (storage order)."""
    axes = ("z", "y", "x")[-len(material_grid.shape) :]
    public = {"x": 0, "y": 1, "z": 2}
    masks = []
    for axis in axes:
        edges = np.asarray(material_grid.grid.axis_edges(axis), dtype=float)
        mid = 0.5 * (edges[:-1] + edges[1:])
        c, s = float(center[public[axis]]), float(size[public[axis]])
        if s > 0.0:
            mask = (edges[1:] > c - 0.5 * s) & (edges[:-1] < c + 0.5 * s)
        else:
            index = int(np.clip(np.searchsorted(edges, c) - 1, 0, mid.size - 1))
            mask = np.abs(np.arange(mid.size) - index) <= 1
        masks.append(mask)
    out = masks[0]
    for mask in masks[1:]:
        out = out[..., None] & mask
    return out
