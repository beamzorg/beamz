"""Frequency-domain views of a material grid for modal solves.

Mode solvers need static coefficients. For grids with tensor poles, these
views fold ``Re chi(frequency)`` into every constitutive support so a CW mode
solved at that frequency sees the medium the time stepper propagates. Electric
poles fold into permittivity; magnetic tensor poles fold into permeability on
the H supports and cell centers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from beamz.constitutive import collocate_yee_component, expand_packed
from beamz.lattice import component_shapes


def _axis_index(component: str) -> int:
    return "xyz".index(component[-1].lower())


def foldable(medium) -> bool:
    """Whether a medium's poles can be frozen into static mode-solver tensors.

    Symmetric electric and magnetic tensor poles can. Gyrotropic
    (non-reciprocal) poles cannot; ports must avoid such media, which
    compilation enforces.
    """
    from beamz.design.anisotropic import GyrotropicPole

    return not any(
        isinstance(p, GyrotropicPole)
        for p in (*medium.electric_poles, *medium.magnetic_poles)
    )


def _pole_arrays(medium, family: str = "E"):
    """Pole parameters of one family as plain arrays (resonance, damping, W)."""
    poles = medium.electric_poles if family == "E" else medium.magnetic_poles
    return (
        np.array([p.resonance for p in poles], dtype=np.float64),
        np.array([p.damping for p in poles], dtype=np.float64),
        np.stack([np.asarray(p.weight, dtype=np.float64) for p in poles])
        if poles
        else np.zeros((0, 6)),
    )


def _packed_susceptibility(resonance, damping, weight, frequency) -> np.ndarray:
    """Real part of sum_p W_p / (w_p^2 - w^2 - i g_p w), packed (6,)."""
    omega = 2.0 * np.pi * float(frequency)
    response = 1.0 / (resonance**2 - omega**2 - 1j * damping * omega)
    return np.real(np.sum(response[:, None] * weight, axis=0))


@dataclass(frozen=True)
class DispersionFolding:
    """Precomputed data to freeze electric tensor poles at any frequency.

    Base coefficients and each medium's volume fraction on every support (cells,
    Yee sites, shared nodes) are collocated once on the full grid. Folding a
    frequency is then elementwise: ``eps(f) = eps_inf + sum_m fraction_m
    Re chi_m(f)``. ``crop`` keeps the same exact values for a monitor-local
    snapshot. Only arrays and numbers are stored, so results archives can
    serialize it.
    """

    components: tuple
    poles: tuple  # per medium: (resonance, damping, weight (P, 6))
    cell_fraction: tuple  # per medium: cell array
    site_fraction: tuple  # per medium: {component: support array}
    node_fraction: tuple  # per medium: node array or None
    base_cell: np.ndarray  # packed (6, ...) cell eps_inf
    base_yee_materials: Mapping  # {"eps_x": array, ...} present in the source
    base_yee_tensors: Mapping  # packed support tensors, incl. eps_node, or {}
    # Magnetic poles, present only when some folded medium has them. Cell
    # coefficients reuse ``cell_fraction``; H sites carry their own fractions.
    magnetic_poles: tuple = ()  # per medium: (resonance, damping, weight (P, 6))
    h_site_fraction: tuple = ()  # per medium: {component: support array}
    base_magnetic_tensors: Mapping = field(default_factory=dict)  # mu_h*, mu_cell
    base_mu_yee_materials: Mapping = field(default_factory=dict)  # {"mu_hx": ...}

    @classmethod
    def from_material_grid(cls, material_grid) -> DispersionFolding:
        polarization = material_grid.polarization
        if len(material_grid.shape) == 3:
            components = ("Ex", "Ey", "Ez")
        elif polarization == "tm":
            components = ("Ez",)
        else:
            components = ("Ex", "Ey")
        shapes = component_shapes(tuple(material_grid.shape), polarization or "tm")
        node_shape = tuple(n + 1 for n in material_grid.shape)

        def at(values, target, shape):
            return np.asarray(
                collocate_yee_component(jnp.asarray(values), "Cell", target, shape),
                dtype=np.float64,
            )

        media = tuple((m, f) for m, f in material_grid.tensor_dispersion if foldable(m))
        poles = tuple(_pole_arrays(m) for m, _ in media)
        coupled_capable = len(components) > 1 and (
            bool(material_grid.yee_tensors)
            or any(np.any(weight[:, 3:] != 0.0) for _, _, weight in poles)
        )
        cell_fraction, site_fraction, node_fraction = [], [], []
        for _medium, fraction in media:
            fraction = np.asarray(fraction, dtype=np.float64)
            cell_fraction.append(fraction)
            site_fraction.append({c: at(fraction, c, shapes[c]) for c in components})
            node_fraction.append(
                at(fraction, "Node", node_shape) if coupled_capable else None
            )
        base_cell = expand_packed(material_grid.tensors["epsilon"])
        names = {c: "eps_" + c[-1].lower() for c in components}
        base_yee_materials = {
            name: np.asarray(material_grid.yee_materials[name], dtype=np.float64)
            for name in names.values()
            if name in material_grid.yee_materials
        }
        base_yee_tensors = {}
        if coupled_capable:
            for component, name in names.items():
                if name in material_grid.yee_tensors:
                    base = expand_packed(material_grid.yee_tensors[name])
                else:
                    base = np.stack(
                        [at(v, component, shapes[component]) for v in base_cell]
                    )
                    if name in base_yee_materials:
                        # Keep the smoothed self-diagonal the propagation uses.
                        base[_axis_index(component)] = base_yee_materials[name]
                base_yee_tensors[name] = base
            base_yee_tensors["eps_node"] = (
                expand_packed(material_grid.yee_tensors["eps_node"])
                if "eps_node" in material_grid.yee_tensors
                else np.stack([at(v, "Node", node_shape) for v in base_cell])
            )
        return cls(
            components,
            poles,
            tuple(cell_fraction),
            tuple(site_fraction),
            tuple(node_fraction),
            base_cell,
            MappingProxyType(base_yee_materials),
            MappingProxyType(base_yee_tensors),
            **_magnetic_folding(material_grid, media, shapes),
        )

    def overlaps(self) -> bool:
        return any(np.any(f > 1e-7) for f in self.cell_fraction)

    def crop(self, cells: tuple, supports: tuple) -> DispersionFolding:
        """Restrict to a snapshot: ``cells`` for cell arrays, ``supports`` else."""

        def cut(array, parts, packed=False):
            if array is None:
                return None
            index = ((slice(None),) if packed else ()) + tuple(parts)
            return np.asarray(array)[index].copy()

        return DispersionFolding(
            self.components,
            self.poles,
            tuple(cut(f, cells) for f in self.cell_fraction),
            tuple(
                {c: cut(a, supports) for c, a in sites.items()}
                for sites in self.site_fraction
            ),
            tuple(cut(f, supports) for f in self.node_fraction),
            np.asarray(self.base_cell)[(slice(None), *cells)].copy(),
            MappingProxyType(
                {k: cut(v, supports) for k, v in self.base_yee_materials.items()}
            ),
            MappingProxyType(
                {
                    k: cut(v, supports, packed=True)
                    for k, v in self.base_yee_tensors.items()
                }
            ),
            self.magnetic_poles,
            tuple(
                {c: cut(a, supports) for c, a in sites.items()}
                for sites in self.h_site_fraction
            ),
            MappingProxyType(
                {
                    k: cut(v, cells if k == "mu_cell" else supports, packed=True)
                    for k, v in self.base_magnetic_tensors.items()
                }
            ),
            MappingProxyType(
                {k: cut(v, supports) for k, v in self.base_mu_yee_materials.items()}
            ),
        )

    def fold(self, frequency: float) -> dict:
        """Static coefficients at ``frequency``: scalar, cell, Yee, and node arrays."""
        chis = [
            _packed_susceptibility(resonance, damping, weight, frequency)
            for resonance, damping, weight in self.poles
        ]

        def add_packed(base, fractions):
            out = np.array(base, dtype=np.float64, copy=True)
            for chi, fraction in zip(chis, fractions, strict=True):
                out = out + chi.reshape((6,) + (1,) * fraction.ndim) * fraction
            return out

        def add_diagonal(base, fractions, index):
            out = np.array(base, dtype=np.float64, copy=True)
            for chi, fraction in zip(chis, fractions, strict=True):
                out = out + chi[index] * fraction
            return out

        def site_fractions(name):
            return [sites["E" + name[-1]] for sites in self.site_fraction]

        cell = add_packed(self.base_cell, self.cell_fraction)
        scalar = np.mean(cell[[_axis_index(c) for c in self.components]], axis=0)
        yee_materials = {
            name: add_diagonal(base, site_fractions(name), _axis_index(name))
            for name, base in self.base_yee_materials.items()
        }
        yee_tensors = {}
        for name, base in self.base_yee_tensors.items():
            if name == "eps_node":
                fractions = self.node_fraction
            else:
                fractions = site_fractions(name)
            yee_tensors[name] = add_packed(base, fractions)
        folded = {
            "permittivity": scalar,
            "epsilon": cell,
            "yee_materials": yee_materials,
            "yee_tensors": yee_tensors,
        }
        if self.magnetic_poles:
            folded.update(self._fold_magnetic(frequency, yee_materials))
        return folded

    def _fold_magnetic(self, frequency: float, yee_materials: dict) -> dict:
        """Permeability ``mu_inf + sum_m fraction_m Re chi_m(f)`` on every support."""
        chis = [
            _packed_susceptibility(resonance, damping, weight, frequency)
            for resonance, damping, weight in self.magnetic_poles
        ]

        def add(base, fractions, index=None):
            out = np.array(base, dtype=np.float64, copy=True)
            for chi, fraction in zip(chis, fractions, strict=True):
                if index is None:
                    out = out + chi.reshape((6,) + (1,) * fraction.ndim) * fraction
                else:
                    out = out + chi[index] * fraction
            return out

        def sites(component):
            return [fractions[component] for fractions in self.h_site_fraction]

        magnetic = {
            name: add(
                base,
                self.cell_fraction if name == "mu_cell" else sites("H" + name[-1]),
            )
            for name, base in self.base_magnetic_tensors.items()
        }
        yee_materials.update(
            {
                name: add(base, sites("H" + name[-1]), "xyz".index(name[-1]))
                for name, base in self.base_mu_yee_materials.items()
            }
        )
        cell = magnetic["mu_cell"]
        active = ["xyz".index(name[-1]) for name in magnetic if name != "mu_cell"]
        return {
            "mu": cell,
            "permeability": np.mean(cell[active], axis=0),
            "magnetic_tensors": magnetic,
        }


def _magnetic_folding(material_grid, media, shapes) -> dict:
    """Magnetic-pole folding data, or nothing when no folded medium has them."""
    if not any(medium.magnetic_poles for medium, _ in media):
        return {}
    if len(material_grid.shape) == 3:
        components = ("Hx", "Hy", "Hz")
    elif material_grid.polarization == "tm":
        components = ("Hx", "Hy")
    else:
        components = ("Hz",)
    tensors = material_grid.magnetic_tensors
    identity = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def base(name, shape):
        if name in tensors:
            return np.array(expand_packed(tensors[name]), dtype=np.float64)
        shape = tuple(shape)
        return np.array(
            np.broadcast_to(identity.reshape((6,) + (1,) * len(shape)), (6, *shape)),
            dtype=np.float64,
        )

    names = {c: "mu_h" + c[-1].lower() for c in components}
    base_magnetic = {name: base(name, shapes[c]) for c, name in names.items()}
    base_magnetic["mu_cell"] = base("mu_cell", material_grid.shape)
    h_site_fraction = tuple(
        {
            c: np.asarray(
                collocate_yee_component(jnp.asarray(fraction), "Cell", c, shapes[c]),
                dtype=np.float64,
            )
            for c in components
        }
        for _medium, fraction in media
    )
    return {
        "magnetic_poles": tuple(_pole_arrays(m, "H") for m, _ in media),
        "h_site_fraction": h_site_fraction,
        "base_magnetic_tensors": MappingProxyType(base_magnetic),
        "base_mu_yee_materials": MappingProxyType(
            {
                name: np.asarray(material_grid.yee_materials[name], dtype=np.float64)
                for name in names.values()
                if name in material_grid.yee_materials
            }
        ),
    }


def static_material_grid(material_grid, frequency: float):
    """Fold tensor poles into static coefficients at one frequency.

    The returned grid carries ``eps_inf + Re chi(frequency)`` on every
    constitutive support (cells, Yee sites, and the shared nodes), which is the
    material a CW mode solve at that frequency must see. Grids without tensor
    poles are returned unchanged.
    """

    if not material_grid.tensor_dispersion:
        return material_grid
    folded = DispersionFolding.from_material_grid(material_grid).fold(frequency)
    tensors = dict(material_grid.tensors)
    tensors["epsilon"] = folded["epsilon"]
    yee_materials = dict(material_grid.yee_materials)
    yee_materials.update(folded["yee_materials"])
    yee_tensors = dict(material_grid.yee_tensors)
    yee_tensors.update(folded["yee_tensors"])
    magnetic = {}
    if "magnetic_tensors" in folded:
        tensors["mu"] = folded["mu"]
        magnetic = dict(
            permeability=folded["permeability"],
            magnetic_tensors=folded["magnetic_tensors"],
        )
    return replace(
        material_grid,
        permittivity=folded["permittivity"],
        tensors=tensors,
        yee_materials=yee_materials,
        yee_tensors=yee_tensors,
        tensor_dispersion=(),
        **magnetic,
    )


class FrequencyFields:
    """Compiled-lattice view whose materials are frozen at one frequency."""

    def __init__(self, fields, frequency: float):
        self._fields = fields
        self.material_grid = static_material_grid(fields.material_grid, frequency)
        self.permittivity = jnp.asarray(self.material_grid.permittivity)
        self.permeability = jnp.asarray(self.material_grid.permeability)

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
