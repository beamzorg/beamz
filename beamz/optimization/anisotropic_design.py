"""Density-dependent anisotropic design materials for topology optimization.

The design density ``rho`` blends two materials cell by cell,

    material(rho) = (1 - rho) * material_min + rho * material_max,

where either material may be an ``AnisotropicMedium``: its high-frequency
permittivity tensor blends linearly, and the volume fraction of its tensor
poles is ``rho`` (or ``1 - rho``). Outside the design mask the compiled
simulation is unchanged. At runtime the binding rebuilds, from traced ``rho``,

* the electric update coefficients (inverse permittivity tensors on Yee sites
  and shared nodes for full-tensor programs, or the diagonal coefficients), and
* the electric tensor-pole regions of the anisotropic plan (pole weights and
  their inverse operators),

so autodiff differentiates through every material path. Supports straddling
the mask boundary blend design and fixed coefficients by their in-mask
fraction, exactly like the scalar topology binding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from beamz.constitutive import (
    collocate_yee_component,
    expand_packed,
    inverse_permittivity_components,
)
from beamz.design.anisotropic import AnisotropicMedium, GyrotropicPole, TensorPole
from beamz.design.materials import Material
from beamz.lattice import component_shapes


def _axis_index(component: str) -> int:
    return "xyz".index(component[-1].lower())


def _support_name(component: str) -> str:
    return "eps_" + component[-1].lower()


def _validate_design_material(material, name: str) -> None:
    if not isinstance(material, Material):
        raise TypeError(f"{name} must be a Material or AnisotropicMedium.")
    if not np.allclose(material.mu_r, (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)):
        raise ValueError(f"{name} must be nonmagnetic (permeability one).")
    if np.any(np.asarray(material.conductivity) != 0.0):
        raise ValueError(
            f"{name} must be lossless (zero conductivity); use poles for loss."
        )
    if isinstance(material, AnisotropicMedium):
        if material.magnetic_poles:
            raise ValueError(f"{name} cannot have magnetic poles.")
        if any(isinstance(p, GyrotropicPole) for p in material.electric_poles):
            raise ValueError(f"{name} cannot have gyrotropic poles.")
        if material.requires_static_correction:
            raise ValueError(
                f"{name} cannot have non-symmetric tensors or conductivity tensors."
            )
    elif type(material) is not Material:
        raise TypeError(
            f"{name} must be a static Material or an AnisotropicMedium; scalar "
            "pole-residue media are not supported as design materials."
        )


def _has_poles(material) -> bool:
    return isinstance(material, AnisotropicMedium) and bool(material.electric_poles)


def _components(material_grid) -> tuple[str, ...]:
    if len(material_grid.shape) == 3:
        return ("Ex", "Ey", "Ez")
    return ("Ez",) if material_grid.polarization == "tm" else ("Ex", "Ey")


def _collocate_numpy(values, target, shape):
    return np.asarray(
        collocate_yee_component(jnp.asarray(values), "Cell", target, shape),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _MediumEntry:
    medium: AnisotropicMedium
    base_fraction: np.ndarray  # cell fraction outside the mask
    role: str | None  # "min", "max", or None for fixed media


class AnisotropicDesignBinding:
    """Bind a density field to design materials on a fixed compiled plan."""

    def __init__(self, material_grid, mask, material_min, material_max):
        _validate_design_material(material_min, "material_min")
        _validate_design_material(material_max, "material_max")
        self.material_grid = material_grid
        self.mask = np.asarray(mask, dtype=bool)
        self.components = _components(material_grid)
        self.material_min = material_min
        self.material_max = material_max
        shapes = component_shapes(
            tuple(material_grid.shape), material_grid.polarization or "tm"
        )
        self.shapes = {c: shapes[c] for c in self.components}
        self.node_shape = tuple(n + 1 for n in material_grid.shape)
        self.coupled = len(self.components) > 1
        if "epsilon" in material_grid.tensors:
            cell = expand_packed(material_grid.tensors["epsilon"])
        else:
            cell = expand_packed(np.asarray(material_grid.permittivity)[None])
        self.base_cell = cell
        self.base_supports = {}
        for component in self.components:
            name = _support_name(component)
            if name in material_grid.yee_tensors:
                base = expand_packed(material_grid.yee_tensors[name])
            else:
                base = np.stack(
                    [
                        _collocate_numpy(v, component, self.shapes[component])
                        for v in cell
                    ]
                )
                if name in material_grid.yee_materials:
                    base[_axis_index(component)] = material_grid.yee_materials[name]
            self.base_supports[component] = base
        if self.coupled:
            self.base_supports["Node"] = (
                expand_packed(material_grid.yee_tensors["eps_node"])
                if "eps_node" in material_grid.yee_tensors
                else np.stack(
                    [_collocate_numpy(v, "Node", self.node_shape) for v in cell]
                )
            )
        mask_cells = self.mask.astype(np.float64)
        self.mask_supports = {
            c: _collocate_numpy(mask_cells, c, self.shapes[c]) for c in self.components
        }
        if self.coupled:
            self.mask_supports["Node"] = _collocate_numpy(
                mask_cells, "Node", self.node_shape
            )
        self.eps_min = np.asarray(material_min.epsilon_r, dtype=np.float64)
        self.eps_max = np.asarray(material_max.epsilon_r, dtype=np.float64)
        self.entries = self._media_entries()

    # -- media bookkeeping -------------------------------------------------

    def _media_entries(self) -> tuple[_MediumEntry, ...]:
        roles = {
            role: material
            for role, material in (
                ("min", self.material_min),
                ("max", self.material_max),
            )
            if _has_poles(material)
        }
        claimed = set()
        outside = ~self.mask
        entries = []
        for medium, fraction in self.material_grid.tensor_dispersion:
            fraction = np.asarray(fraction, dtype=np.float64)
            role = next(
                (
                    r
                    for r, material in roles.items()
                    if r not in claimed and material.cache_spec() == medium.cache_spec()
                ),
                None,
            )
            if role is not None:
                claimed.add(role)
            entries.append(_MediumEntry(medium, fraction * outside, role))
        for role, material in roles.items():
            if role not in claimed:
                entries.append(
                    _MediumEntry(material, np.zeros(self.material_grid.shape), role)
                )
        return tuple(entries)

    def _fraction(self, entry: _MediumEntry, rho):
        if entry.role is None:
            return entry.base_fraction
        weight = rho if entry.role == "max" else 1.0 - rho
        return entry.base_fraction + self.mask * weight

    # -- constitutive fields ----------------------------------------------

    def _design_cell(self, rho, xp):
        rho = xp.asarray(rho)
        lower = xp.asarray(self.eps_min).reshape((6,) + (1,) * rho.ndim)
        upper = xp.asarray(self.eps_max).reshape((6,) + (1,) * rho.ndim)
        return lower + (upper - lower) * rho[None]

    def _cell_tensor(self, rho):
        mask = self.mask.astype(np.float64)
        return self.base_cell * (1.0 - mask) + self._design_cell(rho, np) * mask

    def supports(self, rho):
        """Packed permittivity tensors on each active Yee site (and nodes)."""
        design = self._design_cell(rho, jnp) * jnp.asarray(self.mask, jnp.float32)
        out = {}
        for support, base in self.base_supports.items():
            shape = self.node_shape if support == "Node" else self.shapes[support]
            sampled = jax.vmap(
                lambda v, support=support, shape=shape: collocate_yee_component(
                    v, "Cell", support, shape
                )
            )(design)
            out[support] = (
                jnp.asarray(base) * (1.0 - self.mask_supports[support]) + sampled
            )
        return out

    # -- compile-time grid --------------------------------------------------

    def compile_grid(self):
        """Material grid at rho = 1/2 inside the mask.

        Compiling at the midpoint gives the program a full-tensor kernel when
        either design material needs one, and pole regions covering the mask
        for every design medium with poles. Runtime values come from ``rho``.
        """
        rho = np.full(self.material_grid.shape, 0.5) * self.mask
        return self.material_grid_at(rho)

    def material_grid_at(self, rho):
        """Ordinary (detached) material grid for a physical density field."""
        rho = np.asarray(rho, dtype=np.float64)
        supports = {
            k: np.asarray(v, dtype=np.float64) for k, v in self.supports(rho).items()
        }
        cell = np.asarray(self._cell_tensor(rho), dtype=np.float64)
        indices = [_axis_index(c) for c in self.components]
        grid = self.material_grid
        tensors = dict(grid.tensors)
        tensors["epsilon"] = cell
        yee_materials = dict(grid.yee_materials)
        for component in self.components:
            yee_materials[_support_name(component)] = supports[component][
                _axis_index(component)
            ]
        full = any(np.any(np.abs(v[3:]) > 1e-10) for v in supports.values())
        yee_tensors = dict(grid.yee_tensors)
        if full:
            for component in self.components:
                yee_tensors[_support_name(component)] = supports[component]
            if self.coupled:
                yee_tensors["eps_node"] = supports["Node"]
        fractions = [
            (entry.medium, np.asarray(self._fraction(entry, rho)))
            for entry in self.entries
        ]
        tensor_dispersion = tuple(
            (medium, np.clip(fraction, 0.0, 1.0))
            for medium, fraction in fractions
            if np.any(fraction > 1e-7)
        )
        return replace(
            grid,
            permittivity=np.mean(cell[indices], axis=0),
            tensors=tensors,
            yee_materials=yee_materials,
            yee_tensors=yee_tensors,
            tensor_dispersion=tensor_dispersion,
            smoothing="farjadpour_full" if full else "farjadpour_diagonal",
        )

    # -- runtime binding ----------------------------------------------------

    def coefficients(self, program, supports):
        """Electric update coefficients for traced support tensors."""
        plan = program.dispersion.anisotropic if program.dispersion else None
        if plan is not None and plan.electric is not None:
            # Those media need vacuum E coefficients on their box, which a
            # rebuild from the design blend would overwrite.
            raise ValueError(
                "Anisotropic designs cannot yet share a simulation with "
                "non-symmetric or off-diagonal-conductive media."
            )
        coefficients = program.coefficients
        updates = {}
        if coefficients.e_inverse_offdiagonal.size:
            for component in self.components:
                axis = component[-1].lower()
                inverse = inverse_permittivity_components(supports[component], xp=jnp)
                updates[f"e_inverse_diagonal_{axis}"] = inverse[
                    ..., _axis_index(component)
                ]
            if self.coupled:
                updates["e_inverse_offdiagonal"] = inverse_permittivity_components(
                    supports["Node"], xp=jnp
                )[..., 3:]
            return coefficients._replace(**updates)
        from beamz.simulation.kernels import precompute_e_update_coefficients

        grid = program.grid
        for component in self.components:
            axis = component[-1].lower()
            permittivity = supports[component][_axis_index(component)]
            if program.config.is_3d:
                updates[f"e_permittivity_{axis}"] = permittivity
                continue
            decay, source = precompute_e_update_coefficients(
                shape=grid.component_shapes[component],
                conductivity=getattr(grid, f"sig_{axis}"),
                permittivity=permittivity,
                dt=program.config.dt,
                region=getattr(grid, f"region_{axis}"),
            )
            updates[f"e_decay_{axis}"] = decay
            updates[f"e_source_{axis}"] = source
        return coefficients._replace(**updates)

    def plan(self, program, rho, supports):
        """Anisotropic plan with design-dependent electric pole regions."""
        from beamz.simulation.anisotropic import _pole_coefficients, tensor_pole_weights

        base = program.dispersion.anisotropic
        if base is None:
            return None
        full = bool(program.coefficients.e_inverse_offdiagonal.size)
        dt = float(program.config.dt)
        owners = self._electric_region_owners(program)
        regions = []
        for region, entry in zip(base.electric_regions, owners, strict=True):
            if entry is None or entry.role is None:
                regions.append(region)
                continue
            box = region.box
            poles = [
                p for p in entry.medium.electric_poles if isinstance(p, TensorPole)
            ]
            _c1, _c2, c3 = _pole_coefficients(poles, dt)
            drive = np.stack([np.asarray(p.weight) for p in poles]) * c3[:, None]
            fraction = jnp.asarray(self._fraction(entry, rho))[box.cell_slices]
            diagonal, offdiagonal = tensor_pole_weights(fraction, drive, box)
            regions.append(
                replace(
                    region,
                    diagonal=diagonal,
                    offdiagonal=offdiagonal,
                    inverse=self._inverse(box, supports, full, region.inverse),
                )
            )
        return replace(base, electric_regions=tuple(regions))

    def _inverse(self, box, supports, full, previous):
        from beamz.simulation.anisotropic import InverseOperator

        diagonal = []
        for component, part in zip(box.components, box.slices, strict=True):
            local = supports[component][(slice(None), *part)]
            if full:
                diagonal.append(
                    inverse_permittivity_components(local, xp=jnp)[
                        ..., _axis_index(component)
                    ]
                )
            else:
                diagonal.append(1.0 / local[_axis_index(component)])
        offdiagonal = None
        if full and previous.offdiagonal is not None:
            node = supports["Node"][(slice(None), *box.shared_slices)]
            offdiagonal = inverse_permittivity_components(node, xp=jnp)[..., 3:]
        return InverseOperator(tuple(diagonal), offdiagonal)

    def _electric_region_owners(self, program):
        """Map each compiled electric region to its medium entry (plan order)."""
        grid_media = program.grid.material_grid.tensor_dispersion
        owners = []
        for medium, fraction in grid_media:
            if not medium.electric_poles or not np.any(np.asarray(fraction) > 1e-7):
                continue
            entry = next(
                (
                    e
                    for e in self.entries
                    if e.medium.cache_spec() == medium.cache_spec()
                ),
                None,
            )
            if any(isinstance(p, TensorPole) for p in medium.electric_poles):
                owners.append(entry)
            if any(isinstance(p, GyrotropicPole) for p in medium.electric_poles):
                owners.append(None)
        return owners
