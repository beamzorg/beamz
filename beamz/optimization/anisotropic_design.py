"""Density-dependent anisotropic design materials for topology optimization.

A design region chooses among ``K >= 2`` materials ``m_0, ..., m_{K-1}``. Its
physical density has ``K - 1`` nested channels ``c_1, ..., c_{K-1}`` in
[0, 1] (a single field when ``K = 2``). Material ``k`` or a later one fills
the fraction ``Q_k = c_1 c_2 ... c_k`` of a cell (``Q_0 = 1``, ``Q_K = 0``),
so material ``k`` alone has weight ``w_k = Q_k - Q_{k+1}``:

    material(c) = sum_k w_k m_k = m_0 + sum_{k>=1} Q_k (m_k - m_{k-1}).

The weights are nonnegative and sum to one, binary channels select exactly one
material, and ``K = 2`` gives ``(1 - rho) * m_0 + rho * m_1``. Any material may
be an ``AnisotropicMedium``: its high-frequency permittivity tensor blends
linearly, and the volume fraction of its tensor poles is ``w_k``. Outside the
design mask the compiled simulation is unchanged. At runtime the binding
rebuilds, from traced ``rho``,

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

from beamz._cache_tokens import cache_token
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


def validate_design_materials(materials) -> tuple:
    """Check a sequence of at least two design materials and return it."""
    materials = tuple(materials)
    if len(materials) < 2:
        raise ValueError("A design region needs at least two materials.")
    names = (
        ("material_min", "material_max")
        if len(materials) == 2
        else tuple(f"materials[{k}]" for k in range(len(materials)))
    )
    for material, name in zip(materials, names, strict=True):
        _validate_design_material(material, name)
    return materials


def solid_fractions(rho, count: int) -> list:
    """Nested fractions ``Q_1, ..., Q_{K-1}`` of physical channels ``rho``.

    ``rho`` has the mask shape for ``count == 2`` and a leading axis of
    ``count - 1`` channels otherwise. ``Q_k`` is the fraction of the cell
    holding material ``k`` or a later one.
    """
    channels = [rho] if count == 2 else [rho[k] for k in range(count - 1)]
    fractions = [channels[0]]
    for channel in channels[1:]:
        fractions.append(fractions[-1] * channel)
    return fractions


def material_weights(rho, count: int) -> list:
    """Per-material volume fractions ``w_k = Q_k - Q_{k+1}``; they sum to one."""
    q = solid_fractions(rho, count)
    return [1.0 - q[0], *(q[k - 1] - q[k] for k in range(1, count - 1)), q[-1]]


@dataclass(frozen=True)
class _MediumEntry:
    medium: AnisotropicMedium
    base_fraction: np.ndarray  # cell fraction outside the mask
    indices: tuple[int, ...] | None  # design materials, or None for fixed media


class AnisotropicDesignBinding:
    """Bind a density field to design materials on a fixed compiled plan."""

    def __init__(self, material_grid, mask, materials):
        self.materials = validate_design_materials(materials)
        self.material_grid = material_grid
        self.mask = np.asarray(mask, dtype=bool)
        self.components = _components(material_grid)
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
        self.eps = [
            np.asarray(material.epsilon_r, dtype=np.float64)
            for material in self.materials
        ]
        self.entries = self._media_entries()

    @property
    def density_shape(self) -> tuple[int, ...]:
        """Physical density shape: the mask, with ``K - 1`` channels for K > 2."""
        if len(self.materials) == 2:
            return self.mask.shape
        return (len(self.materials) - 1, *self.mask.shape)

    @property
    def min_permittivity(self) -> float:
        """Smallest principal high-frequency permittivity of any design material."""
        values = []
        for xx, yy, zz, xy, xz, yz in self.eps:
            tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            values.append(float(np.min(np.linalg.eigvalsh(tensor))))
        return min(values)

    # -- media bookkeeping -------------------------------------------------

    def _media_entries(self) -> tuple[_MediumEntry, ...]:
        # Identical pole media share one entry whose fraction sums their weights.
        media, unclaimed = {}, {}
        for index, material in enumerate(self.materials):
            if _has_poles(material):
                key = cache_token(material.cache_spec())
                media.setdefault(key, material)
                unclaimed[key] = (*unclaimed.get(key, ()), index)
        outside = ~self.mask
        entries = []
        for medium, fraction in self.material_grid.tensor_dispersion:
            fraction = np.asarray(fraction, dtype=np.float64)
            indices = unclaimed.pop(cache_token(medium.cache_spec()), None)
            entries.append(_MediumEntry(medium, fraction * outside, indices))
        for key, indices in unclaimed.items():
            zero = np.zeros(self.material_grid.shape)
            entries.append(_MediumEntry(media[key], zero, indices))
        return tuple(entries)

    def _fraction(self, entry: _MediumEntry, rho):
        if entry.indices is None:
            return entry.base_fraction
        weights = material_weights(rho, len(self.materials))
        weight = weights[entry.indices[0]]
        for index in entry.indices[1:]:
            weight = weight + weights[index]
        return entry.base_fraction + self.mask * weight

    # -- constitutive fields ----------------------------------------------

    def _design_cell(self, rho, xp):
        rho = xp.asarray(rho)
        shape = (6,) + (1,) * self.mask.ndim
        tensors = [xp.asarray(eps).reshape(shape) for eps in self.eps]
        cell = tensors[0]
        for k, fraction in enumerate(solid_fractions(rho, len(self.materials)), 1):
            cell = cell + (tensors[k] - tensors[k - 1]) * fraction[None]
        return cell

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
        """Material grid with every density channel at 1/2 inside the mask.

        Every material then has a nonzero weight, so the program gets a
        full-tensor kernel when any design material needs one, and pole regions
        covering the mask for every design medium with poles. Runtime values
        come from ``rho``.
        """
        rho = np.full(self.density_shape, 0.5) * self.mask
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
            if entry is None or entry.indices is None:
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
