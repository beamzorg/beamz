"""Lower immutable boundary specifications into grid-aligned numerical arrays."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from beamz._region_array import SeparableMask, tiles, uniform_value
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.design.discretization import MaterialGrid
from beamz.devices.boundaries import (
    PEC,
    PML,
    Absorber,
    Periodic,
    edges_for_dimension,
    normalize_boundaries,
    periodic_storage_axes,
)

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _rectilinear_geometry(fields):
    geometry = getattr(fields, "geometry", None)
    if geometry is None:
        return None
    axes = ("x", "y", "z") if fields.permittivity.ndim == 3 else ("x", "y")
    return geometry if geometry.metric_kind_for(axes) != "isotropic_uniform" else None


@dataclass(frozen=True, slots=True)
class CpmlDerivativeSpec:
    """One derivative profile that Devices lowers onto a Yee component support."""

    name: str
    target_component: str
    derivative_axis: str


CPML_3D_H_DERIVATIVES = (
    CpmlDerivativeSpec("Hxy", "Hx", "y"),
    CpmlDerivativeSpec("Hxz", "Hx", "z"),
    CpmlDerivativeSpec("Hyz", "Hy", "z"),
    CpmlDerivativeSpec("Hyx", "Hy", "x"),
    CpmlDerivativeSpec("Hzx", "Hz", "x"),
    CpmlDerivativeSpec("Hzy", "Hz", "y"),
)

CPML_3D_E_DERIVATIVES = (
    CpmlDerivativeSpec("Exy", "Ex", "y"),
    CpmlDerivativeSpec("Exz", "Ex", "z"),
    CpmlDerivativeSpec("Eyz", "Ey", "z"),
    CpmlDerivativeSpec("Eyx", "Ey", "x"),
    CpmlDerivativeSpec("Ezx", "Ez", "x"),
    CpmlDerivativeSpec("Ezy", "Ez", "y"),
)


@dataclass(frozen=True, slots=True)
class BoundaryData:
    """Numerical material, absorber, and PEC data produced by boundary lowering."""

    permittivity: Any
    conductivity: Any
    permeability: Any
    yee_materials: Mapping[str, Any]
    profiles: Mapping[str, Any] | None
    masks: Mapping[str, Any]
    metallic_edges: frozenset[str]
    periodic_axes: frozenset[int]


def _scalar_constants_allowed(material_grid, boundaries) -> bool:
    """Return whether uniform loss and permeability may be stored as scalars."""
    # Deliberately conservative scope: CPML keeps its damping in psi state and its
    # material extrusion leaves uniform arrays unchanged, so the scalar form is
    # safe here. Sponge layers, metallic walls and 2D programs would likely work
    # too, but keep their established arrays until they are measured.
    return len(material_grid.shape) == 3 and all(
        isinstance(boundary, PML) and boundary.formulation == "cpml"
        for boundary in boundaries
    )


class _BoundaryGrid:
    """Mutable material workspace without allocating solver field arrays."""

    def __init__(
        self,
        material_grid,
        component_shapes,
        *,
        region_setup=False,
        scalar_constants=False,
    ) -> None:
        self.region_setup = region_setup

        def array(value, *, constant=False):
            # A uniform array is stored as its one value; broadcasting reproduces it
            # and avoids volume-sized copies through upload.
            if scalar_constants and constant:
                uniform = uniform_value(value)
                if uniform is not None:
                    value = uniform
            if not region_setup:
                return jnp.asarray(value)
            source = np.asarray(value)
            return np.asarray(source, dtype=jax.dtypes.canonicalize_dtype(source.dtype))

        self.permittivity = array(material_grid.permittivity)
        self.conductivity = array(material_grid.conductivity, constant=True)
        self.permeability = array(material_grid.permeability, constant=True)
        self.polarization_2d = material_grid.polarization or "tm"
        # Preserve the established uniform profile algebra exactly; stretched grids
        # opt into physical-coordinate grading.
        self.geometry = (
            material_grid.grid
            if material_grid.metric_kind != "isotropic_uniform"
            else None
        )
        self.yee_materials = {
            name: array(value, constant=name.startswith(("sig_", "mu_")))
            for name, value in material_grid.yee_materials.items()
        }
        for component, shape in component_shapes.items():
            setattr(
                self, component, SimpleNamespace(shape=tuple(int(v) for v in shape))
            )


def resolve_metallic_edges(boundaries, is_3d: bool) -> frozenset[str]:
    """Apply boundary precedence and return walls that remain metallic."""
    metallic: set[str] = set()
    for boundary in normalize_boundaries(boundaries):
        if isinstance(boundary, Periodic):
            continue
        edges = edges_for_dimension(boundary.edges, bool(is_3d))
        if isinstance(boundary, PEC):
            metallic.update(edges)
        else:
            metallic.difference_update(edges)
    return frozenset(metallic)


def compile_metallic_masks(
    component_shapes,
    material_shape,
    boundaries,
    *,
    polarization_2d: str = "tm",
    region_setup=False,
) -> dict[str, jnp.ndarray | SeparableMask]:
    """Compile boundary specifications into component-aligned PEC masks."""
    masks = (
        {}
        if region_setup
        else {
            name: np.zeros(tuple(component_shapes[name]), dtype=bool)
            for name in _COMPONENTS
        }
    )
    is_3d = len(material_shape) == 3
    metallic_edges = resolve_metallic_edges(boundaries, is_3d)
    wall_specs = (
        {
            "front": (0, 0, ("Ex", "Ey", "Hz")),
            "back": (0, -1, ("Ex", "Ey", "Hz")),
            "bottom": (1, 0, ("Ex", "Ez", "Hy")),
            "top": (1, -1, ("Ex", "Ez", "Hy")),
            "left": (2, 0, ("Ey", "Ez", "Hx")),
            "right": (2, -1, ("Ey", "Ez", "Hx")),
        }
        if is_3d
        else (
            {
                "bottom": (0, 0, ("Ez", "Hy")),
                "top": (0, -1, ("Ez", "Hy")),
                "left": (1, 0, ("Ez", "Hx")),
                "right": (1, -1, ("Ez", "Hx")),
            }
            if polarization_2d == "tm"
            else {
                "bottom": (0, 0, ("Ex",)),
                "top": (0, -1, ("Ex",)),
                "left": (1, 0, ("Ey",)),
                "right": (1, -1, ("Ey",)),
            }
        )
    )
    if region_setup:
        profiles = {
            name: [np.zeros(n, bool) for n in component_shapes[name]]
            for name in _COMPONENTS
        }
        for edge in metallic_edges:
            axis, index, components = wall_specs[edge]
            for component in components:
                profiles[component][axis][index] = True
        return {name: SeparableMask(values) for name, values in profiles.items()}
    for edge in metallic_edges:
        axis, index, components = wall_specs[edge]
        selection: list[slice | int] = [slice(None)] * len(material_shape)
        selection[axis] = index
        for component in components:
            masks[component][tuple(selection)] = True
    return {name: jnp.asarray(mask) for name, mask in masks.items()}


def _merge_profiles(lhs, rhs, *, key: str | None = None):
    """Merge contributions using the physical operation for each coefficient family."""
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        merged = dict(lhs)
        for child_key, child_value in rhs.items():
            merged[child_key] = (
                _merge_profiles(merged[child_key], child_value, key=child_key)
                if child_key in merged
                else child_value
            )
        return merged
    key = "" if key is None else str(key)
    if key == "formulation":
        if str(lhs).lower() != str(rhs).lower():
            raise ValueError(
                "Cannot merge boundary profiles with different formulations: "
                f"{lhs!r} != {rhs!r}."
            )
        return str(lhs).lower()
    lhs_arr, rhs_arr = jnp.asarray(lhs), jnp.asarray(rhs)
    if key == "mask":
        return lhs_arr.astype(bool) | rhs_arr.astype(bool)
    if "kappa" in key or "alpha" in key:
        return jnp.maximum(lhs_arr, rhs_arr)
    return lhs_arr + rhs_arr


def _edge_regions(shape, axis, side, count):
    slab = [slice(None)] * len(shape)
    reference = slab.copy()
    start = 0 if side == "low" else shape[axis] - count
    cell = count if side == "low" else start - 1
    slab[axis] = slice(start, start + count)
    reference[axis] = slice(cell, cell + 1)
    return tuple(slab), tuple(reference)


class _AbsorberCompiler:
    """Lower one immutable Device specification into grid-aware solver arrays."""

    def __init__(self, spec: PML | Absorber, *, cell_thickness: int | None = None):
        self.spec = spec
        self.cell_thickness = cell_thickness

    def _physical_thickness(self) -> float:
        thickness = self.spec.thickness
        if thickness is None:
            raise ValueError("PML thickness must be resolved before profile sampling.")
        return float(thickness)

    def compile_absorber_regions(self, fields, domain_size, resolution, dt):
        """Create grid-aligned masks and graded conductivity profiles."""
        # Resolve geometry-dependent arrays now so runtime code stays branch-free and
        # shape-static.
        profile = self._resolved_profile_boundary(fields, resolution, dt)

        if self.spec.formulation == "cpml":
            out = profile._create_cpml_profiles(fields, domain_size)
            self._extend_cpml_materials_to_absorber(fields, out)
            return out

        out = profile._create_sponge_profiles(fields, domain_size)
        self._warn_if_material_not_extruded(fields, out)
        return out

    def _resolved_profile_boundary(self, fields, resolution, dt):
        # Build this profile on its exact staggered support to avoid interpolation
        # inside timestep kernels.
        cell_thickness = (
            self.spec.DEFAULT_CELLS
            if isinstance(self.spec, PML) and self.spec.thickness is None
            else None
        )
        spec = (
            self.spec.updated_copy(thickness=cell_thickness * float(resolution))
            if cell_thickness is not None
            else self.spec
        )
        profile = type(self)(spec, cell_thickness=cell_thickness)
        sigma_max, alpha_max = profile._resolved_profile_params(fields, resolution, dt)
        changes: dict[str, float | None] = {"sigma_max": sigma_max}
        if isinstance(spec, PML):
            changes["alpha_max"] = alpha_max
        return type(self)(spec.updated_copy(**changes), cell_thickness=cell_thickness)

    def _resolved_profile_params(self, fields, resolution, dt):
        sigma_max = self.spec.sigma_max
        if sigma_max is None:
            eta = np.sqrt(MU_0 / (EPS_0 * 1.0))
            thickness = max(self._physical_thickness(), float(resolution))
            sigma_max = (
                -(self.spec.m + 1)
                * np.log(max(self.spec.target_reflection, 1e-16))
                / (2.0 * eta * thickness)
            )
        alpha_max = self.spec.alpha_max
        if self.spec.formulation == "cpml" and alpha_max is None:
            # Use the physical layer transit time so mesh/timestep refinement
            # preserves the shift. Zero-thickness layers have no active samples.
            thickness = self._physical_thickness() or float(resolution)
            alpha_max = 0.1 * EPS_0 * LIGHT_SPEED / thickness
        return float(sigma_max), None if alpha_max is None else float(alpha_max)

    def _pml_material_variation_edges(self, fields, pml_data):
        """Return edges where material changes along the PML normal."""
        # Map storage axes to named physical walls once so every boundary path uses
        # the same convention.
        material = np.asarray(fields.permittivity)
        if material.size == 0:
            return []
        return [
            edge
            for edge, axis, side, count in self._pml_material_edge_counts(
                material, pml_data
            )
            if 0 < count < material.shape[axis]
            and self._pml_edge_material_varies(material, axis, side, count)
        ]

    def _pml_material_edge_counts(self, material, pml_data):
        """Return active absorber-cell counts for material extrusion checks."""
        # 1. Map public wall names to storage axes for the current dimensionality; 2D
        # planes omit one physical axis and therefore need a plane-specific mapping.
        mat = np.asarray(material)
        edge_axes = {
            "left": ("x", "low"),
            "right": ("x", "high"),
            "bottom": ("y", "low"),
            "top": ("y", "high"),
            "front": ("z", "low"),
            "back": ("z", "high"),
        }
        is_3d = mat.ndim == 3
        axis_map = {axis: i for i, axis in enumerate("zyx" if is_3d else "yx")}
        # 2. For each active wall, collapse its sigma grid transversely to a 1D activity
        # profile and count contiguous absorber cells inward from that wall.
        edge_counts = []
        for edge in edges_for_dimension(self.spec.edges, is_3d):
            axis_name, side = edge_axes.get(edge, (None, None))
            if axis_name not in axis_map:
                continue
            sigma = pml_data.get(f"sigma_{axis_name}")
            if sigma is None:
                continue
            axis = axis_map[axis_name]
            active_1d = np.any(
                np.asarray(sigma) > 0.0,
                axis=tuple(i for i in range(material.ndim) if i != axis),
            )
            inward = active_1d if side == "low" else active_1d[::-1]
            count = int(np.argmax(~inward)) if np.any(~inward) else 0
            edge_counts.append((edge, axis, side, count))
        return edge_counts

    def _warn_if_material_not_extruded(self, fields, pml_data):
        """Warn when material changes along the PML normal inside the absorber."""
        # Diagnose impedance changes inside the absorber because they create avoidable reflection.
        bad_edges = self._pml_material_variation_edges(fields, pml_data)
        if bad_edges:
            warnings.warn(
                f"{type(self.spec).__name__} material varies along the absorber normal on edges "
                f"{bad_edges}. For lower reflection, extrude the boundary material "
                "profile through the PML or keep geometry clear of the absorber.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _extend_cpml_materials_to_absorber(self, fields, pml_data):
        """Copy first-interior material values through active CPML slabs."""
        counts = self._pml_material_edge_counts(fields.permittivity, pml_data)
        host = getattr(fields, "region_setup", False)
        yee_materials = getattr(fields, "yee_materials", {})
        materials = {
            name: getattr(fields, name)
            for name in ("permittivity", "conductivity", "permeability")
        } | yee_materials
        for name, source in materials.items():
            source = np.asarray(source)
            if source.ndim == 0 or (
                host and not _extrusion_changes_material(source, counts)
            ):
                continue
            material = np.array(source, copy=True)
            for _edge, axis, side, count in counts:
                if count <= 0 or count >= material.shape[axis]:
                    continue
                destination, reference = _edge_regions(
                    material.shape, axis, side, count
                )
                material[destination] = material[reference]
            value = material if host else jnp.asarray(material, dtype=source.dtype)
            if name in yee_materials:
                yee_materials[name] = value
            else:
                setattr(fields, name, value)

    @staticmethod
    def _pml_edge_material_varies(material, axis: int, side: str, count: int) -> bool:
        # Map storage axes to named physical walls once so every boundary path uses
        # the same convention.
        mat = np.asarray(material)
        slab, reference = _edge_regions(mat.shape, axis, side, count)
        return bool(np.any(np.abs(mat[slab] - mat[reference]) > 1e-9))

    def _compute_1d_profile(self, coords, length, low_active, high_active):
        """Compute 1D graded-sigma profile along an axis."""
        # Build this profile on its exact staggered support to avoid interpolation
        # inside timestep kernels.
        sigma = jnp.zeros_like(coords)
        thickness = self._physical_thickness()
        if self.spec.sigma_max is None:
            raise ValueError("PML profile parameters must be resolved before sampling.")

        for active, distance in (
            (low_active, thickness - coords),
            (high_active, coords - (length - thickness)),
        ):
            if active:
                sigma = (
                    sigma
                    + self.spec.sigma_max
                    * (jnp.clip(distance, 0.0, None) / thickness) ** self.spec.m
                )

        return sigma

    def _compute_1d_cpml_profile(self, coords, length, low_active, high_active):
        """Compute 1D CPML sigma/kappa/alpha profiles."""
        coords = np.asarray(coords, dtype=jax.dtypes.canonicalize_dtype(coords.dtype))
        # Build this profile on its exact staggered support to avoid interpolation
        # inside timestep kernels.
        if self.cell_thickness is not None:
            # A cell-count default must retain exactly the same number of layers on
            # uniform, axis-uniform, and fully rectilinear grids. Grade in logical
            # cell coordinates; explicit physical thicknesses continue to use the
            # realized metric coordinates supplied by the caller.
            cells = int(coords.shape[0])
            coords = np.arange(cells, dtype=np.float32) + 0.5
            length = float(cells)
        sigma = np.zeros_like(coords)
        kappa = np.ones_like(coords)
        alpha = np.zeros_like(coords)
        thickness = max(
            float(self.cell_thickness)
            if self.cell_thickness is not None
            else self._physical_thickness(),
            1e-30,
        )
        if self.spec.sigma_max is None or self.spec.alpha_max is None:
            raise ValueError(
                "CPML profile parameters must be resolved before sampling."
            )
        sigma_max = self.spec.sigma_max
        alpha_max = self.spec.alpha_max

        def apply_side(dist):
            # Grade sigma/kappa inward while alpha decreases toward the interior interface.
            graded = np.clip(dist / thickness, 0.0, 1.0)
            side_sigma = sigma_max * graded**self.spec.m
            side_kappa = 1.0 + (self.spec.kappa_max - 1.0) * graded**self.spec.m
            side_alpha = alpha_max * (1.0 - graded)
            mask = dist > 0
            return mask, side_sigma, side_kappa, side_alpha

        for active, dist in (
            (low_active, thickness - coords),
            (high_active, coords - (length - thickness)),
        ):
            if active:
                mask, side_sigma, side_kappa, side_alpha = apply_side(
                    np.clip(dist, 0.0, None)
                )
                sigma = sigma + side_sigma
                kappa = np.where(mask, side_kappa, kappa)
                alpha = np.where(mask, side_alpha, alpha)

        return sigma, kappa, alpha

    def _compute_fdtdx_staggered_profile_1d(
        self,
        total_samples,
        spacing,
        low_active,
        high_active,
        *,
        sample_kind,
        domain_cells=None,
        sigma_order=None,
        kappa_order=None,
        alpha_order=None,
        sample_coordinates=None,
        domain_bounds=None,
    ):
        """Compute a 1D CPML profile using FDTDX-style discrete Yee offsets."""

        # 1. Start from vacuum CPML values and convert the physical thickness to an integer
        # cell count used by FDTDX's discrete grading convention.
        sigma = np.zeros((int(total_samples),), dtype=np.float32)
        kappa = np.ones((int(total_samples),), dtype=np.float32)
        alpha = np.zeros((int(total_samples),), dtype=np.float32)

        pml_cells = max(
            int(self.cell_thickness)
            if self.cell_thickness is not None
            else int(round(self._physical_thickness() / max(float(spacing), 1e-30))),
            1,
        )
        sigma_order = float(self.spec.m if sigma_order is None else sigma_order)
        kappa_order = float(self.spec.m if kappa_order is None else kappa_order)
        alpha_order = float(1.0 if alpha_order is None else alpha_order)

        def apply_u(u):
            # Evaluate normalized polynomial profiles once for either low or high side distance.
            dtype = sigma.dtype
            u = np.clip(np.asarray(u, dtype=dtype), 0.0, 1.0)
            one = np.asarray(1.0, dtype=dtype)
            return (
                np.asarray(self.spec.sigma_max, dtype=dtype) * np.power(u, sigma_order),
                one
                + (np.asarray(self.spec.kappa_max, dtype=dtype) - one)
                * np.power(u, kappa_order),
                np.asarray(self.spec.alpha_max, dtype=dtype)
                * np.power(one - u, alpha_order),
            )

        # 2. Place the profile on physical Yee coordinates. Complete E supports have
        # wall-aligned samples at i*dx for i=0..N, while H supports have one sample per
        # material cell at (i+1/2)*dx for i=0..N-1. Deriving both absorber sides from
        # these coordinates avoids separate low/high recipes drifting by half a cell.
        if sample_kind == "E":
            offset = 0.0
            inferred_domain_cells = int(total_samples) - 1
        elif sample_kind == "H":
            offset = 0.5
            inferred_domain_cells = int(total_samples)
        else:
            raise ValueError(f"Unsupported CPML sample kind {sample_kind!r}")
        domain_cells = (
            inferred_domain_cells if domain_cells is None else int(domain_cells)
        )
        if domain_cells <= 0:
            raise ValueError("CPML domain_cells must be positive.")
        coords = np.arange(int(total_samples), dtype=np.float32) + np.asarray(
            offset, dtype=np.float32
        )

        physical_coordinates = (
            None
            if sample_coordinates is None or self.cell_thickness is not None
            else np.asarray(sample_coordinates, dtype=np.float32)
        )
        physical_thickness = (
            0.0 if physical_coordinates is None else self._physical_thickness()
        )
        distance_scale = (
            max(float(pml_cells), 1e-30)
            if physical_coordinates is None
            else max(physical_thickness, 1e-30)
        )

        def apply_side(dist):
            mask = dist > 0.0
            side_sigma, side_kappa, side_alpha = apply_u(dist / distance_scale)
            return mask, side_sigma, side_kappa, side_alpha

        # 3. Evaluate each active side from the same domain coordinate system. The
        # interface sample has zero CPML strength and remains an identity update.
        if physical_coordinates is not None:
            if domain_bounds is None:
                raise ValueError(
                    "Physical CPML sample coordinates require domain_bounds."
                )
            lower_bound, upper_bound = (float(value) for value in domain_bounds)
            coords = physical_coordinates

        low_dist, high_dist, limit = (
            (
                float(pml_cells) - coords,
                coords - float(domain_cells - pml_cells),
                float(pml_cells),
            )
            if physical_coordinates is None
            else (
                physical_thickness - (coords - lower_bound),
                coords - (upper_bound - physical_thickness),
                physical_thickness,
            )
        )
        for active, distance in ((low_active, low_dist), (high_active, high_dist)):
            if active:
                mask, side_sigma, side_kappa, side_alpha = apply_side(
                    np.clip(distance, 0.0, limit)
                )
                sigma = np.where(mask, np.maximum(sigma, side_sigma), sigma)
                kappa = np.where(mask, np.maximum(kappa, side_kappa), kappa)
                alpha = np.where(mask, np.maximum(alpha, side_alpha), alpha)

        # 5. Return aligned sigma, kappa, and alpha vectors for broadcasting onto the
        # corresponding directional derivative.
        return sigma, kappa, alpha

    def _create_sponge_profiles(self, fields, domain_size):
        """Create sponge profiles from one axis-driven 2D/3D implementation."""
        shape = tuple(fields.permittivity.shape)
        width, height, depth = domain_size
        if len(shape) == 3:
            axes = (
                ("z", depth, 0, "front", "back"),
                ("y", height, 1, "bottom", "top"),
                ("x", width, 2, "left", "right"),
            )
        else:
            axes = (
                ("y", height, 0, "bottom", "top"),
                ("x", width, 1, "left", "right"),
            )

        edges = set(edges_for_dimension(self.spec.edges, len(shape) == 3))
        profiles = {f"sigma_{name}": jnp.zeros(shape) for name in "xyz"}
        mask = jnp.zeros(shape, dtype=bool)
        for name, length, axis, low_edge, high_edge in axes:
            geometry = _rectilinear_geometry(fields)
            coords = (
                jnp.asarray(geometry.centers(name) - geometry.axis_edges(name)[0])
                if geometry is not None
                else jnp.linspace(0, length, shape[axis])
            )
            sigma = self._compute_1d_profile(
                coords, length, low_edge in edges, high_edge in edges
            )
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = shape[axis]
            grid = jnp.broadcast_to(jnp.reshape(sigma, broadcast_shape), shape)
            profiles[f"sigma_{name}"] = grid
            mask |= grid > 0
        return {"formulation": "sponge", "mask": mask, **profiles}

    def _create_cpml_profiles(self, fields, domain_size):
        """Sample static CPML profiles on cell and Yee supports in 2D or 3D."""
        shape = fields.permittivity.shape
        dimensions = len(shape)
        is_3d = dimensions == 3
        axis_index = dict(
            zip("xyz"[:dimensions], reversed(range(dimensions)), strict=True)
        )
        walls = {"x": ("left", "right"), "y": ("bottom", "top"), "z": ("front", "back")}
        axes = {
            axis: (shape[index], domain_size["xyz".index(axis)], *walls[axis])
            for axis, index in axis_index.items()
        }
        edges = set(edges_for_dimension(self.spec.edges, is_3d))
        geometry = _rectilinear_geometry(fields)
        families = ("sigma", "kappa", "alpha")

        def expand(axis, value):
            support = [1] * dimensions
            support[axis_index[axis]] = value.size
            return value.reshape(support)

        out: dict[str, Any] = {"formulation": "cpml"}
        for axis, (cells, length, low, high) in axes.items():
            coords = (
                geometry.centers(axis) - geometry.axis_edges(axis)[0]
                if geometry is not None
                else np.linspace(0, length, cells)
            )
            values = self._compute_1d_cpml_profile(
                coords, length, low in edges, high in edges
            )
            for family, value in zip(families, values, strict=True):
                out[f"{family}_{axis}"] = expand(axis, value)

        if is_3d:
            supports = tuple(
                (
                    f"cpml3d_{spec.name}",
                    spec.derivative_axis,
                    getattr(fields, spec.target_component).shape[
                        axis_index[spec.derivative_axis]
                    ],
                    spec.target_component[0],
                )
                for spec in (*CPML_3D_H_DERIVATIVES, *CPML_3D_E_DERIVATIVES)
            )
            staggered = out
        else:
            polarization = fields.polarization_2d
            components = (
                ("Ez", "Ez", "Hx", "Hy")
                if polarization == "tm"
                else ("Hz", "Hz", "Ex", "Ey")
            )
            supports = tuple(
                (
                    f"{component}_{axis}",
                    axis,
                    getattr(fields, component).shape[axis_index[axis]],
                    component[0],
                )
                for component, axis in zip(components, "xyyx", strict=True)
            )
            out.update(
                sigma_z=np.zeros((1, 1)),
                kappa_z=np.ones((1, 1)),
                alpha_z=np.zeros((1, 1)),
            )
            out["mask"] = (out["sigma_x"] > 0) | (out["sigma_y"] > 0)
            staggered = {}
            out[f"{polarization}_xy_cpml"] = staggered
        for name, axis, samples, kind in supports:
            cells, length, low, high = axes[axis]
            values = self._compute_fdtdx_staggered_profile_1d(
                samples,
                float(length) / max(cells, 1),
                low in edges,
                high in edges,
                sample_kind=kind,
                domain_cells=cells,
                sample_coordinates=(
                    geometry.axis_edges(axis)
                    if geometry is not None and kind == "E"
                    else (geometry.centers(axis) if geometry is not None else None)
                ),
                domain_bounds=(
                    (geometry.axis_edges(axis)[0], geometry.axis_edges(axis)[-1])
                    if geometry is not None
                    else None
                ),
            )
            for family, value in zip(families, values, strict=True):
                staggered[f"{name}_{family}"] = expand(axis, value)
        return out


def compile_absorber_regions(
    spec, fields, domain_size, resolution, dt
) -> Mapping[str, Any]:
    """Compile an absorbing Device boundary for one discretized field grid."""
    return _AbsorberCompiler(spec).compile_absorber_regions(
        fields, domain_size, resolution, dt
    )


def _extrusion_changes_material(material, edge_counts):
    """Check exact CPML extension without a full-volume copy or slab scratch."""
    for _edge, axis, side, count in edge_counts:
        if count <= 0 or count >= material.shape[axis]:
            continue
        slab, reference = _edge_regions(material.shape, axis, side, count)
        values = material[slab]
        ref = np.broadcast_to(material[reference], values.shape)
        for tile in tiles(values.shape, values.dtype):
            a, b = values[tile], ref[tile]
            if not np.array_equal(a, b) and not np.array_equal(a, b, equal_nan=True):
                return True
    return False


def lower_boundaries(
    material_grid: MaterialGrid,
    component_shapes,
    boundaries,
    domain_size,
    dt: float,
    *,
    polarization_2d: str = "tm",
    plane_2d: str = "xy",
    region_setup=False,
) -> BoundaryData:
    """Lower the complete boundary tuple once for Simulation compilation."""
    boundaries = normalize_boundaries(boundaries)
    workspace = _BoundaryGrid(
        material_grid,
        component_shapes,
        region_setup=region_setup,
        scalar_constants=_scalar_constants_allowed(material_grid, boundaries),
    )
    profiles: Mapping[str, Any] | None = None
    for boundary in boundaries:
        if not isinstance(boundary, (PML, Absorber)):
            continue
        contribution = compile_absorber_regions(
            boundary,
            workspace,
            tuple(float(value) for value in domain_size),
            material_grid.resolution,
            dt,
        )
        profiles = (
            dict(contribution)
            if profiles is None
            else cast(Mapping[str, Any], _merge_profiles(profiles, contribution))
        )
    return BoundaryData(
        workspace.permittivity,
        workspace.conductivity,
        workspace.permeability,
        workspace.yee_materials,
        profiles,
        compile_metallic_masks(
            component_shapes,
            material_grid.shape,
            boundaries,
            polarization_2d=polarization_2d,
            region_setup=region_setup,
        ),
        resolve_metallic_edges(boundaries, len(material_grid.shape) == 3),
        periodic_storage_axes(
            boundaries,
            is_3d=len(material_grid.shape) == 3,
            plane_2d=plane_2d,
        ),
    )
