"""Internal simulation configuration operations."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from beamz._cache_tokens import cache_token
from beamz.lattice import (
    grid_vector_to_physical_2d,
    physical_vector_to_grid_2d,
)
from beamz.simulation.preparation_trace import trace_preparation

from .preparation import (
    _resolved_raster_options,
    _shift_device_to_domain,
    _shift_grid_spec_to_domain,
)


class _SimulationConfiguration:
    def updated_copy(self: Any, **changes):
        """Return a validated simulation with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Any supported constructor field, including ``design``, ``sources``,
            ``monitors``, ``boundaries``, ``resolution``, ``time``, ``plane_2d``,
            ``grid_spec``, ``run_time``, ``setup_device``, and
            ``normalize_source``, ``material_grid``, and ``raster_options``.

        Returns
        -------
        Simulation
            A new immutable specification. The original simulation is unchanged.

        Raises
        ------
        TypeError
            If a replacement key is not a supported simulation field.
        ValueError
            If the reconstructed configuration violates a simulation invariant.

        Examples
        --------
        >>> longer = sim.updated_copy(time=np.arange(200) * sim.dt)
        >>> raw = sim.updated_copy(normalize_source=None)

        Notes
        -----
        Replacement goes through the normal constructor validation and therefore
        produces independent compilation-cache identity when numerical inputs change.
        """
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.init and item.compare
        }
        values["setup_device"] = self.setup_device_policy
        del values["setup_device_policy"]
        values["design"] = None if self.material_grid is not None else self.design
        allowed = values.keys()
        unknown = set(changes) - allowed
        if unknown:
            raise TypeError(
                f"Unknown Simulation field(s): {', '.join(sorted(unknown))}."
            )
        # 2. Preserve the existing coordinate translation unless the design itself changes; replacement devices must be shifted into that normalized domain.
        material_source_changed = "design" in changes or "material_grid" in changes
        offset = (0.0, 0.0, 0.0) if material_source_changed else self.coordinate_offset
        if offset != (0.0, 0.0, 0.0):
            for name in ("sources", "monitors"):
                if name in changes:
                    changes[name] = tuple(
                        _shift_device_to_domain(value, offset, plane_2d=self.plane_2d)
                        for value in changes[name] or ()
                    )
        values.update(changes)
        if "design" in changes and "material_grid" not in changes:
            values["material_grid"] = None
        if "material_grid" in changes and "design" not in changes:
            values["design"] = None
            if changes["material_grid"] is not None and "raster_options" not in changes:
                values["raster_options"] = None
        if self.material_grid is not None:
            inverse_offset = tuple(-value for value in self.coordinate_offset)
            for name in ("sources", "monitors"):
                if material_source_changed and name in changes:
                    continue
                values[name] = tuple(
                    _shift_device_to_domain(
                        value, inverse_offset, plane_2d=self.plane_2d
                    )
                    for value in values[name] or ()
                )
        # 4. Keep the two mutually exclusive time specifications mutually
        # exclusive during reconstruction. A simulation originally created from
        # run_time must regenerate its time grid when other fields change (for
        # example, resolution can change dt). An explicit replacement time grid
        # takes precedence over the stored run_time convenience specification.
        if "time" in changes and "run_time" not in changes:
            values["run_time"] = None
        elif "time" not in changes and (
            "run_time" in changes or self.run_time is not None
        ):
            values["time"] = None
        public_grid_spec = values["grid_spec"]
        if (
            self.material_grid is None
            and not material_source_changed
            and public_grid_spec is not None
            and offset != (0.0, 0.0, 0.0)
        ):
            # The retained design is already in domain coordinates, while mesh
            # requests remain in public coordinates, just like replacement devices.
            grid_offset = (
                offset
                if self.is_3d
                else physical_vector_to_grid_2d(offset, self.plane_2d)
            )
            values["grid_spec"] = _shift_grid_spec_to_domain(
                public_grid_spec, grid_offset
            )
        result = type(self)(**values)
        # Keep constraints public so subsequent copies translate them exactly once.
        object.__setattr__(result, "grid_spec", public_grid_spec)
        if self.material_grid is None and not material_source_changed:
            preserved_offset = offset
            target_plane = str(values["plane_2d"])
            if not self.is_3d and target_plane != self.plane_2d:
                preserved_offset = grid_vector_to_physical_2d(
                    physical_vector_to_grid_2d(offset, self.plane_2d), target_plane
                )
            object.__setattr__(result, "coordinate_offset", preserved_offset)
        return result

    @trace_preparation("material_grid")
    def _material_grid(self: Any, *, progress: bool = False):
        """Return Design's immutable cell-centered material raster."""
        from beamz.simulation import api as api_runtime

        if self.material_grid is not None:
            return self.material_grid
        token = self._material_grid_token()
        cached = api_runtime._MATERIAL_GRID_CACHE.get(token)
        if cached is None:
            options = _resolved_raster_options(
                self.raster_options,
                self.sources,
                boundaries=self.boundaries,
                is_3d=self.is_3d,
                plane_2d=self.plane_2d,
            )
            raster_kwargs = {
                "quality": options.quality,
                "smoothing": options.smoothing,
                "metal_smoothing": options.metal_smoothing,
                "reference_frequency": options.reference_frequency,
                "periodic_axes": options.periodic_axes,
            }
            with api_runtime._resolved_setup_device_context(self.setup_device_resolved):
                cached = api_runtime.build_material_grid(
                    self.design,
                    self.grid if self._uses_realized_grid else self.resolution,
                    progress=progress,
                    polarization=self.polarization,
                    **raster_kwargs,
                )
            api_runtime._MATERIAL_GRID_CACHE[token] = cached
            if len(api_runtime._MATERIAL_GRID_CACHE) > api_runtime._MAX_MATERIAL_GRIDS:
                api_runtime._MATERIAL_GRID_CACHE.popitem(last=False)
        else:
            api_runtime._MATERIAL_GRID_CACHE.move_to_end(token)
        return cached

    def _material_grid_token(self: Any):
        """Hash only values that change Design's cell-centered raster."""
        if self.material_grid is not None:
            return cache_token(self.material_grid)
        design = self.design
        return cache_token(
            (
                self.resolution,
                self.grid,
                design.width,
                design.height,
                design.depth,
                design.background,
                design.structures,
                _resolved_raster_options(
                    self.raster_options,
                    self.sources,
                    boundaries=self.boundaries,
                    is_3d=self.is_3d,
                    plane_2d=self.plane_2d,
                ),
                self.polarization,
            )
        )

    @property
    def pml_data(self: Any):
        """Return compiled absorber profiles for the current lattice.

        Returns
        -------
        mapping or None
            Grid-aligned PML/absorber coefficient arrays, or ``None`` when no
            absorber profile is present.

        Notes
        -----
        Accessing this property may rasterize and compile the simulation. It is
        intended for diagnostics; normal execution does not require it.
        """
        return self.compile().grid.pml_data
