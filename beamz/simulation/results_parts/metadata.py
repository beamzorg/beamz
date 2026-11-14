"""Detached raw outputs produced by simulation execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType

import numpy as np

from beamz._cache_tokens import cache_token
from beamz.design.grid import RectilinearGrid
from beamz.devices._immutable import immutable_snapshot, readonly_array
from beamz.lattice import canonical_component_2d
from beamz.simulation.model import CompiledGrid


@dataclass(frozen=True, slots=True)
class SimulationPerformance:
    """JIT- and setup-free execution statistics for one simulation run.

    ``runtime_s`` measures only the ready compiled executable, from dispatch until
    its device work completes. Plan construction, JIT compilation, runtime setup,
    and result decoding are excluded.
    """

    runtime_s: float
    cells: int
    steps: int

    def __post_init__(self) -> None:
        runtime_s = float(self.runtime_s)
        if not np.isfinite(runtime_s) or runtime_s <= 0.0:
            raise ValueError(
                "SimulationPerformance.runtime_s must be positive and finite."
            )
        for name in ("cells", "steps"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) <= 0
            ):
                raise ValueError(
                    f"SimulationPerformance.{name} must be a positive integer."
                )
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "runtime_s", runtime_s)

    @property
    def gcups(self) -> float:
        """Billions of cell updates per second for this run."""
        return self.cells * self.steps / self.runtime_s / 1e9


@dataclass(frozen=True, slots=True)
class RunTermination:
    """Describe why a bounded simulation run stopped and its final residuals.

    Attributes
    ----------
    reason : {"converged", "time_limit", "wall_time_limit", "nonfinite", "diverged"}
        Terminal condition reached by the execution controller.
    steps : int
        Number of timesteps executed.
    time : float
        Physical time of the final runtime state, in seconds.
    converged : bool
        Whether every applicable convergence criterion passed.
    field_decay, monitor_change, source_decay : float or None
        Final normalized field-energy, monitor-change, and remaining-source residuals.
    energy, peak_energy, max_field : float or None
        Final integrated energy, recorded peak energy, and maximum absolute field.
    consecutive_checks : int
        Number of successful checks accumulated at termination.
    """

    reason: str
    steps: int
    time: float
    converged: bool
    field_decay: float | None = None
    monitor_change: float | None = None
    source_decay: float | None = None
    energy: float | None = None
    peak_energy: float | None = None
    max_field: float | None = None
    consecutive_checks: int = 0

    def __post_init__(self) -> None:
        reasons = {
            "converged",
            "time_limit",
            "wall_time_limit",
            "nonfinite",
            "diverged",
        }
        reason = str(self.reason)
        if reason not in reasons:
            raise ValueError(f"RunTermination.reason must be one of {sorted(reasons)}.")
        object.__setattr__(self, "reason", reason)
        for name in ("steps", "consecutive_checks"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(
                    f"RunTermination.{name} must be a non-negative integer."
                )
            object.__setattr__(self, name, int(value))
        time = float(self.time)
        if not np.isfinite(time):
            raise ValueError("RunTermination.time must be finite.")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "converged", bool(self.converged))
        for name in (
            "field_decay",
            "monitor_change",
            "source_decay",
            "energy",
            "peak_energy",
            "max_field",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))


def _analysis_function(module: str, name: str):
    """Resolve analysis lazily so result values do not import it eagerly."""
    return getattr(import_module(f"beamz.analysis.{module}"), name)


@dataclass(frozen=True, slots=True)
class MaterialRegion:
    """Detached material neighborhood retained for deferred analysis.

    Tensor and Yee coefficients preserve the material model used by mode sources
    and propagation. Rebuilding them from scalar permittivity loses interface
    smoothing and anisotropy. All arrays use the same global ``origin``; Yee
    arrays may include an extra upper sample for node-aligned components.
    """

    permittivity: np.ndarray
    permeability: np.ndarray
    origin: tuple[int, ...]
    full_shape: tuple[int, ...]
    material_tensors: Mapping[str, np.ndarray] = field(default_factory=dict)
    yee_materials: Mapping[str, np.ndarray] = field(default_factory=dict)
    yee_tensors: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "permittivity", readonly_array(self.permittivity))
        object.__setattr__(self, "permeability", readonly_array(self.permeability))
        object.__setattr__(self, "origin", tuple(int(value) for value in self.origin))
        object.__setattr__(
            self, "full_shape", tuple(int(value) for value in self.full_shape)
        )
        for name in ("material_tensors", "yee_materials", "yee_tensors"):
            object.__setattr__(self, name, immutable_snapshot(getattr(self, name)))


def _snapshot_material_region(runtime_fields, crop, origin):
    """Keep the coefficients used by the source's 3D mode solve, including halos."""
    permittivity = np.asarray(runtime_fields.permittivity)
    permeability = np.asarray(runtime_fields.permeability)
    material_grid = runtime_fields.material_grid
    tensors = {}
    yee = {}
    # Node-aligned coefficients retain one upper sample beyond the voxel crop.
    component_crop = tuple(
        slice(index.start, None if index.stop is None else index.stop + 1)
        for index in crop
    )
    if permittivity.ndim in (2, 3):
        tensors = {
            name: np.asarray(values)[(slice(None), *crop)]
            for name, values in material_grid.tensors.items()
            if name in {"epsilon", "mu"}
        }
        if material_grid.uses_direct_yee_materials:
            yee = {
                name: np.asarray(values)[component_crop]
                for name, values in material_grid.yee_materials.items()
                if name in {"eps_x", "eps_y", "eps_z", "mu_hx", "mu_hy", "mu_hz"}
            }
    support_tensors = {
        name: np.asarray(values)[(slice(None), *component_crop)]
        for name, values in material_grid.yee_tensors.items()
    }
    return MaterialRegion(
        permittivity[crop],
        permeability if permeability.ndim == 0 else permeability[crop],
        tuple(origin),
        permittivity.shape,
        material_tensors=tensors,
        yee_materials=yee,
        yee_tensors=support_tensors,
    )


def material_region_for_monitor(result, *, runtime_fields):
    """Retain the material halo around the compiler's realized sampling plane."""
    region = result.sample_region
    if not result.dft_frequencies.size or region is None:
        return None
    full_shape = runtime_fields.permittivity.shape
    axes = ("z", "y", "x") if len(full_shape) == 3 else ("y", "x")
    if region.normal_axis not in axes:
        return None
    dim = axes.index(region.normal_axis)
    plane = int(region.plane_index)
    plane = int(np.clip(plane, 0, full_shape[dim] - 1))
    from beamz.devices.modes.plane import TENSOR_MODE_NORMAL_HALO

    halo = TENSOR_MODE_NORMAL_HALO if runtime_fields.material_grid.yee_tensors else 0
    start, stop = max(0, plane - 2 - halo), min(full_shape[dim], plane + 3 + halo)
    crop = [slice(None)] * len(full_shape)
    crop[dim] = slice(start, stop)
    origin = [0] * len(full_shape)
    origin[dim] = start
    return _snapshot_material_region(runtime_fields, tuple(crop), origin)


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """Describe stored field-array shapes and optional material data.

    Attributes
    ----------
    grid_shape : tuple of int
        Shape of the cell-centered material grid in canonical storage order.
    component_shapes : mapping of str to tuple of int
        Public ``Ex`` through ``Hz`` component shapes.
    materials : MaterialRegion or None
        Optional immutable material snapshot retained for analysis.
    """

    grid_shape: tuple[int, ...]
    component_shapes: Mapping[str, tuple[int, ...]]
    materials: MaterialRegion | None = None

    def __post_init__(self):
        if self.materials is not None and not isinstance(
            self.materials, MaterialRegion
        ):
            raise TypeError("FieldMetadata.materials must be a MaterialRegion or None.")
        if not isinstance(self.component_shapes, Mapping):
            raise TypeError("FieldMetadata.component_shapes must be a mapping.")
        object.__setattr__(self, "grid_shape", tuple(int(v) for v in self.grid_shape))
        object.__setattr__(
            self,
            "component_shapes",
            MappingProxyType(
                {
                    str(name): tuple(int(v) for v in shape)
                    for name, shape in self.component_shapes.items()
                }
            ),
        )

    @property
    def permittivity(self):
        """Return stored relative permittivity, if available.

        Returns
        -------
        numpy.ndarray or None
            Read-only material-region permittivity, or ``None`` when material data
            was not retained.
        """
        return None if self.materials is None else self.materials.permittivity

    @property
    def permeability(self):
        """Return stored relative permeability, if available.

        Returns
        -------
        numpy.ndarray or None
            Read-only material-region permeability, or ``None`` when material data
            was not retained.
        """
        return None if self.materials is None else self.materials.permeability


@dataclass(frozen=True, slots=True, eq=False)
class SimulationMetadata:
    """Store compact immutable context required to analyze results.

    Attributes
    ----------
    dt, resolution : float
        Simulation timestep in seconds and spatial cell size in metres.
    is_3d : bool
        Whether the source simulation used a three-dimensional lattice.
    plane_2d : {"xy", "xz", "yz"}
        Physical plane represented by two-dimensional field arrays.
    polarization_2d : {"tm", "te"}
        Polarization used to interpret two-dimensional field arrays.
    coordinate_offset : tuple of float
        Translation added to public coordinates to obtain solver-local coordinates.
    time : numpy.ndarray
        Complete immutable simulation time grid in seconds.
    width, height, depth : float
        Domain extents in metres.
    fields : FieldMetadata
        Field shapes and optional material information.
    grid : RectilinearGrid, optional
        Exact physical grid used by the simulation. Older manually constructed
        metadata may omit it and fall back to ``resolution``.

    Notes
    -----
    Metadata intentionally contains values needed for interpretation rather than a
    reference to the originating ``Simulation``. Results therefore remain usable
    after the simulation object is deleted.
    """

    dt: float
    resolution: float
    is_3d: bool
    plane_2d: str
    coordinate_offset: tuple[float, float, float]
    time: np.ndarray
    width: float
    height: float
    depth: float
    fields: FieldMetadata
    polarization_2d: str = "tm"
    grid: RectilinearGrid | None = None

    def __post_init__(self):
        if not isinstance(self.fields, FieldMetadata):
            raise TypeError("SimulationMetadata.fields must be FieldMetadata.")
        if self.grid is not None and not isinstance(self.grid, RectilinearGrid):
            raise TypeError("SimulationMetadata.grid must be RectilinearGrid or None.")
        offset = tuple(float(value) for value in self.coordinate_offset)
        if len(offset) != 3:
            raise ValueError(
                "SimulationMetadata.coordinate_offset must have three values."
            )
        time = readonly_array(self.time, dtype=float)
        if time.ndim != 1 or not np.all(np.isfinite(time)):
            raise ValueError("SimulationMetadata.time must be a finite 1D array.")
        plane_2d = str(self.plane_2d).lower()
        if plane_2d not in {"xy", "xz", "yz"}:
            raise ValueError("SimulationMetadata.plane_2d must be 'xy', 'xz', or 'yz'.")
        polarization_2d = str(self.polarization_2d).lower()
        if polarization_2d not in {"tm", "te"}:
            raise ValueError("SimulationMetadata.polarization_2d must be 'tm' or 'te'.")
        for name in ("dt", "resolution", "width", "height", "depth"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"SimulationMetadata.{name} must be finite.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "is_3d", bool(self.is_3d))
        object.__setattr__(self, "plane_2d", plane_2d)
        object.__setattr__(self, "polarization_2d", polarization_2d)
        object.__setattr__(self, "coordinate_offset", offset)
        object.__setattr__(self, "time", time)

    def canonical_spec(self):
        """Return immutable values defining metadata equality and hashing.

        Returns
        -------
        tuple
            Canonical scalar, coordinate, time-grid, and field metadata values.
        """
        return (
            self.dt,
            self.resolution,
            self.is_3d,
            self.plane_2d,
            self.polarization_2d,
            self.coordinate_offset,
            self.time,
            self.width,
            self.height,
            self.depth,
            self.fields,
            self.grid,
        )

    def __eq__(self, other):
        if not isinstance(other, SimulationMetadata):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))

    @classmethod
    def from_simulation(
        cls,
        simulation,
        *,
        runtime_fields: CompiledGrid,
        store_full_materials: bool = False,
    ) -> "SimulationMetadata":
        """Detach analysis metadata from a simulation and compiled lattice.

        Parameters
        ----------
        simulation : Simulation
            Immutable source specification.
        runtime_fields : CompiledGrid
            Compiled lattice providing material and Yee-component shapes.
        store_full_materials : bool, default=False
            Retain full-domain permittivity and permeability arrays.

        Returns
        -------
        SimulationMetadata
            Independent, immutable analysis context.

        Raises
        ------
        TypeError
            If either input is not a canonical BeamZ simulation or compiled grid.

        Notes
        -----
        Execution uses this constructor internally. Normal applications read the
        ``metadata`` attribute of ``SimulationResults``.
        """
        from beamz.simulation.api import Simulation

        if not isinstance(simulation, Simulation):
            raise TypeError("Result metadata requires a canonical Simulation.")
        if not isinstance(runtime_fields, CompiledGrid):
            raise TypeError("Result metadata requires a canonical CompiledGrid.")
        fields = runtime_fields
        design = simulation.design
        offset = simulation.coordinate_offset
        grid_shape = tuple(int(v) for v in np.asarray(fields.permittivity).shape)
        components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        component_shapes = {}
        for name in components:
            canonical = (
                name
                if simulation.is_3d
                else canonical_component_2d(
                    name, simulation.plane_2d, simulation.polarization
                )
            )
            component = None if canonical is None else getattr(fields, canonical, None)
            component_shapes[name] = (
                (1, 1) if component is None else tuple(int(v) for v in component.shape)
            )
        materials = (
            _snapshot_material_region(
                fields, (slice(None),) * len(grid_shape), (0,) * len(grid_shape)
            )
            if store_full_materials
            else None
        )
        return cls(
            dt=float(simulation.dt),
            resolution=float(simulation.resolution),
            is_3d=bool(simulation.is_3d),
            plane_2d=str(simulation.plane_2d),
            coordinate_offset=(offset[0], offset[1], offset[2]),
            time=simulation.time,
            width=float(design.width),
            height=float(design.height),
            depth=float(design.depth),
            fields=FieldMetadata(
                grid_shape=grid_shape,
                component_shapes=component_shapes,
                materials=materials,
            ),
            polarization_2d=simulation.polarization,
            # Field arrays are indexed on the compiler's normalized geometry. Imported
            # material grids may retain a nonzero public origin on ``simulation.grid``.
            grid=runtime_fields.geometry,
        )
