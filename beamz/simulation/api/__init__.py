"""Public immutable simulation specification and convenience API."""

from __future__ import annotations

# Retain the names available from this module before its implementation was split.
import os as os
from collections import Counter as Counter
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from dataclasses import replace as replace
from importlib import import_module as import_module
from typing import Any, Literal

import numpy as np

from beamz._cache_tokens import cache_token
from beamz._helpers import env_bool as env_bool
from beamz._helpers import positive_float

# ``Simulation`` is an immutable specification; explicit ``SimulationState`` values
# carry all mutable runtime data.
from beamz.const import µm
from beamz.design.core import Design
from beamz.design.discretization import MaterialGrid
from beamz.design.discretization import build_material_grid as build_material_grid
from beamz.design.grid import RectilinearGrid
from beamz.design.grid_spec import (
    GridSpec,
)
from beamz.design.materials import Material as Material
from beamz.design.materials import MaterialProtocol as MaterialProtocol
from beamz.design.structures import Box as Box
from beamz.devices.boundaries import (
    normalize_boundaries,
    validate_boundary_compatibility,
)
from beamz.devices.boundaries import periodic_storage_axes as periodic_storage_axes
from beamz.devices.monitors.monitors import (
    FieldMonitor,
    FieldRecorder,
    FluxMonitor,
    ModeMonitor,
    _Monitor,
)
from beamz.devices.sources import CANONICAL_SOURCE_TYPES as CANONICAL_SOURCE_TYPES
from beamz.devices.sources import GaussianBeamSource as GaussianBeamSource
from beamz.devices.sources import PlaneWaveSource as PlaneWaveSource
from beamz.lattice import grid_vector_to_physical_2d as grid_vector_to_physical_2d
from beamz.lattice import in_plane_vector_2d as in_plane_vector_2d
from beamz.lattice import (
    normalize_polarization_2d,
)
from beamz.lattice import physical_vector_to_grid_2d as physical_vector_to_grid_2d
from beamz.simulation.api.configuration import _SimulationConfiguration
from beamz.simulation.api.plotting import _SimulationPresentation
from beamz.simulation.api.preparation import (
    _analysis_function,
    _design_depth,
    _design_domain,
    _design_is_3d,
    _devices_require_uniform_grid,
    _material_index,
    _max_index_for_specs,
    _normalize_devices,
    _normalize_domain,
    _normalize_plane_2d,
    _normalize_time,
    _prepare_design,
    _prepare_material_grid,
    _rasterize_scene_for_simulation,
    _resolve_design_time_and_grid,
    _resolve_grid_resolution,
    _resolve_setup_device,
    _resolved_raster_options,
    _setup_device_policy_label,
    _shift_device_to_domain,
    _shift_grid_spec_to_domain,
    _structure_to_domain,
    _time_from_run_time,
)
from beamz.simulation.api.workflow import _SimulationExecution
from beamz.simulation.compile import CompiledProgramKey as CompiledProgramKey
from beamz.simulation.compile import (
    _resolved_setup_device_context as _resolved_setup_device_context,
)
from beamz.simulation.compile import clear_program_cache as clear_program_cache
from beamz.simulation.compile import compile_program as compile_program
from beamz.simulation.compile import compile_simulation as compile_simulation
from beamz.simulation.compile import simulation_memory_estimate as memory_estimate
from beamz.simulation.compile.backend import trace_preparation as trace_preparation
from beamz.simulation.execute import clear_execution_cache as clear_execution_cache
from beamz.simulation.execute import (
    compiled_xla_memory_analysis as analyze_compiled_xla_memory,
)
from beamz.simulation.execute import execute_step as execute_step
from beamz.simulation.execute import run_simulation_program as run_simulation_program
from beamz.simulation.execute import run_until_terminated as run_until_terminated
from beamz.simulation.execute.sharding import (
    normalize_sharding_config as normalize_sharding_config,
)
from beamz.simulation.model import AutoTermination as AutoTermination
from beamz.simulation.model import CompiledProgram as CompiledProgram
from beamz.simulation.model import DomainSpec as DomainSpec
from beamz.simulation.model import RunConfig as RunConfig
from beamz.simulation.model import SimulationRequest as SimulationRequest
from beamz.simulation.model import SimulationState as SimulationState
from beamz.simulation.results import MonitorResults as MonitorResults
from beamz.simulation.results import SimulationResults as SimulationResults
from beamz.simulation.results import SimulationRun as SimulationRun

_MONITOR_TYPES = (FieldMonitor, FieldRecorder, FluxMonitor, ModeMonitor)
_MAX_MATERIAL_GRIDS = 4
_MATERIAL_GRID_CACHE: OrderedDict[Any, Any] = OrderedDict()


@dataclass(frozen=True, init=False, eq=False)
class Simulation:
    """Define an immutable FDTD simulation specification.

    ``Simulation`` contains configuration only: geometry, devices, boundaries,
    discretization, and the time grid. It never stores evolving electromagnetic
    fields. Use :meth:`run` for the normal complete-simulation workflow and
    :meth:`advance` only when an application needs explicit continuation state.

    Parameters
    ----------
    design : Design, optional
        Physical geometry and material distribution. Supply exactly one of
        ``design``, ``material_grid``, or ``scene``.
    material_grid : MaterialGrid, optional
        A compatible uniform or rectilinear solver grid, such as one produced by
        ``MaterialGrid.from_raster_result()``. This is the simulation bridge for
        imported GDS, STL, Gmsh, and raw-mesh scenes.
    scene : beamz.design.raster.Scene, optional
        Reusable scene produced by an importer. Supply it with ``raster_grid``;
        BeamZ rasterizes and validates it during simulation setup.
    raster_grid : beamz.design.raster.Grid, optional
        Explicit uniform or nonuniform rectilinear solver grid for ``scene``.
    sources : sequence, optional
        Immutable source specifications to inject during execution.
    monitors : sequence of monitor specifications, optional
        Quantities to record. Results are keyed by monitor name.
    boundaries : sequence of PEC, PML, Absorber, Periodic, or Bloch, optional
        Domain boundary conditions. An all-edge PEC boundary is used when omitted.
    resolution : float, default=0.02 * um
        Uniform cell spacing in metres when ``grid_spec`` does not override it.
        Geometry-aware specifications expose their exact mesh through ``grid``;
        this attribute remains a representative scalar for compatibility.
    time : numpy.ndarray, optional
        One-dimensional, uniformly spaced time samples in seconds. Exactly one of
        ``time`` and ``run_time`` may be supplied.
    plane_2d : {"xy", "xz", "yz"}, default="xy"
        Physical plane represented by a two-dimensional design.
    polarization : {"tm", "te"}, default="tm"
        Electromagnetic polarization for a 2D simulation. ``"tm"`` advances the
        out-of-plane electric field; ``"te"`` advances the out-of-plane magnetic
        field. Three-dimensional simulations require the default.
    domain, size : sequence of float, optional
        Alternative domain extents in public ``(x, y, z)`` order. Passing both is
        invalid. Two values create a 2D domain.
    background : Material, optional
        Background material used when constructing a design from ``domain`` or
        ``size``.
    grid_spec : GridSpec, optional
        Automatic or explicit spatial-discretization policy.
    run_time : float, optional
        Total physical duration in seconds. BeamZ derives a stable time grid when
        this is supplied instead of ``time``.
    setup_device : {"auto", "cpu", "default"}, optional
        Device policy for setup-time rasterization and lowering. Runtime placement
        is controlled independently by ``sharding`` on execution methods.
    normalize_source : int or None, default=0
        Source index used to normalize frequency-domain monitor data. Use ``None``
        to retain raw acquisitions.
    raster_options : RasterOptions, optional
        Native raster quality and smoothing policy. Component selection remains
        automatic for the simulation dimensionality. Simulations default to
        ``farjadpour_diagonal`` for dielectrics and staircasing for metals
        (Re(epsilon_r) < 1 at the mean source center frequency). Select
        ``metal_smoothing="inherit"`` to smooth metals too, or
        ``farjadpour_full`` to retain lossless
        off-diagonal electric coupling.

    Notes
    -----
    The three execution methods have deliberately different contracts:

    - :meth:`run` executes the complete time grid and returns only durable,
      immutable :class:`SimulationResults`.
    - :meth:`advance` executes a fresh or continued segment and returns a
      :class:`SimulationRun` containing both results and the next
      :class:`SimulationState`.
    - :meth:`step` advances state by exactly one timestep without constructing
      analysis results. It is intended for debugging and numerical verification.

    Compilation is lazy and cached. Calling :meth:`run` does not require a prior
    call to :meth:`compile`.

    Examples
    --------
    Run a complete simulation and inspect a named monitor:

    >>> import numpy as np
    >>> import beamz as bz
    >>> sim = bz.Simulation(
    ...     domain=(4 * bz.um, 3 * bz.um),
    ...     time=np.arange(100) * 1e-16,
    ...     monitors=[bz.FieldRecorder(("Ez",), name="fields")],
    ... )
    >>> results = sim.run()
    >>> frames = results["fields"].fields["Ez"]

    Execute two continuation segments while preserving the branch point:

    >>> first = sim.advance(num_steps=40)
    >>> second = sim.advance(state=first.state, num_steps=30)
    >>> alternative = sim.advance(state=first.state, num_steps=20)
    """

    # The class stores semantic configuration only; mutable compilation and execution resources are intentionally external.
    design: Design
    material_grid: MaterialGrid | None
    sources: tuple[object, ...]
    monitors: tuple[_Monitor, ...]
    boundaries: tuple[object, ...]
    resolution: float
    grid: RectilinearGrid = field(init=False)
    time: np.ndarray
    plane_2d: str
    polarization: str
    grid_spec: GridSpec | None
    run_time: float | None
    setup_device_policy: str
    setup_device_resolved: str = field(compare=False)
    normalize_source: int | None
    raster_options: Any
    coordinate_offset: tuple[float, float, float] = field(compare=False)
    _uses_realized_grid: bool = field(compare=False)

    def __init__(
        self,
        design: Design | None = None,
        sources: Sequence[Any] | None = None,
        monitors: Sequence[_Monitor] | None = None,
        boundaries: Sequence[Any] | None = None,
        resolution: float = 0.02 * µm,
        time: np.ndarray | None = None,
        plane_2d: str = "xy",
        polarization: Literal["tm", "te"] = "tm",
        *,
        domain=None,
        size=None,
        background=None,
        material_grid: MaterialGrid | None = None,
        scene=None,
        raster_grid=None,
        grid_spec=None,
        run_time: float | None = None,
        setup_device: Literal["auto", "cpu", "default"] | None = None,
        normalize_source: int | None = 0,
        raster_options=None,
    ):
        polarization = normalize_polarization_2d(polarization)
        plane_2d = _normalize_plane_2d(plane_2d)
        if time is not None and run_time is not None:
            raise ValueError("Pass only one of time=... or run_time=....")
        resolution = positive_float(resolution, name="Simulation resolution")
        if run_time is not None:
            run_time = positive_float(run_time, name="run_time")
        if scene is not None or raster_grid is not None:
            if scene is None or raster_grid is None:
                raise ValueError("Pass scene and raster_grid together.")
            if design is not None or material_grid is not None:
                raise ValueError(
                    "Pass exactly one material source: design, material_grid, or scene."
                )
            if background is not None:
                raise ValueError("background cannot be combined with scene.")
            if grid_spec is not None:
                raise ValueError(
                    "grid_spec cannot be combined with scene; use raster_grid."
                )
            # Raster topology and boundary compilation must see the same input,
            # including when boundaries were supplied as a one-shot iterable.
            boundaries = normalize_boundaries(boundaries)
            material_grid = _rasterize_scene_for_simulation(
                scene,
                raster_grid,
                raster_options,
                polarization,
                sources=sources,
                boundaries=boundaries,
                plane_2d=plane_2d,
            )
            raster_options = None
        require_uniform_grid = _devices_require_uniform_grid(sources, monitors)
        # 1. Normalize alternative domain, grid, and time forms into one concrete design;
        # this also returns any coordinate translation introduced by size-based domains.
        if material_grid is None:
            design, resolution, grid, time, offset, uses_realized_grid = (
                _prepare_design(
                    design=design,
                    domain=domain,
                    size=size,
                    background=background,
                    grid_spec=grid_spec,
                    resolution=resolution,
                    time=time,
                    run_time=run_time,
                    require_uniform_grid=require_uniform_grid,
                    plane_2d=plane_2d,
                )
            )
        else:
            design, resolution, grid, time, offset, uses_realized_grid = (
                _prepare_material_grid(
                    material_grid,
                    design=design,
                    domain=domain,
                    size=size,
                    background=background,
                    grid_spec=grid_spec,
                    raster_options=raster_options,
                    time=time,
                    run_time=run_time,
                    plane_2d=plane_2d,
                )
            )
        # 2. Shift sources and monitors into the normalized domain coordinate system so their public positions remain physically unchanged.
        if offset != (0.0, 0.0, 0.0):
            sources = tuple(
                _shift_device_to_domain(s, offset, plane_2d=plane_2d)
                for s in (sources or ())
            )
            monitors = tuple(
                _shift_device_to_domain(m, offset, plane_2d=plane_2d)
                for m in (monitors or ())
            )

        # 3. Canonicalize devices, time, plane, and boundaries before resolving the setup
        # device; subsequent cache identity depends on these normalized values.
        is_3d = _design_is_3d(design)
        if is_3d and polarization != "tm":
            raise ValueError("polarization applies only to 2D simulations.")
        if (
            material_grid is not None
            and not is_3d
            and material_grid.polarization != polarization
        ):
            raise ValueError(
                "MaterialGrid polarization does not match Simulation polarization."
            )
        sources, monitors = _normalize_devices(
            sources=sources or (), monitors=monitors or ()
        )
        if normalize_source is not None:
            normalize_source = int(normalize_source)
            if normalize_source < 0:
                raise ValueError(
                    "normalize_source must be a non-negative index or None."
                )
            if sources and normalize_source >= len(sources):
                raise ValueError(
                    f"normalize_source={normalize_source} exceeds {len(sources)} sources."
                )
        time = _normalize_time(time)
        boundaries = normalize_boundaries(boundaries)
        validate_boundary_compatibility(boundaries, is_3d=is_3d, plane_2d=plane_2d)
        if raster_options is not None:
            from beamz.design.raster import RasterOptions

            if not isinstance(raster_options, RasterOptions):
                raise TypeError("raster_options must be RasterOptions or None.")
        setup_device_policy = _setup_device_policy_label(setup_device)
        setup_device_resolved = _resolve_setup_device(setup_device_policy)
        # 4. Assign the fully normalized specification atomically through the frozen
        # dataclass escape hatch so no partially initialized simulation is observable.
        coordinate_offset = tuple(float(v) for v in offset)
        _uses_realized_grid = uses_realized_grid
        values = locals()
        for item in fields(self):
            object.__setattr__(self, item.name, values[item.name])

    @property
    def size(self):
        """Return the domain extents in public ``(x, y, z)`` order.

        Returns
        -------
        tuple of float
            Width, height, and depth in metres. A 2D simulation has zero depth.
        """
        return _design_domain(self.design)

    @property
    def domain(self):
        """Return the domain extents; an alias of :attr:`size`.

        Returns
        -------
        tuple of float
            Domain extents in public ``(x, y, z)`` order.
        """
        return self.size

    @property
    def is_3d(self):
        """Return whether the simulation has a positive z extent.

        Returns
        -------
        bool
            ``True`` for a three-dimensional lattice and ``False`` for a 2D plane.
        """
        return _design_is_3d(self.design)

    @property
    def dt(self):
        """Return the uniform simulation timestep.

        Returns
        -------
        float
            Difference between consecutive entries of :attr:`time`, in seconds.
        """
        return float(self.time[1] - self.time[0])

    @property
    def num_steps(self):
        """Return the number of timesteps executed by :meth:`run`.

        Returns
        -------
        int
            Length of the simulation time grid.
        """
        return int(self.time.size)

    def canonical_spec(self):
        """Return the immutable values defining simulation identity.

        Returns
        -------
        tuple
            Canonical design, device, boundary, grid, time, and normalization
            values used by equality, hashing, and compilation caches.

        Notes
        -----
        This is an advanced introspection method. Use :meth:`updated_copy` to
        construct a modified simulation instead of editing the returned values.
        """
        return tuple(getattr(self, item.name) for item in fields(self) if item.compare)

    def __eq__(self, other):
        # Compare canonical cache tokens so array content, not object identity, defines equality.
        if not isinstance(other, Simulation):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        # Hash the same canonical specification used by equality to preserve the hash contract.
        return hash(cache_token(self.canonical_spec()))

    updated_copy = _SimulationConfiguration.updated_copy

    _material_grid = _SimulationConfiguration._material_grid

    _material_grid_token = _SimulationConfiguration._material_grid_token

    pml_data = _SimulationConfiguration.pml_data

    initial_state = _SimulationExecution.initial_state

    step = _SimulationExecution.step

    to_request = _SimulationExecution.to_request

    compile = _SimulationExecution.compile

    differentiable = _SimulationExecution.differentiable

    clear_compiled_cache = _SimulationExecution.clear_compiled_cache

    memory_estimate = _SimulationExecution.memory_estimate  # noqa: F811

    compiled_xla_memory_analysis = _SimulationExecution.compiled_xla_memory_analysis

    advance = _SimulationPresentation.advance

    run = _SimulationPresentation.run

    plot = _SimulationPresentation.plot

    plot_eps = _SimulationPresentation.plot_eps

    show = _SimulationPresentation.show

    view3d = _SimulationPresentation.view3d

    show3d = _SimulationPresentation.show3d
