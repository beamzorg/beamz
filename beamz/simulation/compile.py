"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

import beamz.simulation.kernels as ops
from beamz._cache_tokens import HashToken, cache_token
from beamz._region_array import RegionArray, constant_array, tiles, uniform_value
from beamz.devices._boundary_compile import (
    CPML_3D_E_DERIVATIVES,
    CPML_3D_H_DERIVATIVES,
    BoundaryData,
    lower_boundaries,
)
from beamz.devices.boundaries import Periodic, bloch_storage_phases, bloch_wavevector
from beamz.devices.monitors.compiler import compile_monitor_specs
from beamz.devices.sources.compiler import compile_source_specs
from beamz.lattice import (
    build_material_coefficients,
    component_shapes,
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_e_component_3d_centered,
)
from beamz.simulation.model import (
    BoundaryPlan,
    CompiledGrid,
    CompiledProgram,
    CpmlPlan,
    DerivativeMetricPlan,
    MetallicPlan,
    RunConfig,
    ShardingConfig,
    ShardingPlan,
    ShardingToken,
    SimulationRequest,
    UpdateCoefficients,
)
from beamz.simulation.preparation_trace import preparation_phase, trace_preparation
from beamz.simulation.region_materials import (
    build_region_materials,
    electric_coefficients,
    magnetic_coefficients,
)

from .memory import cuda_memory_policy_from_env
from .observe import MONITOR_FIELDS, empty_monitor_values
from .sharding import (
    build_sharding_plan,
    lower_compiled_arrays,
    lower_derivative_metrics,
    normalize_sharding_config,
    sharding_cache_token,
)

# Cache identity contains only values that affect generated code or static storage;
# runtime field values stay excluded so one executable can serve many states.


@dataclass(frozen=True, slots=True)
class CompiledProgramKey:
    # Group these values because their positional and shape relationships form one
    # invariant.
    num_steps: int
    total_steps: int
    t0: float
    dt: float
    is_3d: bool
    plane_2d: str
    polarization_2d: str
    loop_kind: str
    source_single_slab_dense: bool
    backend: str
    cuda_flags: int
    cuda_graph_cache_capacity: int
    cuda_storage_axes: tuple[int, int, int]
    sharding: ShardingToken
    materials: HashToken
    sources: tuple[HashToken, ...]
    monitors: tuple[HashToken, ...]
    boundaries: tuple[HashToken, ...]
    cuda_tuning_policy: tuple = ()
    cuda_memory_policy: str = "auto"
    cuda_schedule: str = "streamed"
    cuda_expected_runs: int = 1

    @classmethod
    def from_request(cls, request: SimulationRequest) -> CompiledProgramKey:
        # Reduce rich objects to semantic tokens; large arrays must not become key members.
        return cls(
            request.run.num_steps,
            request.run.total_steps,
            request.run.t0,
            request.run.dt,
            request.domain.is_3d,
            request.domain.plane_2d,
            request.domain.polarization_2d,
            request.run.loop_kind,
            request.run.source_single_slab_dense,
            request.run.backend,
            request.run.cuda_flags,
            request.run.cuda_graph_cache_capacity,
            request.run.cuda_storage_axes,
            request.run.sharding,
            cache_token(request.materials),
            tuple(cache_token(source) for source in request.sources),
            tuple(cache_token(monitor) for monitor in request.monitors),
            tuple(cache_token(boundary) for boundary in request.boundaries),
            cuda_memory_policy=request.run.cuda_memory_policy,
            cuda_schedule=request.run.cuda_schedule,
            cuda_expected_runs=request.run.cuda_expected_runs,
        )


def _preparation_key(key: CompiledProgramKey):
    # Horizon and observation/source plans do not change constitutive arrays.
    # CUDA schedules share constitutive arrays. Kernel/workspace choices must
    # not create extra material banks when trying several launch configurations.
    from ._cuda_abi import (
        CUDA_BF16_PSI,
        CUDA_FIELD_PAD32,
        CUDA_FIELD_PAD64,
        CUDA_FIELD_PAD_Y8,
    )

    cuda_backend = key.backend in {"cuda_streamed", "cuda_mstep"}

    return replace(
        key,
        num_steps=0,
        total_steps=0,
        t0=0,
        sources=(),
        monitors=(),
        cuda_tuning_policy=(),
        cuda_schedule="streamed",
        cuda_expected_runs=1,
        backend="cuda_streamed" if cuda_backend else key.backend,
        cuda_flags=key.cuda_flags
        & (CUDA_BF16_PSI | CUDA_FIELD_PAD32 | CUDA_FIELD_PAD64 | CUDA_FIELD_PAD_Y8)
        if cuda_backend
        else key.cuda_flags,
        cuda_memory_policy="auto",
        cuda_graph_cache_capacity=0,
        # Storage permutations are applied by the runtime to these canonical arrays.
        cuda_storage_axes=(0, 1, 2) if cuda_backend else key.cuda_storage_axes,
    )


_MAX_COMPILED_PROGRAMS = 4
_PROGRAM_CACHE: OrderedDict[CompiledProgramKey, CompiledProgram] = OrderedDict()


def _resolved_setup_device_context(resolved_device):
    """Place setup work on CPU only when explicitly requested."""
    if str(resolved_device).strip().lower() != "cpu":
        return nullcontext(None)
    try:
        devices = jax.devices("cpu")
        return jax.default_device(devices[0]) if devices else nullcontext(None)
    except Exception:
        return nullcontext(None)


def clear_program_cache() -> None:
    """Drop cached immutable compiled plans."""
    _PROGRAM_CACHE.clear()


def _elide_zero_conductivity_grid(value):
    """Return a scalar zero when a conductivity-like grid is identically zero."""

    # Zero padding/disabled state explicitly so non-physical cells cannot inject
    # energy.
    if isinstance(value, RegionArray):
        for tile in tiles(value.shape, value.dtype):
            if np.any(value[tile] != 0):
                return value
        return jnp.asarray(0, dtype=value.dtype)
    arr_np = np.asarray(value)
    if arr_np.size and not bool(np.any(arr_np != 0.0)):
        return jnp.asarray(0.0, dtype=getattr(value, "dtype", jnp.float32))
    return value


def _elide_uniform_grid(value):
    """Represent an exactly uniform CUDA coefficient with one scalar value."""

    first = uniform_value(value)
    if first is None:
        return value
    return jnp.asarray(first, dtype=value.dtype)


_CUDA_CODEBOOK_CHUNK_CELLS = 1 << 20


def _pack_cuda_coefficient_ids(value, *, max_values: int = 256):
    """Pack four low-cardinality FP32 coefficient IDs into each int32 word.

    The CUDA streamed kernel recognizes the resulting one-dimensional S32 array
    and uses the returned exact FP32 table for lookup.  Keeping this transform at
    compile time avoids carrying three dense material coefficient grids through
    DRAM on every electric update.
    """

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.size == 0:
        return None
    # Discover the small sorted table without full-volume int64 inverse/sort
    # arrays. A second bounded pass writes directly into the final packed buffer.
    # This is setup-only chunking; the time-stepping executable is unchanged.
    flat = array.ravel()
    chunk = _CUDA_CODEBOOK_CHUNK_CELLS  # A multiple of four packed IDs.
    table = np.empty(0, dtype=np.float32)
    for start in range(0, flat.size, chunk):
        table = np.union1d(table, np.unique(flat[start : start + chunk]))
        if table.size > min(max_values, 256):
            return None
    packed = np.empty((flat.size + 3) // 4, dtype=np.uint32)
    for start in range(0, flat.size, chunk):
        block = flat[start : start + chunk]
        ids = np.zeros((block.size + 3) & ~3, dtype=np.uint32)
        ids[: block.size] = np.searchsorted(table, block)
        packed[start // 4 : (start + block.size + 3) // 4] = (
            ids[0::4]
            | (ids[1::4] << np.uint32(8))
            | (ids[2::4] << np.uint32(16))
            | (ids[3::4] << np.uint32(24))
        )
    return jnp.asarray(table, dtype=jnp.float32), jnp.asarray(packed.view(np.int32))


def _pack_cuda_lossless_e_coefficients(decays, sources, *, blocks=False):
    """Return table/ID material buffers for a lossless low-cardinality E update."""

    decay_arrays = tuple(np.asarray(value) for value in decays)
    if not all(
        value.ndim == 0 and np.float32(value) == np.float32(1.0)
        for value in decay_arrays
    ):
        return None
    packed = tuple(_pack_cuda_coefficient_ids(value) for value in sources)
    values = tuple(value for value in packed if value is not None)
    if blocks:
        from .cuda.material_blocks import pack_material_blocks

        baseline_bytes = (
            sum(a.nbytes + b.nbytes for a, b in values)
            if len(values) == len(packed)
            else sum(value.nbytes for value in sources)
        )
        tiled = pack_material_blocks(sources, max_bytes=0.75 * baseline_bytes)
        if tiled is not None:
            return tiled
    if any(value is None for value in packed):
        return None
    return tuple(value[0] for value in values), tuple(value[1] for value in values)


def _inverse_permittivity_components(values):
    """Invert a packed symmetric tensor field into six trailing components."""

    from beamz.constitutive import inverse_permittivity_components

    return jnp.asarray(inverse_permittivity_components(values))


def _compile_derivative_metrics(
    material_grid, periodic_axes: frozenset[int] = frozenset()
) -> DerivativeMetricPlan:
    """Precompute O(nx + ny + nz) staggered inverse-distance metrics."""
    kind = material_grid.metric_kind
    empty = jnp.zeros((0,), dtype=jnp.float32)
    if kind == "isotropic_uniform":
        return DerivativeMetricPlan(*(empty for _ in range(6)))

    assert material_grid.grid is not None
    active_axes = ("x", "y", "z") if len(material_grid.shape) == 3 else ("x", "y")
    forward = {}
    backward = {}
    storage_axes = ("z", "y", "x") if len(material_grid.shape) == 3 else ("y", "x")
    periodic_names = {storage_axes[index] for index in periodic_axes}
    for axis in active_axes:
        widths = material_grid.grid.cell_widths(axis)
        if kind == "axis_uniform":
            forward[axis] = backward[axis] = jnp.asarray(
                1.0 / float(widths[0]), dtype=jnp.float32
            )
            continue
        inverse_forward = 1.0 / widths
        inverse_backward = np.empty(widths.size + 1, dtype=np.float64)
        inverse_backward[0] = 1.0 / widths[0]
        inverse_backward[-1] = 1.0 / widths[-1]
        if widths.size > 1:
            inverse_backward[1:-1] = 2.0 / (widths[:-1] + widths[1:])
        if axis in periodic_names:
            seam_inverse = 2.0 / (widths[-1] + widths[0])
            inverse_backward[0] = seam_inverse
            inverse_backward[-1] = seam_inverse
        forward[axis] = jnp.asarray(inverse_forward, dtype=jnp.float32)
        backward[axis] = jnp.asarray(inverse_backward, dtype=jnp.float32)
    return DerivativeMetricPlan(
        *(forward.get(axis, empty) for axis in ("x", "y", "z")),
        *(backward.get(axis, empty) for axis in ("x", "y", "z")),
    )


@trace_preparation("cpml_plan")
def _compile_cpml_plan(
    fields, *, dt, is_3d, metallic_edges, polarization_2d: str = "tm"
) -> CpmlPlan:
    """Compile every active derivative into the same packed coefficient record."""
    data = getattr(fields, "pml_data", None)
    enabled = bool(getattr(fields, "has_cpml", False) and data)
    if not enabled:
        return CpmlPlan(False, frozenset(metallic_edges), (), ())
    assert data is not None

    def term(prefix, component, axis, sign, shape):
        if data is None:
            raise RuntimeError("CPML profiles disappeared during compilation.")
        return ops.compile_cpml_term(
            component=component,
            axis=axis,
            sign=sign,
            sigma=data[f"{prefix}_sigma"],
            kappa=data[f"{prefix}_kappa"],
            alpha=data[f"{prefix}_alpha"],
            dt=dt,
            full_shape=tuple(int(value) for value in shape),
        )

    if is_3d:
        axis_index = {"z": 0, "y": 1, "x": 2}

        def terms_for(specs):
            return tuple(
                term(
                    f"cpml3d_{spec.name}",
                    spec.target_component,
                    axis_index[spec.derivative_axis],
                    1.0 if index % 2 == 0 else -1.0,
                    getattr(fields, spec.target_component).shape,
                )
                for index, spec in enumerate(specs)
            )

        h_terms = terms_for(CPML_3D_H_DERIVATIVES)
        e_terms = terms_for(CPML_3D_E_DERIVATIVES)
    else:
        profiles = data.get(f"{polarization_2d}_xy_cpml")
        if profiles is None:
            return CpmlPlan(False, frozenset(metallic_edges), (), ())
        data = profiles
        if polarization_2d == "tm":
            h_terms = (
                term("Hx_y", "Hx", 0, 1.0, fields.Hx.shape),
                term("Hy_x", "Hy", 1, -1.0, fields.Hy.shape),
            )
            e_terms = (
                term("Ez_x", "Ez", 1, 1.0, fields.Ez.shape),
                term("Ez_y", "Ez", 0, -1.0, fields.Ez.shape),
            )
        else:
            h_terms = (
                term("Hz_x", "Hz", 1, 1.0, fields.Hz.shape),
                term("Hz_y", "Hz", 0, -1.0, fields.Hz.shape),
            )
            e_terms = (
                term("Ex_y", "Ex", 0, 1.0, fields.Ex.shape),
                term("Ey_x", "Ey", 1, -1.0, fields.Ey.shape),
            )
    return CpmlPlan(True, frozenset(metallic_edges), h_terms, e_terms)


@dataclass(frozen=True, slots=True)
class _CompileSetup:
    """Shape-defining values shared by all compilation phases."""

    dt: float
    sharding: ShardingPlan
    source_specs: tuple[Any, ...]
    monitor_specs: tuple[Any, ...]
    config: RunConfig


@trace_preparation("sources_monitors_and_layout")
def _prepare_compilation(
    request: SimulationRequest, logical_fields: CompiledGrid, sharding: ShardingPlan
) -> _CompileSetup:
    """Resolve shapes, devices, sources, monitors, and scalar run configuration."""
    resolution = float(request.materials.resolution)
    dt = float(request.run.dt)
    num_steps = int(request.run.num_steps)
    sharding_cfg = ShardingConfig(*request.run.sharding)
    sharding_layout = sharding.layout
    effective_sharding = (
        replace(
            sharding_cfg,
            enabled=True,
            axis=cast(Literal["auto", "z", "y", "x"], sharding_layout.axis_name),
            num_devices=sharding_layout.num_devices,
        )
        if sharding_layout.enabled
        else replace(sharding_cfg, enabled=False)
    )
    from beamz.devices.sources.specs import PlaneWaveSource

    vector = bloch_wavevector(
        request.boundaries, is_3d=request.domain.is_3d, plane_2d=request.domain.plane_2d
    )
    for source in request.sources:
        if isinstance(source, PlaneWaveSource) and not np.allclose(
            source.transverse_wavevector, vector, rtol=1e-10, atol=1e-10
        ):
            raise ValueError(
                "PlaneWaveSource transverse_wavevector must match the Bloch boundaries."
            )
    source_specs = compile_source_specs(
        request.sources,
        logical_fields,
        dt,
        resolution,
        num_steps,
        request.run.t0,
        request.run.total_steps,
        request.domain,
        grid=logical_fields.geometry,
    )
    monitor_fields = SimpleNamespace(
        **{
            name: SimpleNamespace(shape=shape)
            for name, shape in sharding_layout.padded_shapes.items()
        },
        permittivity=logical_fields.permittivity,
        plane_2d=logical_fields.plane_2d,
        polarization_2d=logical_fields.polarization_2d,
        _logical_component_shapes=sharding_layout.logical_shapes,
    )
    monitor_specs, _ = compile_monitor_specs(
        request.monitors,
        monitor_fields,
        resolution,
        num_steps,
        dt,
        request.domain.plane_2d,
        request.domain.polarization_2d,
        grid=logical_fields.geometry,
    )
    loop_aliases = {
        "fori": "fori_loop",
        "fori_loop": "fori_loop",
        "fori-loop": "fori_loop",
        "scan": "scan",
    }
    try:
        loop_kind = loop_aliases[str(request.run.loop_kind).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "Invalid compiled loop kind. Use one of: scan, fori_loop."
        ) from exc
    config = RunConfig(
        resolution=resolution,
        dt=dt,
        num_steps=num_steps,
        plane_2d=request.domain.plane_2d,
        is_3d=bool(request.domain.is_3d),
        t0=float(request.run.t0),
        metric_kind=request.materials.metric_kind,
        polarization_2d=request.domain.polarization_2d,
        loop_kind=loop_kind,
        source_single_slab_dense=bool(request.run.source_single_slab_dense),
        backend=str(request.run.backend),
        sharding=effective_sharding,
        cuda_flags=int(request.run.cuda_flags),
        cuda_graph_cache_capacity=int(request.run.cuda_graph_cache_capacity),
        cuda_storage_axes=request.run.cuda_storage_axes,
        cuda_memory_policy=request.run.cuda_memory_policy,
        cuda_schedule=request.run.cuda_schedule,
        cuda_expected_runs=request.run.cuda_expected_runs,
    )
    return _CompileSetup(
        dt,
        sharding,
        source_specs,
        monitor_specs,
        config,
    )


@trace_preparation("yee_materials")
def _compile_grid(
    request: SimulationRequest, boundary_data: BoundaryData, *, region_setup=False
) -> CompiledGrid:
    """Lower cell materials and boundary data into one frozen logical Yee lattice."""
    material_grid = request.materials
    assert material_grid.grid is not None
    grid_origin = material_grid.grid.origin
    # Simulation has already shifted devices into solver-local coordinates. Normalize
    # the realized grid from its own origin so centered-domain offsets are not applied
    # a second time, while imported grids retain the same local frame.
    local_geometry = material_grid.grid.translated(
        (-grid_origin[0], -grid_origin[1], -grid_origin[2])
    )
    material_source = boundary_data
    profiles = boundary_data.profiles
    shapes = MappingProxyType(
        dict(component_shapes(material_grid.shape, request.domain.polarization_2d))
    )
    vector = bloch_wavevector(
        request.boundaries, is_3d=request.domain.is_3d, plane_2d=request.domain.plane_2d
    )
    field_dtype = jnp.complex64 if any(vector) else jnp.float32
    components = {
        component: (
            constant_array(shape, 0)
            if region_setup
            else jnp.zeros(shape, dtype=field_dtype)
        )
        for component, shape in shapes.items()
    }
    pml_data = (
        None
        if profiles is None
        else MappingProxyType(
            {
                key: jnp.asarray(value) if hasattr(value, "__array__") else value
                for key, value in profiles.items()
            }
        )
    )
    array = np.asarray if region_setup else jnp.asarray
    values = {
        "material_grid": material_grid,
        "geometry": local_geometry,
        "component_shapes": shapes,
        "resolution": material_grid.resolution,
        "plane_2d": "xy" if not request.domain.is_3d else request.domain.plane_2d,
        "polarization_2d": request.domain.polarization_2d,
        "permittivity": array(material_source.permittivity),
        "conductivity": array(material_source.conductivity),
        "permeability": array(material_source.permeability),
        "metallic_masks": MappingProxyType(dict(boundary_data.masks)),
        "boundaries": tuple(request.boundaries),
        "has_pml": profiles is not None,
        "has_cpml": bool(
            profiles is not None
            and str(profiles.get("formulation", "sponge")).lower() == "cpml"
        ),
        "pml_data": pml_data,
        **components,
    }
    assembly = SimpleNamespace(**values)
    materials = (
        build_region_materials(assembly)
        if region_setup
        else build_material_coefficients(assembly)
    )
    direct = (
        dict(boundary_data.yee_materials)
        if material_grid.uses_direct_yee_materials
        else {}
    )
    if direct:
        values_by_name = dict(materials.items())
        for source, target in (
            ("eps_x", "eps_x"),
            ("eps_y", "eps_y"),
            ("eps_z", "eps_z"),
            ("sig_x", "sig_x"),
            ("sig_y", "sig_y"),
            ("sig_z", "sig_z"),
            ("mu_hx", "mu_hx"),
            ("mu_hy", "mu_hy"),
            ("mu_hz", "mu_hz"),
        ):
            if source in direct:
                value = array(direct[source])
                if region_setup and value.ndim:
                    value = RegionArray(value.shape, value.dtype, value.__getitem__)
                if (
                    source.startswith("sig_")
                    and assembly.has_pml
                    and not assembly.has_cpml
                ):
                    component = {"sig_x": "Ex", "sig_y": "Ey", "sig_z": "Ez"}[source]
                    if assembly.conductivity.ndim == 0:
                        centered = assembly.conductivity
                    elif assembly.permittivity.ndim == 3:
                        centered = sample_voxel_grid_at_e_component_3d_centered(
                            assembly.conductivity,
                            component,
                        )
                    else:
                        centered = sample_voxel_grid_at_component_2d(
                            assembly.conductivity,
                            component,
                            "xy",
                            assembly.polarization_2d,
                        )
                    value = value + values_by_name[target] - centered
                values_by_name[target] = value
        values_by_name.update(
            eps_ex=values_by_name["eps_x"],
            eps_ey=values_by_name["eps_y"],
            eps_ez=values_by_name["eps_z"],
        )
        materials = type(materials)(**values_by_name)
    if boundary_data.periodic_axes:
        from beamz.simulation.dispersion import periodic_support_average

        # Native periodic rasters already have equal complete supports. This
        # also joins clipped coefficients from the legacy static voxel path.
        arrays = dict(materials.items())
        for axis in "xyz":
            for prefix in ("eps_", "sig_"):
                key = prefix + axis
                arrays[key] = jnp.asarray(
                    periodic_support_average(
                        arrays[key],
                        material_grid.shape,
                        boundary_data.periodic_axes,
                        material_grid.grid,
                    )
                )
            arrays["eps_e" + axis] = arrays["eps_" + axis]
        materials = type(materials)(**arrays)
    return CompiledGrid(
        **values,
        materials=materials,
        **dict(materials.items()),
    )


def _compile_boundary(
    fields, cpml, boundary_data, *, is_3d: bool, plane_2d="xy"
) -> BoundaryPlan:
    """Assemble canonical CPML and metallic values on the logical lattice."""
    masks = fields.metallic_masks
    return BoundaryPlan(
        metallic_edges_2d=(frozenset() if is_3d else boundary_data.metallic_edges),
        periodic_axes=boundary_data.periodic_axes,
        periodic_phases=bloch_storage_phases(
            fields.boundaries, fields.geometry, is_3d=is_3d, plane_2d=plane_2d
        ),
        cpml=cpml,
        metallic=MetallicPlan(
            masks["Ex"],
            masks["Ey"],
            masks["Ez"],
            masks["Hx"],
            masks["Hy"],
            masks["Hz"],
        ),
        material_shape=tuple(int(value) for value in fields.material_grid.shape),
        logical_component_shapes=fields.component_shapes,
    )


@trace_preparation("compile_plan")
def compile_simulation(request: SimulationRequest) -> CompiledProgram:
    """Build an immutable executable plan from a simulation request."""
    vector = bloch_wavevector(
        request.boundaries, is_3d=request.domain.is_3d, plane_2d=request.domain.plane_2d
    )
    if any(vector) and (request.run.backend != "jax" or request.run.sharding[0]):
        raise ValueError(
            "Nonzero Bloch wavevectors require single-device JAX execution."
        )
    if any(vector):
        from beamz.devices.monitors.monitors import ModeMonitor
        from beamz.devices.sources.specs import ModeSource

        if any(isinstance(s, ModeSource) for s in request.sources) or any(
            isinstance(m, ModeMonitor) for m in request.monitors
        ):
            raise ValueError(
                "Bloch mode sources/monitors are not supported; use field monitors."
            )
    if request.materials.uses_full_permittivity:
        from beamz.devices.monitors.monitors import ModeMonitor
        from beamz.devices.sources.specs import ModeSource

        if (
            any(isinstance(s, ModeSource) for s in request.sources)
            or any(isinstance(m, ModeMonitor) for m in request.monitors)
        ) and (
            request.run.backend not in {"jax", "cuda_streamed"}
            or request.run.sharding[0]
        ):
            raise ValueError(
                "Tensor modal workflows require single-device JAX or streamed CUDA execution"
            )
    if request.materials.dispersion:
        from beamz.devices.monitors.monitors import ModeMonitor
        from beamz.devices.sources.specs import ModeSource

        if any(isinstance(source, ModeSource) for source in request.sources) or any(
            isinstance(monitor, ModeMonitor) for monitor in request.monitors
        ):
            raise ValueError(
                "Dispersive mode sources and mode monitors are not yet supported; use plane-wave excitation and field/flux monitors."
            )
        if request.run.backend != "jax" or request.run.sharding[0]:
            raise ValueError("Dispersive media require single-device JAX execution.")
        if request.materials.uses_full_permittivity:
            raise ValueError(
                "Dispersive media cannot yet be combined with full-tensor permittivity."
            )
        material_geometry = request.materials.grid
        if material_geometry is None:
            raise ValueError("Dispersive media require physical grid metrics.")
        cfl_dt = material_geometry.cfl_time_step(
            1.0, active_axes=("x", "y", "z") if request.domain.is_3d else ("x", "y")
        )
        if request.run.dt > cfl_dt * (1 + 1e-6):
            raise ValueError(
                "Dispersive timestep exceeds the vacuum CFL stability bound."
            )
    # Resolve the actual mesh using shape metadata before any volume allocation.
    shapes = component_shapes(request.materials.shape, request.domain.polarization_2d)
    metadata = SimpleNamespace(
        permittivity=SimpleNamespace(shape=request.materials.shape),
        **{name: SimpleNamespace(shape=shape) for name, shape in shapes.items()},
    )
    early_sharding = build_sharding_plan(
        metadata,
        ShardingConfig(*request.run.sharding),
        is_3d=request.domain.is_3d,
        aligned_components=True,
    )
    region_setup = (
        early_sharding.layout.enabled
        and not jax.config.read("jax_enable_x64")
        and request.domain.is_3d
        and not request.materials.uses_full_permittivity
        and all(getattr(b, "formulation", "cpml") == "cpml" for b in request.boundaries)
    )
    # 1. Lower all boundaries once, then collocate materials on the resulting lattice.
    with preparation_phase("boundaries"):
        boundary_data = lower_boundaries(
            request.materials,
            component_shapes(request.materials.shape, request.domain.polarization_2d),
            request.boundaries,
            request.domain.size,
            request.run.dt,
            polarization_2d=request.domain.polarization_2d,
            region_setup=region_setup,
            plane_2d=request.domain.plane_2d,
        )
    logical_grid = _compile_grid(request, boundary_data, region_setup=region_setup)
    setup = _prepare_compilation(request, logical_grid, early_sharding)
    fields = logical_grid
    dt = setup.dt
    sharding = setup.sharding
    sharding_layout = sharding.layout
    source_specs, monitor_specs, config = (
        setup.source_specs,
        setup.monitor_specs,
        setup.config,
    )
    # Precompute each Yee support once. CUDA consumes update coefficients;
    # 3D JAX consumes constitutive grids directly instead.
    native = config.backend in {"cuda_streamed", "cuda_mstep"}
    empty3 = jnp.zeros((0, 0, 0), dtype=jnp.float32)
    precompute_h = (
        magnetic_coefficients if region_setup else ops.precompute_h_update_coefficients
    )
    coefficient_values = {}
    e_decays, e_sources = [], []
    for axis in "xyz":
        conductivity = getattr(fields, f"sig_{axis}")
        permittivity = getattr(fields, f"eps_{axis}")
        sigma_m = getattr(fields, f"sigma_m_h{axis}")
        h_decay = h_source = e_decay = e_source = empty3
        if not config.is_3d or native:
            h_decay, h_source = precompute_h(sigma_m, dt)
            e_decay, e_source = (
                electric_coefficients(
                    getattr(fields, "E" + axis).shape, conductivity, permittivity, dt
                )
                if region_setup
                else ops.precompute_e_update_coefficients(
                    shape=getattr(fields, "E" + axis).shape,
                    conductivity=conductivity,
                    permittivity=permittivity,
                    dt=dt,
                    region=getattr(fields, f"region_{axis}"),
                )
            )
            if native:
                h_decay, h_source, e_decay, e_source = map(
                    _elide_uniform_grid, (h_decay, h_source, e_decay, e_source)
                )
        e_decays.append(e_decay)
        e_sources.append(e_source)
        sigma_m, conductivity, permittivity = (
            (
                _elide_zero_conductivity_grid(sigma_m),
                _elide_zero_conductivity_grid(conductivity),
                permittivity,
            )
            if config.is_3d and not native
            else (empty3,) * 3
        )
        coefficient_values.update(
            {
                f"h_decay_{axis}": h_decay,
                f"h_source_{axis}": h_source,
                f"e_decay_{axis}": e_decay,
                f"e_source_{axis}": e_source,
                f"h_sigma_m_{axis}": sigma_m,
                f"e_conductivity_{axis}": conductivity,
                f"e_permittivity_{axis}": permittivity,
            }
        )
    if (
        native
        and not sharding_layout.enabled
        and not request.materials.uses_full_permittivity
    ):
        packed_e = _pack_cuda_lossless_e_coefficients(
            tuple(e_decays),
            tuple(e_sources),
            blocks=config.backend == "cuda_streamed"
            and config.cuda_storage_axes == (0, 1, 2),
        )
        if packed_e is not None:
            coefficient_values.update(
                zip(
                    (
                        "e_decay_x",
                        "e_decay_y",
                        "e_decay_z",
                        "e_source_x",
                        "e_source_y",
                        "e_source_z",
                    ),
                    (*packed_e[0], *packed_e[1]),
                    strict=True,
                )
            )
    empty_inverse = jnp.zeros((0,), dtype=jnp.float32)
    e_inverse_offdiagonal = empty_inverse
    support_tensors = request.materials.yee_tensors
    if request.materials.uses_full_permittivity:
        if any(
            np.any(np.asarray(getattr(fields, f"sig_{axis}")) != 0.0) for axis in "xyz"
        ):
            raise ValueError(
                "Full-tensor permittivity requires lossless electric updates; "
                "use CPML rather than conductive or sponge-PML materials."
            )
        e_inverse_offdiagonal = _inverse_permittivity_components(
            support_tensors["eps_node"]
        )[..., 3:]
    for index, axis in enumerate("xyz"):
        key = f"eps_{axis}"
        coefficient_values[f"e_inverse_diagonal_{axis}"] = (
            _inverse_permittivity_components(support_tensors[key])[..., index]
            if request.materials.uses_full_permittivity and key in support_tensors
            else empty_inverse
        )
    coefficient_values["e_inverse_offdiagonal"] = e_inverse_offdiagonal
    # 5. Precompute the same packed recurrence record for every 2D or 3D derivative.
    cpml = _compile_cpml_plan(
        fields,
        dt=dt,
        is_3d=bool(request.domain.is_3d),
        metallic_edges=boundary_data.metallic_edges,
        polarization_2d=request.domain.polarization_2d,
    )

    update_coefficients = UpdateCoefficients(**coefficient_values)
    boundary = _compile_boundary(
        fields,
        cpml,
        boundary_data,
        is_3d=bool(request.domain.is_3d),
        plane_2d=request.domain.plane_2d,
    )
    update_coefficients, boundary = lower_compiled_arrays(
        update_coefficients, boundary, sharding_layout
    )
    if (
        config.backend in {"jax", "cuda_streamed"}
        and not sharding_layout.enabled
        and e_inverse_offdiagonal.size
    ):
        from .tensor_regions import compact_tensor_region

        update_coefficients = update_coefficients._replace(
            e_inverse_offdiagonal=compact_tensor_region(e_inverse_offdiagonal)
        )
        if config.backend == "cuda_streamed":
            from .cuda.material_blocks import pack_material_blocks

            inverse = tuple(
                _elide_uniform_grid(
                    getattr(update_coefficients, f"e_inverse_diagonal_{c}")
                )
                for c in "xyz"
            )
            packed_inverse = pack_material_blocks(
                inverse, max_bytes=0.75 * sum(v.nbytes for v in inverse)
            )
            changes = {
                f"e_inverse_diagonal_{c}": value
                for c, value in zip(
                    "xyz",
                    inverse if packed_inverse is None else packed_inverse[0],
                    strict=True,
                )
            }
            if packed_inverse is not None:
                changes.update(
                    {
                        f"e_source_{c}": value
                        for c, value in zip("xyz", packed_inverse[1], strict=True)
                    }
                )
            update_coefficients = update_coefficients._replace(**changes)
    if (
        config.backend == "jax"
        and request.domain.is_3d
        and not sharding_layout.enabled
        and not request.materials.dispersion
        and not request.materials.dispersion_interfaces
    ):
        from .material_blocks import compact_material

        update_coefficients = update_coefficients._replace(
            **{
                prefix + c: compact_material(getattr(update_coefficients, prefix + c))
                for prefix in ("e_permittivity_", "e_inverse_diagonal_")
                for c in "xyz"
            }
        )
    metrics = lower_derivative_metrics(
        _compile_derivative_metrics(request.materials, boundary_data.periodic_axes),
        sharding_layout,
    )
    if config.backend != "jax" and sharding_layout.enabled:
        from .cuda.sharding import validate_sharded_config

        validate_sharded_config(config, boundary, sharding)
    from beamz.simulation.dispersion import compile_dispersion

    dispersion = compile_dispersion(
        logical_grid, dt, boundary_data.periodic_axes, complex_fields=any(vector)
    )
    program = CompiledProgram(
        grid=logical_grid,
        config=config,
        coefficients=update_coefficients,
        dispersion=dispersion,
        metrics=metrics,
        boundary=boundary,
        sources=source_specs,
        monitors=monitor_specs,
        sharding=sharding,
    )

    if region_setup:
        # Recipes retain only immutable inputs; the program owns one final runtime
        # representation. Continuation must never regenerate or transfer it.
        from .sharding import place_tree

        with preparation_phase("coefficient_generation_and_placement"):
            program = replace(
                program, coefficients=place_tree(program, program.coefficients)
            )
    return program


@trace_preparation("public_compile")
def compile_program(
    simulation,
    *,
    num_steps: int | None = None,
    sharding=None,
    backend: str | None = "auto",
    progress: bool = False,
    schedule: str | None = None,
    expected_runs: int = 1,
    setup_context_factory=None,
    compile_factory=None,
) -> CompiledProgram:
    """Compile or reuse the immutable program for one execution horizon."""
    steps = int(simulation.num_steps if num_steps is None else num_steps)
    if steps <= 0:
        raise ValueError("num_steps must be > 0")

    # Environment switches are compilation inputs, so normalize them before building
    # the request and its semantic cache key.
    loop_env = os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan").strip().lower()
    if loop_env in {"fori", "fori_loop", "fori-loop"}:
        loop_kind = "fori_loop"
    elif loop_env == "scan":
        loop_kind = "scan"
    else:
        raise ValueError("Invalid BEAMZ_COMPILED_LOOP_KIND (use: scan, fori_loop).")
    dense = os.getenv("BEAMZ_SOURCE_SINGLE_SLAB_DENSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    from .backend import (
        CudaBackendUnavailable,
        cuda_flags_from_env,
        cuda_graph_cache_capacity_from_env,
        cuda_storage_axes_from_env,
        normalize_backend,
        resolve_backend,
    )
    from .sharding import _jax_devices_for_config

    requested_backend = normalize_backend(backend)
    from .cuda.schedule import normalize_schedule_request

    requested_schedule, expected_runs = normalize_schedule_request(
        requested_backend,
        schedule,
        expected_runs,
    )
    sharding_token = sharding_cache_token(sharding)
    metric_kind = simulation.grid.metric_kind_for(
        ("x", "y", "z") if simulation.is_3d else ("x", "y")
    )
    material_grid = simulation._material_grid(progress=progress)
    if material_grid.uses_full_permittivity and metric_kind != "isotropic_uniform":
        if sharding_token[0] or requested_backend not in {
            "auto",
            "jax",
            "cuda",
            "cuda_streamed",
        }:
            raise ValueError(
                "Rectilinear full-tensor updates require single-device JAX or streamed CUDA."
            )
        if any(isinstance(boundary, Periodic) for boundary in simulation.boundaries):
            raise ValueError(
                "Rectilinear full-tensor updates do not yet support periodic/Bloch "
                "constitutive averaging; use open or CPML boundaries."
            )
    has_dispersion = bool(material_grid.dispersion)
    if has_dispersion and sharding_token[0]:
        raise ValueError(
            "Dispersive media currently support single-device execution only."
        )
    if has_dispersion and requested_backend not in {"auto", "jax"}:
        raise CudaBackendUnavailable(
            "Dispersive media require backend='jax' (including JAX GPU)."
        )
    multi_device = sharding_token[0] and sharding_token[2] != 1
    # One ordered capability ledger drives both automatic fallback and explicit
    # rejection. Schedule-specific restrictions remain with their own planner.
    streamed_request = requested_backend in {"cuda", "cuda_streamed"}
    cuda_requirements = (
        (
            simulation.is_3d,
            "CUDA execution currently requires a 3D simulation; this simulation "
            f"requires {metric_kind!r} metrics. Use backend='jax'.",
        ),
        (
            not material_grid.uses_full_permittivity or streamed_request,
            "Full-tensor permittivity requires backend='jax' or "
            "backend='cuda_streamed'; the CUDA mstep schedule is not supported.",
        ),
        # Auto retains JAX for multi-device runs pending CUDA hardware validation.
        (
            not multi_device or streamed_request,
            "CUDA multi-device sharding requires backend='cuda_streamed'; "
            "use backend='jax' for other sharded configurations.",
        ),
        (
            not any(
                isinstance(boundary, Periodic) for boundary in simulation.boundaries
            ),
            "CUDA execution does not yet support periodic boundaries; use backend='jax'.",
        ),
    )
    unsupported = tuple(
        reason for supported, reason in cuda_requirements if not supported
    )
    if requested_backend not in {"auto", "jax"} and unsupported:
        raise CudaBackendUnavailable(unsupported[0])
    cuda_problem_supported = not unsupported and not has_dispersion
    resolved_backend = (
        "jax"
        if requested_backend == "auto" and not cuda_problem_supported
        else resolve_backend(
            "cuda_mstep" if requested_schedule == "mstep" else requested_backend
        )
    )
    if requested_schedule == "mstep" and (
        not cuda_problem_supported or multi_device or metric_kind != "isotropic_uniform"
    ):
        raise CudaBackendUnavailable(
            "cuda_mstep requires an unsharded isotropic uniform 3D grid"
        )
    cuda_flags = cuda_flags_from_env() if resolved_backend != "jax" else 0
    if requested_schedule == "mstep":
        from .cuda.mstep_config import ALLOWED_FLAGS

        cuda_flags &= ALLOWED_FLAGS
    cuda_storage_axes = (
        cuda_storage_axes_from_env(
            # Automatic sharding can resolve to a single-device execution plan.
            sharded=bool(_jax_devices_for_config(normalize_sharding_config(sharding)))
        )
        if resolved_backend != "jax"
        else (0, 1, 2)
    )
    if cuda_storage_axes != (0, 1, 2) and resolved_backend not in {
        "cuda_streamed",
        "cuda_mstep",
    }:
        raise ValueError("CUDA storage-axis permutations require a CUDA schedule")
    cuda_graph_cache_capacity = (
        cuda_graph_cache_capacity_from_env() if resolved_backend != "jax" else 0
    )
    if (
        material_grid.uses_full_permittivity
        and resolved_backend == "cuda_streamed"
        and not multi_device
    ):
        from . import _cuda_abi as abi

        if cuda_storage_axes != (0, 1, 2) or cuda_flags & (
            abi.CUDA_FIELD_PAD32 | abi.CUDA_FIELD_PAD64 | abi.CUDA_FIELD_PAD_Y8
        ):
            raise ValueError(
                "Full-tensor streamed CUDA requires canonical, unpadded field storage."
            )
    request = simulation.to_request(
        num_steps=steps,
        loop_kind=loop_kind,
        source_single_slab_dense=dense,
        backend=resolved_backend,
        sharding=sharding_token,
        progress=progress,
    )
    request = replace(
        request,
        run=replace(
            request.run,
            cuda_schedule=requested_schedule,
            cuda_expected_runs=expected_runs,
            cuda_flags=cuda_flags,
            cuda_graph_cache_capacity=cuda_graph_cache_capacity,
            cuda_storage_axes=cuda_storage_axes,
            cuda_memory_policy=(
                cuda_memory_policy_from_env() if resolved_backend != "jax" else "auto"
            ),
        ),
    )
    signature = CompiledProgramKey.from_request(request)
    tuning_policy = None
    if resolved_backend == "cuda_streamed" and requested_schedule != "mstep":
        from .cuda.tuning import tuning_policy_from_env

        tuning_policy = tuning_policy_from_env()
        if (
            metric_kind == "isotropic_uniform"
            and steps >= 16
            and steps * expected_runs >= 32
            and np.prod(material_grid.shape) >= 8 * 1024 * 1024
        ):
            signature = replace(signature, cuda_tuning_policy=tuning_policy)
    if cached := _PROGRAM_CACHE.get(signature):
        _PROGRAM_CACHE.move_to_end(signature)
        return cached

    setup_context_factory = setup_context_factory or _resolved_setup_device_context
    compile_factory = compile_factory or compile_simulation
    with setup_context_factory(simulation.setup_device_resolved):
        reusable = None
        if compile_factory is compile_simulation:
            preparation_key = _preparation_key(signature)
            reusable = next(
                (
                    old_program
                    for old_key, old_program in reversed(_PROGRAM_CACHE.items())
                    if _preparation_key(old_key) == preparation_key
                ),
                None,
            )
        if reusable is None:
            program = compile_factory(request)
        else:
            # Reuse existing ownership; changing a continuation segment's length
            # must not allocate a second near-capacity coefficient bank.
            with preparation_phase("reuse_material_preparation"):
                setup = _prepare_compilation(request, reusable.grid, reusable.sharding)
                program = replace(
                    reusable,
                    config=setup.config,
                    sources=setup.source_specs,
                    monitors=setup.monitor_specs,
                )

    if tuning_policy is not None:
        from .cuda.tuning import select_program

        program = select_program(program, signature, tuning_policy)
    _PROGRAM_CACHE[signature] = program
    if len(_PROGRAM_CACHE) > _MAX_COMPILED_PROGRAMS:
        _PROGRAM_CACHE.popitem(last=False)
    return program


_FIELD_GROUPS = {
    "yee_fields": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    "material_center_grids": (
        "permittivity",
        "conductivity",
        "permeability",
        "total_conductivity",
    ),
    "component_material_grids": (
        "eps_x",
        "eps_y",
        "eps_z",
        "sig_x",
        "sig_y",
        "sig_z",
        "sigma_m_hx",
        "sigma_m_hy",
        "sigma_m_hz",
    ),
    "field_masks": (
        "tm_ez_mask",
        "tm_hx_mask",
        "tm_hy_mask",
        "ex_metal_mask",
        "ey_metal_mask",
        "ez_metal_mask",
        "hx_metal_mask",
        "hy_metal_mask",
        "hz_metal_mask",
    ),
}

_REFERENCED_COEFFICIENTS = {
    "h_sigma_m_x",
    "h_sigma_m_y",
    "h_sigma_m_z",
    "e_conductivity_x",
    "e_conductivity_y",
    "e_conductivity_z",
    "e_permittivity_x",
    "e_permittivity_y",
    "e_permittivity_z",
    "e_inverse_diagonal_x",
    "e_inverse_diagonal_y",
    "e_inverse_diagonal_z",
    "e_inverse_offdiagonal",
}


def _add_memory_arrays(entries, name, value, category, residency="persistent"):
    """Flatten array tuples into stable allocation-report entries."""
    from .material_blocks import MaterialBlocks
    from .tensor_regions import TensorRegion

    if isinstance(value, MaterialBlocks):
        for part in ("values", "headers"):
            _add_memory_arrays(
                entries, name + "." + part, getattr(value, part), category, residency
            )
        return
    if isinstance(value, TensorRegion):
        _add_memory_arrays(entries, name + ".values", value.values, category, residency)
        return
    try:
        shape, dtype = tuple(int(v) for v in value.shape), np.dtype(value.dtype)
    except Exception:
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                _add_memory_arrays(
                    entries, f"{name}[{index}]", item, category, residency
                )
        return
    entries.append(
        {
            "name": str(name),
            "category": str(category),
            "residency": str(residency),
            "shape": list(shape),
            "dtype": str(dtype),
            "bytes": int(np.prod(shape, dtype=np.int64) * dtype.itemsize),
        }
    )


def _memory_summary(entries):
    categories, residencies = {}, {}
    for entry in entries:
        size = int(entry["bytes"])
        category, residency = entry["category"], entry["residency"]
        categories[category] = categories.get(category, 0) + size
        residencies[residency] = residencies.get(residency, 0) + size
    total = sum(int(entry["bytes"]) for entry in entries)
    return {
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "totals_by_category": categories,
        "totals_by_residency": residencies,
        "entries": entries,
    }


def _compiled_memory_entries(program):
    from .material_blocks import MaterialBlocks

    owned, referenced = [], []
    for name, value in program.coefficients._asdict().items():
        target = (
            referenced
            if name in _REFERENCED_COEFFICIENTS
            and not isinstance(value, MaterialBlocks)
            else owned
        )
        _add_memory_arrays(
            target,
            name,
            value,
            "compiled_referenced_inputs"
            if target is referenced
            else "compiled_update_coefficients",
            "reference" if target is referenced else "persistent",
        )
    dispersion = getattr(program, "dispersion", None)
    if dispersion is not None:
        for index, region in enumerate(dispersion.regions):
            for name in ("a", "b", "weights"):
                _add_memory_arrays(
                    owned,
                    f"dispersion[{index}].{name}",
                    getattr(region, name),
                    "dispersion_coefficients",
                )
            _add_memory_arrays(
                owned,
                f"polarization[{index}]",
                SimpleNamespace(shape=region.shape, dtype=np.complex64),
                "polarization_state",
                "runtime",
            )
        for index, interface in enumerate(dispersion.interfaces):
            for name in (
                "indices",
                "fractions",
                "weight",
                "epsilon",
                "a",
                "b",
                "response",
                "harmonic_inf",
                "harmonic_step",
                "delta",
            ):
                _add_memory_arrays(
                    owned,
                    f"interface[{index}].{name}",
                    getattr(interface, name),
                    "dispersion_coefficients",
                )
            _add_memory_arrays(
                owned,
                f"interface_polarization[{index}]",
                SimpleNamespace(shape=interface.shape, dtype=np.complex64),
                "polarization_state",
                "runtime",
            )
        for component, ratio, inverse, response in dispersion.coefficients:
            for name, value in (
                ("ratio", ratio),
                ("inverse", inverse),
                ("response", response),
            ):
                _add_memory_arrays(
                    owned,
                    f"dispersion.{component}.{name}",
                    value,
                    "dispersion_coefficients",
                )
    for phase, terms in (
        ("h", program.boundary.cpml.h_terms),
        ("e", program.boundary.cpml.e_terms),
    ):
        for index, term in enumerate(terms):
            for coefficient in ("a", "b", "inv_kappa"):
                _add_memory_arrays(
                    owned,
                    f"cpml_{phase}[{index}].{coefficient}",
                    getattr(term, coefficient),
                    "compiled_static_terms",
                )
    for field in dataclass_fields(program.boundary.metallic):
        _add_memory_arrays(
            owned,
            field.name.replace("_mask", "_metal_mask"),
            getattr(program.boundary.metallic, field.name),
            "compiled_static_terms",
        )
    for index, source in enumerate(program.sources):
        for name in ("coeff", "waveform"):
            _add_memory_arrays(
                owned, f"sources[{index}].{name}", getattr(source, name), "sources"
            )
    for index, monitor in enumerate(program.monitors):
        for field in dataclass_fields(monitor):
            _add_memory_arrays(
                owned,
                f"monitors[{index}].{field.name}",
                getattr(monitor, field.name),
                "monitors",
            )
    return owned, referenced


def _compiled_memory_report(program):
    owned, referenced = _compiled_memory_entries(program)
    runtime = empty_monitor_values(program)
    for name in MONITOR_FIELDS:
        _add_memory_arrays(
            owned, f"monitor_state.{name}", runtime[name], "monitor_state", "runtime"
        )

    layout, config = program.sharding.layout, program.config
    report = _memory_summary(owned)
    report["referenced_inputs"] = _memory_summary(referenced)
    report["config"] = {
        "num_steps": int(config.num_steps),
        "is_3d": bool(config.is_3d),
        "loop_kind": config.loop_kind,
        "use_cpml_3d": bool(config.is_3d and program.boundary.cpml.enabled),
        "sharding": {
            "enabled": bool(layout.enabled),
            "axis": layout.axis_name,
            "num_devices": int(layout.num_devices),
            "backend": layout.backend,
            "logical_shapes": {k: list(v) for k, v in layout.logical_shapes.items()},
            "padded_shapes": {k: list(v) for k, v in layout.padded_shapes.items()},
        },
    }
    if layout.enabled:
        report["per_device_total_bytes"] = int(
            np.ceil(report["total_bytes"] / layout.num_devices)
        )
        report["per_device_total_gib"] = report["per_device_total_bytes"] / 1024**3
    return report


def simulation_memory_estimate(
    simulation,
    *,
    include_compiled: bool = True,
    num_steps: int | None = None,
    sharding=None,
) -> dict:
    """Return deterministic array storage derived from a compiled program."""
    program = simulation.compile(num_steps=num_steps, sharding=sharding)
    fields, entries = program.grid, []
    for category, names in _FIELD_GROUPS.items():
        for name in names:
            if hasattr(fields, name):
                _add_memory_arrays(
                    entries, f"fields.{name}", getattr(fields, name), category
                )
    pml_data = getattr(fields, "pml_data", None)
    if isinstance(pml_data, Mapping):
        for name, value in pml_data.items():
            _add_memory_arrays(entries, f"pml_data.{name}", value, "pml_data")

    report = _memory_summary(entries)
    report.update(
        grid_shape_zyx=[int(v) for v in fields.permittivity.shape],
        is_3d=bool(simulation.is_3d),
    )
    if include_compiled:
        compiled = _compiled_memory_report(program)
        report["compiled"] = compiled
        report["total_with_compiled_bytes"] = (
            report["total_bytes"] + compiled["total_bytes"]
        )
        report["total_with_compiled_gib"] = (
            report["total_with_compiled_bytes"] / 1024**3
        )
    return report
