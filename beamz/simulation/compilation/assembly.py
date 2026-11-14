"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np

import beamz.simulation.kernels as ops
from beamz._region_array import RegionArray, constant_array, tiles, uniform_value
from beamz.devices._boundary_compile import (
    CPML_3D_E_DERIVATIVES,
    CPML_3D_H_DERIVATIVES,
    BoundaryData,
)
from beamz.devices._immutable import immutable_snapshot
from beamz.devices.boundaries import bloch_storage_phases, bloch_wavevector
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
    CpmlPlan,
    DerivativeMetricPlan,
    MetallicPlan,
    RunConfig,
    ShardingPlan,
    SimulationRequest,
)
from beamz.simulation.preparation_trace import trace_preparation


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
        from beamz.simulation.cuda.material_blocks import pack_material_blocks

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
    sharding_cfg = request.run.sharding
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
    config = replace(
        request.run,
        resolution=resolution,
        plane_2d=request.domain.plane_2d,
        is_3d=request.domain.is_3d,
        metric_kind=request.materials.metric_kind,
        polarization_2d=request.domain.polarization_2d,
        loop_kind=loop_kind,
        sharding=effective_sharding,
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
            constant_array(shape, 0, dtype=field_dtype)
            if region_setup or request.run.backend in {"cuda_streamed", "cuda_mstep"}
            else jnp.zeros(shape, dtype=field_dtype)
        )
        for component, shape in shapes.items()
    }
    array = np.asarray if region_setup else jnp.asarray
    pml_data = immutable_snapshot(profiles)
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
    materials = build_material_coefficients(assembly, lazy=region_setup)
    direct = (
        dict(boundary_data.yee_materials)
        if material_grid.uses_direct_yee_materials
        else {}
    )
    if direct:
        values_by_name = dict(materials.items())
        for source, direct_value in direct.items():
            value: Any = array(direct_value)
            if region_setup and value.ndim:
                value = RegionArray(value.shape, value.dtype, value.__getitem__)
            if source.startswith("sig_") and assembly.has_pml and not assembly.has_cpml:
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
                value = value + values_by_name[source] - centered
            values_by_name[source] = value
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
