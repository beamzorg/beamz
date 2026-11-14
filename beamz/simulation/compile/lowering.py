"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import beamz.simulation.compile.kernels as ops
from beamz.const import EPS_0
from beamz.constitutive import advance_e_tensor_dense, collocate_yee_component
from beamz.devices._boundary_compile import (
    lower_boundaries,
)
from beamz.devices.boundaries import bloch_wavevector
from beamz.lattice import (
    component_axis_offsets_3d,
    component_shapes,
)
from beamz.simulation.compile.assembly import (
    _compile_boundary,
    _compile_cpml_plan,
    _compile_derivative_metrics,
    _compile_grid,
    _elide_uniform_grid,
    _elide_zero_conductivity_grid,
    _inverse_permittivity_components,
    _pack_cuda_lossless_e_coefficients,
    _prepare_compilation,
    electric_coefficients,
    magnetic_coefficients,
)
from beamz.simulation.compile.backend import preparation_phase, trace_preparation
from beamz.simulation.compile.material_blocks import TensorRegion, material_values
from beamz.simulation.execute.sharding import (
    build_sharding_plan,
    lower_compiled_arrays,
    lower_derivative_metrics,
)
from beamz.simulation.model import (
    CompiledProgram,
    SimulationRequest,
    UpdateCoefficients,
)


@trace_preparation("compile_plan")
def compile_simulation(request: SimulationRequest) -> CompiledProgram:
    """Build an immutable executable plan from a simulation request."""
    vector = bloch_wavevector(
        request.boundaries, is_3d=request.domain.is_3d, plane_2d=request.domain.plane_2d
    )
    if any(vector) and (request.run.backend != "jax" or request.run.sharding.enabled):
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
            or request.run.sharding.enabled
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
        if request.run.backend != "jax" or request.run.sharding.enabled:
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
        request.run.sharding,
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
        from beamz.simulation.compile.material_blocks import compact_tensor_region

        update_coefficients = update_coefficients._replace(
            e_inverse_offdiagonal=compact_tensor_region(e_inverse_offdiagonal)
        )
        if config.backend == "cuda_streamed":
            from beamz.simulation.cuda.storage import pack_material_blocks

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
        from beamz.simulation.compile.material_blocks import compact_material

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
        from beamz.simulation.cuda.native.sharding import validate_sharded_config

        validate_sharded_config(config, boundary, sharding)
    from beamz.simulation.compile.material_blocks import compile_dispersion

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
        from beamz.simulation.execute.sharding import place_tree

        with preparation_phase("coefficient_generation_and_placement"):
            program = replace(
                program, coefficients=place_tree(program, program.coefficients)
            )
    return program


def advance_e_centered_tensor(
    fields,
    curls,
    inverse_diagonals,
    inverse_offdiagonal,
    components,
    dt,
    *,
    cell_widths=None,
):
    """Advance E with diagonal terms on E sites and cross terms on dual centers."""

    inverse_diagonals = tuple(material_values(v) for v in inverse_diagonals)
    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    if isinstance(inverse_offdiagonal, TensorRegion):
        region = inverse_offdiagonal
        axes = ("z", "y", "x")[-fields[0].ndim :]
        bounds = tuple(
            (max(0, start - 1), min(count, stop + 1))
            for (start, stop), count in zip(
                region.bounds, region.node_shape, strict=True
            )
        )
        values = jnp.pad(
            region.values,
            tuple(
                (start - outer_start, outer_stop - stop)
                for (start, stop), (outer_start, outer_stop) in zip(
                    region.bounds, bounds, strict=True
                )
            )
            + ((0, 0),),
        )
        slices = tuple(
            tuple(
                slice(
                    start, stop - int(component_axis_offsets_3d(component)[axis] == 0.5)
                )
                for axis, (start, stop) in zip(axes, bounds, strict=True)
            )
            for component in components
        )
        widths = cell_widths
        if widths is not None:
            widths = list(widths)
            for axis, (start, stop) in zip(axes, bounds, strict=True):
                index = "xyz".index(axis)
                if jnp.ndim(widths[index]) and jnp.size(widths[index]):
                    widths[index] = widths[index][start : stop - 1]
            widths = tuple(widths)
        local_curls = tuple(
            curl[part] for curl, part in zip(curls, slices, strict=True)
        )
        local = _tensor_cross_curls(local_curls, values, components, widths)
        return tuple(
            field
            + scale
            * (
                diagonal * curl
                + jnp.pad(
                    value,
                    tuple(
                        (part.start, size - part.stop)
                        for part, size in zip(parts, field.shape, strict=True)
                    ),
                )
            )
            for field, diagonal, curl, parts, value in zip(
                fields, inverse_diagonals, curls, slices, local, strict=True
            )
        )
    return advance_e_tensor_dense(
        fields,
        curls,
        inverse_diagonals,
        inverse_offdiagonal,
        components,
        dt,
        cell_widths=cell_widths,
    )


def _tensor_cross_curls(curls, inverse_offdiagonal, components, cell_widths):
    """Sum cross terms at their shared support before interpolating once per E."""
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths
        )
        for curl, source in zip(curls, components, strict=True)
    )
    cross_component = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    updated = []
    for row, (curl, target) in enumerate(zip(curls, components, strict=True)):
        centered = sum(
            inverse_offdiagonal[..., cross_component[(min(row, col), max(row, col))]]
            * value
            for col, value in enumerate(centered_curls)
            if col != row
        )
        updated.append(
            collocate_yee_component(
                centered, "Node", target, curl.shape, cell_widths=cell_widths
            )
        )
    return tuple(updated)
