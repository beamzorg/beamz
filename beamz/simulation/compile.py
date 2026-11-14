"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

import os
from collections import OrderedDict

# Retain the names available from this module before its implementation was split.
from collections.abc import Mapping as Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields  # noqa: F401
from types import MappingProxyType as MappingProxyType
from types import SimpleNamespace as SimpleNamespace
from typing import Any as Any
from typing import Literal as Literal
from typing import cast as cast

import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401

import beamz.simulation.kernels as ops  # noqa: F401
from beamz._cache_tokens import HashToken, cache_token
from beamz._region_array import RegionArray as RegionArray
from beamz._region_array import constant_array as constant_array
from beamz._region_array import tiles as tiles
from beamz._region_array import uniform_value as uniform_value
from beamz.design.discretization import MaterialGrid
from beamz.devices._boundary_compile import (
    CPML_3D_E_DERIVATIVES as CPML_3D_E_DERIVATIVES,
)
from beamz.devices._boundary_compile import (
    CPML_3D_H_DERIVATIVES as CPML_3D_H_DERIVATIVES,
)
from beamz.devices._boundary_compile import BoundaryData as BoundaryData
from beamz.devices._boundary_compile import lower_boundaries as lower_boundaries
from beamz.devices._immutable import immutable_snapshot as immutable_snapshot
from beamz.devices.boundaries import Periodic
from beamz.devices.boundaries import bloch_storage_phases as bloch_storage_phases
from beamz.devices.boundaries import bloch_wavevector as bloch_wavevector
from beamz.devices.monitors.compiler import (
    compile_monitor_specs as compile_monitor_specs,
)
from beamz.devices.sources.compiler import compile_source_specs as compile_source_specs
from beamz.lattice import build_material_coefficients as build_material_coefficients
from beamz.lattice import component_shapes as component_shapes
from beamz.lattice import (
    sample_voxel_grid_at_component_2d as sample_voxel_grid_at_component_2d,
)
from beamz.lattice import (
    sample_voxel_grid_at_e_component_3d_centered as sample_voxel_grid_at_e_component_3d_centered,
)
from beamz.simulation.model import BoundaryPlan as BoundaryPlan
from beamz.simulation.model import CompiledGrid as CompiledGrid
from beamz.simulation.model import (
    CompiledProgram,
    RunConfig,
    SimulationRequest,
)
from beamz.simulation.model import CpmlPlan as CpmlPlan
from beamz.simulation.model import DerivativeMetricPlan as DerivativeMetricPlan
from beamz.simulation.model import MetallicPlan as MetallicPlan
from beamz.simulation.model import ShardingPlan as ShardingPlan
from beamz.simulation.model import UpdateCoefficients as UpdateCoefficients
from beamz.simulation.preparation_trace import preparation_phase, trace_preparation
from beamz.simulation.region_materials import (
    electric_coefficients as electric_coefficients,
)
from beamz.simulation.region_materials import (
    magnetic_coefficients as magnetic_coefficients,
)

from .compilation.assembly import (  # noqa: F401
    _compile_boundary,
    _compile_cpml_plan,
    _compile_derivative_metrics,
    _compile_grid,
    _CompileSetup,
    _elide_uniform_grid,
    _elide_zero_conductivity_grid,
    _inverse_permittivity_components,
    _pack_cuda_coefficient_ids,
    _pack_cuda_lossless_e_coefficients,
    _prepare_compilation,
)
from .compilation.lower import compile_simulation as compile_simulation
from .compilation.storage import (  # noqa: F401
    _add_memory_arrays,
    _compiled_memory_entries,
    _compiled_memory_report,
    _memory_summary,
    simulation_memory_estimate,
)
from .memory import cuda_memory_policy_from_env
from .observe import MONITOR_FIELDS as MONITOR_FIELDS
from .observe import empty_monitor_values as empty_monitor_values
from .sharding import build_sharding_plan as build_sharding_plan
from .sharding import lower_compiled_arrays as lower_compiled_arrays
from .sharding import lower_derivative_metrics as lower_derivative_metrics
from .sharding import (
    normalize_sharding_config,
    sharding_cache_token,
)

# Cache identity contains only values that affect generated code or static storage;
# runtime field values stay excluded so one executable can serve many states.


@dataclass(frozen=True, slots=True)
class CompiledProgramKey:
    """Semantic input tokens plus the canonical immutable execution controls."""

    run: RunConfig
    domain: tuple[bool, str, str]
    materials: MaterialGrid
    sources: tuple[HashToken, ...]
    monitors: tuple[HashToken, ...]
    boundaries: tuple[HashToken, ...]
    cuda_tuning_policy: tuple = ()

    @classmethod
    def from_request(cls, request: SimulationRequest) -> CompiledProgramKey:
        # Owned material snapshots compare contents lazily when identities differ.
        return cls(
            request.run,
            (
                request.domain.is_3d,
                request.domain.plane_2d,
                request.domain.polarization_2d,
            ),
            request.materials,
            tuple(cache_token(source) for source in request.sources),
            tuple(cache_token(monitor) for monitor in request.monitors),
            tuple(cache_token(boundary) for boundary in request.boundaries),
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

    run = key.run
    cuda_backend = run.backend in {"cuda_streamed", "cuda_mstep"}

    return replace(
        key,
        run=replace(
            run,
            num_steps=0,
            total_steps=0,
            t0=0,
            cuda_schedule="streamed",
            cuda_expected_runs=1,
            backend="cuda_streamed" if cuda_backend else run.backend,
            cuda_flags=run.cuda_flags
            & (CUDA_BF16_PSI | CUDA_FIELD_PAD32 | CUDA_FIELD_PAD64 | CUDA_FIELD_PAD_Y8)
            if cuda_backend
            else run.cuda_flags,
            cuda_memory_policy="auto",
            cuda_graph_cache_capacity=0,
            cuda_storage_axes=(0, 1, 2) if cuda_backend else run.cuda_storage_axes,
        ),
        sources=(),
        monitors=(),
        cuda_tuning_policy=(),
    )


_MAX_COMPILED_PROGRAMS = 4
_PROGRAM_CACHE: OrderedDict[CompiledProgramKey, CompiledProgram] = OrderedDict()


def _resolved_setup_device_context(resolved_device):
    """Place setup work on CPU only when the resolved policy requests it."""
    return (
        jax.default_device(jax.devices("cpu")[0])
        if resolved_device == "cpu"
        else nullcontext(None)
    )


def clear_program_cache() -> None:
    """Drop cached immutable compiled plans."""
    _PROGRAM_CACHE.clear()


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
