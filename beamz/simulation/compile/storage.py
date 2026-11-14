"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import jax
import numpy as np

from beamz.simulation.cuda.native import abi
from beamz.simulation.results.observe import MONITOR_FIELDS, empty_monitor_values

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
    from beamz.simulation.compile.material_blocks import MaterialBlocks, TensorRegion

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
    from beamz.simulation.compile.material_blocks import MaterialBlocks

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
    fields = program.grid
    entries: list[dict] = []
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


def cuda_memory_policy_from_env() -> str:
    policy = os.environ.get("BEAMZ_CUDA_MEMORY_POLICY", "auto").strip().lower()
    if policy not in {"auto", "speed", "capacity"}:
        raise ValueError("BEAMZ_CUDA_MEMORY_POLICY must be auto, speed, or capacity")
    return policy


def cuda_workspace_estimate(
    program, *, donate_state: bool, mstep_plan=None, mstep=False, low_memory=False
) -> int:
    """Estimate optional fast-schedule workspace on the busiest device.

    Include both field banks for pair kernels, rotated materials for layout
    transforms, replicated normal CPML slabs, and retained input ownership.
    This intentionally overestimates rather than promising an exact peak.
    With mstep=True and no plan, count only unavoidable mstep allocations. This
    is a lower bound for early rejection, not permission to skip the full check.
    """
    layout = program.sharding.layout
    count = layout.num_devices if layout.enabled else 1
    fields = (
        sum(math.prod(shape) * 4 for shape in layout.padded_shapes.values()) // count
    )
    psi = sum(
        math.prod(term.slab.shape)
        * 4
        // (count if layout.enabled and term.axis != layout.axis else 1)
        for term in (*program.boundary.cpml.h_terms, *program.boundary.cpml.e_terms)
    )
    state = fields + psi
    # Frozen-input streamed graphs may use an alternate bank even without the
    # temporal-pair flag. Pair graphs can use a third bank. Count those possible
    # banks conservatively; capacity mode uses only the in-place output state.
    bank_count = (
        1
        if low_memory
        else 3
        if program.config.cuda_flags & abi.CUDA_TEMPORAL_PAIR
        else 2
    )
    if mstep or mstep_plan is not None:
        # Output state, alternate bank, compact intermediate fields and static
        # dependency arrays. Retained caller state is accounted for below.
        workspace = state + fields
        if mstep_plan is not None:
            summary = mstep_plan.summary()
            workspace += summary["scratch_bytes"] + summary["plan_bytes"]
        workspace += sum(
            int(m.freq_count) * int(m.dft_point_count) * 6 * 2 * 4
            for m in program.monitors
        )
    else:
        workspace = bank_count * state
    cross = program.coefficients.e_inverse_offdiagonal
    if cross.size and not layout.enabled and program.config.backend == "cuda_streamed":
        from beamz.simulation.compile.material_blocks import TensorRegion

        shape = (
            tuple(
                min(n, hi + 1) - max(0, lo - 1)
                for (lo, hi), n in zip(cross.bounds, cross.node_shape, strict=True)
            )
            if isinstance(cross, TensorRegion)
            else cross.shape[:3]
        )
        # The tensor graph is in-place. Include its local corrected curls,
        # padded cross coefficients, and pre-scaled inverse diagonal arrays.
        workspace = state + 4 * (
            sum(
                math.prod(n - int(a == c) for a, n in enumerate(shape))
                for c in range(3)
            )
            + 3 * math.prod(shape)
            + sum(
                getattr(program.coefficients, f"e_inverse_diagonal_{c}").size
                for c in "xyz"
            )
        )
    if (layout.enabled and layout.axis == 2) or program.config.cuda_storage_axes != (
        0,
        1,
        2,
    ):
        materials = (
            sum(value.size * value.dtype.itemsize for value in program.coefficients)
            // count
        )
        workspace += 2 * state + materials
    if not donate_state:
        workspace += state
    return workspace


def cuda_capacity_schedule(program, *, donate_state: bool) -> bool:
    """Keep fast kernels when they fit; use existing in-place kernels otherwise."""
    if program.config.backend != "cuda_streamed":
        return False
    policy = program.config.cuda_memory_policy
    if policy != "auto":
        return policy == "capacity"
    available = cuda_available_bytes(program)
    # CPU contract tests and older backends may not expose allocator statistics.
    # Explicit capacity mode is available without querying or initializing a GPU.
    if available is None:
        return False
    return (
        cuda_workspace_estimate(program, donate_state=donate_state) + (64 << 20)
        > available
    )


def cuda_available_bytes(program) -> int | None:
    """Return the smallest remaining CUDA allocator budget, when available."""
    mesh = program.sharding.mesh
    devices = tuple(mesh.devices.flat) if mesh is not None else tuple(jax.devices())
    budgets = []
    for device in devices:
        if device.platform != "gpu":
            continue
        stats = device.memory_stats()
        if stats and "bytes_limit" in stats and "bytes_in_use" in stats:
            budgets.append(int(stats["bytes_limit"]) - int(stats["bytes_in_use"]))
    return min(budgets) if budgets else None


def check_mstep_workspace(program, plan, *, donate_state):
    """Check a forced temporal workspace before lowering its allocations."""
    from beamz.simulation.compile.backend import CudaBackendUnavailable

    required = cuda_workspace_estimate(
        program, donate_state=donate_state, mstep_plan=plan
    )
    available = cuda_available_bytes(program)
    if program.config.cuda_memory_policy == "capacity":
        raise CudaBackendUnavailable(
            "schedule='mstep' requires additional field workspace; "
            "memory policy 'capacity' requires schedule='streamed' or 'auto'"
        )
    if available is not None and required + (64 << 20) > available:
        raise CudaBackendUnavailable(
            f"schedule='mstep' estimates {required / 2**20:.0f} MiB of workspace "
            f"plus 64 MiB headroom, but only {available / 2**20:.0f} MiB remains; "
            "use schedule='streamed' or 'auto', or reduce the simulation size"
        )
    return required, available
