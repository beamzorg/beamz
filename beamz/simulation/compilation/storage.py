"""Lower discretized simulation requests into immutable execution programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace

import numpy as np

from beamz.simulation.observe import MONITOR_FIELDS, empty_monitor_values

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
    from beamz.simulation.material_blocks import MaterialBlocks
    from beamz.simulation.tensor_regions import TensorRegion

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
    from beamz.simulation.material_blocks import MaterialBlocks

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
