"""Explicit legacy-ABI overrides; automatic planning belongs to the CUDA engine."""

from __future__ import annotations

import operator


def normalize_schedule_request(backend, schedule, expected_runs):
    aliases = {
        "cuda_streamed": "streamed",
        "cuda_mstep": "mstep",
        "cuda-mstep": "mstep",
    }
    default = aliases.get(backend, "auto")
    requested = default if schedule is None else str(schedule).strip().lower()
    if requested not in {"auto", "streamed", "mstep"}:
        raise ValueError("schedule must be auto, streamed, or mstep")
    if backend in aliases and requested != aliases[backend]:
        raise ValueError(
            f"backend={backend!r} requires schedule={aliases[backend]!r}; use backend='cuda' to choose another schedule"
        )
    if backend == "jax" and requested != "auto":
        raise ValueError("CUDA schedules require backend='cuda' or 'auto'")
    return requested, normalize_expected_runs(expected_runs)


def normalize_expected_runs(expected_runs):
    """Validate the caller's reuse hint before progress multiplies it."""
    try:
        runs = operator.index(expected_runs)
    except TypeError as exc:
        raise ValueError("expected_runs must be a positive integer") from exc
    if isinstance(expected_runs, bool) or runs < 1:
        raise ValueError("expected_runs must be a positive integer")
    return runs


def choose_execution_schedule(
    program, groups, packed_monitors, *, native_supported, donate_state
):
    """Freeze capability, cost and memory choices before allocating workspaces."""
    from beamz.simulation.backend import CudaBackendUnavailable
    from beamz.simulation.cuda.mstep import build_mstep_geometry, build_mstep_plan
    from beamz.simulation.memory import (
        check_mstep_workspace,
    )

    cfg = program.config
    requested = "mstep" if cfg.backend == "cuda_mstep" else cfg.cuda_schedule
    decision = {
        "requested": requested,
        "selected": "streamed",
        "expected_runs": cfg.cuda_expected_runs,
        "reason": "explicit streamed schedule",
    }
    if cfg.backend == "jax":
        return None, {**decision, "selected": "jax", "reason": "JAX execution"}
    if requested == "streamed":
        return None, decision
    supported = (
        native_supported
        and cfg.is_3d
        and not cfg.sharding.enabled
        and cfg.metric_kind == "isotropic_uniform"
        and not program.coefficients.e_inverse_offdiagonal.size
    )
    if not supported:
        if requested == "mstep":
            raise CudaBackendUnavailable(
                "cuda_mstep requires unsharded isotropic float32 3D fields, "
                "batched slab sources and vector DFT monitors; recorders, flux/power "
                "monitors and other configurations require schedule='streamed' or jax"
            )
        return None, {**decision, "reason": "features require streamed execution"}
    if cfg.num_steps < 2:
        return None, {**decision, "reason": "single step"}
    if requested == "auto":
        return None, {
            **decision,
            "reason": "ordinary CUDA interoperability; native planning owns temporal selection",
        }
    from beamz.simulation.cuda.storage import mstep_plan_inputs

    shapes, cpml, planning_groups, monitor_indices = mstep_plan_inputs(
        program, groups, packed_monitors
    )
    geometry = build_mstep_geometry(shapes, cpml, planning_groups)
    decision = {**decision, **geometry.summary()}
    if not geometry.core_cells:
        return None, {**decision, "reason": "no eligible columns"}
    reason = "explicit mstep schedule"
    plan = build_mstep_plan(
        shapes,
        cpml,
        planning_groups,
        monitor_indices,
    )
    required, available = check_mstep_workspace(
        program, plan, donate_state=donate_state
    )
    return plan, {
        **decision,
        **plan.summary(),
        "selected": "mstep",
        "reason": reason,
        "workspace_bytes": required,
        "available_bytes": available,
    }
