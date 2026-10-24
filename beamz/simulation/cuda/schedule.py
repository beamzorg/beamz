"""Conservative schedule selection from the recorded RTX3090 cost envelope.

No timing trial runs during selection. Forced schedules remain available on all
supported GPUs; automatic mstep promotion stays inside the measured envelope.
"""

from __future__ import annotations

import operator

import numpy as np

# (physical cells, streamed GCUPS, mstep GCUPS, extra startup seconds).
# Startup includes planning, lowering, XLA compilation and first-execution
# overhead after shared material setup. Source: cuda-schedule-space.json.gz,
# main-anchors after restoring streamed field addressing (archive at commit
# 5c389a04; linked from docs/reviews/pr334-post-merge-validation-2026-10-17.md).
_COSTS = {
    "all": (
        (16777216, 8.2242, 9.0441, 0.1353),
        (67108864, 8.5313, 10.7144, 0.5245),
    ),
    "right": (
        (16777216, 8.1159, 12.4346, 0.1353),
        (67108864, 7.4307, 12.9735, 0.4739),
    ),
    "pec": (
        (16777216, 8.4208, 13.5445, 0.1511),
        (67108864, 8.2482, 13.8656, 0.4838),
    ),
}
_MODEL = "rtx3090-column-v3"


def _enough_work(cfg):
    # Measured 16-step calls can repay preparation when the same executable is
    # known to be reused. Do not query a GPU or plan columns for a short one-off.
    return cfg.num_steps >= 16 and cfg.num_steps * cfg.cuda_expected_runs >= 256


def normalize_schedule_request(backend, schedule, expected_runs):
    aliases = {
        "cuda_streamed": "streamed",
        "cuda_mstep": "mstep",
        "cuda-mstep": "mstep",
    }
    # Existing calls retain main's streamed policy. The new cost selector is an
    # explicit schedule='auto' opt-in, separate from backend availability='auto'.
    default = "auto" if backend == "jax" else aliases.get(backend, "streamed")
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


def cuda_device_kind():
    from beamz.simulation.backend import _gpu_devices

    devices = _gpu_devices()
    return str(devices[0].device_kind) if len(devices) == 1 else ""


def measured_boundary_family(cpml):
    if cpml.magnetic_edges:
        return None
    if not cpml.enabled:
        return "pec" if len(cpml.metallic_edges) == 6 else None
    terms = (*cpml.h_terms, *cpml.e_terms)
    if len(terms) != 12:
        return None
    if all(t.slab.low == t.slab.high == 12 for t in terms):
        return "all"
    if all(
        t.slab.low == 0 and t.slab.high == (12 if t.axis == 2 else 0) for t in terms
    ):
        return (
            "right"
            if cpml.metallic_edges
            == frozenset({"front", "back", "bottom", "top", "left"})
            else None
        )
    return None


def predict_cost(cells, family, steps, expected_runs):
    """Interpolate measured sizes; amortize preparation and initial capture."""
    small, large = _COSTS[family]
    fraction = np.clip((np.log2(cells) - np.log2(small[0])) / 2, 0, 1)
    streamed, mstep, extra_startup = (
        float(a + fraction * (b - a)) for a, b in zip(small[1:], large[1:], strict=True)
    )
    saved = cells * steps * expected_runs * (1 / streamed - 1 / mstep) / 1e9
    return {
        "model": _MODEL,
        "predicted_saved_execution_s": saved,
        "extra_startup_s": extra_startup,
        "expected_runs": expected_runs,
        "streamed_gcups": streamed,
        "mstep_gcups": mstep,
    }


def automatic_candidate(program, geometry, *, device_kind):
    """Return a reason and cost report; unsupported/unmeasured cases stay streamed."""
    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.backend import CUDA_DEFAULT_FLAGS
    from beamz.simulation.cuda.runtime import _packed_e_material

    cfg = program.config
    if cfg.cuda_storage_axes != (0, 1, 2):
        return False, "rotated mstep has no measured selection model", {}
    if not device_kind.endswith("RTX 3090"):
        return False, "GPU has no measured mstep selection model", {}
    if not _enough_work(cfg):
        return False, "short invocation outside the measured selection envelope", {}
    if cfg.cuda_flags != CUDA_DEFAULT_FLAGS:
        return False, "explicit precision or kernel settings", {}
    if cfg.cuda_graph_cache_capacity < 1:
        return False, "graph caching has no measured selection model", {}
    if not _packed_e_material(program.coefficients) or any(
        getattr(program.coefficients, f"h_{kind}_{axis}").ndim != 0
        for kind in ("decay", "source")
        for axis in "xyz"
    ):
        return False, "coefficient representation has no measured selection model", {}
    family = measured_boundary_family(program.boundary.cpml)
    if family is None:
        return False, "boundary layout has no measured selection model", {}
    cells = int(np.prod(tuple(n - 1 for n in geometry.shape)))
    details = geometry.summary()
    if not (
        16777216 <= cells <= 67108864
        and details["active_tiles"] >= 164
        and details["column_depth"] >= 64
        and details["core_fraction"] >= 0.6
    ):
        return False, "column geometry outside the measured selection envelope", details
    if (
        len(program.monitors) > 2
        or any(m.freq_count > 3 for m in program.monitors)
        or sum(m.dft_point_count for m in program.monitors) > 0.02 * cells
    ):
        return (
            False,
            "monitor workload outside the measured selection envelope",
            details,
        )
    costs = predict_cost(cells, family, cfg.num_steps, cfg.cuda_expected_runs)
    variant = (
        abi.CUDA_MSTEP_TILE8X16 | abi.CUDA_MSTEP_SEGMENT32
        if family in {"pec", "right"}
        else 0
    )
    costs["mstep_cuda_flags"] = abi.CUDA_GRAPH_CACHE | variant
    details = {**details, "boundary_family": family, **costs}
    if costs["predicted_saved_execution_s"] <= 1.25 * costs["extra_startup_s"]:
        return False, "predicted execution savings do not repay startup", details
    return True, "measured column geometry and amortized execution cost", details


def choose_execution_schedule(
    program, groups, packed_monitors, *, native_supported, donate_state
):
    """Freeze capability, cost and memory choices before allocating workspaces."""
    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.backend import CudaBackendUnavailable, cuda_backend_status
    from beamz.simulation.cuda.mstep import build_mstep_geometry, build_mstep_plan
    from beamz.simulation.memory import (
        check_mstep_workspace,
        cuda_available_bytes,
        cuda_workspace_estimate,
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
    device_kind = ""
    if requested == "auto":
        if cfg.cuda_memory_policy == "capacity":
            return None, {
                **decision,
                "reason": "capacity policy requires in-place execution",
            }
        if not _enough_work(cfg):
            return None, {
                **decision,
                "reason": "short invocation outside the measured selection envelope",
            }
        device_kind = cuda_device_kind()
        if not device_kind.endswith("RTX 3090"):
            return None, {
                **decision,
                "reason": "GPU has no measured mstep selection model",
                "device": device_kind,
            }
        status = cuda_backend_status(register=False)
        if abi.CUDA_MSTEP_TARGET not in status.targets:
            return None, {**decision, "reason": "optional mstep target is unavailable"}
        available = cuda_available_bytes(program)
        if available is None:
            return None, {
                **decision,
                "available_bytes": None,
                "reason": "mstep requires known memory headroom",
            }
        minimum = cuda_workspace_estimate(
            program, donate_state=donate_state, mstep=True
        )
        if minimum + (64 << 20) > available:
            return None, {
                **decision,
                "mstep_workspace_minimum_bytes": minimum,
                "available_bytes": available,
                "reason": "minimum mstep workspace exceeds known memory headroom",
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
    if requested == "auto":
        candidate, reason, costs = automatic_candidate(
            program, geometry, device_kind=device_kind
        )
        decision = {**decision, **costs}
        if not candidate:
            return None, {**decision, "reason": reason}
    plan = build_mstep_plan(
        shapes,
        cpml,
        planning_groups,
        monitor_indices,
    )
    if requested == "mstep":
        required, available = check_mstep_workspace(
            program, plan, donate_state=donate_state
        )
    else:
        required = cuda_workspace_estimate(
            program, donate_state=donate_state, mstep_plan=plan
        )
        available = cuda_available_bytes(program)
        decision = {
            **decision,
            "workspace_bytes": required,
            "available_bytes": available,
        }
        if available is None or required + (64 << 20) > available:
            return None, {
                **decision,
                "mstep_workspace_bytes": required,
                "reason": "mstep workspace exceeds known memory headroom",
            }
    return plan, {
        **decision,
        **plan.summary(),
        "selected": "mstep",
        "reason": reason,
        "workspace_bytes": required,
        "available_bytes": available,
    }
