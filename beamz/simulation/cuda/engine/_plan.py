"""The sole production CUDA schedule decision: deterministic and immutable."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ._capabilities import _Unsupported
from ._contracts import (
    _CPML,
    _FIELDS,
    _PHASES,
    _YEE,
    _Buffer,
    _ResourcePlanner,
)


@dataclass(frozen=True, slots=True)
class _Plan:
    backend: str = "cuda"
    binding: str = "native"
    state_ownership: str = "private"
    schedule: str = "ordinary"
    ordinary_general: bool = False
    depth: int = 1
    rolling_band: int = 0
    direct_volume_recording: bool = False
    boundary_gather_bytes: int = 0
    raw_slots: int = 0
    macro_slab_count: int = 0
    macro_level_stride: int = 0
    scheduling_sms: int = 0
    observation_slots: int = 0
    payload_bytes: int = 0
    memory_budget_bytes: int = 0
    memory_policy: str = "auto"
    reserve_bytes: int = 512 << 20
    reason: str = ""
    physics_sha256: str = ""
    estimated_ordinary_s: float = 0.0
    estimated_first_solve_s: float = 0.0

    def _report(self) -> dict[str, Any]:
        return asdict(self)


def _resident_packet_bytes(inputs, module, layout):
    """Count physical allocations once under the native packet aliasing contract."""
    electric = layout.inputs["e_decay_x"]
    compact_indices = getattr(module, "compiled_compact_temporal_indices", False)
    aliasing = getattr(module, "compiled_immutable_aliases", False)
    seen = set()
    payload = 4
    for index, value in enumerate(inputs):
        if index in layout.deferred:
            continue
        mutable = index in layout.zeroable or layout.monitor + 10 <= index < layout.monitor + 16
        capture = (
            (6 <= index < 12 and not value.shape)
            or electric <= index < electric + 6
            or index == layout.monitor
            or (index == layout.monitor + 4 and compact_indices)
            or (layout.source <= index < layout.monitor and (index - layout.source) % 3 == 2)
        )
        key = (id(value), capture) if aliasing and not mutable else index
        if key not in seen:
            payload += value.nbytes
            seen.add(key)
    return payload


def _rolling_band(shape, device, steps):
    """Width of a sliding z-band for measured large, regular GPU workloads."""
    if steps < 128 or math.prod(shape) < 100_000_000 or min(shape) < 128:
        return 0
    name = device.get("name", "")
    if name in ("NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3") and shape[0] >= 512:
        return 256 if shape[0] >= 1024 else 128
    if name == "NVIDIA GeForce RTX 5090":
        return 128 if shape[0] < 1024 else 256
    return 0


def _admit_rolling(resource, cells, preferred, available, *, envelope=None):
    """Price the widest measured rolling band against the live frontier and the device limit."""
    # A compact group of mode apertures can profit from rolling even when it
    # observes several components. Dense box monitors favor the wavefront.
    if resource.get("observation_keys", 0) * 128 > cells or resource.get("scatter_sweep_planes", 0) > 2:
        return 0, None
    if envelope is None:
        envelope = resource["extra_gpu_bytes"]
        if not resource.get("scatter_sweep_planes"):
            envelope += resource.get("field_frontier_bytes", 0)
    choices = resource.get("rolling_band_bytes")
    if choices is None:
        choices = {preferred: resource.get("rolling_extra_bytes")}
    for width, extra in sorted(choices.items(), reverse=True):
        if width <= preferred and extra is not None and extra <= min(available, envelope):
            return width, extra
    return 0, None


def _volume_frame_density(recorders):
    return sum(1.0 / interval for _, interval, _, indices, _, history in recorders
               if not indices.size and history.shape[0])


def _single_rolling_band(shape, device, steps, module, layout, attributes, effects, direct_volume=False):
    """Check the regular-grid rolling contract and any full-frame acquisition."""
    if (layout is not _CPML or effects.get("tensor")
            or attributes.get("boundary_code") != 12 << 8 or attributes.get("metric_kind", 0)
            or not getattr(module, "compiled_rolling_tiles", False)):
        return 0
    recorders = effects.get("recorders")
    scatters = effects.get("scatters")
    if not recorders and not scatters:
        return _rolling_band(shape, device, steps)
    if (device.get("name") != "NVIDIA GeForce RTX 3090"
            or (device.get("major"), device.get("minor")) != (8, 6)
            or steps < 128 or min(shape) < 256 or math.prod(shape) < 100_000_000):
        return 0
    if scatters:
        return 64 if not recorders and getattr(module, "compiled_rolling_source_bounds", False) else 0
    if not getattr(module, "compiled_rolling_volume_windows", False):
        return 0
    # A slice contributes sparse observation keys to native memory admission.
    # Whole-volume records share the fast bulk between acquired frames.
    density = 1 / min(spec[1] for spec in recorders) if direct_volume else _volume_frame_density(recorders)
    return 64 if 1 / 64 <= density <= 1 / 17 else 0


def _boundary_recorder_bytes(recorders, shape, module):
    """Price a completed-field acquisition on the common recorder clock."""
    interval = min(record[1] for record in recorders)
    if interval % 16 > 1 or any(
        cadence % interval or not history.shape[0] or not history.flags.c_contiguous
        or not history.flags.writeable or history.dtype != np.float32
        for _, cadence, _, _, _, history in recorders
    ):
        return None
    if all(not indices.size for _, _, _, indices, _, _ in recorders):
        return 0 if _volume_frame_density(recorders) <= 6 / 64 else None
    if (interval % 16 or not 32 <= interval <= 64 or not getattr(module, "compiled_boundary_acquisition", False)
            or any(not indices.size or history.shape[1] > math.prod(sorted(shape)[-2:])
                   for _, _, _, indices, _, history in recorders)):
        return None
    return sum(indices.nbytes + weights.nbytes + history.shape[1] * history.itemsize
               for _, _, _, indices, weights, history in recorders)


def _choose(inputs: tuple[_Buffer, ...] = (), module: _ResourcePlanner | str = "",
            device: Mapping[str, Any] = MappingProxyType({}), *, recorders: tuple = (),
            recorder_staging_bytes: int = 0, **options) -> _Plan:
    """Compare a boundary download with ordinary/staged execution at its own memory cost."""
    if (recorders and 0 < recorder_staging_bytes <= options.get("extra_payload_bytes", 0)
            and device.get("name") == "NVIDIA GeForce RTX 3090"
            and options.get("requested") != "ordinary" and options.get("memory_policy") != "capacity"):
        boundary_bytes = _boundary_recorder_bytes(recorders, inputs[0].shape, module)
        if boundary_bytes is not None and boundary_bytes <= recorder_staging_bytes:
            direct = dict(options, direct_volume_recording=boundary_bytes == 0,
                          boundary_gather_bytes=boundary_bytes,
                          extra_payload_bytes=options["extra_payload_bytes"] - recorder_staging_bytes + boundary_bytes)
            prior = None
            if boundary_bytes:
                # Sparse recorder keys can make ordinary CUDA the smaller prior
                # schedule. Charge a new band to that actual plan.
                try:
                    prior = _schedule(inputs, module, device, recorders=recorders, **options)
                except MemoryError:
                    pass
                else:
                    direct["rolling_payload_limit"] = prior.payload_bytes
            try:
                plan = _schedule(inputs, module, device, recorders=recorders, **direct)
            except MemoryError:
                pass
            else:
                if plan.rolling_band:
                    return plan
            if prior is not None:
                return prior
    return _schedule(inputs, module, device, recorders=recorders, **options)


def _schedule(
    inputs: tuple[_Buffer, ...] = (),
    module: _ResourcePlanner | str = "",
    device: Mapping[str, Any] = MappingProxyType({}),
    *,
    steps: int = 0,
    requested: str = "auto",
    budget: int = 0,
    temporal_reason: str = "",
    memory_policy: str = "auto",
    extra_payload_bytes: int = 0,
    scatters: tuple = (),
    recorders: tuple = (),
    tensor: bool = False,
    direct_volume_recording: bool = False, boundary_gather_bytes: int = 0,
    rolling_payload_limit: int | None = None,
    attributes: Mapping[str, Any] = MappingProxyType({}),
) -> _Plan:
    if isinstance(module, str): raise ValueError("A native CUDA resource planner is required")
    if memory_policy == "capacity" and requested == "temporal":
        from beamz.simulation.backend import CudaBackendUnavailable
        raise CudaBackendUnavailable("Temporal CUDA requires extra workspace; memory policy 'capacity' requires ordinary execution")
    layout = _CPML if len(inputs) == _CPML.input_count else _YEE
    indices = layout.inputs
    payload = _resident_packet_bytes(inputs, module, layout) + extra_payload_bytes
    budget = min(budget or int(device["total_memory_bytes"] * .9), int(device.get("free_memory_bytes", device["total_memory_bytes"])))
    reserve = 512 << 20
    if payload + reserve > budget:
        raise MemoryError(f"CUDA ordinary payload {payload} plus reserve {reserve} exceeds budget {budget}")
    shape = tuple(min(v.shape[a] for v in inputs[:len(_FIELDS)]) for a in range(3))
    cells = math.prod(shape)
    ordinary = cells * steps / 6e9 + steps * 6e-6
    base = _Plan(payload_bytes=payload, memory_budget_bytes=budget, memory_policy=memory_policy, physics_sha256=str(module.compiled_program_sha256),
        estimated_ordinary_s=ordinary, estimated_first_solve_s=ordinary)
    if layout is _YEE and not getattr(module, "compiled_temporal_boundary_widths", False):
        temporal_reason = "temporal execution requires CPML12"
    if not getattr(module, "compiled_temporal_sources", False) and any(inputs[indices[f"source_{_PHASES[0]}_{axis}_coefficients"]].shape[0] for axis in "xyz"):
        temporal_reason = "pre-electric sources require ordinary execution"
    if not temporal_reason and not getattr(module, "compiled_temporal_diagonal_materials", False):
        if any(inputs[i].shape for i in range(6, 12)):
            temporal_reason = "spatial magnetic coefficients require ordinary execution"
        elif any(inputs[indices["e_source_x"]+i].dtype != np.int32 for i in range(3)):
            temporal_reason = "dense electric coefficients require ordinary execution"
    if temporal_reason:
        if requested == "temporal": raise _Unsupported("Temporal CUDA unsupported: " + temporal_reason)
        return replace(base, ordinary_general=True, reason=temporal_reason)
    if memory_policy == "capacity": return replace(base, reason="capacity policy selects minimum-memory ordinary execution")
    if requested == "ordinary": return replace(base, reason="ordinary explicitly requested")
    if tensor and requested != "temporal":
        return replace(base, reason="ordinary CUDA is faster for measured full tensor workloads")
    model = str(device.get("name", "")).removeprefix("NVIDIA ").removeprefix("GeForce ")
    capability = (int(device.get("major", 0)), int(device.get("minor", 0)))
    ampere_3090 = model == "RTX 3090" and capability == (8, 6)
    # Cold setup scans material blocks and builds the graph. Measured costs
    # grow with the grid and the distinct field values gathered by monitors.
    startup = 0.04 + 4.5e-9 * cells if ampere_3090 else 2.0
    if requested != "temporal" and ordinary <= startup:
        return replace(base, reason="ordinary execution cannot amortize temporal startup")
    # Preferred (time depth, simultaneous wavefronts) measured on independent
    # CPML/DFT workloads. The resource planner still owns the memory decision.
    profile = {
        ("RTX 3090", (8, 6)): (16, 4), ("RTX 5090", (12, 0)): (32, 4),
        ("A100-SXM4-80GB", (8, 0)): (16, 6), ("H100 NVL", (9, 0)): (16, 6),
        ("H100 80GB HBM3", (9, 0)): (16, 6),
    }.get((model, capability))
    if requested != "temporal" and profile is None:
        return replace(base, reason="temporal cost model unvalidated for this device")
    # Measured RTX3090 tile-step priors: bulk, CPML-only, and remaining source/observer work.
    # Counts and frontier costs come from the shared native planner, never geometry names.
    effects = {name: values for name, values in (("scatters", scatters), ("recorders", recorders)) if values}
    # Once the preferred depth exhausts its overlap choices, a shorter
    # block can keep the field updates tiled inside the same device budget.
    compiled_depths = getattr(module, "compiled_depths", (16,))
    preferred_depth, preferred_slots = profile if profile and requested == "auto" else (16, 4)
    if profile and memory_policy == "speed" and requested == "auto":
        # Wider overlap improved warm runs on both the MMI and large dielectric
        # grid. The native resource planner can reduce it to remain in budget.
        preferred_slots = {"H100 80GB HBM3": 12, "RTX 5090": 8}.get(model, preferred_slots)
    if tensor:
        depths = (4,)
        effects["tensor"] = True
    else:
        depths = tuple(depth for depth in (32, 16, 12, 8, 4) if depth <= preferred_depth and depth in compiled_depths)
    # The native planner can reuse one physical partition for all overlaps.
    if getattr(module, "compiled_budgeted_overlap", False):
        effects.update(budget=budget - payload - reserve, minimum_overlap=2)
    sms = int(device.get("multiprocessors", 82))
    if (model == "RTX 3090" and requested == "auto" and memory_policy == "auto"
            and getattr(module, "compiled_observation_spacing", False)):
        effects["scheduling_sms"] = sms
    band = _single_rolling_band(shape, device, steps, module, layout, attributes, effects, direct_volume_recording or bool(boundary_gather_bytes))
    if direct_volume_recording or boundary_gather_bytes:
        if not band:
            return base
        # Completed fields supply each sample at the acquisition step. Weighted
        # gathers use their priced frame buffer, with no per-step recorder keys.
        effects.pop("recorders")
    for depth in depths:
        if getattr(module, "compiled_rolling_tiles", False):
            effects["macro_slab_count"] = max(2, min(16, (steps // depth) // 2 * 2))
        if band and depth % 8 == 0:
            effects.update(rolling_band=band, boundary_code=attributes["boundary_code"])
        else:
            effects.pop("rolling_band", None)
        slots = preferred_slots
        while slots >= 2:
            resource = module.moving_resource_plan(inputs, depth=depth, raw_slots=slots, **effects)
            slots = int(resource.get("raw_slots", slots))
            # Driver-owned graph storage grows with the number of captured
            # wavefront levels. The RTX 3090 process peaks include this cost.
            graph_reserve = reserve
            if profile is not None:
                graph_reserve = max(reserve, (320 << 20) + int(resource["macro_levels"]) * (256 << 10))
            selected_band, roll_bytes = (0, None)
            if effects.get("rolling_band"):
                envelope = resource["extra_gpu_bytes"] if boundary_gather_bytes else None
                available = min(budget - graph_reserve, budget if rolling_payload_limit is None else rolling_payload_limit) - payload
                selected_band, roll_bytes = _admit_rolling(resource, cells, band, available, envelope=envelope)
            total = payload + (roll_bytes if roll_bytes is not None else int(resource["extra_gpu_bytes"]))
            if total + graph_reserve <= budget:
                break
            slots -= 1
        else:
            continue
        policies = resource["policy_tile_counts"]
        tile_step = policies[0] * 1.2e-7 + policies[1] * 1.6e-7 + sum(policies[2:]) * 4.5e-7
        frontier = resource["max_frontier"]
        cold = 2.0
        if ampere_3090 and "rank_count" in resource:
            # Sparse ends of each wavefront also execute. A peak frontier can
            # overstate occupancy on small cubes even with several macros live.
            frontier = min(frontier, resource["tile_count"] / max(1, resource["rank_count"]))
            cold = startup + 4e-7 * resource.get("observation_keys", 0)
        utilization = max(1.0, sms / max(1, frontier))
        waves = 1 + sms * resource["macro_levels"] / max(1, resource["tile_count"] * 32)
        # The reference tile costs were collected at depth 16 on Ampere.
        # At depth 32 the measured Blackwell cost per tile-step is lower.
        if model == "RTX 5090" and capability == (12, 0) and depth == 32:
            tile_step *= 0.7
        temporal = cold + tile_step * (steps // depth * depth) * utilization * waves + cells * (steps % depth) / 6e9
        prefer_ordinary = requested != "temporal" and temporal >= ordinary * (1.0 if memory_policy == "speed" else .9)
        if roll_bytes is not None and not (scatters and prefer_ordinary):
            rolling_rate = {(8, 0): 23e9, (9, 0): 38e9}.get(capability, 49e9)
            estimated = 2.0 + cells * steps / rolling_rate
            if scatters or recorders:
                # Large Ampere windows pay the band rate plus transfer time for
                # each full FP32 frame. A narrow current adds no frame transfer.
                per_cell = 0.075e-9 if selected_band == 32 else 0.0625e-9
                estimated = startup + cells * steps * (per_cell + 0.75e-9 * _volume_frame_density(recorders))
            return replace(base, schedule="temporal", depth=depth, raw_slots=slots,
                direct_volume_recording=direct_volume_recording, boundary_gather_bytes=boundary_gather_bytes,
                macro_slab_count=int(resource["macro_slab_count"]), rolling_band=selected_band,
                observation_slots=int(bool(resource.get("observation_keys"))),
                payload_bytes=total, reserve_bytes=graph_reserve,
                estimated_first_solve_s=estimated,
                reason="four-step rolling tiles fit the measured workload and memory envelope")
        if prefer_ordinary:
            return replace(base, reason="modeled temporal startup, tiles and tails exceed ordinary cost" if memory_policy == "speed" else "modeled temporal first-solve improvement is below the 10% automatic benefit threshold")
        return replace(base, schedule="temporal", depth=depth, raw_slots=slots, reserve_bytes=graph_reserve,
            observation_slots=int(resource.get("observation_slots", slots)), macro_slab_count=int(resource.get("macro_slab_count", 16)),
            macro_level_stride=int(resource.get("macro_level_stride", 0)), scheduling_sms=effects.get("scheduling_sms", 0),
            payload_bytes=total, estimated_first_solve_s=temporal,
            reason="explicit temporal request" if requested=="temporal" else "modeled temporal startup, tiles and tails improve first-solve cost within budget")
    if requested == "temporal": raise MemoryError(f"No bounded CUDA temporal overlap fits budget {budget}")
    return replace(base, reason="temporal storage exceeds the declared memory budget")
