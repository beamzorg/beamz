"""Capacity-aware CUDA scheduling, independent of input-buffer ownership.

This is a conservative workspace estimate, not an allocator reservation. A
choice is frozen when a scan is built; it does not change inside a timestep.
"""

from __future__ import annotations

import math
import os

import jax

from beamz.simulation import _cuda_abi as abi


def cuda_memory_policy_from_env() -> str:
    policy = os.environ.get("BEAMZ_CUDA_MEMORY_POLICY", "auto").strip().lower()
    if policy not in {"auto", "speed", "capacity"}:
        raise ValueError("BEAMZ_CUDA_MEMORY_POLICY must be auto, speed, or capacity")
    return policy


def cuda_workspace_estimate(program, *, donate_state: bool) -> int:
    """Estimate optional fast-schedule workspace on the busiest device.

    Include both field banks for pair kernels, rotated materials for layout
    transforms, replicated normal CPML slabs, and retained input ownership.
    This intentionally overestimates rather than promising an exact peak.
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
    bank_count = 2 if program.config.cuda_flags & abi.CUDA_TEMPORAL_PAIR else 1
    workspace = bank_count * state
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
    mesh = program.sharding.mesh
    devices = tuple(mesh.devices.flat) if mesh is not None else tuple(jax.devices())
    budgets = []
    for device in devices:
        if device.platform != "gpu":
            continue
        stats = device.memory_stats()
        if stats and "bytes_limit" in stats and "bytes_in_use" in stats:
            budgets.append(int(stats["bytes_limit"]) - int(stats["bytes_in_use"]))
    # CPU contract tests and older backends may not expose allocator statistics.
    # Explicit capacity mode is available without querying or initializing a GPU.
    if not budgets:
        return False
    return cuda_workspace_estimate(program, donate_state=donate_state) + (
        64 << 20
    ) > min(budgets)
