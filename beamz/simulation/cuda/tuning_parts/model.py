"""Host-side CUDA layout prediction and startup cost model."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

_AXES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_GEOMETRY_MODEL = "cpml12-queue-v1"
_MINIMUM_WORK_REDUCTION = 0.25
# (physical cells, canonical GCUPS, rotated GCUPS, additional startup seconds).
# Compact monitor-block ordering; two fresh rounds at 256 steps, checked at
# 4096 steps. Source: docs/reviews/cuda-schedule-space.json.gz, main-anchors
# after restoring streamed field addressing; horizons are earlier profiles.
# Archive at commit 5c389a04, linked from
# docs/reviews/pr334-post-merge-validation-2026-10-17.md.
_LAYOUT_COSTS = (
    (8388608, 5.2982, 6.9365, 1.1399),
    (16777216, 5.3617, 6.8345, 1.8993),
)


def _queue_work(shape):
    """Count 64x4 queue thread slots per phase, including inactive lanes.

    Mirrors LaunchCpmlQueueTile in cuda/src/update.cu for CPML12. Across
    either Yee triplet, the per-axis maximum extent is n+1 and minimum is n.
    The queue partitions z faces, then y faces, then x faces, then the core.
    This is a geometric work proxy, not a prediction of memory traffic or time;
    native fusion can replace the core with a different kernel.
    """
    z, y, x = shape
    iz, iy, ix = z - 24, y - 24, x - 24

    def tiles(n, width):
        return (n + width - 1) // width

    z_blocks = tiles(x + 1, 64) * tiles(y + 1, 4) * 25
    y_blocks = tiles(x + 1, 64) * tiles(25, 4) * iz
    x_blocks = tiles(25, 64) * tiles(iy, 4) * iz
    core_blocks = tiles(ix, 64) * tiles(iy, 4) * iz
    shell_slots = 256 * (z_blocks + y_blocks + x_blocks)
    core_slots = 256 * core_blocks
    return dict(
        shell_slots=shell_slots,
        core_slots=core_slots,
        total_slots=shell_slots + core_slots,
        slots_per_cell=(shell_slots + core_slots) / (z * y * x),
    )


def predict_layout(shape) -> dict[str, Any]:
    """Estimate an RTX3090 CPML12 storage order from logical (z,y,x) extents.

    Pure host arithmetic: no device queries, trials or persistent profiles.
    Only right-handed cyclic permutations are considered. Require a 25% queue
    work reduction to change layout: smaller proxy wins have not reliably
    predicted timing wins in the recorded shape study. This margin is a policy
    choice, not a measured speedup or statistical confidence bound. Keep 64x4:
    32x8 has no established benefit after repairing the narrow-x layout.

    Runtime selection separately checks device and supported workload scope.
    See docs/reviews/cuda-layout-prediction-2026-09-18.md for evidence and limits.
    """
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise ValueError("Expected three integer domain extents") from exc
    if len(shape) != 3 or min(shape) <= 24:
        raise ValueError("CPML12 layout prediction needs three extents above 24")
    candidates: dict[str, dict[str, Any]] = {
        "".join(map(str, axes)): dict(
            stored_shape=tuple(shape[a] for a in axes),
            **_queue_work(tuple(shape[a] for a in axes)),
        )
        for axes in _AXES
    }
    baseline = candidates["012"]["total_slots"]
    best = min(candidates, key=lambda name: candidates[name]["total_slots"])
    reduction = 1 - candidates[best]["total_slots"] / baseline
    selected = best if reduction >= _MINIMUM_WORK_REDUCTION else "012"
    return dict(
        model=_GEOMETRY_MODEL,
        choice=dict(axes=tuple(map(int, selected)), shell_tile="64x4"),
        logical_shape=shape,
        candidate_work=candidates,
        minimum_work_reduction=_MINIMUM_WORK_REDUCTION,
        best_work_reduction=reduction,
        reason=(
            "Substantial reduction in CPML queue thread slots"
            if selected != "012"
            else "Canonical layout retained; no substantial proxy improvement"
        ),
    )


def predict_layout_time_to_result(shape, steps, expected_runs=1):
    """Repay measured rotation startup; keep canonical outside this cost scope.

    Geometry remains a candidate generator. The measured cost family is the
    narrow-x, CPML12, packed-E case; runtime additionally checks the workload.
    """
    try:
        work_steps, runs = operator.index(steps), operator.index(expected_runs)
    except TypeError as exc:
        raise ValueError("steps and expected_runs must be positive integers") from exc
    if (
        isinstance(steps, bool)
        or isinstance(expected_runs, bool)
        or min(work_steps, runs) < 1
    ):
        raise ValueError("steps and expected_runs must be positive integers")
    report = predict_layout(shape)
    candidate = report["choice"]
    report = {
        **report,
        "candidate_choice": candidate,
        "choice": dict(axes=(0, 1, 2), shell_tile="64x4"),
    }
    z, y, x = report["logical_shape"]
    if candidate["axes"] == (0, 1, 2):
        return report
    if not (
        candidate["axes"] == (1, 2, 0)
        and 512 <= z <= 1024
        and y == 256
        and 63 <= x <= 65
    ):
        return {**report, "reason": "Rotated layout has no measured startup model"}
    cells = z * y * x
    small, large = _LAYOUT_COSTS
    fraction = np.clip(np.log2(cells / small[0]), 0, 1)
    canonical, rotated, extra = (
        float(a + fraction * (b - a)) for a, b in zip(small[1:], large[1:], strict=True)
    )
    saved = cells * work_steps * runs * (1 / canonical - 1 / rotated) / 1e9
    admitted = saved > 1.25 * extra
    return {
        **report,
        "cost_model": "rtx3090-layout-startup-v2",
        "choice": candidate if admitted else report["choice"],
        "expected_runs": expected_runs,
        "predicted_saved_execution_s": saved,
        "extra_startup_s": extra,
        "reason": "Predicted rotation savings repay startup"
        if admitted
        else "Predicted rotation savings do not repay startup",
    }
