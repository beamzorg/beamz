"""Cacheable experimental launch choices for the shared two-step planner."""

import os

from beamz.simulation import _cuda_abi as abi

TILES = {
    "16x32": 0,
    "8x32": abi.CUDA_MSTEP_TILE8X32,
    "8x16": abi.CUDA_MSTEP_TILE8X16,
    "16x16": abi.CUDA_MSTEP_TILE16X16,
}
SEGMENTS = {
    "whole": 0,
    "32": abi.CUDA_MSTEP_SEGMENT32,
    "64": abi.CUDA_MSTEP_SEGMENT64,
    "128": abi.CUDA_MSTEP_SEGMENT128,
    "256": abi.CUDA_MSTEP_SEGMENT256,
}
STEPS = {"2": 0, "4": abi.CUDA_MSTEP_STEPS4}
VARIANT_FLAGS = sum(TILES.values()) | sum(SEGMENTS.values()) | sum(STEPS.values())
ALLOWED_FLAGS = abi.CUDA_GRAPH_CACHE | abi.CUDA_BF16_PSI | VARIANT_FLAGS


def variant_flags_from_env():
    tile = os.environ.get("BEAMZ_CUDA_MSTEP_TILE", "16x32").strip().lower()
    segment = os.environ.get("BEAMZ_CUDA_MSTEP_SEGMENT", "whole").strip().lower()
    steps = os.environ.get("BEAMZ_CUDA_MSTEP_STEPS", "2").strip()
    if tile not in TILES:
        raise ValueError("BEAMZ_CUDA_MSTEP_TILE must be 16x32, 8x32, 8x16 or 16x16")
    if segment not in SEGMENTS:
        raise ValueError("BEAMZ_CUDA_MSTEP_SEGMENT must be whole, 32, 64, 128 or 256")
    if steps not in STEPS:
        raise ValueError("BEAMZ_CUDA_MSTEP_STEPS must be 2 or 4")
    return TILES[tile] | SEGMENTS[segment] | STEPS[steps]


def variant_summary(flags):
    tiles = [name for name, bit in TILES.items() if bit and flags & bit]
    segments = [name for name, bit in SEGMENTS.items() if bit and flags & bit]
    if len(tiles) > 1 or len(segments) > 1:
        raise ValueError("Conflicting mstep tile or segment flags")
    tile = tiles[0] if tiles else "16x32"
    segment = int(segments[0]) if segments else 0
    return {
        "tile": tuple(map(int, tile.split("x"))),
        "column_segment": segment,
        "temporal_steps": steps_from_flags(flags),
    }


def steps_from_flags(flags):
    """Steps each mstep pass advances: 4 with CUDA_MSTEP_STEPS4, else 2."""
    return 4 if flags & abi.CUDA_MSTEP_STEPS4 else 2


def with_steps(flags, steps):
    """Set the steps bit to match a plan built for `steps` steps per pass."""
    flags &= ~abi.CUDA_MSTEP_STEPS4
    return flags | (abi.CUDA_MSTEP_STEPS4 if steps == 4 else 0)
