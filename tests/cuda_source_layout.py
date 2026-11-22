"""Explicit include contract shared by standalone C++ correctness harnesses."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_INCLUDE_FLAGS = tuple(
    flag
    for directory in [
        "cuda/src/bindings",
        "cuda/src/common",
        "cuda/src/ordinary",
        "cuda/src/ordinary/mstep",
        "cuda/src/physics",
        "cuda/src/runtime",
        "cuda/src/runtime/deep",
        "cuda/src/runtime/sharded",
        "cuda/src/temporal",
        "cuda/src/temporal/host",
        "cuda/src/temporal/host/observation",
    ]
    for flag in ("-I", str(ROOT / directory))
)
