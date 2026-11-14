"""Optimization loop, schedules, and checkpoint state."""

from importlib import import_module
from typing import Any

__all__ = ["Optimizer"]


def __getattr__(name: str = "") -> Any:
    if name in {
        "Optimizer",
        "_optimize",
        "_function_token",
        "TOPOLOGY_NUMERICS_VERSION",
    }:
        return getattr(import_module("beamz.optimization.optimizer.optimizer"), name)
    raise AttributeError(name)
