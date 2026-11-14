"""Native finite-difference mode solving for BeamZ devices."""

from __future__ import annotations

from beamz.devices import _install_legacy_modules

_install_legacy_modules(
    __name__,
    (
        ("constants", "beamz.devices.modes.backend.constants"),
        ("solver", "beamz.devices.modes.backend.solver"),
        ("_scipy", "beamz.devices.modes.backend._scipy"),
        ("_tensor_yee", "beamz.devices.modes.backend._tensor_yee"),
        ("_yee", "beamz.devices.modes.backend._yee"),
        ("discrete", "beamz.devices.modes.spatial.discrete"),
        ("fields", "beamz.devices.modes.spatial.fields"),
        ("line", "beamz.devices.modes.spatial.line"),
        ("plane", "beamz.devices.modes.spatial.plane"),
        ("_sampling", "beamz.devices.modes.spatial._sampling"),
        ("constitutive", "beamz.devices.modes.support.constitutive"),
        ("policy", "beamz.devices.modes.support.policy"),
        ("_bend", "beamz.devices.modes.support._bend"),
        ("_cache", "beamz.devices.modes.support._cache"),
        ("_symmetry", "beamz.devices.modes.support._symmetry"),
    ),
)

# Re-export the small public API from package root so users do not need to know
# the internal module split.
from beamz.devices.modes.backend.solver import solve_grid  # noqa: E402

from .models import Materials  # noqa: E402
from .result import Result  # noqa: E402
from .specs import ModeData, ModeSpec  # noqa: E402

# BeamZ users configure modes through ModeSpec/ModeData. ``solve_grid`` and its
# Result consume an explicitly validated Materials value. Placement contracts
# and sparse numerical implementation details remain internal.
__all__ = [
    "Materials",
    "ModeData",
    "ModeSpec",
    "Result",
    "solve_grid",
]
