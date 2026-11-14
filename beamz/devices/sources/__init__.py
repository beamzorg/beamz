from beamz.devices import _install_legacy_modules

_install_legacy_modules(
    __name__,
    (
        ("bloch", "beamz.devices.sources.launch.bloch"),
        ("mode_launch", "beamz.devices.sources.launch.mode_launch"),
        ("planar_tfsf", "beamz.devices.sources.launch.planar_tfsf"),
        ("tensor_preview", "beamz.devices.sources.launch.tensor_preview"),
    ),
)

from beamz.devices.sources.specs import PlaneWaveSource  # noqa: E402

from .specs import (  # noqa: E402
    CustomSource,
    GaussianBeamSource,
    GaussianSource,
    ModeData,
    ModeSource,
    ModeSpec,
)
from .time import GaussianPulse, SampledSignal  # noqa: E402

CANONICAL_SOURCE_TYPES = (
    CustomSource,
    GaussianSource,
    GaussianBeamSource,
    PlaneWaveSource,
    ModeSource,
)

__all__ = [
    "ModeSource",
    "ModeSpec",
    "ModeData",
    "GaussianPulse",
    "SampledSignal",
    "GaussianSource",
    "GaussianBeamSource",
    "PlaneWaveSource",
    "CustomSource",
]
