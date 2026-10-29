"""Canonical modal ports."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

import numpy as np

from beamz.devices._immutable import (
    _Replaceable,
    finite_tuple,
    immutable_snapshot,
    nonnegative_finite_extents,
    shifted_coordinates,
)
from beamz.devices.modes.specs import ModeSpec

if TYPE_CHECKING:
    from beamz.devices.monitors import ModeMonitor
    from beamz.devices.sources import ModeSource

PortDirection = Literal["+", "-"]


@dataclass(frozen=True)
class Port(_Replaceable):
    """Modal S-parameter port on a finite plane.

    The zero extent in ``size`` selects the axis. ``direction`` points into the
    device. Use :meth:`to_source` and :meth:`to_monitor` to build matching devices.
    ``monitor_name`` defaults to ``name``.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    name: str
    direction: PortDirection
    mode_spec: ModeSpec = field(default_factory=lambda: ModeSpec(polarization="te"))
    monitor_name: str | None = None

    def __post_init__(self) -> None:
        center = finite_tuple(self.center, name="Port center")
        size = nonnegative_finite_extents(self.size, name="Port size")
        if len(center) != 3 or len(size) != 3:
            raise ValueError("Port center and size must contain three values.")
        zero_axes = [index for index, value in enumerate(size) if abs(value) <= 1e-15]
        if len(zero_axes) != 1:
            raise ValueError("Port size must contain exactly one zero extent.")
        name = str(self.name)
        if not name:
            raise ValueError("Port name cannot be empty.")
        direction = str(self.direction)
        if direction not in {"+", "-"}:
            raise ValueError("Port direction must be '+' or '-'.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        monitor_name = name if self.monitor_name is None else str(self.monitor_name)
        if not monitor_name:
            raise ValueError("Port monitor_name cannot be empty.")
        object.__setattr__(self, "monitor_name", monitor_name)
        mode_spec = immutable_snapshot(self.mode_spec)
        if not isinstance(mode_spec, ModeSpec):
            raise TypeError("Port mode_spec must be a ModeSpec.")
        object.__setattr__(self, "mode_spec", mode_spec)

    @property
    def axis(self) -> str:
        """Return the plane-normal axis.

        Returns
        -------
        {"x", "y", "z"}
            Axis selected by the zero extent in :attr:`size`.
        """
        index = min(range(3), key=lambda index: abs(self.size[index]))
        return ("x", "y", "z")[index]

    @property
    def signed_direction(self) -> str:
        """Return the signed input propagation direction.

        Returns
        -------
        str
            Direction such as ``"+x"`` or ``"-z"``.
        """
        return f"{self.direction}{self.axis}"

    @property
    def num_modes(self) -> int:
        """Return the number of candidate modes represented by this port.

        Returns
        -------
        int
            Positive mode count from :attr:`mode_spec`.
        """
        return int(self.mode_spec.num_modes)

    @property
    def mode_index(self) -> int:
        """Return the zero-based mode selected for modal projection.

        Returns
        -------
        int
            Selected index from :attr:`mode_spec`.
        """
        return int(self.mode_spec.mode_index)

    @property
    def polarization(self) -> str | None:
        """Return the configured polarization; the default port selects TE."""
        return self.mode_spec.polarization

    @property
    def projection_direction(self) -> str:
        """Return the positive-axis modal basis, such as ``"+x"``."""
        return f"+{self.axis}"

    def shifted(self, offset: Iterable[float]) -> Port:
        """Return a translated port. ``offset`` must contain three metres."""
        return replace(
            self,
            center=shifted_coordinates(
                self.center, offset, mismatch="Port offsets must contain three values."
            ),
        )

    def to_monitor(self, freqs: Sequence[float] | np.ndarray) -> ModeMonitor:
        """Create a mode monitor with this geometry, name, and mode specification."""
        from beamz.devices.monitors import ModeMonitor

        return ModeMonitor(
            center=self.center,
            size=self.size,
            freqs=np.asarray(freqs, dtype=float),
            mode_spec=self.mode_spec,
            name=self.monitor_name,
        )

    def to_source(
        self,
        freq0: float,
        fwidth: float,
        mode_index: int | None = None,
        num_freqs: int = 1,
        *,
        source_time: Any | None = None,
        power: float = 1.0,
    ) -> ModeSource:
        """Create a matching mode source.

        ``freq0`` and ``fwidth`` are hertz and build a ``GaussianPulse`` when
        ``source_time`` is omitted. ``mode_index`` defaults to :attr:`mode_spec`.
        ``num_freqs`` is the broadband reconstruction count. ``power`` is watts.
        """
        from beamz.devices.sources import GaussianPulse, ModeSource

        if source_time is None:
            source_time = GaussianPulse(freq0=float(freq0), fwidth=float(fwidth))
        mode_spec = replace(
            self.mode_spec,
            mode_index=(
                self.mode_spec.mode_index if mode_index is None else int(mode_index)
            ),
            num_freqs=max(1, int(num_freqs)),
        )
        return ModeSource(
            center=self.center,
            size=self.size,
            source_time=source_time,
            direction=self.direction,
            mode_spec=mode_spec,
            power=power,
        )


__all__ = ["Port"]
