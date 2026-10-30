"""Canonical source specifications and immutable field-profile data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np

from beamz.devices._immutable import (
    _Replaceable,
    canonical_tuple,
    finite_tuple,
    immutable_snapshot,
    nonnegative_finite_extents,
    normalize_source_signal,
    shifted_coordinates,
)
from beamz.devices.modes.specs import ModeData, ModeSpec

from .time import chebyshev_frequency_nodes

Direction3D = Literal["+x", "-x", "+y", "-y", "+z", "-z"]

FieldAxis3D = Literal["x", "y", "z"]


FieldIndex3D = tuple[int | slice, int | slice, int | slice]


@dataclass(frozen=True)
class FieldProfile3D:
    """Discrete 3D E/H profile on BeamZ's Yee component grids."""

    components: Mapping[str, np.ndarray]
    indices: Mapping[str, FieldIndex3D]
    axis: FieldAxis3D
    direction_sign: float
    omega: float
    k_axis: float | None
    phase_ref_coord: float
    phase_plane_coord: float
    grid: Any | None = None
    power_weights: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "components", immutable_snapshot(self.components))
        object.__setattr__(self, "indices", immutable_snapshot(self.indices))
        object.__setattr__(
            self, "power_weights", immutable_snapshot(self.power_weights)
        )


SourceDirection = Literal["+", "-"]


def _center_size(
    value: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = tuple(float(item) for item in value.center)
    size = tuple(float(item) for item in value.size)
    if len(center) != 3 or len(size) != 3:
        raise ValueError("Planar objects require three-dimensional center and size.")
    return center, size


def _temporal_spectrum(
    source_time, freqs, *, normalize: bool, retry_without_normalize=False
):
    """Return a source-time spectrum, or None when the waveform has none."""
    if source_time is None:
        return None
    frequency = np.asarray(freqs, dtype=float)
    if normalize and hasattr(source_time, "dft_normalization_spectrum"):
        return np.asarray(
            source_time.dft_normalization_spectrum(frequency), dtype=np.complex128
        )
    if not hasattr(source_time, "spectrum"):
        return None
    try:
        spectrum = source_time.spectrum(frequency, normalize=normalize)
    except TypeError:
        if not retry_without_normalize:
            raise
        spectrum = source_time.spectrum(frequency)
    return np.asarray(spectrum, dtype=np.complex128)


def plane_axis_and_spans(plane):
    """Return the normal axis, center, and tangential extents of a plane."""
    center, size = _center_size(plane)
    zero_axes = np.flatnonzero(np.isclose(size, 0.0, rtol=0.0, atol=1e-15))
    if zero_axes.size != 1:
        raise ValueError("A modal plane size must contain exactly one zero extent.")
    normal = int(zero_axes[0])
    axis = ("x", "y", "z")[normal]
    spans = tuple(size[index] for index in range(3) if index != normal)
    return axis, center, spans


@dataclass(frozen=True)
class ModeSource(_Replaceable):
    """Inject a solved waveguide mode through a finite plane.

    The zero extent in ``size`` selects the axis. ``direction`` is ``"+"`` or
    ``"-"``. ``power`` is watts in the source plane's discrete Poynting
    quadrature. In 3D that quadrature matches ``ModeMonitorData.power``, including
    normal Yee interpolation; aperture and discretization error can remain.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    source_time: Any
    direction: SourceDirection
    mode_spec: ModeSpec = field(default_factory=ModeSpec)
    power: float = 1.0

    def __post_init__(self) -> None:
        center = finite_tuple(self.center, name="ModeSource center")
        size = nonnegative_finite_extents(self.size, name="ModeSource size")
        if len(center) != 3 or len(size) != 3:
            raise ValueError("ModeSource center and size must contain three values.")
        direction = str(self.direction)
        if direction not in {"+", "-"}:
            raise ValueError("ModeSource direction must be '+' or '-'.")
        power = float(self.power)
        if not np.isfinite(power) or power < 0.0:
            raise ValueError("ModeSource power must be non-negative and finite.")
        source_time = immutable_snapshot(self.source_time)
        if source_time is None or not hasattr(source_time, "sample"):
            raise TypeError("ModeSource source_time must provide sample(time).")
        freq0 = getattr(source_time, "freq0", None)
        if freq0 is None or not np.isfinite(float(freq0)) or float(freq0) <= 0.0:
            raise ValueError("ModeSource source_time must define a positive freq0.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "direction", direction)
        mode_spec = immutable_snapshot(self.mode_spec)
        if not isinstance(mode_spec, ModeSpec):
            raise TypeError("ModeSource mode_spec must be a ModeSpec.")
        object.__setattr__(self, "mode_spec", mode_spec)
        object.__setattr__(self, "power", power)
        plane_axis_and_spans(self)

    @property
    def axis(self) -> str:
        """Return the source-plane normal axis."""
        return plane_axis_and_spans(self)[0]

    @property
    def signed_direction(self) -> str:
        """Return the internal signed-axis propagation direction."""
        return f"{self.direction}{self.axis}"

    @property
    def frequency(self) -> float:
        """Return the source-time carrier frequency in hertz."""
        return float(self.source_time.freq0)

    @property
    def transverse_size(self) -> tuple[float, float]:
        """Return the two tangential source-plane extents."""
        spans = plane_axis_and_spans(self)[2]
        return float(spans[0]), float(spans[1])

    def profile_frequencies(self) -> np.ndarray:
        """Return mode-profile frequencies requested by ``mode_spec``."""
        count = int(self.mode_spec.num_freqs or 1)
        fwidth = float(getattr(self.source_time, "fwidth", self.frequency / 10.0))
        return chebyshev_frequency_nodes(self.frequency, fwidth, count)

    def shifted(self, offset):
        """Return a translated source."""
        return replace(
            self,
            center=shifted_coordinates(
                self.center,
                offset,
                mismatch="ModeSource offsets must contain three values.",
            ),
        )

    def solve_modes(self, simulation, freqs=None, mode_spec=None) -> ModeData:
        """Solve modal fields on this source plane."""
        from .solve import solve_mode_plane

        frequencies = (
            np.asarray([self.frequency], dtype=float)
            if freqs is None
            else np.asarray(freqs, dtype=float).reshape(-1)
        )
        if frequencies.size == 0:
            raise ValueError("ModeSource mode solving requires a frequency.")
        return solve_mode_plane(
            simulation=simulation,
            plane=self,
            mode_spec=self.mode_spec if mode_spec is None else mode_spec,
            freqs=frequencies,
            direction=self.signed_direction,
        )

    def source_spectrum(self, freqs, *, normalize: bool = True) -> np.ndarray | None:
        """Return the complex temporal spectrum at frequencies in hertz."""
        spectrum = _temporal_spectrum(self.source_time, freqs, normalize=normalize)
        if (
            spectrum is not None
            and normalize
            and hasattr(self.source_time, "dft_normalization_spectrum")
        ):
            frequency = np.asarray(freqs, dtype=float)
            spectrum = spectrum * np.sqrt(np.maximum(frequency, 0.0) / self.frequency)
        return spectrum

    def launch_power_normalization_spectrum(self, freqs, *, fields=None, dt=None):
        """Return frequency-dependent launched-power normalization."""
        del freqs, fields, dt
        return None


@dataclass(frozen=True, slots=True)
class GaussianSource(_Replaceable):
    """Inject a Gaussian spatial profile into an FDTD simulation.

    In 2D it excites the active out-of-plane component: ``Ez`` for TMz and
    ``Hz`` for TEz. In 3D it retains the existing compatible electric-component
    behavior. Position and width remain differentiable through JAX.

    Parameters
    ----------
    position : tuple of float
        Source center in ``(x, y)`` or ``(x, y, z)`` order, in metres.
    width : float
        Positive Gaussian standard-width scale in metres.
    signal : array-like
        Immutable temporal samples evaluated at the simulation time step.

    Examples
    --------
    >>> import numpy as np
    >>> import beamz as bz
    >>> time = np.linspace(0.0, 1e-13, 100)
    >>> source = bz.GaussianSource(
    ...     position=(1 * bz.um, 1.5 * bz.um),
    ...     width=0.2 * bz.um,
    ...     signal=np.sin(2 * np.pi * 2e14 * time),
    ... )
    """

    position: tuple[float, ...]
    width: float
    signal: Any = field(compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        position = canonical_tuple(self.position, dtype=float)
        if len(position) not in {2, 3}:
            raise ValueError("GaussianSource position must be a 2D or 3D coordinate.")
        width = float(self.width)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("GaussianSource width must be a positive finite value.")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "width", width)
        object.__setattr__(
            self,
            "signal",
            normalize_source_signal(self.signal, name="GaussianSource.signal"),
        )

    def shifted(self, offset):
        """Return the source translated by a physical coordinate offset."""
        return replace(
            self,
            position=shifted_coordinates(
                self.position,
                offset,
                mismatch="GaussianSource offset dimensionality must match its position.",
            ),
        )


_VALID_DIRECTIONS_3D = {"+x", "-x", "+y", "-y", "+z", "-z"}


def _as_xyz(value, *, name: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise ValueError(f"GaussianBeamSource {name} must be a 3D coordinate.")
    return (values[0], values[1], values[2])


def _as_direction(value) -> Direction3D:
    direction = str(value)
    if direction not in _VALID_DIRECTIONS_3D:
        raise ValueError(f"Unsupported GaussianBeamSource direction {direction!r}.")
    return cast(Direction3D, direction)


@dataclass(frozen=True, slots=True)
class GaussianBeamSource(_Replaceable):
    """Angled 3D Gaussian beam launched through a planar TF/SF boundary.

    ``size`` has one zero extent, the plane normal. Angles are radians.
    ``waist_radius`` is derived from the aperture when omitted. ``power`` is
    watts. ``wavelength`` is the vacuum wavelength, otherwise taken from
    ``source_time``. ``max_shift`` is the largest Yee-plane residual shift.
    """

    center: tuple[float, float, float]
    size: Any
    source_time: Any
    direction: Direction3D = "-z"
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    waist_radius: float | None = None
    waist_distance: float = 0.0
    power: float = 1.0
    background_index: float = 1.0
    wavelength: float | None = None
    max_shift: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _as_xyz(self.center, name="center"))
        object.__setattr__(self, "size", canonical_tuple(self.size, dtype=float))
        object.__setattr__(self, "source_time", immutable_snapshot(self.source_time))
        object.__setattr__(self, "direction", _as_direction(self.direction))
        for name in ("angle_theta", "angle_phi", "pol_angle", "waist_distance"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(
            self,
            "waist_radius",
            None if self.waist_radius is None else float(self.waist_radius),
        )
        power_value = float(self.power)
        if not np.isfinite(power_value) or power_value < 0.0:
            raise ValueError(
                "GaussianBeamSource power must be a non-negative finite value, "
                f"got {self.power!r}."
            )
        object.__setattr__(self, "power", power_value)
        object.__setattr__(self, "background_index", float(self.background_index))
        object.__setattr__(
            self,
            "wavelength",
            None if self.wavelength is None else float(self.wavelength),
        )
        object.__setattr__(self, "max_shift", int(max(1, self.max_shift)))

    def shifted(self, offset):
        """Return the beam plane translated by an ``(x, y, z)`` offset."""
        return replace(
            self,
            center=shifted_coordinates(
                self.center,
                offset,
                mismatch="GaussianBeamSource offsets must contain three values.",
            ),
        )

    def source_spectrum(self, freqs, *, normalize: bool = True) -> np.ndarray | None:
        """Return the complex temporal spectrum at frequencies in hertz."""
        return _temporal_spectrum(
            self.source_time, freqs, normalize=normalize, retry_without_normalize=True
        )


@dataclass(frozen=True, slots=True)
class CustomSource(_Replaceable):
    """Immutable precomputed field injection.

    ``timing`` is ``"e"`` or ``"h"``. ``index`` selects cells, ``coeff`` is the
    spatial weight, and ``waveform`` holds per-step samples. Inputs are
    snapshotted so later mutation cannot change a compiled source.
    """

    component: str
    timing: str
    index: tuple[Any, ...]
    coeff: Any
    waveform: Any
    target_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", str(self.component))
        object.__setattr__(self, "timing", str(self.timing))
        object.__setattr__(self, "index", immutable_snapshot(self.index))
        object.__setattr__(self, "coeff", immutable_snapshot(self.coeff))
        object.__setattr__(self, "waveform", immutable_snapshot(self.waveform))
        object.__setattr__(
            self, "target_shape", tuple(int(v) for v in self.target_shape)
        )

    def shifted(self, offset):
        """Keep grid-indexed injections unchanged by coordinate normalization."""
        del offset
        return self


@dataclass(frozen=True, slots=True)
class PlaneWaveSource(GaussianBeamSource):
    """Uniform plane wave with an optional transverse wavevector in rad/m.

    Injection and power match GaussianBeamSource, with a uniform envelope.
    Oblique incidence requires a Bloch-matched ``transverse_wavevector`` and a
    narrowband quadrature ``source_time`` (``fwidth <= 0.1 * freq0``). The
    wavevector is fixed across the pulse; fixed-angle spectra need separate
    runs. ``power`` is the total power through the aperture. ``angle_theta`` and
    ``waist_distance`` must stay zero.
    """

    transverse_wavevector: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        GaussianBeamSource.__post_init__(self)
        if not np.isfinite(self.power) or self.power < 0:
            raise ValueError("PlaneWaveSource power must be finite and nonnegative.")
        if not np.isfinite(self.background_index) or self.background_index <= 0:
            raise ValueError(
                "PlaneWaveSource background_index must be positive and finite."
            )
        vector = tuple(float(k) for k in self.transverse_wavevector)
        if len(vector) != 3 or not all(np.isfinite(k) for k in vector):
            raise ValueError(
                "transverse_wavevector must contain three finite values in rad/m."
            )
        if vector["xyz".index(self.direction[-1])] != 0:
            raise ValueError(
                "transverse_wavevector must be zero along the source normal."
            )
        object.__setattr__(self, "transverse_wavevector", vector)
        if self.angle_theta != 0 or self.waist_distance != 0:
            raise ValueError(
                "Use transverse_wavevector for PlaneWaveSource incidence; angle_theta and waist_distance must be zero."
            )
        object.__setattr__(self, "waist_radius", float("inf"))
