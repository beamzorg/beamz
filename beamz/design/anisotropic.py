"""Anisotropic, optionally dispersive, electric and magnetic tensor media.

A medium is described by symmetric high-frequency tensors ``eps_inf`` and
``mu_inf`` plus any number of second-order resonances whose strengths are
symmetric positive-semidefinite tensors. With the exp(-i omega t) convention,

    eps(omega) = eps_inf + sum_p W_p / (omega_p**2 - omega**2 - i gamma_p omega)

and likewise for the permeability. ``W_p`` carries units of rad^2/s^2: a
Lorentz oscillator has ``W = delta_eps * omega_p**2`` and a Drude term has
``omega_p = 0`` and ``W = omega_plasma**2``. Because every ``W_p`` is a
tensor, rotating a medium produces frequency-dependent off-diagonal terms
(oriented poles), which grid-aligned per-axis poles cannot describe.

Hyperbolic media — tensors with principal values of opposite sign — need
these poles: a static negative permittivity is unconditionally unstable in
FDTD, so the negative branch must come from a resonance or a Drude term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.design.materials import Material

_PACKED_INDEX = ((0, 3, 4), (3, 1, 5), (4, 5, 2))


def _packed_from_any(value: Any, name: str) -> np.ndarray:
    """Return a symmetric tensor as a packed ``(xx, yy, zz, xy, xz, yz)`` array."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return np.array([array, array, array, 0.0, 0.0, 0.0], dtype=np.float64)
    if array.shape == (3,):
        return np.array([*array, 0.0, 0.0, 0.0], dtype=np.float64)
    if array.shape == (6,):
        return array.copy()
    if array.shape == (3, 3):
        if not np.allclose(
            array, array.T, rtol=1e-12, atol=1e-14 * np.abs(array).max(initial=1.0)
        ):
            raise ValueError(f"{name} must be symmetric.")
        return np.array(
            [
                array[0, 0],
                array[1, 1],
                array[2, 2],
                array[0, 1],
                array[0, 2],
                array[1, 2],
            ]
        )
    raise ValueError(
        f"{name} must be a scalar, three diagonal values, six packed symmetric "
        "values, or a symmetric 3x3 matrix."
    )


def packed_to_matrix(packed) -> np.ndarray:
    """Expand packed ``(xx, yy, zz, xy, xz, yz)`` values into a 3x3 matrix."""

    packed = np.asarray(packed)
    return packed[..., np.asarray(_PACKED_INDEX)]


def rotation_matrix(rotation) -> np.ndarray:
    """Return a proper rotation from a 3x3 matrix or ``(alpha, beta, gamma)``.

    Euler angles are in radians and applied extrinsically as
    ``R = Rz(gamma) @ Ry(beta) @ Rx(alpha)``.
    """

    array = np.asarray(rotation, dtype=np.float64)
    if array.shape == (3,):
        alpha, beta, gamma = array
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        rx = np.array(((1.0, 0.0, 0.0), (0.0, ca, -sa), (0.0, sa, ca)))
        ry = np.array(((cb, 0.0, sb), (0.0, 1.0, 0.0), (-sb, 0.0, cb)))
        rz = np.array(((cg, -sg, 0.0), (sg, cg, 0.0), (0.0, 0.0, 1.0)))
        return rz @ ry @ rx
    if array.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix or three Euler angles.")
    if (
        not np.allclose(array @ array.T, np.eye(3), atol=1e-10)
        or np.linalg.det(array) < 0
    ):
        raise ValueError("rotation must be a proper orthogonal matrix.")
    return array


def _rotate_packed(packed: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float64)
    if np.all(packed[3:] == 0.0) and packed[0] == packed[1] == packed[2]:
        return packed.copy()  # isotropic tensors are rotation invariant
    rotated = _packed_from_any(
        rotation @ packed_to_matrix(packed) @ rotation.T, "tensor"
    )
    # Drop roundoff so grid-aligned rotations keep a compact diagonal tensor.
    rotated[np.abs(rotated) < 1e-13 * np.abs(packed).max(initial=0.0)] = 0.0
    return rotated


def _require_semidefinite(packed: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(packed_to_matrix(packed))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, float(np.abs(eigenvalues).max()))
    if np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite for a passive medium.")


@dataclass(frozen=True, init=False)
class TensorPole:
    """One passive second-order resonance with a symmetric tensor strength.

    ``chi(omega) = weight / (resonance**2 - omega**2 - i * damping * omega)``,
    with ``resonance`` and ``damping`` in rad/s and ``weight`` in rad^2/s^2.
    Use :meth:`lorentz` or :meth:`drude` rather than the raw constructor.
    """

    resonance: float
    damping: float
    weight: tuple[float, float, float, float, float, float]

    def __init__(self, resonance: float, damping: float, weight: Any) -> None:
        resonance, damping = float(resonance), float(damping)
        if not np.isfinite(resonance) or resonance < 0.0:
            raise ValueError("Pole resonance must be finite and nonnegative.")
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("Pole damping must be finite and nonnegative.")
        packed = _packed_from_any(weight, "Pole weight")
        if not np.isfinite(packed).all():
            raise ValueError("Pole weight must be finite.")
        _require_semidefinite(packed, "Pole weight")
        object.__setattr__(self, "resonance", resonance)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "weight", tuple(float(v) for v in packed))

    @classmethod
    def lorentz(
        cls, delta_epsilon: Any, *, resonance: float, damping: float
    ) -> TensorPole:
        """Lorentz oscillator with tensor strength ``delta_epsilon`` (dimensionless)."""

        if float(resonance) <= 0.0:
            raise ValueError("A Lorentz pole needs a positive resonance; use drude().")
        strength = _packed_from_any(delta_epsilon, "delta_epsilon")
        return cls(resonance, damping, strength * float(resonance) ** 2)

    @classmethod
    def drude(cls, plasma_frequency: Any, *, damping: float) -> TensorPole:
        """Drude term; ``plasma_frequency`` (rad/s) may be per-axis.

        A tensor argument is interpreted as the symmetric tensor of squared
        plasma frequencies, so pass ``(wp_x**2, wp_y**2, ...)`` explicitly
        through the raw constructor if off-diagonal terms are needed.
        """

        frequencies = np.asarray(plasma_frequency, dtype=np.float64)
        if frequencies.shape not in ((), (3,)):
            raise ValueError(
                "plasma_frequency must be a scalar or three per-axis values; "
                "rotate the medium for off-diagonal Drude response."
            )
        if np.any(frequencies < 0):
            raise ValueError("Plasma frequencies must be nonnegative.")
        return cls(0.0, damping, frequencies**2)

    @property
    def weight_matrix(self) -> np.ndarray:
        return packed_to_matrix(np.asarray(self.weight))

    def susceptibility(self, frequencies) -> np.ndarray:
        """Complex 3x3 susceptibility at positive frequencies in Hz."""

        omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64)
        response = 1.0 / (self.resonance**2 - omega**2 - 1j * self.damping * omega)
        return response[..., None, None] * self.weight_matrix

    def rotated(self, rotation) -> TensorPole:
        return TensorPole(
            self.resonance,
            self.damping,
            _rotate_packed(np.asarray(self.weight), rotation_matrix(rotation)),
        )


def _cross_matrix(vector) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


@dataclass(frozen=True, init=False)
class GyrotropicPole:
    """Isotropic resonance whose current precesses about a gyration vector.

    ``p'' + damping p' + resonance^2 p = strength F + gyration x p'``, with all
    rates in rad/s and ``strength`` in rad^2/s^2. A magnetized electron plasma
    (``resonance = 0``, ``gyration`` = cyclotron vector e B / m) and
    magneto-optic Lorentz media are special cases. The susceptibility is
    ``strength (A I + i omega [gyration]x)^-1`` with
    ``A = resonance^2 - omega^2 - i damping omega``: Hermitian when lossless
    and non-symmetric, so the medium is non-reciprocal.
    """

    strength: float
    resonance: float
    damping: float
    gyration: tuple[float, float, float]

    def __init__(self, strength, resonance, damping, gyration) -> None:
        values = [float(strength), float(resonance), float(damping)]
        if not np.isfinite(values).all() or min(values) < 0.0:
            raise ValueError("Gyrotropic strength, resonance and damping must be >= 0.")
        vector = np.asarray(gyration, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("gyration must be a finite 3-vector in rad/s.")
        object.__setattr__(self, "strength", values[0])
        object.__setattr__(self, "resonance", values[1])
        object.__setattr__(self, "damping", values[2])
        object.__setattr__(self, "gyration", tuple(float(v) for v in vector))

    @classmethod
    def magnetized_plasma(
        cls, plasma_frequency: float, cyclotron, *, damping: float
    ) -> GyrotropicPole:
        """Drude plasma in a static field; ``cyclotron`` is e B / m as a vector."""

        return cls(float(plasma_frequency) ** 2, 0.0, damping, cyclotron)

    @classmethod
    def lorentz(
        cls, delta_epsilon: float, *, resonance: float, damping: float, gyration
    ) -> GyrotropicPole:
        """Magneto-optic Lorentz oscillator with scalar strength ``delta_epsilon``."""

        if float(resonance) <= 0.0:
            raise ValueError("A Lorentz pole needs a positive resonance.")
        return cls(
            float(delta_epsilon) * float(resonance) ** 2, resonance, damping, gyration
        )

    @property
    def weight(self) -> tuple[float, ...]:
        return (self.strength, self.strength, self.strength, 0.0, 0.0, 0.0)

    @property
    def weight_matrix(self) -> np.ndarray:
        return self.strength * np.eye(3)

    def susceptibility(self, frequencies) -> np.ndarray:
        omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64)
        diagonal = self.resonance**2 - omega**2 - 1j * self.damping * omega
        matrix = diagonal[..., None, None] * np.eye(3) + 1j * omega[
            ..., None, None
        ] * _cross_matrix(self.gyration)
        return self.strength * np.linalg.inv(matrix)

    def rotated(self, rotation) -> GyrotropicPole:
        # A proper rotation maps the axial gyration vector like any vector.
        vector = rotation_matrix(rotation) @ np.asarray(self.gyration)
        return GyrotropicPole(self.strength, self.resonance, self.damping, vector)


def _poles(values, name: str) -> tuple:
    poles = tuple(values)
    if any(not isinstance(pole, TensorPole | GyrotropicPole) for pole in poles):
        raise TypeError(f"{name} must contain TensorPole or GyrotropicPole values.")
    return poles


def _pole_spec(pole):
    if isinstance(pole, GyrotropicPole):
        return {
            "gyrotropic": [
                pole.strength,
                pole.resonance,
                pole.damping,
                list(pole.gyration),
            ]
        }
    return [pole.resonance, pole.damping, list(pole.weight)]


def _pole_from_spec(spec):
    if isinstance(spec, dict):
        return GyrotropicPole(*spec["gyrotropic"])
    return TensorPole(*spec)


@dataclass(frozen=True, init=False)
class AnisotropicMedium(Material):
    """Electric and magnetic tensor medium with optional tensor-valued poles.

    Parameters
    ----------
    permittivity : scalar, 3, 6, or symmetric 3x3
        High-frequency relative permittivity ``eps_inf``; positive definite.
    permeability : scalar, 3, 6, or symmetric 3x3
        High-frequency relative permeability ``mu_inf``; positive definite.
    electric_poles, magnetic_poles : sequence of TensorPole or GyrotropicPole
        Resonances added to the permittivity and permeability respectively.
    conductivity : scalar or 3 values, default=0
        Diagonal electrical conductivity in S/m.
    frequency_range : (float, float), optional
        Band in Hz used to report ``max_permittivity`` for meshing.

    Notes
    -----
    Off-diagonal tensors require ``volume`` or ``farjadpour_full`` raster
    smoothing. Dispersive tensor poles and non-unit permeability currently run
    on the single-device JAX backend only.

    Examples
    --------
    A hyperbolic Drude slab whose optic axis is tilted by 30 degrees in the
    xy plane:

    >>> import numpy as np
    >>> import beamz as bz
    >>> crystal = bz.AnisotropicMedium(
    ...     permittivity=2.0,
    ...     electric_poles=[bz.TensorPole.drude((2e15, 0.0, 0.0), damping=1e13)],
    ... )
    >>> tilted = crystal.rotated((0.0, 0.0, np.radians(30)))
    """

    electric_poles: tuple[TensorPole, ...]
    magnetic_poles: tuple[TensorPole, ...]
    frequency_range: tuple[float, float] | None

    def __init__(
        self,
        permittivity: Any = 1.0,
        permeability: Any = 1.0,
        *,
        electric_poles=(),
        magnetic_poles=(),
        conductivity: Any = 0.0,
        frequency_range=None,
    ) -> None:
        super().__init__(
            permittivity=permittivity,
            permeability=permeability,
            conductivity=conductivity,
        )
        if any(abs(v) > 0 for v in self._conductivity[3:]):
            raise ValueError("AnisotropicMedium supports diagonal conductivity only.")
        object.__setattr__(
            self, "electric_poles", _poles(electric_poles, "electric_poles")
        )
        object.__setattr__(
            self, "magnetic_poles", _poles(magnetic_poles, "magnetic_poles")
        )
        if frequency_range is not None:
            band = tuple(float(f) for f in frequency_range)
            if (
                len(band) != 2
                or not 0 < band[0] < band[1]
                or not np.isfinite(band).all()
            ):
                raise ValueError("frequency_range must contain increasing positive Hz.")
            frequency_range = band
        object.__setattr__(self, "frequency_range", frequency_range)

    @property
    def is_dispersive(self) -> bool:
        return bool(self.electric_poles or self.magnetic_poles)

    def permittivity_tensor(self, frequencies) -> np.ndarray:
        """Complex relative permittivity tensor(s), shape ``(..., 3, 3)``."""

        return self._response(self._epsilon_r, self.electric_poles, frequencies)

    def permeability_tensor(self, frequencies) -> np.ndarray:
        """Complex relative permeability tensor(s), shape ``(..., 3, 3)``."""

        return self._response(self._mu_r, self.magnetic_poles, frequencies)

    @staticmethod
    def _response(static, poles, frequencies) -> np.ndarray:
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("Frequencies must be finite and positive.")
        value = np.broadcast_to(
            packed_to_matrix(np.asarray(static)), (*frequencies.shape, 3, 3)
        ).astype(complex)
        for pole in poles:
            value = value + pole.susceptibility(frequencies)
        return value

    def rotated(self, rotation) -> AnisotropicMedium:
        """Return this medium with its crystal axes rotated into the lab frame.

        ``rotation`` is a proper 3x3 matrix or Euler angles ``(alpha, beta,
        gamma)`` in radians applied as ``Rz(gamma) @ Ry(beta) @ Rx(alpha)``.
        Every tensor transforms as ``R T R^T``.
        """

        matrix = rotation_matrix(rotation)
        conductivity = _rotate_packed(np.asarray(self._conductivity), matrix)
        if np.any(
            np.abs(conductivity[3:]) > 1e-12 * max(1.0, np.abs(conductivity).max())
        ):
            raise ValueError("Rotating anisotropic conductivity is not supported.")
        return AnisotropicMedium(
            _rotate_packed(np.asarray(self._epsilon_r), matrix),
            _rotate_packed(np.asarray(self._mu_r), matrix),
            electric_poles=tuple(p.rotated(matrix) for p in self.electric_poles),
            magnetic_poles=tuple(p.rotated(matrix) for p in self.magnetic_poles),
            conductivity=conductivity[:3],
            frequency_range=self.frequency_range,
        )

    def is_indefinite(self) -> bool:
        """Whether the Hermitian part of eps or mu is ever indefinite.

        Such (hyperbolic) responses carry backward waves. The search covers
        ``frequency_range`` when declared, otherwise a wide band around every
        pole.
        """

        if not self.is_dispersive:
            return False
        if self.frequency_range is not None:
            frequencies = np.linspace(*self.frequency_range, 512)
        else:
            poles = (*self.electric_poles, *self.magnetic_poles)
            scales = [
                max(
                    p.resonance,
                    p.damping,
                    np.sqrt(np.abs(p.weight_matrix).max()),
                    float(np.linalg.norm(getattr(p, "gyration", (0.0, 0.0, 0.0)))),
                )
                for p in poles
            ]
            frequencies = np.geomspace(min(scales) / 1e3, max(scales) * 1e3, 4096) / (
                2 * np.pi
            )
        for tensor in (
            self.permittivity_tensor(frequencies),
            self.permeability_tensor(frequencies),
        ):
            hermitian = 0.5 * (tensor + np.conj(np.swapaxes(tensor, -1, -2)))
            eigenvalues = np.linalg.eigvalsh(hermitian)
            if np.any((eigenvalues.min(axis=-1) < 0) & (eigenvalues.max(axis=-1) > 0)):
                return True
        return False

    @property
    def max_permittivity(self) -> float:
        static = super().max_permittivity
        if self.frequency_range is None or not self.is_dispersive:
            return static
        frequencies = np.linspace(*self.frequency_range, 256)
        eps = np.abs(np.linalg.eigvals(self.permittivity_tensor(frequencies))).max()
        mu = np.abs(np.linalg.eigvals(self.permeability_tensor(frequencies))).max()
        return max(static, float(eps * mu))

    def cache_spec(self):
        return (
            super().cache_spec(),
            tuple(str(_pole_spec(p)) for p in self.electric_poles),
            tuple(str(_pole_spec(p)) for p in self.magnetic_poles),
            self.frequency_range,
        )

    def to_spec(self) -> dict:
        """JSON-compatible complete specification (SI units, packed tensors)."""

        def poles(values):
            return [_pole_spec(p) for p in values]

        return {
            "permittivity": list(self._epsilon_r),
            "permeability": list(self._mu_r),
            "conductivity": list(self._conductivity[:3]),
            "electric_poles": poles(self.electric_poles),
            "magnetic_poles": poles(self.magnetic_poles),
            "frequency_range": (
                None if self.frequency_range is None else list(self.frequency_range)
            ),
        }

    @classmethod
    def from_spec(cls, spec) -> AnisotropicMedium:
        return cls(
            spec["permittivity"],
            spec["permeability"],
            electric_poles=[_pole_from_spec(p) for p in spec.get("electric_poles", ())],
            magnetic_poles=[_pole_from_spec(p) for p in spec.get("magnetic_poles", ())],
            conductivity=spec.get("conductivity", 0.0),
            frequency_range=spec.get("frequency_range"),
        )

    def __repr__(self) -> str:
        return (
            f"AnisotropicMedium(permittivity={self.permittivity!r}, "
            f"permeability={self.permeability!r}, "
            f"electric_poles={len(self.electric_poles)}, "
            f"magnetic_poles={len(self.magnetic_poles)})"
        )

    def updated_copy(self, **changes):
        values = dict(
            permittivity=self.permittivity,
            permeability=self.permeability,
            electric_poles=self.electric_poles,
            magnetic_poles=self.magnetic_poles,
            conductivity=self.conductivity,
            frequency_range=self.frequency_range,
        )
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(
                f"Unknown AnisotropicMedium field(s): {', '.join(sorted(unknown))}."
            )
        return type(self)(**(values | changes))
