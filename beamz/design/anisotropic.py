"""Anisotropic, optionally dispersive, electric and magnetic tensor media.

A medium is described by symmetric high-frequency tensors ``eps_inf`` and
``mu_inf`` plus any number of second-order resonances whose strengths are
symmetric positive-semidefinite tensors. With the exp(-i omega t) convention,

    eps(omega) = eps_inf + sum_p W_p / (omega_p**2 - omega**2 - i gamma_p omega)

and likewise for the permeability. ``W_p`` carries units of rad^2/s^2: a
Lorentz oscillator has ``W = delta_eps * omega_p**2`` and a Drude term has
``omega_p = 0`` and ``W = omega_plasma**2``. General complex-conjugate
pole-residue (CCPR) poles add a second symmetric tensor ``B_p`` (rad/s) that
couples to the field's time derivative,
``chi_p = (W_p - i omega B_p) / (omega_p**2 - omega**2 - i gamma_p omega)``,
which represents any fitted pole-residue pair (see :meth:`TensorPole.ccpr`). Because every ``W_p`` is a
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
    """One second-order resonance with symmetric tensor strengths.

    ``chi(omega) = (weight - i omega edot) / (resonance**2 - omega**2 - i
    damping omega)``, with ``resonance`` and ``damping`` in rad/s, ``weight``
    in rad^2/s^2, and ``edot`` in rad/s. In the time domain the normalized
    polarization obeys ``p'' + damping p' + resonance**2 p = weight E + edot E'``.

    Lorentz and Drude poles (:meth:`lorentz`, :meth:`drude`) have no ``edot``
    term and a positive-semidefinite ``weight``. General complex-conjugate
    pole-residue (CCPR) poles (:meth:`ccpr`, :meth:`critical_point`,
    :func:`pole_residue_tensor_poles`) carry ``edot``. One fitted CCPR pole
    need not be passive on its own, so their tensors are not required to be
    semidefinite; :class:`AnisotropicMedium` checks the passivity of the sum.
    """

    resonance: float
    damping: float
    weight: tuple[float, float, float, float, float, float]
    edot: tuple[float, float, float, float, float, float] | None = None

    def __init__(
        self, resonance: float, damping: float, weight: Any, edot: Any = None
    ) -> None:
        resonance, damping = float(resonance), float(damping)
        if not np.isfinite(resonance) or resonance < 0.0:
            raise ValueError("Pole resonance must be finite and nonnegative.")
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("Pole damping must be finite and nonnegative.")
        packed = _packed_from_any(weight, "Pole weight")
        if not np.isfinite(packed).all():
            raise ValueError("Pole weight must be finite.")
        if edot is None:
            _require_semidefinite(packed, "Pole weight")
        else:
            edot = _packed_from_any(edot, "Pole edot")
            if not np.isfinite(edot).all():
                raise ValueError("Pole edot must be finite.")
            edot = tuple(float(v) for v in edot)
        object.__setattr__(self, "resonance", resonance)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "weight", tuple(float(v) for v in packed))
        object.__setattr__(self, "edot", edot)

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
        """Drude term; ``plasma_frequency`` (rad/s) is a scalar or three axis values.

        For off-diagonal response, rotate the pole or use the raw constructor
        with a tensor of squared plasma frequencies.
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

    @classmethod
    def ccpr(
        cls, pole: complex, residue: complex, *, tensor: Any = 1.0, orientation=None
    ) -> TensorPole:
        """Complex-conjugate pole-residue pair ``r/(s-q) + r*/(s-q*)``, ``s = -i omega``.

        ``pole`` ``q`` and ``residue`` ``r`` are in rad/s, as in
        :class:`~beamz.PoleResidue` and fdtdx's ``CCPRPole``. Over a common
        denominator the pair is ``(a - i omega b) / (omega0**2 - omega**2 - i
        gamma omega)`` with

            omega0 = |q|,  gamma = -2 Re q,  a = -2 Re(r q*),  b = 2 Re r.

        The scalar response is spread over the axes by ``tensor`` (scalar,
        three diagonal weights, six packed values, or a symmetric 3x3 matrix;
        positive semidefinite) or by a unit ``orientation`` vector ``u``
        (``tensor = u u^T``): ``weight = a * tensor`` and ``edot = b * tensor``.
        Lorentz and Drude pairs have ``b = 0``.
        """

        q, r = complex(pole), complex(residue)
        if not np.isfinite([q, r]).all():
            raise ValueError("CCPR pole and residue must be finite.")
        if q.real > 0.0:
            raise ValueError("A CCPR pole needs Re(pole) <= 0 to be stable.")
        shape = _orientation_tensor(tensor, orientation)
        return cls(
            abs(q),
            -2.0 * q.real,
            -2.0 * (r * q.conjugate()).real * shape,
            2.0 * r.real * shape,
        )

    @classmethod
    def critical_point(
        cls,
        amplitude: float,
        phase: float,
        *,
        resonance: float,
        damping: float,
        tensor: Any = 1.0,
        orientation=None,
    ) -> TensorPole:
        """Critical-point term ``A W [e^{i phi}/(W - w - i G) + e^{-i phi}/(W + w + i G)]``.

        The parameterization reported for fitted metal permittivities, with
        ``resonance`` ``W`` and ``damping`` ``G`` in rad/s, a dimensionless
        ``amplitude`` ``A``, and ``phase`` in radians. It is the CCPR pair with
        ``q = -G - i W`` and ``r = i A W e^{i phi}``.
        """

        q = complex(-float(damping), -float(resonance))
        r = 1j * float(amplitude) * float(resonance) * np.exp(1j * float(phase))
        return cls.ccpr(q, r, tensor=tensor, orientation=orientation)

    @property
    def weight_matrix(self) -> np.ndarray:
        return packed_to_matrix(np.asarray(self.weight))

    @property
    def edot_packed(self) -> np.ndarray:
        """Packed ``E'`` coupling tensor ``B`` (zeros for Lorentz and Drude)."""
        if self.edot is None:
            return np.zeros(6)
        return np.asarray(self.edot, dtype=np.float64)

    @property
    def edot_matrix(self) -> np.ndarray:
        return packed_to_matrix(self.edot_packed)

    @property
    def has_edot(self) -> bool:
        """Whether the pole couples to the field's time derivative (``B != 0``)."""
        return self.edot is not None and any(v != 0.0 for v in self.edot)

    def susceptibility(self, frequencies) -> np.ndarray:
        """Complex 3x3 susceptibility at positive frequencies in Hz."""

        omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64)
        response = 1.0 / (self.resonance**2 - omega**2 - 1j * self.damping * omega)
        value = response[..., None, None] * self.weight_matrix
        if self.edot is not None:
            value = value - (1j * omega * response)[..., None, None] * self.edot_matrix
        return value

    def rotated(self, rotation) -> TensorPole:
        matrix = rotation_matrix(rotation)
        return TensorPole(
            self.resonance,
            self.damping,
            _rotate_packed(np.asarray(self.weight), matrix),
            None if self.edot is None else _rotate_packed(self.edot_packed, matrix),
        )


def _orientation_tensor(tensor, orientation) -> np.ndarray:
    """Packed axis-weighting tensor from ``tensor`` or a unit ``orientation``."""
    if orientation is not None:
        if not (np.ndim(tensor) == 0 and float(tensor) == 1.0):
            raise ValueError("Pass either tensor or orientation, not both.")
        vector = np.asarray(orientation, dtype=np.float64)
        norm = float(np.linalg.norm(vector)) if vector.shape == (3,) else 0.0
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("orientation must be a finite nonzero 3-vector.")
        vector = vector / norm
        tensor = np.outer(vector, vector)
    packed = _packed_from_any(tensor, "tensor")
    if not np.isfinite(packed).all():
        raise ValueError("tensor must be finite.")
    _require_semidefinite(packed, "tensor")
    return packed


def pole_residue_tensor_poles(
    model, *, tensor: Any = 1.0, orientation=None
) -> tuple[TensorPole, ...]:
    """Convert a scalar :class:`~beamz.PoleResidue` fit into tensor poles.

    Complex pole/residue pairs map through :meth:`TensorPole.ccpr`. A real
    pole ``a`` contributes the first-order term ``k / (s - a)`` with
    ``k = 2 Re(c)``. Real poles are combined in the order given, two at a
    time, into one second-order pole

        omega0**2 = a1 a2,  gamma = -(a1 + a2),
        weight = -(k1 a2 + k2 a1),  edot = k1 + k2,

    which turns BeamZ's two-real-pole Drude term back into an exact Drude pole
    (``edot = 0``). A leftover real pole becomes the Debye relaxation
    ``omega0 = 0``, ``gamma = -a``, ``weight = 0``, ``edot = k``. The poles
    carry ``model``'s susceptibility (not ``epsilon_inf``) spread over the axes
    by ``tensor`` or ``orientation`` as in :meth:`TensorPole.ccpr`.
    """

    from beamz.design.dispersion import PoleResidue

    if not isinstance(model, PoleResidue):
        raise TypeError("model must be a PoleResidue.")
    shape = _orientation_tensor(tensor, orientation)
    poles, real = [], []
    for pole, residue in model.poles:
        if pole.imag != 0.0:
            poles.append(TensorPole.ccpr(pole, residue, tensor=shape))
        else:
            real.append((pole.real, 2.0 * residue.real))
    for index in range(0, len(real), 2):
        a1, k1 = real[index]
        a2, k2 = real[index + 1] if index + 1 < len(real) else (0.0, 0.0)
        poles.append(
            TensorPole(
                np.sqrt(abs(a1 * a2)),
                -(a1 + a2),
                -(k1 * a2 + k2 * a1) * shape,
                (k1 + k2) * shape,
            )
        )
    return tuple(poles)


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
    if pole.edot is not None:
        return {
            "ccpr": [pole.resonance, pole.damping, list(pole.weight), list(pole.edot)]
        }
    return [pole.resonance, pole.damping, list(pole.weight)]


def _pole_from_spec(spec):
    if isinstance(spec, dict):
        if "ccpr" in spec:
            return TensorPole(*spec["ccpr"])
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
        if any(
            getattr(p, "edot", None) is not None
            for p in (*self.electric_poles, *self.magnetic_poles)
        ):
            self._require_passive()

    def _require_passive(self) -> None:
        """Sampled passivity of media with CCPR poles, which are not passive per pole.

        The anti-Hermitian part of eps and mu must stay positive semidefinite,
        checked like :class:`~beamz.PoleResidue` on a dense grid spanning every
        pole's scale (and the declared band); a numerical check, not a proof.
        """

        poles = (*self.electric_poles, *self.magnetic_poles)
        scales = [
            s
            for p in poles
            for s in (
                p.resonance,
                p.damping,
                np.sqrt(np.abs(p.weight_matrix).max()),
                float(np.abs(getattr(p, "edot_packed", 0.0)).max()),
            )
            if s > 0.0
        ]
        if not scales:
            return
        frequencies = np.geomspace(min(scales) * 1e-4, max(scales) * 1e4, 8192) / (
            2.0 * np.pi
        )
        if self.frequency_range is not None:
            frequencies = np.r_[frequencies, np.linspace(*self.frequency_range, 512)]
        for name, tensor in (
            ("permittivity", self.permittivity_tensor(frequencies)),
            ("permeability", self.permeability_tensor(frequencies)),
        ):
            loss = (tensor - np.conj(np.swapaxes(tensor, -1, -2))) / 2j
            floor = np.linalg.eigvalsh(loss)[..., 0]
            scale = np.maximum(1.0, np.abs(tensor).max(axis=(-2, -1)))
            if not np.isfinite(tensor).all() or np.any(floor < -1e-7 * scale):
                raise ValueError(
                    f"The {name} poles fail the sampled passivity check (the "
                    "medium would amplify); check the CCPR pole/residue signs."
                )

    @classmethod
    def from_pole_residue(
        cls, x, y=None, z=None, *, permeability: Any = 1.0
    ) -> AnisotropicMedium:
        """Tensor medium from scalar :class:`~beamz.PoleResidue` fits.

        With only ``x`` the medium is isotropic. With ``x``, ``y`` and ``z``
        each model is the response along that principal axis; ``eps_inf``
        becomes ``diag(x.epsilon_inf, y.epsilon_inf, z.epsilon_inf)`` and axes
        sharing one model share its poles. Use :meth:`rotated` to orient the
        principal axes. ``frequency_range`` spans the models' fit bands.
        """

        models = (x, x, x) if y is None and z is None else (x, y, z)
        if any(m is None for m in models):
            raise ValueError("Pass one model (isotropic) or all three axis models.")
        groups: dict = {}
        for axis, model in enumerate(models):
            groups.setdefault(model.cache_spec(), (model, []))[1].append(axis)
        poles = []
        for model, axes in groups.values():
            weights = np.zeros(3)
            weights[axes] = 1.0
            poles.extend(pole_residue_tensor_poles(model, tensor=weights))
        bands = np.array([m.frequency_range for m in models])
        return cls(
            [m.epsilon_inf for m in models],
            permeability,
            electric_poles=poles,
            frequency_range=(bands[:, 0].min(), bands[:, 1].max()),
        )

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

    def fold_fast_poles(self, max_resonance: float) -> AnisotropicMedium:
        """Slow down poles resonating above ``max_resonance`` (rad/s).

        Ultraviolet Sellmeier terms are far above typical simulation bands but
        set the explicit pole timestep limit. Each tensor pole with
        ``resonance > max_resonance`` is replaced by a static shift of the
        high-frequency tensor plus a slower pole at ``max_resonance`` that
        matches the original susceptibility's value and its ``omega^2`` slope
        at low frequency:

            W' = W (w1 / w0)^4,   delta eps_inf = (W / w0^2) (1 - w1^2 / w0^2)

        Both are positive semidefinite, so the result stays passive; damping
        is kept, preserving the leading in-band loss. The remaining error grows
        like ``(omega / w1)^4`` relative to the pole's contribution, so choose
        ``max_resonance`` well above the band of interest and check
        ``permittivity_tensor`` against the original over that band. A declared
        ``frequency_range`` is capped at half of ``max_resonance``. Gyrotropic
        and CCPR poles are left unchanged.
        """

        target = float(max_resonance)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("max_resonance must be finite and positive.")

        def fold(static, poles):
            static = np.asarray(static, dtype=np.float64).copy()
            kept = []
            for pole in poles:
                if (
                    not isinstance(pole, TensorPole)
                    or pole.edot is not None
                    or pole.resonance <= target
                ):
                    kept.append(pole)
                    continue
                ratio = target / pole.resonance
                weight = np.asarray(pole.weight)
                static = static + weight / pole.resonance**2 * (1.0 - ratio**2)
                kept.append(TensorPole(target, pole.damping, weight * ratio**4))
            return static, tuple(kept)

        permittivity, electric = fold(self._epsilon_r, self.electric_poles)
        permeability, magnetic = fold(self._mu_r, self.magnetic_poles)
        band = self.frequency_range
        if band is not None:
            # The folded response is only faithful well below the new resonance.
            upper = min(band[1], 0.5 * target / (2.0 * np.pi))
            band = (band[0], upper) if upper > band[0] else None
        return AnisotropicMedium(
            permittivity,
            permeability,
            electric_poles=electric,
            magnetic_poles=magnetic,
            conductivity=np.asarray(self._conductivity[:3]),
            frequency_range=band,
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
                    float(np.abs(getattr(p, "edot_packed", 0.0)).max()),
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
