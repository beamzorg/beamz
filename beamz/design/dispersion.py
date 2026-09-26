"""Causal scalar optical media, using SI pole/residue coefficients.

The exp(-i omega t) convention is used. Each supplied (a, c) represents a
conjugate pair: epsilon = epsilon_inf + c/(-i omega-a) + c*/(-i omega-a*).
Real poles therefore contribute twice, too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, nnls

from beamz.const import LIGHT_SPEED
from beamz.design.materials import Material


@dataclass(frozen=True, init=False)
class PoleResidue(Material):
    """Scalar dispersive material with stable conjugate pole/residue pairs.

    ``poles`` contains (pole, residue) in rad/s. ``frequency_range`` (Hz)
    declares the validated fit band. Passivity is checked on a dense frequency
    grid; this is a numerical check, not a global mathematical certificate.
    Interfaces use volume averaging. Only JAX single-device execution is supported.
    """

    poles: tuple[tuple[complex, complex], ...]
    frequency_range: tuple[float, float]

    def __init__(self, epsilon_inf, poles, *, frequency_range):
        if np.ndim(epsilon_inf) != 0 or float(epsilon_inf) < 1:
            raise ValueError(
                "PoleResidue epsilon_inf must be scalar and >= 1 for the vacuum CFL bound."
            )
        super().__init__(permittivity=epsilon_inf)
        pairs = tuple((complex(a), complex(c)) for a, c in poles)
        if not pairs or any(
            not np.isfinite([a, c]).all() or a.real > 0 for a, c in pairs
        ):
            raise ValueError("Poles must be finite, nonempty, and have real part <= 0.")
        band = tuple(float(f) for f in frequency_range)
        if len(band) != 2 or not 0 < band[0] < band[1] or not np.isfinite(band).all():
            raise ValueError("frequency_range must contain increasing positive Hz.")
        object.__setattr__(self, "poles", pairs)
        object.__setattr__(self, "frequency_range", band)
        # Check beyond the fit band as well: a fitted medium must not turn into
        # an amplifier in the pulse tails. Zero-frequency poles are evaluated
        # using a positive limiting frequency.
        frequencies = np.unique(
            np.r_[
                np.linspace(*band, 4096),
                np.geomspace(band[0] * 1e-4, band[1] * 1e4, 4096),
            ]
        )
        eps = self.eps_model(frequencies)
        if not np.isfinite(eps).all() or np.any(
            eps.imag < -1e-7 * np.maximum(1.0, abs(eps))
        ):
            raise ValueError(
                "Pole-residue model fails sampled passivity/finite-response checks."
            )

    @property
    def epsilon_inf(self):
        return self.permittivity

    def eps_model(self, frequencies):
        """Complex relative permittivity at positive frequencies in Hz."""
        freq = np.asarray(frequencies, dtype=float)
        if np.any(~np.isfinite(freq)) or np.any(freq <= 0):
            raise ValueError("Frequencies must be finite and positive.")
        s = -2j * np.pi * freq
        eps = np.full(freq.shape, self.epsilon_inf, dtype=complex)
        for a, c in self.poles:
            eps += c / (s - a) + c.conjugate() / (s - a.conjugate())
        return eps

    @property
    def max_permittivity(self):
        # abs(n)**2 also accounts for metal extinction when choosing a mesh.
        return max(
            float(self.epsilon_inf),
            float(
                np.max(abs(self.eps_model(np.linspace(*self.frequency_range, 2048))))
            ),
        )

    def cache_spec(self):
        return super().cache_spec(), self.poles, self.frequency_range

    def to_spec(self):
        """JSON-compatible complete material specification (SI units)."""
        return {
            "epsilon_inf": self.epsilon_inf,
            "frequency_range": list(self.frequency_range),
            "poles": [[[a.real, a.imag], [c.real, c.imag]] for a, c in self.poles],
        }

    @classmethod
    def from_spec(cls, spec):
        return cls(
            spec["epsilon_inf"],
            [(complex(*a), complex(*c)) for a, c in spec["poles"]],
            frequency_range=spec["frequency_range"],
        )

    def updated_copy(self, **changes):
        values = dict(
            epsilon_inf=self.epsilon_inf,
            poles=self.poles,
            frequency_range=self.frequency_range,
        )
        if set(changes) - set(values):
            raise TypeError(
                "PoleResidue updates accept epsilon_inf, poles, frequency_range."
            )
        return type(self)(**(values | changes))

    @classmethod
    def lorentz(cls, epsilon_inf, *, strength, resonance, damping, frequency_range):
        """One passive Lorentz oscillator; resonance and damping are in rad/s.

        susceptibility = strength*resonance**2 /
                         (resonance**2 - omega**2 - i*damping*omega).
        """
        if strength <= 0 or resonance <= 0 or damping < 0 or damping >= 2 * resonance:
            raise ValueError("Require strength>0, resonance>0, 0<=damping<2*resonance.")
        b = np.sqrt(resonance**2 - damping**2 / 4)
        return cls(
            epsilon_inf,
            [(-damping / 2 - 1j * b, 1j * strength * resonance**2 / (2 * b))],
            frequency_range=frequency_range,
        )

    @classmethod
    def drude(cls, epsilon_inf, *, plasma_frequency, damping, frequency_range):
        """Passive Drude model; plasma_frequency and damping are in rad/s."""
        if plasma_frequency <= 0 or damping <= 0:
            raise ValueError("Drude plasma frequency and damping must be positive.")
        c = plasma_frequency**2 / (2 * damping)
        return cls(
            epsilon_inf, [(0, c), (-damping, -c)], frequency_range=frequency_range
        )


def fit_nk(wavelengths, n, k, *, num_poles=9, weights=(1.0, 1.0), max_nfev=300):
    """Fit passive Lorentz oscillators to tabulated n,k (wavelengths in metres).

    Returns ``(material, diagnostics)``. Positive strengths and damping guarantee
    passive Lorentz terms. The weighted complex-index residual is minimized;
    an error report is returned rather than silently claiming a target tolerance.
    """
    wl, n, k = [np.asarray(x, dtype=float) for x in (wavelengths, n, k)]
    if (
        wl.ndim != 1
        or wl.shape != n.shape
        or wl.shape != k.shape
        or wl.size < 4
        or not np.isfinite([wl, n, k]).all()
        or np.any(wl <= 0)
        or np.any(n <= 0)
        or np.any(k < 0)
    ):
        raise ValueError(
            "Expected matching finite wavelength,n,k vectors with wavelength,n>0 and k>=0."
        )
    if num_poles < 1 or len(weights) != 2 or min(weights) <= 0:
        raise ValueError("Require positive pole count and residual weights.")
    f = LIGHT_SPEED / wl
    scale = 2 * np.pi * np.sqrt(f.min() * f.max())
    w = 2 * np.pi * f / scale
    target = (n + 1j * k) ** 2
    resonances = np.linspace(w.min() * 0.8, w.max() * 1.3, num_poles)
    if num_poles == 1:
        resonances[:] = w.max() * 1.3
    dampings = np.full(num_poles, (w.max() - w.min()) / num_poles * 1.5)

    def basis(r, g):
        return r[None, :] ** 2 / (
            r[None, :] ** 2 - w[:, None] ** 2 - 1j * w[:, None] * g[None, :]
        )

    b = np.c_[np.ones(w.size), basis(resonances, dampings)]
    amplitudes = nnls(
        np.r_[b.real * weights[0], b.imag * weights[1]],
        np.r_[target.real * weights[0], target.imag * weights[1]],
    )[0]
    x0 = np.log(
        np.r_[
            max(amplitudes[0], 1.00001),
            np.maximum(amplitudes[1:], 1e-5),
            resonances,
            dampings,
        ]
    )

    def unpack(x):
        e, s, r, g = np.split(np.exp(x), [1, 1 + num_poles, 1 + 2 * num_poles])
        return e[0], s, r, g

    def residual(x):
        e, s, r, g = unpack(x)
        delta = np.sqrt(e + basis(r, g) @ s) - (n + 1j * k)
        return np.r_[delta.real * weights[0], delta.imag * weights[1]]

    result = least_squares(
        residual,
        x0,
        bounds=(np.r_[0.0, np.full(3 * num_poles, -18.0)], 8),
        max_nfev=max_nfev,
    )
    e, s, r, g = unpack(result.x)
    poles = []
    for strength, resonance, damping in zip(s, r * scale, g * scale, strict=True):
        # Both overdamped and underdamped real-coefficient Lorentz oscillators.
        root = np.sqrt(complex(damping**2 - 4 * resonance**2))
        a1, a2 = (-damping + root) / 2, (-damping - root) / 2
        if abs(root) < 1e-12 * resonance:
            raise ValueError(
                "Critically damped fit is ill-conditioned; retry with fewer poles."
            )
        c = strength * resonance**2 / root
        if root.imag:
            poles.append((a1, c))
        else:
            poles.extend([(a1, c / 2), (a2, -c / 2)])
    material = PoleResidue(e, poles, frequency_range=(f.min(), f.max()))
    fitted_nk = np.sqrt(material.eps_model(f))
    return material, {
        "rms_epsilon": float(
            np.sqrt(np.mean(abs(material.eps_model(f) - target) ** 2))
        ),
        "rms_n": float(np.sqrt(np.mean((fitted_nk.real - n) ** 2))),
        "rms_k": float(np.sqrt(np.mean((fitted_nk.imag - k) ** 2))),
        "optimizer_success": bool(result.success),
        "evaluations": int(result.nfev),
    }
