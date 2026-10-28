"""Mode-field container, power, and Lorentz phase normalization.

Moved unchanged from the MicroMode ``scipy_reference.py`` import
(``_scipy.py``; see ``UPSTREAM.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _ModeFields:
    """Mutable six-component field container used during normalization."""

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray

    def components(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return field components in electric-then-magnetic order."""
        return self.ex, self.ey, self.ez, self.hx, self.hy, self.hz

    def add_scaled(self, other: _ModeFields, scale: complex) -> None:
        """Add another mode into this mode with a complex scale factor."""
        # Orthogonalization operates in-place to avoid repeatedly allocating six
        # large component arrays per mode.
        for left, right in zip(self.components(), other.components(), strict=True):
            left += scale * right


def _lorentz_orthogonalize_and_normalize(
    modes: list[_ModeFields], cell_areas: np.ndarray
) -> dict[str, object]:
    """Normalize modes and enforce Lorentz orthogonality."""
    # Begin with unit transverse power so projection coefficients are stable.
    for mode in modes:
        _normalize_to_unit_power(mode, cell_areas)

    # Modified Gram-Schmidt using the unconjugated Lorentz overlap.
    for mode_index, mode in enumerate(modes):
        for previous in modes[:mode_index]:
            denom = _lorentz_overlap(previous, previous, cell_areas)
            if abs(denom) <= np.finfo(float).eps:
                continue
            coeff = _lorentz_overlap(previous, mode, cell_areas) / denom
            mode.add_scaled(previous, -coeff)

        # Renormalize and pin phase after each projection step.
        _normalize_to_unit_power(mode, cell_areas)
        _apply_dominant_e_phase_convention(mode)

    # Diagnostics expose both power normalization and residual Lorentz coupling.
    power_norms = np.asarray(
        [abs(_transverse_power(mode, cell_areas)) for mode in modes], dtype=float
    )
    lorentz_norms = np.asarray(
        [_lorentz_overlap(mode, mode, cell_areas) for mode in modes],
        dtype=np.complex128,
    )
    error = 0.0
    for left_index, left in enumerate(modes):
        for right_index, right in enumerate(modes):
            if left_index == right_index:
                continue
            denom = float(
                np.sqrt(
                    abs(lorentz_norms[left_index]) * abs(lorentz_norms[right_index])
                )
            )
            if denom <= np.finfo(float).eps:
                continue
            # Track the largest normalized off-diagonal overlap.
            error = max(error, abs(_lorentz_overlap(left, right, cell_areas)) / denom)
    return {
        "power_norms": power_norms,
        "lorentz_norms": lorentz_norms,
        "lorentz_orthogonality_error": float(error),
    }


def _normalize_to_unit_power(mode: _ModeFields, cell_areas: np.ndarray) -> float:
    """Scale a mode to unit transverse power."""
    # Use magnitude so forward/backward sign conventions do not prevent scaling.
    norm = abs(_transverse_power(mode, cell_areas))
    if norm <= np.finfo(float).eps:
        return 0.0
    scale = 1.0 / np.sqrt(norm)

    # Apply one scalar to all six components to preserve field ratios.
    for component in mode.components():
        component *= scale
    return abs(_transverse_power(mode, cell_areas))


def _transverse_power(mode: _ModeFields, cell_areas: np.ndarray) -> complex:
    """Compute conjugated transverse power flux."""
    # For local z propagation, transverse flux is (E x H*) . z.
    return complex(
        np.sum((mode.ex * np.conj(mode.hy) - mode.ey * np.conj(mode.hx)) * cell_areas)
    )


def _lorentz_overlap(
    left: _ModeFields, right: _ModeFields, cell_areas: np.ndarray
) -> complex:
    """Compute the unconjugated reciprocal-product overlap."""
    # Symmetrize the unconjugated cross product to match reciprocal-mode theory.
    left_cross_right = np.sum((left.ex * right.hy - left.ey * right.hx) * cell_areas)
    right_cross_left = np.sum((right.ex * left.hy - right.ey * left.hx) * cell_areas)
    return complex(0.5 * (left_cross_right + right_cross_left))


def _apply_dominant_e_phase_convention(mode: _ModeFields) -> None:
    """Rotate a mode so its largest electric component is real-positive."""
    # Choose a robust phase anchor from the largest electric-field entry.
    electric = np.concatenate((mode.ex, mode.ey, mode.ez))
    if electric.size == 0:
        return
    anchor = electric[int(np.argmax(np.abs(electric) ** 2))]
    if abs(anchor) <= np.finfo(float).eps:
        return

    # Multiplying by conj(anchor)/|anchor| rotates the anchor onto the positive
    # real axis without changing norms.
    phase = np.conj(anchor) / abs(anchor)
    for component in mode.components():
        component *= phase
