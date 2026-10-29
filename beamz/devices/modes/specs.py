"""BeamZ-facing immutable mode selection and result values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from beamz.devices._immutable import immutable_snapshot, readonly_array


@dataclass(frozen=True)
class ModeSpec:
    """Mode selection for previews, sources, and monitors.

    ``num_modes`` is the minimum candidate count and the number of monitor
    amplitudes. ``mode_index`` selects one mode after polarization-aware ordering.
    ``num_freqs`` is the broadband reconstruction count, independent of monitor
    frequencies. ``force_symmetry`` projects 2D transverse planes about validated
    mirrors before Yee interpolation; line profiles stay unprojected.
    """

    num_modes: int = 1
    mode_index: int = 0
    polarization: str | None = None
    target_neff: float | None = None
    num_freqs: int | None = None
    force_symmetry: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.force_symmetry, bool):
            raise TypeError("force_symmetry must be a bool")
        object.__setattr__(self, "num_modes", max(1, int(self.num_modes)))
        object.__setattr__(self, "mode_index", max(0, int(self.mode_index)))
        if self.num_freqs is not None:
            object.__setattr__(self, "num_freqs", max(1, int(self.num_freqs)))
        if self.polarization is not None:
            pol = str(self.polarization).lower()
            if pol not in {"te", "tm"}:
                raise ValueError(f"polarization must be 'te' or 'tm', got {pol!r}.")
            object.__setattr__(self, "polarization", pol)


@dataclass(frozen=True)
class ModeData:
    """Immutable modal fields from a mode solve.

    Leading axes of ``neffs``, ``e_fields``, and ``h_fields`` are frequency then
    mode. ``grid_edges`` are physical cell edges in the trailing field-axis
    order, named by ``transverse_axes``. Arrays are copied and made read-only.
    ``diagnostics`` snapshots requested and applied symmetry per frequency.
    """

    frequencies: np.ndarray
    neffs: np.ndarray
    e_fields: np.ndarray
    h_fields: np.ndarray
    eps_profiles: np.ndarray
    resolution: float
    solver_direction: str | None = None
    axis: str | None = None
    center: tuple[float, float, float] | None = None
    plane: Any = field(default=None, compare=False, hash=False, repr=False)
    crop_slices: tuple[slice, ...] | None = field(
        default=None, compare=False, hash=False
    )
    eps_profile_fulls: np.ndarray | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    grid_edges: tuple[np.ndarray, ...] | None = field(
        default=None, compare=False, hash=False, repr=False
    )
    transverse_axes: tuple[str, ...] | None = field(
        default=None, compare=False, hash=False
    )

    diagnostics: tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for name, dtype in (
            ("frequencies", float),
            ("neffs", np.complex128),
            ("e_fields", np.complex128),
            ("h_fields", np.complex128),
            ("eps_profiles", None),
        ):
            object.__setattr__(
                self, name, readonly_array(getattr(self, name), dtype=dtype)
            )
        if self.eps_profile_fulls is not None:
            object.__setattr__(
                self, "eps_profile_fulls", readonly_array(self.eps_profile_fulls)
            )
        if self.grid_edges is not None:
            edges = tuple(
                readonly_array(values, dtype=float) for values in self.grid_edges
            )
            if not edges:
                raise ValueError("grid_edges must contain at least one edge array.")
            field_shape = self.e_fields.shape[-len(edges) :]
            for index, (values, cell_count) in enumerate(
                zip(edges, field_shape, strict=True)
            ):
                if values.ndim != 1 or values.size != int(cell_count) + 1:
                    raise ValueError(
                        "grid_edges must have one more entry than their modal-field "
                        f"axis; axis {index} has {values.shape} for {cell_count} cells."
                    )
                if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                    raise ValueError(
                        "grid_edges must be finite and strictly increasing."
                    )
            object.__setattr__(self, "grid_edges", edges)
        if self.transverse_axes is not None:
            axes = tuple(str(axis).lower() for axis in self.transverse_axes)
            if any(axis not in {"x", "y", "z"} for axis in axes):
                raise ValueError("transverse_axes entries must be 'x', 'y', or 'z'.")
            if self.grid_edges is not None and len(axes) != len(self.grid_edges):
                raise ValueError(
                    "transverse_axes must provide one name per grid edge array."
                )
            object.__setattr__(self, "transverse_axes", axes)
        object.__setattr__(self, "diagnostics", immutable_snapshot(self.diagnostics))
        if self.center is not None:
            object.__setattr__(self, "center", tuple(float(v) for v in self.center[:3]))

    def _frequency_index(self, f=None) -> int:
        if self.frequencies.size == 0:
            raise ValueError("ModeData contains no frequencies.")
        if f is None:
            return 0
        return int(
            np.argmin(np.abs(np.asarray(self.frequencies, dtype=float) - float(f)))
        )

    def selected_mode(self, *, f=None, mode_index=0):
        """Return arrays for the nearest frequency and selected mode index."""
        f_idx = self._frequency_index(f)
        m_idx = int(mode_index)
        neff = np.asarray(self.neffs)[f_idx, m_idx]
        e_field = np.asarray(self.e_fields)[f_idx, m_idx]
        h_field = np.asarray(self.h_fields)[f_idx, m_idx]
        eps_full = (
            np.asarray(self.eps_profile_fulls)[f_idx]
            if self.eps_profile_fulls is not None
            else np.asarray(self.eps_profiles)[f_idx]
        )
        return f_idx, m_idx, neff, e_field, h_field, eps_full
