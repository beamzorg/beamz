"""Complex-coordinate PML stretch factors for the SciPy reference solver.

Moved unchanged from the MicroMode ``scipy_reference.py`` import
(``_scipy.py``; see ``UPSTREAM.md``).

Forward stencils (``_create_sfactor_f``) sample stretch half a cell in from
each edge. Backward stencils (``_create_sfactor_b``) sample on the low edge
and one cell inside the high edge. Both call the same polynomial ``_s_value``.
"""

from __future__ import annotations

import numpy as np

from ._scipy_constants import EPSILON0, ETA0


def _create_s_diagonal_values(
    *,
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dmin_pml: tuple[bool, bool],
    omega: float,
    profile: dict[str, float | int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inverse PML stretch vectors for every derivative stencil."""
    nx, ny = shape

    # Estimate boundary-relative speed so PML sigma scales with local material.
    avg_speed = _average_relative_speed(shape, num_pml, eps_tensor, mu_tensor)

    # Forward and backward derivative stencils sample stretch factors at
    # different Yee locations.
    sx_f = _create_sfactor(
        "f",
        omega,
        np.asarray(dlf[0], dtype=float),
        nx,
        num_pml[0],
        dmin_pml[0],
        avg_speed[:2],
        profile,
    )
    sx_b = _create_sfactor(
        "b",
        omega,
        np.asarray(dlb[0], dtype=float),
        nx,
        num_pml[0],
        dmin_pml[0],
        avg_speed[:2],
        profile,
    )
    sy_f = _create_sfactor(
        "f",
        omega,
        np.asarray(dlf[1], dtype=float),
        ny,
        num_pml[1],
        dmin_pml[1],
        avg_speed[2:],
        profile,
    )
    sy_b = _create_sfactor(
        "b",
        omega,
        np.asarray(dlb[1], dtype=float),
        ny,
        num_pml[1],
        dmin_pml[1],
        avg_speed[2:],
        profile,
    )

    # Expand one-dimensional stretch factors into flattened cell vectors.
    sx_f_vec = np.empty(nx * ny, dtype=np.complex128)
    sx_b_vec = np.empty(nx * ny, dtype=np.complex128)
    sy_f_vec = np.empty(nx * ny, dtype=np.complex128)
    sy_b_vec = np.empty(nx * ny, dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            index = ix * ny + iy
            sx_f_vec[index] = 1.0 / sx_f[ix]
            sx_b_vec[index] = 1.0 / sx_b[ix]
            sy_f_vec[index] = 1.0 / sy_f[iy]
            sy_b_vec[index] = 1.0 / sy_b[iy]
    return sx_f_vec, sx_b_vec, sy_f_vec, sy_b_vec


def _average_relative_speed(
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
) -> np.ndarray:
    """Estimate relative wave speed near PML boundaries."""
    # Average diagonal epsilon and mu on each side, then convert to relative
    # speed for damping-strength scaling.
    eps_avg = _pml_average_all_sides(shape, num_pml, eps_tensor)
    mu_avg = _pml_average_all_sides(shape, num_pml, mu_tensor)
    return 1.0 / np.sqrt(eps_avg * mu_avg)


def _pml_average_all_sides(
    shape: tuple[int, int], num_pml: tuple[int, int], tensor: np.ndarray
) -> np.ndarray:
    """Average diagonal material components on each boundary side."""
    nx, ny = shape

    diagonal = np.asarray(
        [tensor[comp, comp] for comp in range(3)], dtype=np.complex128
    ).reshape(3, nx, ny)
    # Region order is xmin, xmax, ymin, ymax. Clamp high-side starts so zero
    # thickness produces an empty slice and oversized PML covers the full axis.
    regions = (
        diagonal[:, : max(num_pml[0], 0), :],
        diagonal[:, max(nx - num_pml[0], 0) :, :],
        diagonal[:, :, : max(num_pml[1], 0)],
        diagonal[:, :, max(ny - num_pml[1], 0) :],
    )
    return np.asarray(
        [region.mean() if region.size else 1.0 for region in regions],
        dtype=np.complex128,
    )


def _create_sfactor(
    direction: str,
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    """Dispatch PML stretch-factor construction for a derivative direction."""
    # No PML means identity stretching along this axis.
    if n_pml == 0:
        return np.ones(n, dtype=np.complex128)

    # Forward and backward stencils use half-cell-shifted PML depths.
    if direction == "f":
        return _create_sfactor_f(omega, dls, n, n_pml, dmin_pml, avg_speed, profile)
    if direction == "b":
        return _create_sfactor_b(omega, dls, n, n_pml, dmin_pml, avg_speed, profile)
    raise ValueError(f"direction value {direction} not recognized")


def _create_sfactor_f(
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    """Build forward-grid PML stretch factors."""
    # Start with identity stretch and overwrite only PML cells.
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        # Low-edge PML can be disabled by symmetry walls.
        if i < n_pml and dmin_pml:
            sfactor[i] = _s_value(
                dls[0], (n_pml - i - 0.5) / n_pml, omega, avg_speed[0], profile
            )
        elif i >= n - n_pml:
            sfactor[i] = _s_value(
                dls[-1], (i - (n - n_pml) + 0.5) / n_pml, omega, avg_speed[1], profile
            )
    return sfactor


def _create_sfactor_b(
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    """Build backward-grid PML stretch factors."""
    # Backward stencils sample exactly on the low edge, not half a cell inward.
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        if i < n_pml and dmin_pml:
            sfactor[i] = _s_value(
                dls[0], (n_pml - i) / n_pml, omega, avg_speed[0], profile
            )
        elif i > n - n_pml:
            sfactor[i] = _s_value(
                dls[-1], (i - (n - n_pml)) / n_pml, omega, avg_speed[1], profile
            )
    return sfactor


def _s_value(
    dl: float,
    step: float,
    omega: float,
    avg_speed: complex,
    profile: dict[str, float | int] | None,
) -> complex:
    """Evaluate one polynomial PML stretch factor."""
    # Defaults match PmlSpec and are overridden by a validated profile dict.
    values = {
        "sigma_max": 2.0,
        "kappa_min": 1.0,
        "kappa_max": 3.0,
        "order": 3,
    }
    if profile is not None:
        values.update(profile)

    # Polynomial depth controls both attenuation sigma and coordinate kappa.
    step_power = step ** int(values["order"])
    kappa = (
        float(values["kappa_min"])
        + (float(values["kappa_max"]) - float(values["kappa_min"])) * step_power
    )
    sigma = avg_speed * (float(values["sigma_max"]) / (ETA0 * dl) * step_power)
    return complex(kappa, 0.0) + 1j * sigma / (omega * EPSILON0)
