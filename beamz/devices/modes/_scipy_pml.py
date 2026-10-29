"""Complex-coordinate PML stretch factors for the SciPy reference solver.

Forward stencils sample stretch half a cell in from each edge. Backward
stencils sample on the low edge and one cell inside the high edge.
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
    avg_speed = _average_relative_speed(shape, num_pml, eps_tensor, mu_tensor)
    specs = (
        ("f", dlf[0], nx, num_pml[0], dmin_pml[0], avg_speed[:2]),
        ("b", dlb[0], nx, num_pml[0], dmin_pml[0], avg_speed[:2]),
        ("f", dlf[1], ny, num_pml[1], dmin_pml[1], avg_speed[2:]),
        ("b", dlb[1], ny, num_pml[1], dmin_pml[1], avg_speed[2:]),
    )
    sx_f, sx_b, sy_f, sy_b = (
        _create_sfactor(
            direction,
            omega,
            np.asarray(steps, dtype=float),
            count,
            n_pml,
            dmin,
            speeds,
            profile,
        )
        for direction, steps, count, n_pml, dmin, speeds in specs
    )
    return (
        np.repeat(1.0 / sx_f, ny),
        np.repeat(1.0 / sx_b, ny),
        np.tile(1.0 / sy_f, nx),
        np.tile(1.0 / sy_b, nx),
    )


def _average_relative_speed(
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
) -> np.ndarray:
    """Estimate relative wave speed near each PML boundary."""
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
    # Order is xmin, xmax, ymin, ymax. Clamp so zero thickness is an empty slice.
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
    """Build PML stretch factors for one derivative direction."""
    if n_pml == 0:
        return np.ones(n, dtype=np.complex128)
    if direction not in {"f", "b"}:
        raise ValueError(f"direction value {direction} not recognized")
    backward = direction == "b"
    inset = 0.0 if backward else 0.5
    high_start = n - n_pml + int(backward)
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        if i < n_pml and dmin_pml:
            depth = (n_pml - i - inset) / n_pml
            step, speed = dls[0], avg_speed[0]
        elif i >= high_start:
            depth = (i - (n - n_pml) + inset) / n_pml
            step, speed = dls[-1], avg_speed[1]
        else:
            continue
        sfactor[i] = _s_value(step, depth, omega, speed, profile)
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
    return _create_sfactor("b", omega, dls, n, n_pml, dmin_pml, avg_speed, profile)


def _s_value(
    dl: float,
    step: float,
    omega: float,
    avg_speed: complex,
    profile: dict[str, float | int] | None,
) -> complex:
    """Evaluate one polynomial PML stretch factor."""
    values = {"sigma_max": 2.0, "kappa_min": 1.0, "kappa_max": 3.0, "order": 3}
    if profile is not None:
        values.update(profile)
    step_power = step ** int(values["order"])
    kappa = (
        float(values["kappa_min"])
        + (float(values["kappa_max"]) - float(values["kappa_min"])) * step_power
    )
    sigma = avg_speed * (float(values["sigma_max"]) / (ETA0 * dl) * step_power)
    return complex(kappa, 0.0) + 1j * sigma / (omega * EPSILON0)
