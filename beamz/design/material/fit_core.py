"""Design implementation."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, OptimizeResult, minimize, minimize_scalar

from beamz.design.dispersion import (
    _stack,
)


def _basis(omega, poles) -> NDArray[np.complex128]:
    """Real residue coordinates: one per real pole, two per conjugate pair."""
    s = -1j * np.asarray(omega)
    columns = []
    for pole in poles:
        plus = 1 / (s - pole)
        if pole.imag == 0:
            columns.append(plus)
        else:
            minus = 1 / (s - pole.conjugate())
            columns.extend((plus + minus, 1j * (plus - minus)))
    return np.column_stack(columns)


def _expand(poles, coefficients):
    expanded, residues = [], []
    index = 0
    for pole in poles:
        if pole.imag == 0:
            expanded.append(pole)
            residues.append(complex(coefficients[index]))
            index += 1
        else:
            c = complex(*coefficients[index : index + 2])
            expanded.extend((pole, pole.conjugate()))
            residues.extend((c, c.conjugate()))
            index += 2
    return np.asarray(expanded), np.asarray(residues)


def _canonical_poles(roots):
    """Collapse conjugate roots, reflect unstable poles, preserve real roots."""
    poles = []
    remaining = list(roots)
    while remaining:
        root = remaining.pop(0)
        if abs(root.imag) <= 1e-8 * max(1, abs(root)):
            poles.append(complex(-max(abs(root.real), 1e-10)))
        else:
            match = int(np.argmin(abs(np.asarray(remaining) - root.conjugate())))
            conjugate = remaining.pop(match)
            if abs(root - conjugate.conjugate()) > 1e-5 * max(1, abs(root)):
                raise ValueError("Pole relocation lost conjugate symmetry.")
            root = (root + conjugate.conjugate()) / 2
            poles.append(complex(-max(abs(root.real), 1e-10), -abs(root.imag)))
    return np.asarray(sorted(poles, key=lambda p: (abs(p.imag), p.real)))


def _least_squares(matrix, rhs):
    norms = np.maximum(np.linalg.norm(matrix, axis=0), 1e-30)
    return np.linalg.lstsq(matrix / norms, rhs, rcond=1e-12)[0] / norms


def _relocate(omega, target, poles, epsilon_inf, weights, relaxed):
    basis = _basis(omega, poles)
    matrix = np.column_stack((basis, -target[:, None] * basis))
    count = basis.shape[1]
    if relaxed:
        matrix = np.column_stack((matrix, -(target - epsilon_inf)))
        matrix = _stack(matrix, weights)
        # Mean(real(sigma)) = 1 removes the homogeneous solution while allowing
        # sigma's asymptotic value to vary (relaxed pole relocation).
        strength = np.linalg.norm(_stack(target, weights))
        row = np.r_[np.zeros(count), basis.real.mean(axis=0), 1] * strength
        solution = _least_squares(
            np.vstack((matrix, row)), np.r_[np.zeros(matrix.shape[0]), strength]
        )
        constant = solution[-1]
        if abs(constant) < 1e-8:
            return _relocate(omega, target, poles, epsilon_inf, weights, False)
    else:
        solution = _least_squares(
            _stack(matrix, weights), _stack(target - epsilon_inf, weights)
        )
        constant = 1.0
    full_poles, residues = _expand(poles, solution[count : 2 * count])
    # Zeros of sigma(s) are eigenvalues of diag(a) - 1*c^T/d.
    roots = np.linalg.eigvals(np.diag(full_poles) - residues[None, :] / constant)
    return _canonical_poles(roots)


def _trial(omega, target, poles, epsilon_inf, weights, relaxed, num_iters):
    # Retain several iterations: the lowest unconstrained residual need not
    # give the best fit after passivity enforcement.
    candidates = []
    max_entries = len(poles)
    for iteration in range(num_iters):
        basis = _basis(omega, poles)
        coefficients = _least_squares(
            _stack(basis, weights), _stack(target - epsilon_inf, weights)
        )
        error = float(
            np.mean(_stack(basis @ coefficients + epsilon_inf - target, weights) ** 2)
        )
        if len(poles) <= max_entries and not any(
            abs(error - c[0]) < 1e-8 * max(error, 1e-20) for c in candidates
        ):
            candidates.append((error, poles.copy(), coefficients, iteration + 1))
            candidates.sort(key=lambda candidate: candidate[0])
            candidates = candidates[:4]
        if error < 1e-26:
            break
        try:
            updated = _relocate(omega, target, poles, epsilon_inf, weights, relaxed)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            break
        if not np.isfinite(updated).all():
            break
        if updated.shape == poles.shape and np.allclose(
            updated, poles, rtol=1e-9, atol=1e-12
        ):
            break
        poles = updated
    return candidates


def _audit_grid(omega, poles):
    # Resolve narrow resonances as well as both asymptotic tails.
    local = [abs(p.imag) + abs(p.real) * np.linspace(-30, 30, 601) for p in poles]
    grid = np.concatenate(
        [
            np.geomspace(omega.min() * 1e-7, omega.max() * 1e7, 20001),
            np.linspace(omega.min() * 0.5, omega.max() * 2, 10001),
            *local,
        ]
    )
    return np.unique(grid[grid > 0])


def _local_minima(loss, limit):
    return (
        np.flatnonzero(
            (loss[1:-1] < loss[:-2]) & (loss[1:-1] < loss[2:]) & (loss[1:-1] < limit)
        )
        + 1
    )


def _refined_minima(poles, coefficients, grid, loss):
    """Locate between-sample loss minima in log frequency."""
    indices = _local_minima(loss, 1e-7)
    # Ignore roundoff-only oscillations in the asymptotic tails unless negative.
    indices = indices[np.argsort(loss[indices])[:64]]
    frequencies, values = [], []
    for index in indices:

        def objective(log_frequency):
            w = np.exp(log_frequency)
            return float(
                (_basis([w], poles).imag @ coefficients)[0] / (w / (1 + w * w))
            )

        result = cast(
            OptimizeResult,
            minimize_scalar(
                objective,
                bounds=np.log(grid[index - 1 : index + 2 : 2]),
                method="bounded",
                options={"xatol": 1e-12},
            ),
        )
        frequencies.append(float(np.exp(result.x)))
        values.append(float(result.fun))
    return np.asarray(frequencies), np.asarray(values)


def _passive_residues(omega, target, poles, coefficients, epsilon_inf, weights):
    basis = _basis(omega, poles)
    matrix = _stack(basis, weights)
    rhs = _stack(target - epsilon_inf, weights)
    audit = _audit_grid(omega, poles)
    audit_basis = _basis(audit, poles).imag / (audit / (1 + audit**2))[:, None]
    loss = audit_basis @ coefficients
    extra, minima = _refined_minima(poles, coefficients, audit, loss)
    floor = float(min(loss.min(), minima.min(initial=np.inf)))
    if floor >= -1e-10:
        return coefficients, floor
    # Whiten the quadratic objective so near-collinear real-pole bases do not
    # make SLSQP operate on a poorly conditioned Hessian.
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if singular[-1] < singular[0] * 1e-12:
        return None
    transform = vt.T / singular
    center = u.T @ rhs
    x = singular * (vt @ coefficients)
    checks = np.unique(np.r_[np.geomspace(audit.min(), audit.max(), 256), omega, extra])
    for _ in range(12):
        constraint = _basis(checks, poles).imag / (checks / (1 + checks**2))[:, None]
        constraint = constraint @ transform
        constraint /= np.maximum(np.linalg.norm(constraint, axis=1), 1e-30)[:, None]
        result = minimize(
            lambda z: (0.5 * np.sum((z - center) ** 2), z - center),
            x,
            jac=True,
            method="SLSQP",
            constraints=[LinearConstraint(constraint, 1e-10, np.inf)],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        x = result.x
        loss = audit_basis @ (transform @ x)
        extra, refined = _refined_minima(poles, transform @ x, audit, loss)
        minimum = min(loss.min(), refined.min(initial=np.inf))
        if result.success and minimum >= -1e-10:
            return transform @ x, float(minimum)
        minima = _local_minima(loss, 0)
        checks = np.unique(np.r_[checks, extra, audit[minima], audit[np.argmin(loss)]])
    return None
