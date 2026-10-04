"""Independent scalar vector fitting, with the exp(-i omega t) convention.

Implements pole relocation from Gustavsen & Semlyen (1999), DOI
10.1109/61.772353, and relaxed normalization from Gustavsen (2006), DOI
10.1109/TPWRD.2005.860281. Real coefficients enforce conjugate symmetry.
Passivity is enforced by constrained residue fitting and audited numerically;
it is not a global mathematical certificate. No external fitter is used.
"""

from itertools import product
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, OptimizeResult, minimize, minimize_scalar

from beamz.const import LIGHT_SPEED
from beamz.design.dispersion import PoleResidue


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


def _stack(values, weights):
    return np.concatenate((values.real * weights[0], values.imag * weights[1]))


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


def _refined_minima(poles, coefficients, grid, loss):
    """Locate between-sample loss minima in log frequency."""
    indices = (
        np.flatnonzero(
            (loss[1:-1] < loss[:-2]) & (loss[1:-1] < loss[2:]) & (loss[1:-1] < 1e-7)
        )
        + 1
    )
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
    if min(loss.min(), minima.min(initial=np.inf)) >= -1e-10:
        return coefficients, float(min(loss.min(), minima.min(initial=np.inf)))
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
        minima = (
            np.flatnonzero(
                (loss[1:-1] < loss[:-2]) & (loss[1:-1] < loss[2:]) & (loss[1:-1] < 0)
            )
            + 1
        )
        checks = np.unique(np.r_[checks, extra, audit[minima], audit[np.argmin(loss)]])
    return None


def fit_nk_vector(
    wavelengths,
    n,
    k,
    *,
    max_poles=9,
    epsilon_inf=1.0,
    weights=(1.0, 1.0),
    num_iters=200,
    tolerance_rms=0.02,
):
    """Fit passive pole/residue media by scalar vector fitting using NumPy/SciPy.

    Wavelengths are metres. Weights apply to Re(epsilon), Im(epsilon), normalized
    to mean one; weighted RMS is sqrt(mean((wr*dRe)**2+(wi*dIm)**2)). Tries
    1..max_poles stored poles (each complex entry represents a conjugate pair),
    both standard/relaxed relocation, and real/complex initial poles with
    linear/log spacing. Real poles are stored with half their residue to match
    the PoleResidue convention.
    Returns the best passive model and diagnostics even if tolerance is missed.
    Passivity is sampled over seven decades on either side of the fit band,
    with extra samples around resonances. No global passivity proof is claimed.
    """
    wl, n, k = [np.asarray(v, dtype=float) for v in (wavelengths, n, k)]
    if (
        wl.ndim != 1
        or wl.shape != n.shape
        or wl.shape != k.shape
        or wl.size < 4
        or not np.isfinite([wl, n, k]).all()
        or np.any(wl <= 0)
        or np.any(n <= 0)
        or np.any(k < 0)
        or len(np.unique(wl)) != len(wl)
    ):
        raise ValueError("Require matching finite, unique wavelengths>0, n>0, k>=0.")
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (2,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.isfinite(epsilon_inf)
        or epsilon_inf < 1
        or not isinstance(max_poles, (int, np.integer))
        or max_poles < 1
        or not isinstance(num_iters, (int, np.integer))
        or num_iters < 1
        or not np.isfinite(tolerance_rms)
        or tolerance_rms < 0
    ):
        raise ValueError(
            "Require positive integer orders/iterations, positive weights, epsilon_inf>=1, tolerance>=0."
        )
    weights = weights / weights.mean()
    order = np.argsort(wl)
    frequency = LIGHT_SPEED / wl[order]
    scale = 2 * np.pi * np.sqrt(frequency.min() * frequency.max())
    omega = 2 * np.pi * frequency / scale
    target = (n[order] + 1j * k[order]) ** 2
    best = None
    trials = 0
    for count in range(1, max_poles + 1):
        for relaxed, real, logarithmic in product((False, True), repeat=3):
            spacing = np.geomspace if logarithmic else np.linspace
            initial = spacing(omega.min(), omega.max(), count)
            poles = -initial.astype(complex) if real else -initial * (0.01 + 1j)
            candidates = _trial(
                omega, target, poles, epsilon_inf, weights, relaxed, num_iters
            )
            trials += 1
            for lower_error, poles, coefficients, iterations in candidates:
                if len(poles) > count or (
                    best is not None and np.sqrt(2 * lower_error) >= best[0]
                ):
                    continue
                passive = _passive_residues(
                    omega, target, poles, coefficients, epsilon_inf, weights
                )
                if passive is None:
                    continue
                coefficients, minimum = passive
                error = float(
                    np.sqrt(
                        2
                        * np.mean(
                            _stack(
                                epsilon_inf
                                + _basis(omega, poles) @ coefficients
                                - target,
                                weights,
                            )
                            ** 2
                        )
                    )
                )
                if best is None or error < best[0]:
                    best = (
                        error,
                        poles,
                        coefficients,
                        minimum,
                        iterations,
                        relaxed,
                        real,
                        logarithmic,
                    )
        if best is not None and best[0] <= tolerance_rms:
            break
    if best is None:
        raise RuntimeError("No fit passed residue optimization and passivity checks.")
    error, poles, coefficients, minimum, iterations, relaxed, real, logarithmic = best
    full_poles, residues = _expand(poles, coefficients)
    pairs = [
        (p * scale, c * scale / (2 if p.imag == 0 else 1))
        for p, c in zip(full_poles, residues, strict=True)
        if p.imag <= 0
    ]
    model = PoleResidue(
        epsilon_inf, pairs, frequency_range=(frequency.min(), frequency.max())
    )
    fitted: NDArray[np.complex128] = np.sqrt(model.eps_model(frequency))
    return model, {
        "method": "Independent scalar vector fitting with constrained passive residues",
        "weighted_rms_epsilon": error,
        "tolerance_rms": float(tolerance_rms),
        "tolerance_met": bool(error <= tolerance_rms),
        "weights": weights.tolist(),
        "epsilon_inf": float(epsilon_inf),
        "max_poles": int(max_poles),
        "rational_order": len(full_poles),
        "pole_entries": len(pairs),
        "num_iters": int(num_iters),
        "selected_iteration": iterations,
        "trials": trials,
        "relaxed": relaxed,
        "real_initial_poles": real,
        "logspacing": logarithmic,
        "passivity_audit_min": minimum,
        "optimizer_success": True,
        "rms_n": float(np.sqrt(np.mean((fitted.real - n[order]) ** 2))),
        "rms_k": float(np.sqrt(np.mean((fitted.imag - k[order]) ** 2))),
    }
