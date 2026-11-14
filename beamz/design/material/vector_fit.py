"""Design implementation."""

from itertools import product

import numpy as np
from numpy.typing import NDArray

from beamz.design.dispersion import (
    PoleResidue,
    _optical_samples,
    _stack,
    _validate_nk,
)
from beamz.design.material.fit_core import _basis, _expand, _passive_residues, _trial


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
    wl, n, k = _validate_nk(wavelengths, n, k, unique=True)
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
    frequency, scale, omega, target = _optical_samples(wl[order], n[order], k[order])
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
