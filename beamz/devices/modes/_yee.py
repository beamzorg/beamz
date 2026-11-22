"""Fixed-beta refinement on BeamZ's x-normal Yee cross section."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

_C0 = 299_792_458.0
_EPS0 = 8.854_187_812_8e-12
_MU0 = 1.256_637_062_12e-6

_E_COMPONENTS = ("Ex", "Ey", "Ez")
_H_COMPONENTS = ("Hx", "Hy", "Hz")


def transverse_grid_metrics(indices, grid):
    """Primal/dual curl distances and volume averages on the actual node indices.

    A mode's Ex samples are transverse nodes, not cell centers. In particular,
    averaging their dual widths does not recover the derivative's primal width.
    Neighboring cells outside the aperture still contribute to the dual metric.
    """
    distances, averages = [], []
    for dimension, axis in enumerate(("z", "y")):
        selector = indices["Ex"][dimension]
        widths = np.asarray(grid.cell_widths(axis), dtype=float)
        nodes = np.arange(widths.size + 1)[selector]
        left_width = widths[np.maximum(nodes - 1, 0)]
        right_width = widths[np.minimum(nodes, widths.size - 1)]
        dual = 0.5 * (left_width + right_width)
        left = left_width / (left_width + right_width)
        right = 1.0 - left
        left = np.where(nodes == widths.size, 1.0, left)
        right = np.where(nodes == 0, 1.0, right)
        distances.append((widths[nodes[:-1]], dual))
        averages.append((left, right))
    return tuple(distances), tuple(averages)


def refine_x_mode_at_fixed_beta(
    profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
    omega,
    dt,
    resolution,
    k_num,
    direction_sign,
    transverse_cell_widths=None,
    transverse_metrics=None,
    _correct_beta=True,
    _remaining_corrections=10,
    _prior_factor=None,
):
    """Refine E at the selected beta, then reconstruct H from Faraday's law."""
    from scipy import sparse
    from scipy.sparse import linalg as spla

    out = {
        name: np.asarray(value, dtype=np.complex128) for name, value in profiles.items()
    }
    nz, ny = out["Ex"].shape
    d = float(resolution)
    sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
    delta = -2j * sign * np.sin(0.5 * float(k_num) * d) / d
    curl_e, curl_h, curl_e_beta, curl_h_beta = _yee_curl_operators(
        sparse,
        out,
        resolution=d,
        delta=delta,
        transverse_cell_widths=transverse_cell_widths,
        transverse_metrics=transverse_metrics,
    )
    eps, mu = _component_material_vectors(
        out,
        indices,
        component_permittivity=component_permittivity,
        component_permeability=component_permeability,
    )
    ex_n, ey_n, _ez_n = (out[name].size for name in _E_COMPONENTS)
    hx_n, hy_n, _hz_n = (out[name].size for name in _H_COMPONENTS)
    omega_d = (
        float(omega)
        if dt is None
        else 2.0 * np.sin(0.5 * float(omega) * float(dt)) / float(dt)
    )
    target = (omega_d / _C0) ** 2
    seed = np.concatenate([out[name].reshape(-1) for name in ("Ex", "Ey", "Ez")])
    if _prior_factor is None:
        _prior_factor = {}
    eig = None
    # The reduced transverse system must itself reach the factorization
    # size where a second solve can reuse its numerical factor.
    if seed.size - ex_n >= 16384:
        eig = _transverse_eigenpairs(
            sparse,
            spla,
            curl_h,
            curl_e,
            eps,
            mu,
            delta,
            seed,
            target,
            (nz, ny),
            _prior_factor,
        )
    if eig is None:
        operator = curl_h @ sparse.diags(1.0 / mu) @ curl_e
        eig = _converged_shift_invert_eigenpairs(
            spla,
            operator,
            sparse.diags(eps),
            target=target,
            seed=seed,
            count=min(4, seed.size - 2),
            longitudinal_size=ex_n,
            transverse_shape=(nz, ny),
            prior_factor=_prior_factor,
        )
    values, vectors = eig
    # Generalized eigenvectors have arbitrary scales. Track both E and H:
    # a nearby electric-overlap candidate can belong to a different magnetic branch.
    magnetic_candidates = 1j * (curl_e @ vectors) / (_MU0 * omega_d * mu[:, None])
    magnetic_seed = np.concatenate([out[name].reshape(-1) for name in _H_COMPONENTS])
    overlaps = np.abs(np.conjugate(seed) @ vectors) / np.maximum(
        np.linalg.norm(vectors, axis=0), np.finfo(float).tiny
    )
    overlaps *= np.abs(np.conjugate(magnetic_seed) @ magnetic_candidates) / np.maximum(
        np.linalg.norm(magnetic_candidates, axis=0), np.finfo(float).tiny
    )
    selected = int(np.argmax(overlaps))
    if _prior_factor:
        # Closely spaced polarizations must be reconsidered together after beta moves.
        gaps = np.abs(np.delete(values, selected) - values[selected])
        if not gaps.size or gaps.min() <= max(
            1e-6 * target, 16 * abs(values[selected] - target)
        ):
            _prior_factor.clear()
    electric = np.asarray(vectors[:, selected], dtype=np.complex128)
    phase = np.vdot(seed, electric)
    if abs(phase) > 1e-30:
        electric *= np.exp(-1j * np.angle(phase))
    ex, ey, ez = np.split(electric, (ex_n, ex_n + ey_n))
    out["Ex"], out["Ey"], out["Ez"] = (
        ex.reshape(nz, ny),
        ey.reshape(nz, ny - 1),
        ez.reshape(nz - 1, ny),
    )

    magnetic = 1j * (curl_e @ electric) / (_MU0 * omega_d * mu)
    hx, hy, hz = np.split(magnetic, (hx_n, hx_n + hy_n))
    out["Hx"], out["Hy"], out["Hz"] = (
        hx.reshape(nz - 1, ny - 1),
        hy.reshape(nz - 1, ny),
        hz.reshape(nz, ny - 1),
    )
    residual = np.linalg.norm(
        curl_h @ ((curl_e @ electric) / mu) - values[selected] * (eps * electric)
    )
    residual /= max(
        np.linalg.norm(values[selected] * (eps * electric)), np.finfo(float).eps
    )
    eigenvalue = float(np.real(values[selected]))
    frequency_ratio = float(np.sqrt(max(eigenvalue / target, 0.0)))
    frequency_ratio_initial = frequency_ratio
    # A single beta correction can leave a percent-level frequency mismatch.
    # Retain the selected branch until the discrete frequency converges.
    if _correct_beta and abs(frequency_ratio - 1.0) > 1e-8:
        q_axis = 2.0 * np.sin(0.5 * float(k_num) * d) / d
        q_probe = 0.98 * q_axis
        delta_probe = -1j * sign * q_probe
        delta_change = delta_probe - delta
        probe_h = (curl_e @ electric + delta_change * (curl_e_beta @ electric)) / mu
        probe_e = curl_h @ probe_h + delta_change * (curl_h_beta @ probe_h)
        denominator = np.vdot(electric, eps * electric)
        eigenvalue_probe = float(np.real(np.vdot(electric, probe_e) / denominator))
        slope = (eigenvalue_probe - eigenvalue) / (q_probe**2 - q_axis**2)
        if not np.isfinite(slope) or abs(slope) <= np.finfo(float).eps:
            slope = 0.5
        q_corrected = np.sqrt(max(q_axis**2 - (eigenvalue - target) / slope, 0.0))
        k_corrected = 2.0 * np.arcsin(np.clip(0.5 * q_corrected * d, -1.0, 1.0)) / d
        if np.isfinite(k_corrected) and k_corrected > 0.0:
            refined, residual, frequency_ratio, k_refined, _ = (
                refine_x_mode_at_fixed_beta(
                    out,
                    indices,
                    component_permittivity=component_permittivity,
                    component_permeability=component_permeability,
                    omega=omega,
                    dt=dt,
                    resolution=resolution,
                    k_num=k_corrected,
                    direction_sign=direction_sign,
                    transverse_cell_widths=transverse_cell_widths,
                    transverse_metrics=transverse_metrics,
                    _correct_beta=_remaining_corrections > 1,
                    _remaining_corrections=_remaining_corrections - 1,
                    _prior_factor=_prior_factor,
                )
            )
            return (
                refined,
                residual,
                frequency_ratio,
                float(k_refined),
                frequency_ratio_initial,
            )
    return out, float(residual), frequency_ratio, float(k_num), frequency_ratio


def _transverse_eigenpairs(
    sparse,
    spla,
    curl_h,
    curl_e,
    eps,
    mu,
    delta,
    seed,
    target,
    shape,
    prior_factor,
):
    """Eliminate longitudinal E using the exact Yee divergence of epsilon E."""
    nz, ny = shape
    ex, hx, hy = nz * ny, (nz - 1) * (ny - 1), (nz - 1) * ny
    if (
        np.any(eps.imag)
        or np.any(mu.imag)
        or np.any(eps.real <= 0)
        or np.any(mu.real <= 0)
    ):
        return None
    transverse = seed[ex:]
    phase = np.exp(-1j * np.angle(transverse[np.argmax(np.abs(transverse))]))
    transverse = transverse * phase
    imaginary = np.hypot(
        np.linalg.norm(transverse.imag), np.linalg.norm((seed[:ex] * phase).real)
    )
    if imaginary > 1e-12 * np.linalg.norm(seed):
        return None
    # The x row of curl(H) contains [-Dz, Dy]. Divergence of this curl
    # vanishes, giving delta*eps_x*Ex = -Dy(eps_y*Ey) - Dz(eps_z*Ez).
    divergence = sparse.hstack(
        (curl_h[:ex, hx + hy :], -curl_h[:ex, hx : hx + hy]), format="csc"
    )
    if abs(delta) <= np.sqrt(np.finfo(float).eps) * np.max(
        np.abs(divergence.data), initial=1
    ):
        return None
    lift = (
        divergence.multiply(eps[ex:])
        .multiply((-1 / (delta * eps[:ex]))[:, None])
        .tocsc()
    )
    ampere = curl_h[ex:, :].multiply(1 / mu).tocsc()
    operator = ampere @ curl_e[:, ex:] + (ampere @ curl_e[:, :ex]) @ lift
    if np.any(operator.data.imag):
        return None
    operator = operator.real.tocsc()
    operator.eliminate_zeros()
    mass = sparse.diags(eps[ex:].real, format="csc")
    try:
        values, vectors = _converged_shift_invert_eigenpairs(
            spla,
            operator,
            mass,
            target=target,
            seed=transverse.real,
            count=min(4, operator.shape[0] - 2),
            transverse_shape=shape,
            prior_factor=prior_factor,
        )
    except RuntimeError:
        prior_factor.clear()
        return None
    full = np.vstack((lift @ vectors, vectors))
    full /= np.linalg.norm(full, axis=0)
    rhs = eps[:, None] * full * values
    residual = np.linalg.norm(curl_h @ ((curl_e @ full) / mu[:, None]) - rhs, axis=0)
    # A tracked vector has already met the strict reduced residual. Check its
    # reconstructed Maxwell residual separately before accepting that shortcut.
    tolerance = 1e-10 if len(values) == 1 else 1e-6
    if np.any(~np.isfinite(residual)) or np.any(
        residual > tolerance * np.linalg.norm(rhs, axis=0)
    ):
        prior_factor.clear()
        return None
    return values, full


def _track_corrected_mode(operator, mass, target, seed, factor):
    """Refine the same polarization after a small change of propagation constant."""
    seed_norm = np.linalg.norm(seed)
    if not np.isfinite(seed_norm) or not seed_norm:
        return None
    reference = seed / seed_norm
    vector = reference
    for _ in range(5):
        applied = operator @ vector
        weighted = mass @ vector
        value = np.vdot(vector, applied) / np.vdot(vector, weighted)
        residual = applied - value * weighted
        relative = np.linalg.norm(residual) / np.linalg.norm(value * weighted)
        if (
            not np.isfinite(relative)
            or relative > 1e-3
            or abs(np.vdot(reference, vector)) < 0.99999
        ):
            break
        if relative < 2e-12 and abs(value / target - 1) < 1e-6:
            return value, vector
        # Project out the resonant component of the preceding shift-invert
        # factor. Only the small transverse change in polarization remains.
        direction = factor @ residual
        vector = vector - (direction - vector * np.vdot(vector, direction))
        vector /= np.linalg.norm(vector)
    return None


def _converged_shift_invert_eigenpairs(
    spla,
    operator,
    mass,
    *,
    target,
    seed,
    count,
    longitudinal_size=0,
    transverse_shape=None,
    prior_factor=None,
):
    """Return numerically valid eigenpairs near ``target``.

    Some ARPACK/SciPy combinations can report converged null-space vectors for
    this singular curl-curl operator even though their generalized eigenpair
    residuals are large. Retrying with a wider search reliably exposes the
    physical modes; independently checking the residual prevents a false mode
    from reaching the refinement validator.
    """
    # At real beta the lossless Yee operator couples Ex to Ey/Ez through an
    # imaginary derivative. Rotating Ex by i makes this sparse problem real,
    # even on a nonuniform grid. Keep complex arithmetic for lossy material
    # or a seed whose polarization requires independent real and imaginary parts.
    phases = None
    if longitudinal_size and not np.any(mass.data.imag):
        phase = np.ones(seed.size, dtype=complex)
        phase[:longitudinal_size] = 1j
        seed_in_basis = phase.conj() * seed
        peak = seed_in_basis[np.argmax(np.abs(seed_in_basis))]
        seed_in_basis *= np.exp(-1j * np.angle(peak))
        if np.linalg.norm(seed_in_basis.imag) <= 1e-12 * np.linalg.norm(seed_in_basis):
            rotated = operator.tocoo(copy=True)
            rotated.data *= phase[rotated.row].conj() * phase[rotated.col]
            if not np.any(rotated.data.imag):
                operator = rotated.real.tocsc()
                mass = mass.real
                seed = seed_in_basis.real
                phases = phase
    real = np.isrealobj(operator.data) and np.isrealobj(seed)
    if prior_factor and prior_factor["OPinv"].shape != operator.shape:
        prior_factor.clear()
    if prior_factor and real:
        tracked = _track_corrected_mode(
            operator, mass, target, seed, prior_factor["OPinv"]
        )
        if tracked is not None:
            value, vector = tracked
            return np.array([value]), (
                phases * vector if phases is not None else vector
            )[:, None]
    if prior_factor:
        prior_factor.clear()
    inverse = {}
    if real and transverse_shape:
        from ._ordering import shift_inverse

        nz, ny = transverse_shape
        components = ((nz, ny - 1), (nz - 1, ny))
        if longitudinal_size:
            components = ((nz, ny), *components)
        inverse = shift_inverse(spla, operator, target, components, mass)
    if prior_factor is not None:
        prior_factor.update(inverse)
    attempts = (count, min(8, operator.shape[0] - 2))
    for candidate_count in dict.fromkeys(attempts):
        eigenpairs = spla.eigs(
            operator,
            M=mass,
            k=candidate_count,
            sigma=target,
            v0=seed,
            **inverse,
            tol=1e-9,  # pyright: ignore[reportArgumentType] -- SciPy infers int from its untyped default.
        )
        values, vectors = cast(tuple[np.ndarray, np.ndarray], eigenpairs)
        valid = []
        for index, value in enumerate(values):
            vector = vectors[:, index]
            right = value * (mass @ vector)
            relative_residual = np.linalg.norm(operator @ vector - right) / max(
                np.linalg.norm(right), np.finfo(float).eps
            )
            if np.isfinite(relative_residual) and relative_residual <= 1e-6:
                valid.append(index)
        if valid:
            selected = vectors[:, valid]
            if phases is not None:
                selected = phases[:, None] * selected
            return values[valid], selected
    raise RuntimeError(
        "fixed-beta mode refinement did not produce a converged eigenpair"
    )


def validate_x_mode_refinement(
    seed_profiles,
    candidate_profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
    omega,
    dt,
    resolution,
    k_num,
    direction_sign,
    transverse_cell_widths=None,
    transverse_metrics=None,
    integration_weights=None,
    minimum_electric_overlap=None,
    minimum_magnetic_overlap=0.5,
    maximum_impedance_change=4.0,
    maximum_energy_change=4.0,
    maximum_energy_imbalance=4.0,
    maximum_maxwell_residual=5e-3,
):
    """Validate that a refined x-normal profile remains the same physical mode.

    The refinement eigen-residual alone only establishes that the selected
    electric vector solves the assembled curl-curl problem.  These checks also
    require mode identity, a consistent E/H balance, forward signed power, and
    both first-order discrete Maxwell equations. Electric overlap remains a
    diagnostic by default because Yee refinement can legitimately redistribute
    longitudinal electric components; callers may opt into a hard threshold.
    """
    from scipy import sparse

    seed = {
        name: np.asarray(value, dtype=np.complex128)
        for name, value in seed_profiles.items()
    }
    candidate = {
        name: np.asarray(value, dtype=np.complex128)
        for name, value in candidate_profiles.items()
    }
    diagnostics: dict[str, Any] = {
        "electric_overlap": _normalized_component_overlap(
            seed, candidate, _E_COMPONENTS
        ),
        "magnetic_overlap": _normalized_component_overlap(
            seed, candidate, _H_COMPONENTS
        ),
    }
    seed_impedance = _rms_impedance(seed)
    candidate_impedance = _rms_impedance(candidate)
    diagnostics["seed_rms_impedance"] = seed_impedance
    diagnostics["candidate_rms_impedance"] = candidate_impedance
    diagnostics["impedance_change"] = _ratio_change(candidate_impedance, seed_impedance)

    eps, mu = _component_material_vectors(
        candidate,
        indices,
        component_permittivity=component_permittivity,
        component_permeability=component_permeability,
    )
    seed_energy_ratio = _energy_ratio(
        seed, eps, mu, integration_weights=integration_weights
    )
    candidate_energy_ratio = _energy_ratio(
        candidate, eps, mu, integration_weights=integration_weights
    )
    diagnostics["seed_energy_ratio"] = seed_energy_ratio
    diagnostics["candidate_energy_ratio"] = candidate_energy_ratio
    diagnostics["energy_ratio_change"] = _ratio_change(
        candidate_energy_ratio, seed_energy_ratio
    )
    diagnostics["candidate_energy_imbalance"] = _ratio_change(
        candidate_energy_ratio, 1.0
    )

    seed_power = _signed_x_power(
        seed,
        resolution=resolution,
        direction_sign=direction_sign,
        integration_weights=integration_weights,
    )
    candidate_power = _signed_x_power(
        candidate,
        resolution=resolution,
        direction_sign=direction_sign,
        integration_weights=integration_weights,
    )
    diagnostics["seed_signed_power"] = seed_power
    diagnostics["candidate_signed_power"] = candidate_power

    sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
    delta = (
        -2j * sign * np.sin(0.5 * float(k_num) * float(resolution)) / float(resolution)
    )
    curl_e, curl_h, _curl_e_beta, _curl_h_beta = _yee_curl_operators(
        sparse,
        candidate,
        resolution=float(resolution),
        delta=delta,
        transverse_cell_widths=transverse_cell_widths,
        transverse_metrics=transverse_metrics,
    )
    omega_d = (
        float(omega)
        if dt is None
        else 2.0 * np.sin(0.5 * float(omega) * float(dt)) / float(dt)
    )
    electric = _component_vector(candidate, _E_COMPONENTS)
    magnetic = _component_vector(candidate, _H_COMPONENTS)
    faraday_left = curl_e @ electric
    faraday_right = -1j * _MU0 * omega_d * mu * magnetic
    ampere_left = curl_h @ magnetic
    ampere_right = 1j * _EPS0 * omega_d * eps * electric
    diagnostics["faraday_residual"] = _relative_pair_residual(
        faraday_left, faraday_right
    )
    diagnostics["ampere_residual"] = _relative_pair_residual(ampere_left, ampere_right)

    rejection_reasons = []
    if minimum_electric_overlap is not None and diagnostics["electric_overlap"] < float(
        minimum_electric_overlap
    ):
        rejection_reasons.append("electric overlap")
    if diagnostics["magnetic_overlap"] < float(minimum_magnetic_overlap):
        rejection_reasons.append("magnetic overlap")
    if diagnostics["impedance_change"] > float(maximum_impedance_change):
        rejection_reasons.append("E/H impedance change")
    if diagnostics["energy_ratio_change"] > float(maximum_energy_change):
        rejection_reasons.append("electric/magnetic energy change")
    if diagnostics["candidate_energy_imbalance"] > float(maximum_energy_imbalance):
        rejection_reasons.append("electric/magnetic energy imbalance")
    if not np.isfinite(candidate_power) or candidate_power <= np.finfo(float).tiny:
        rejection_reasons.append("non-forward signed power")
    if diagnostics["faraday_residual"] > float(maximum_maxwell_residual):
        rejection_reasons.append("Faraday residual")
    if diagnostics["ampere_residual"] > float(maximum_maxwell_residual):
        rejection_reasons.append("Ampere residual")
    diagnostics["accepted"] = not rejection_reasons
    diagnostics["rejection_reason"] = ", ".join(rejection_reasons)
    return not rejection_reasons, diagnostics


def _yee_curl_operators(
    sparse,
    profiles,
    *,
    resolution,
    delta,
    transverse_cell_widths=None,
    transverse_metrics=None,
):
    nz, ny = profiles["Ex"].shape
    if transverse_metrics is not None:
        operators = []
        for count, (primal, dual) in zip((nz, ny), transverse_metrics, strict=True):
            primal, dual = np.asarray(primal), np.asarray(dual)
            if primal.shape != (count - 1,) or dual.shape != (count,):
                raise ValueError(
                    "Tensor curl metrics must match the native Yee supports"
                )
            incidence = _forward_difference(sparse, count, 1.0)
            operators.append(
                (
                    sparse.diags(1.0 / primal) @ incidence,
                    -sparse.diags(1.0 / dual) @ incidence.T,
                )
            )
        (dz, bz), (dy, by) = operators
    elif transverse_cell_widths is None:
        dz = _forward_difference(sparse, nz, resolution)
        dy = _forward_difference(sparse, ny, resolution)
        bz, by = -dz.T, -dy.T
    else:
        widths_z, widths_y = (
            np.asarray(values, dtype=float).reshape(-1)
            for values in transverse_cell_widths
        )
        if widths_z.size != nz or widths_y.size != ny:
            raise ValueError(
                "Rectilinear transverse widths must match the x-normal mode plane"
            )
        dz = _center_to_edge_difference(sparse, widths_z)
        dy = _center_to_edge_difference(sparse, widths_y)
        bz = _edge_to_center_difference(sparse, widths_z)
        by = _edge_to_center_difference(sparse, widths_y)
    iz = sparse.eye(nz, format="csc")
    iy = sparse.eye(ny, format="csc")
    izm = sparse.eye(nz - 1, format="csc")
    iym = sparse.eye(ny - 1, format="csc")
    dy_ez = sparse.kron(izm, dy, format="csc")
    dz_ey = sparse.kron(dz, iym, format="csc")
    dz_ex = sparse.kron(dz, iy, format="csc")
    dy_ex = sparse.kron(iz, dy, format="csc")
    by_hz = sparse.kron(iz, by, format="csc")
    bz_hy = sparse.kron(bz, iy, format="csc")
    bz_hx = sparse.kron(bz, iym, format="csc")
    by_hx = sparse.kron(izm, by, format="csc")
    ex_n, ey_n, ez_n = (profiles[name].size for name in _E_COMPONENTS)
    hx_n, hy_n, hz_n = (profiles[name].size for name in _H_COMPONENTS)
    z = sparse.csc_matrix
    curl_e = sparse.bmat(
        [
            [z((hx_n, ex_n)), -dz_ey, dy_ez],
            [dz_ex, z((hy_n, ey_n)), -delta * sparse.eye(ez_n)],
            [-dy_ex, delta * sparse.eye(ey_n), z((hz_n, ez_n))],
        ],
        format="csc",
    )
    curl_h = sparse.bmat(
        [
            [z((ex_n, hx_n)), -bz_hy, by_hz],
            [bz_hx, z((ey_n, hy_n)), -delta * sparse.eye(hz_n)],
            [-by_hx, delta * sparse.eye(hy_n), z((ez_n, hz_n))],
        ],
        format="csc",
    )
    curl_e_beta = sparse.bmat(
        [
            [z((hx_n, ex_n)), z((hx_n, ey_n)), z((hx_n, ez_n))],
            [z((hy_n, ex_n)), z((hy_n, ey_n)), -sparse.eye(ez_n)],
            [z((hz_n, ex_n)), sparse.eye(ey_n), z((hz_n, ez_n))],
        ],
        format="csc",
    )
    curl_h_beta = sparse.bmat(
        [
            [z((ex_n, hx_n)), z((ex_n, hy_n)), z((ex_n, hz_n))],
            [z((ey_n, hx_n)), z((ey_n, hy_n)), -sparse.eye(hz_n)],
            [z((ez_n, hx_n)), sparse.eye(hy_n), z((ez_n, hz_n))],
        ],
        format="csc",
    )
    return curl_e, curl_h, curl_e_beta, curl_h_beta


def _component_material_vectors(
    profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
):
    eps = np.concatenate(
        [
            _material(component_permittivity[name], indices[name], profiles[name].shape)
            for name in _E_COMPONENTS
        ]
    )
    mu = np.concatenate(
        [
            _material(component_permeability[name], indices[name], profiles[name].shape)
            for name in _H_COMPONENTS
        ]
    )
    return eps, mu


def _component_vector(profiles, names):
    return np.concatenate(
        [np.asarray(profiles[name], dtype=np.complex128).reshape(-1) for name in names]
    )


def _normalized_component_overlap(seed, candidate, names):
    left = _component_vector(seed, names)
    right = _component_vector(candidate, names)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return 0.0
    return float(abs(np.vdot(left, right)) / denominator)


def _rms_impedance(profiles):
    electric_norm = np.linalg.norm(_component_vector(profiles, _E_COMPONENTS))
    magnetic_norm = np.linalg.norm(_component_vector(profiles, _H_COMPONENTS))
    if (
        not np.isfinite(electric_norm)
        or not np.isfinite(magnetic_norm)
        or magnetic_norm <= np.finfo(float).tiny
    ):
        return np.inf
    return float(electric_norm / magnetic_norm)


def _energy_ratio(profiles, eps, mu, *, integration_weights=None):
    electric = _component_vector(profiles, _E_COMPONENTS)
    magnetic = _component_vector(profiles, _H_COMPONENTS)
    electric_weights = _component_weight_vector(
        profiles, _E_COMPONENTS, integration_weights
    )
    magnetic_weights = _component_weight_vector(
        profiles, _H_COMPONENTS, integration_weights
    )
    electric_energy = _EPS0 * float(
        np.real(np.vdot(electric, electric_weights * eps * electric))
    )
    magnetic_energy = _MU0 * float(
        np.real(np.vdot(magnetic, magnetic_weights * mu * magnetic))
    )
    if (
        not np.isfinite(electric_energy)
        or not np.isfinite(magnetic_energy)
        or magnetic_energy <= np.finfo(float).tiny
    ):
        return np.inf
    return float(electric_energy / magnetic_energy)


def _ratio_change(candidate, seed):
    if (
        not np.isfinite(candidate)
        or not np.isfinite(seed)
        or min(candidate, seed) <= np.finfo(float).tiny
    ):
        return np.inf
    return float(max(candidate / seed, seed / candidate))


def _signed_x_power(profiles, *, resolution, direction_sign, integration_weights=None):
    if integration_weights is None:
        ey_weights = ez_weights = float(resolution) ** 2
    else:
        ey_weights = np.asarray(integration_weights["Ey"], dtype=float).reshape(-1)
        ez_weights = np.asarray(integration_weights["Ez"], dtype=float).reshape(-1)
    flux = np.vdot(
        profiles["Hz"].reshape(-1),
        ey_weights * profiles["Ey"].reshape(-1),
    )
    flux -= np.vdot(
        profiles["Hy"].reshape(-1),
        ez_weights * profiles["Ez"].reshape(-1),
    )
    return float(0.5 * float(direction_sign) * np.real(flux))


def _component_weight_vector(profiles, names, integration_weights):
    if integration_weights is None:
        return np.ones(sum(np.asarray(profiles[name]).size for name in names))
    return np.concatenate(
        [
            np.asarray(integration_weights[name], dtype=float).reshape(-1)
            for name in names
        ]
    )


def _relative_pair_residual(left, right):
    denominator = max(np.linalg.norm(left), np.linalg.norm(right), np.finfo(float).eps)
    return float(np.linalg.norm(left - right) / denominator)


def _material(values, index, shape):
    arr = np.asarray(values)
    profile = arr[index] if arr.ndim == 3 else arr
    profile = np.asarray(profile, dtype=np.complex128).squeeze()
    if profile.shape != shape:
        raise ValueError(f"Material shape {profile.shape} does not match {shape}")
    return profile.reshape(-1)


def _forward_difference(sparse, count, spacing):
    rows = np.repeat(np.arange(count - 1), 2)
    cols = np.column_stack((np.arange(count - 1), np.arange(1, count))).reshape(-1)
    data = np.tile(np.asarray([-1.0, 1.0]) / spacing, count - 1)
    return sparse.csc_matrix((data, (rows, cols)), shape=(count - 1, count))


def _center_to_edge_difference(sparse, cell_widths):
    widths = np.asarray(cell_widths, dtype=float).reshape(-1)
    spacing = 0.5 * (widths[:-1] + widths[1:])
    rows = np.repeat(np.arange(widths.size - 1), 2)
    cols = np.column_stack(
        (np.arange(widths.size - 1), np.arange(1, widths.size))
    ).reshape(-1)
    data = np.column_stack((-1.0 / spacing, 1.0 / spacing)).reshape(-1)
    return sparse.csc_matrix((data, (rows, cols)), shape=(widths.size - 1, widths.size))


def _edge_to_center_difference(sparse, cell_widths):
    widths = np.asarray(cell_widths, dtype=float).reshape(-1)
    rows = np.concatenate(
        (
            np.asarray([0]),
            np.repeat(np.arange(1, widths.size - 1), 2),
            np.asarray([widths.size - 1]),
        )
    )
    cols = np.concatenate(
        (
            np.asarray([0]),
            np.column_stack(
                (np.arange(widths.size - 2), np.arange(1, widths.size - 1))
            ).reshape(-1),
            np.asarray([widths.size - 2]),
        )
    )
    data = np.concatenate(
        (
            np.asarray([1.0 / widths[0]]),
            np.column_stack(
                (
                    -1.0 / widths[1:-1],
                    1.0 / widths[1:-1],
                )
            ).reshape(-1),
            np.asarray([-1.0 / widths[-1]]),
        )
    )
    return sparse.csc_matrix((data, (rows, cols)), shape=(widths.size, widths.size - 1))
