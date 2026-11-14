"""Fourier reduction of the lossless full-tensor Yee electric update.

The normal-axis symbol of node/edge averaging is cos(beta dx/2), while
that of a centered derivative is -2i sin(beta dx/2)/dx. Keeping both is
necessary: replacing the constitutive operator by a cell tensor loses the
normal averaging even in a spatially homogeneous anisotropic dielectric.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.constitutive import electric_tensor_coefficients
from beamz.devices.modes.backend._yee import _yee_curl_operators
from beamz.devices.modes.backend._yee import (
    transverse_grid_metrics as transverse_metrics,
)


def _average(count):
    rows = np.repeat(np.arange(count - 1), 2)
    columns = np.column_stack((np.arange(count - 1), np.arange(1, count))).reshape(-1)
    return sparse.csc_matrix(
        (np.full(rows.size, 0.5), (rows, columns)), shape=(count - 1, count)
    )


def _extend(count, weights=None):
    """Cell-to-node averaging with the same edge extension as the timestep."""
    if weights is not None:
        left, right = weights
        columns = np.arange(count - 1)
        return sparse.csc_matrix(
            (
                np.concatenate((right[:-1], left[1:])),
                (np.concatenate((columns, columns + 1)), np.tile(columns, 2)),
            ),
            shape=(count, count - 1),
        )
    avg = _average(count).T.tolil()
    avg[0, 0] = 1.0
    avg[-1, -1] = 1.0
    return avg.tocsc()


def inverse_operator(profiles, indices, tensors, *, beta, spacing, grid=None):
    """Assemble E <- inverse-epsilon curl(H) on an x-normal cross section."""
    diagonal, cross = electric_tensor_coefficients(tensors)
    nz, ny = profiles["Ex"].shape
    averages = (None, None) if grid is None else transverse_metrics(indices, grid)[1]
    c = np.cos(0.5 * beta * spacing)
    to_node = (
        c * sparse.eye(nz * ny),
        sparse.kron(sparse.eye(nz), _extend(ny, averages[1])),
        sparse.kron(_extend(nz, averages[0]), sparse.eye(ny)),
    )
    from_node = (
        c * sparse.eye(nz * ny),
        sparse.kron(sparse.eye(nz), _average(ny)),
        sparse.kron(_average(nz), sparse.eye(ny)),
    )
    # At a longitudinally invariant port any normal node has the same tensor.
    node_index = indices["Ex"]
    node_cross = np.asarray(cross[node_index])
    if node_cross.shape != (nz, ny, 3):
        raise ValueError("Tensor node context does not cover the mode aperture")
    pair = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    blocks = []
    for row, component in enumerate(("Ex", "Ey", "Ez")):
        line = []
        for col in range(3):
            if row == col:
                values = np.asarray(diagonal[row][indices[component]])
                if values.shape != profiles[component].shape:
                    raise ValueError(
                        "Tensor electric support does not cover the mode aperture"
                    )
                line.append(sparse.diags(values.reshape(-1)))
            else:
                values = node_cross[..., pair[(min(row, col), max(row, col))]].reshape(
                    -1
                )
                line.append(from_node[row] @ sparse.diags(values) @ to_node[col])
        blocks.append(line)
    return sparse.bmat(blocks, format="csc")


def refine_tensor_mode(
    profiles, indices, tensors, *, omega, dt, spacing, beta, direction_sign, grid=None
):
    """Match the prescribed discrete frequency, tracking the seed by overlap.

    Failure is explicit; a failed coupled solve must never return its continuum
    seed as an allegedly propagation-consistent port mode.
    """
    shapes = {k: np.asarray(v).shape for k, v in profiles.items()}
    electric_names, magnetic_names = ("Ex", "Ey", "Ez"), ("Hx", "Hy", "Hz")
    seed = np.concatenate([profiles[k].reshape(-1) for k in electric_names])
    omega_d = 2 * np.sin(0.5 * omega * dt) / dt
    target = (omega_d / LIGHT_SPEED) ** 2
    sign = float(direction_sign)
    metrics = None if grid is None else transverse_metrics(indices, grid)[0]

    def solve(k, initial):
        delta = -2j * sign * np.sin(0.5 * k * spacing) / spacing
        ce, ch, _, _ = _yee_curl_operators(
            sparse,
            profiles,
            resolution=spacing,
            delta=delta,
            transverse_metrics=metrics,
        )
        inv = inverse_operator(
            profiles, indices, tensors, beta=k, spacing=spacing, grid=grid
        )
        op = inv @ ch @ ce
        count = min(6, op.shape[0] - 2)
        values, vectors = cast(
            tuple[np.ndarray, np.ndarray],
            spla.eigs(
                op / target,
                k=count,
                sigma=1.00001,
                v0=initial,
                tol=1e-10,  # pyright: ignore[reportArgumentType] -- SciPy's untyped default.
                return_eigenvectors=True,
            ),
        )
        overlaps = np.abs(initial.conj() @ vectors) / np.linalg.norm(vectors, axis=0)
        # The curl-curl operator has a large static nullspace. Its longitudinal
        # vectors may overlap a continuum seed but cannot represent a mode at
        # the requested nonzero frequency. Never track one as a port branch.
        physical = (np.real(values) > 1e-6) & (np.abs(np.imag(values)) < 1e-7)
        if not np.any(physical):
            raise ValueError(
                "Tensor mode search found no finite-frequency propagating eigenpair"
            )
        selected = int(np.argmax(np.where(physical, overlaps, -np.inf)))
        value, electric = values[selected], vectors[:, selected]
        residual = np.linalg.norm(op @ electric - value * target * electric) / (
            target * np.linalg.norm(electric)
        )
        if residual > 1e-7 or abs(value.imag) > 1e-7:
            raise ValueError(
                f"Tensor mode eigenpair failed residual check: {residual:.3g}"
            )
        return float(value.real - 1), electric, ce, ch, inv

    k = abs(float(beta))
    previous = None
    for _iteration in range(15):
        error, electric, ce, ch, inv = solve(k, seed)
        if abs(error) < 2e-8:
            break
        if previous is None:
            probe = k * 1.001
            probe_error, *_ = solve(probe, electric)
            slope = (probe_error - error) / (probe - k)
        else:
            slope = (error - previous[1]) / (k - previous[0])
        if not np.isfinite(slope) or abs(slope * k) < 1e-9:
            raise ValueError(
                "Tensor mode propagation-constant solve has a singular slope"
            )
        previous = (k, error)
        k -= np.clip(error / slope, -0.1 * k, 0.1 * k)
        if not 0 < k * spacing < np.pi:
            raise ValueError(
                "Tensor mode lies outside the supported propagating branch"
            )
        seed = electric
    else:
        raise ValueError(
            f"Tensor mode frequency mismatch did not converge: {error:.3g}"
        )
    magnetic = 1j * (ce @ electric) / (MU_0 * omega_d)
    phase = np.angle(magnetic[np.argmax(np.abs(magnetic))])
    electric *= np.exp(-1j * phase)
    magnetic *= np.exp(-1j * phase)
    out = {}
    for names, vector in ((electric_names, electric), (magnetic_names, magnetic)):
        start = 0
        for name in names:
            size = np.prod(shapes[name])
            out[name] = vector[start : start + size].reshape(shapes[name])
            start += size
    # Both first-order equations, rather than only a second-order eigen-residual.
    faraday = ce @ electric + 1j * MU_0 * omega_d * magnetic
    ampere = inv @ (ch @ magnetic) - 1j * EPS_0 * omega_d * electric
    diagnostics = {
        "tensor_frequency_error": abs(error),
        "tensor_iterations": _iteration + 1,
        "tensor_faraday_residual": float(
            np.linalg.norm(faraday) / np.linalg.norm(ce @ electric)
        ),
        "tensor_ampere_residual": float(
            np.linalg.norm(ampere) / (EPS_0 * omega_d * np.linalg.norm(electric))
        ),
    }
    if (
        max(
            diagnostics["tensor_faraday_residual"],
            diagnostics["tensor_ampere_residual"],
        )
        > 1e-7
    ):
        raise ValueError(
            f"Tensor mode failed first-order Maxwell validation: {diagnostics}"
        )
    return out, k, diagnostics
