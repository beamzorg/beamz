"""Timestep stability of tensor-pole media: a von Neumann analysis of the scheme.

The Yee leapfrog coupled to the pole recurrences of
:mod:`beamz.simulation.anisotropic` (Lorentz and Drude), their CCPR ``E'``
extension (:mod:`beamz.simulation.ccpr`) and gyrotropic poles is a linear
recurrence. In the bulk medium each plane wave evolves independently, so the
scheme is stable exactly when the per-step amplification matrix of every
sampled wave vector has spectral radius at most one. The matrix models the
implemented update, not its continuous limit:

* static operators act through the shared-support collocation, which gives the
  off-diagonal entries the factors ``cos(k_j h_j / 2)``;
* the field update applies the pointwise inverse of ``eps_inf`` (``eps'`` with
  CCPR poles), collocated the same way;
* in 2D only the simulated polarization is kept.

The analytic bound ``S^2 + (Omega dt / 2)^2 < 1`` (in-medium Courant number
``S``, fastest coupled pole frequency ``Omega``) neither holds for every
multi-pole anisotropic medium nor applies once ``B != 0``, so it is not used.
The analysis depends on the medium and ``dt`` only, not on the grid size, and
is cached.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from beamz.const import LIGHT_SPEED

_STABLE = 1.0 + 1e-6  # neutral (static and Drude) modes sit on the unit circle


def _collocated(tensor, c):
    """Bulk symbol of a shared-support operator: cross terms scale by c_r c_c.

    ``tensor`` (3, 3) or (samples, 3, 3); ``c`` (samples, 3) of ``cos(k h / 2)``.
    """
    out = tensor * (c[..., :, None] * c[..., None, :])
    index = np.arange(3)
    out[..., index, index] = tensor[..., index, index]
    return out


def _samples(active: int, count: int = 9) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, count)
    axes = [grid] * active + [np.zeros(1)] * (3 - active)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _statics(medium, family):
    """(own static tensor, other family's static tensor, poles) as packed tuples."""
    eps = tuple(float(v) for v in np.asarray(medium.epsilon_r))
    mu = tuple(float(v) for v in np.asarray(medium.mu_r))
    if family == "E":
        return eps, mu, tuple(medium.electric_poles)
    return mu, eps, tuple(medium.magnetic_poles)


def _pole_update(pole, step, cos):
    """Pole recurrence ``p' = C1 p + C2 q + F E + G change`` per wave vector.

    ``step`` is ``c dt`` in metres; ``change`` is ``E^{n+1} - E^n``. Returns
    the four (samples, 3, 3) operators.
    """
    from beamz.design.anisotropic import GyrotropicPole, _cross_matrix

    c = LIGHT_SPEED
    eye = np.broadcast_to(np.eye(3), (len(cos), 3, 3))
    half = 0.5 * pole.damping / c * step
    resonance = (pole.resonance / c * step) ** 2
    weight = _collocated(pole.weight_matrix, cos) * step**2 / c**2
    if isinstance(pole, GyrotropicPole):
        # (1 + h - R / 2) j' = (1 - h + R / 2) j + w F - (w0 dt)^2 p with
        # j = p - q and the precession R reaching across the shared support.
        rotation = _collocated(_cross_matrix(pole.gyration), cos) * (step / c)
        implicit = np.linalg.inv((1.0 + half) * eye - 0.5 * rotation)
        explicit = (1.0 - half) * eye + 0.5 * rotation
        c1 = eye + implicit @ (explicit - resonance * eye)
        return c1, -implicit @ explicit, implicit @ weight, np.zeros_like(c1)
    d = 1.0 + half
    c1 = (2.0 - resonance) / d * eye
    c2 = -(1.0 - half) / d * eye
    edot = _collocated(pole.edot_matrix, cos)
    return c1, c2, weight / d, edot * (step / (c * d))


def _implicit_static(static, poles, step) -> np.ndarray:
    """``eps' = eps_inf + sum_p (dt / d_p) B_p`` over the tensor poles."""
    from beamz.design.anisotropic import TensorPole, packed_to_matrix

    out = packed_to_matrix(np.asarray(static))
    for pole in poles:
        if isinstance(pole, TensorPole) and pole.has_edot:
            d = 1.0 + 0.5 * pole.damping / LIGHT_SPEED * step
            out = out + pole.edot_matrix * step / (LIGHT_SPEED * d)
    return out


@lru_cache(maxsize=512)
def _radius(static, other, poles, family, dt, spacings, polarization) -> float:
    from beamz.design.anisotropic import GyrotropicPole, packed_to_matrix

    step = LIGHT_SPEED * dt  # time in metres
    h = np.asarray(spacings, dtype=np.float64)
    s = _samples(len(h))
    phase = 0.5 * np.pi * s[:, : len(h)]
    kappa = np.zeros((len(s), 3))
    cos = np.ones((len(s), 3))
    kappa[:, : len(h)] = 2.0 / h * np.sin(phase)
    cos[:, : len(h)] = np.cos(phase)
    x, y, z = kappa.T
    zero = np.zeros_like(x)
    curl = 1j * np.stack(
        [
            np.stack([zero, -z, y], -1),
            np.stack([z, zero, -x], -1),
            np.stack([-y, x, zero], -1),
        ],
        -2,
    )
    eye = np.broadcast_to(np.eye(3), (len(s), 3, 3))

    def inverse(tensor):
        return _collocated(np.linalg.inv(tensor), cos)

    # Tensor poles solve with eps' (eps_inf with CCPR poles); gyrotropic
    # regions apply their increment afterwards through eps_inf alone.
    solve = inverse(_implicit_static(static, poles, step))
    plain = inverse(packed_to_matrix(np.asarray(static)))
    other_m = inverse(packed_to_matrix(np.asarray(other)))
    count = len(poles)
    size = 6 + 6 * count
    current = [slice(6 + 3 * i, 9 + 3 * i) for i in range(count)]
    previous = [slice(6 + 3 * (count + i), 9 + 3 * (count + i)) for i in range(count)]
    updates = [_pole_update(p, step, cos) for p in poles]

    # Explicit pole increments delta_i = p_i* - p_i as rows over the state.
    deltas = []
    for (c1, c2, drive, _), p, q in zip(updates, current, previous, strict=True):
        row = np.zeros((len(s), 3, size), dtype=complex)
        row[:, :, 0:3] = drive
        row[:, :, p] = c1 - eye
        row[:, :, q] = c2
        deltas.append(row)
    gyrotropic = [isinstance(p, GyrotropicPole) for p in poles]
    tensor_delta = sum(
        (d for d, g in zip(deltas, gyrotropic, strict=True) if not g),
        np.zeros((len(s), 3, size), dtype=complex),
    )
    # E^{n+1} - E^n before the gyrotropic correction: the change every CCPR
    # pole's implicit B term sees.
    change = np.zeros((len(s), 3, size), dtype=complex)
    change[:, :, 0:3] = -(step**2) * solve @ curl @ other_m @ curl
    change[:, :, 3:6] = step * solve @ curl
    change -= solve @ tensor_delta
    matrix = np.zeros((len(s), size, size), dtype=complex)
    matrix[:, 0:3, :] = change
    matrix[:, 0:3, 0:3] += eye
    for d, g in zip(deltas, gyrotropic, strict=True):
        if g:
            matrix[:, 0:3, :] -= plain @ d
    matrix[:, 3:6, 0:3] = -step * other_m @ curl
    matrix[:, 3:6, 3:6] = eye
    for (_, _, _, edot), delta, p, q in zip(
        updates, deltas, current, previous, strict=True
    ):
        matrix[:, p, :] = delta + edot @ change
        matrix[:, p, p] += eye
        matrix[:, q, p] = eye
    if polarization is not None:
        # In 2D the TE and TM systems decouple; keep the simulated one. The
        # family's own components are in-plane for E/TE and H/TM.
        in_plane = (polarization == "te") == (family == "E")
        own = (0, 1) if in_plane else (2,)
        other_axes = (2,) if in_plane else (0, 1)
        keep = [*own, *(3 + a for a in other_axes)]
        keep += [6 + 3 * i + a for i in range(2 * count) for a in own]
        matrix = matrix[:, keep][:, :, keep]
    return float(np.abs(np.linalg.eigvals(matrix)).max())


def amplification_radius(medium, family, dt, spacings, polarization=None) -> float:
    """Largest amplification factor per step of the scheme in the bulk medium.

    Wave vectors sample the Brillouin zone of the active axes (``spacings`` in
    m). With two axes, ``polarization`` (``"te"`` or ``"tm"``) keeps only the
    simulated field components (both when None). Material losses are omitted,
    which only removes damping. Neutral modes (static fields, Drude) sit at 1.
    """
    static, other, poles = _statics(medium, family)
    return _radius(
        static,
        other,
        poles,
        family,
        float(dt),
        tuple(float(h) for h in spacings),
        polarization,
    )


def _courant_rate(medium, spacings) -> float:
    """(S / dt)^2 for the in-medium Courant number S (CFL at S = 1)."""
    from beamz.design.anisotropic import packed_to_matrix

    eps_floor = np.linalg.eigvalsh(packed_to_matrix(np.asarray(medium.epsilon_r)))[0]
    mu_floor = np.linalg.eigvalsh(packed_to_matrix(np.asarray(medium.mu_r)))[0]
    return float(
        LIGHT_SPEED**2 * sum(1.0 / h**2 for h in spacings) / (eps_floor * mu_floor)
    )


def max_stable_timestep(medium, family, spacings, polarization=None) -> float:
    """Largest stable dt (bisection on :func:`amplification_radius`).

    Bounded above by the in-medium CFL limit of the static tensors.
    """

    def stable(dt):
        radius = amplification_radius(medium, family, dt, spacings, polarization)
        return radius <= _STABLE

    upper = float(1.0 / np.sqrt(_courant_rate(medium, spacings))) * (1.0 - 1e-9)
    if stable(upper):
        return upper
    lower = 0.0
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        if stable(middle):
            lower = middle
        else:
            upper = middle
        if upper - lower < 1e-4 * upper:
            break
    return lower


def check_stability(medium, family, dt, grid) -> None:
    """Reject a timestep at which a medium's pole scheme grows in the bulk."""
    from beamz.design.anisotropic import TensorPole

    static, _, poles = _statics(medium, family)
    step = LIGHT_SPEED * dt
    if np.linalg.eigvalsh(_implicit_static(static, poles, step))[0] <= 0.0:
        raise ValueError(
            "The implicit CCPR update divides by eps_inf + sum_p dt/(1 + gamma_p "
            f"dt/2) B_p, which is not positive definite at dt={dt:.3e} s. "
            "Reduce the timestep or refit the medium."
        )
    material_grid = grid.material_grid
    active = len(material_grid.shape)
    spacings = tuple(grid.geometry.min_spacings[:active])
    polarization = material_grid.polarization if active == 2 else None
    radius = amplification_radius(medium, family, dt, spacings, polarization)
    if radius <= _STABLE:
        return
    limit = max_stable_timestep(medium, family, spacings, polarization)
    hint = ""
    # Pole frequency that would leave a margin at this dt, if the in-medium
    # Courant number alone is still stable; folding slows Lorentz/Drude poles.
    headroom = 1.0 - _courant_rate(medium, spacings) * dt**2
    if headroom > 0.0 and any(
        isinstance(p, TensorPole) and p.edot is None for p in poles
    ):
        hint = (
            " For poles far above the band of interest, medium.fold_fast_poles("
            f"{1.8 * np.sqrt(headroom) / dt:.3e}) slows them enough for this dt."
        )
    raise ValueError(
        f"Tensor poles are unstable at dt={dt:.3e} s (amplification "
        f"{radius:.6f} per step); this medium needs dt < {limit:.3e} s." + hint
    )
