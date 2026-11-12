"""Tensor CCPR poles: the implicit ``B E'`` coupling around the tensor-pole update.

A complex-conjugate pole-residue (CCPR) pole adds a field-derivative drive to
the tensor-pole recurrence of :mod:`beamz.simulation.anisotropic`,

    p'' + gamma p' + omega0^2 p = A E + B E',

discretized like fdtdx's diagonal CCPR update, now with full tensors::

    p^{n+1} = c1 p^n + c2 p^{n-1} + (dt^2 / d) A E^n + (dt / d) B (E^{n+1} - E^n),

with ``d = 1 + gamma dt / 2``. The last term is implicit. Substituting it into
Ampere's law gives, with ``K = sum_p (dt / d_p) B_p`` applied with the same
fraction weighting and Yee collocation as the drive,

    (eps_inf + beta + K) (E^{n+1} - E^n) = X - 2 beta E^n - delta,

where ``X`` is the curl (and source) term, ``beta = sigma dt / (2 eps0)`` and
``delta = sum_p (c1 p + c2 p_prev + (dt^2 / d) A E^n - p^n)`` is the usual
Lorentz increment. ``K`` is symmetric, so ``eps' = eps_inf + K`` is a static
symmetric tensor on the existing constitutive supports. Two equivalent
realizations are used:

* Diagonal kernel (no off-diagonal static permittivity, and every magnetic
  family). The unchanged kernel already produced ``E'`` with
  ``(eps_inf + beta)(E' - E^n) = X - 2 beta E^n``, so the correction recovers
  ``X`` locally::

      E^{n+1} = E^n + Inv_{eps'+beta}((eps_inf + beta)(E' - E^n) - delta).

* Full-tensor kernel. The kernel's own inverse tensors are built from
  ``eps'`` at compile time (:func:`electric_kernel_tensors`), so the ordinary
  correction ``E^{n+1} = E' - Inv_{eps'}(delta)`` applies.

Both give ``E^{n+1} - E^n = Inv_{eps'+beta}(X - 2 beta E^n - delta)``. Then
every pole gains ``(dt / d_p) B_p (E^{n+1} - E^n)``. Gyrotropic poles are
unaffected. Stability is checked with a von Neumann analysis of exactly this
scheme (:mod:`beamz.simulation.pole_stability`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import numpy as np

from beamz.constitutive import (
    expand_packed,
    inverse_permittivity_components,
    lossy_tensor_inverse,
)
from beamz.simulation.anisotropic import (
    InverseOperator,
    TensorPolarizationRegion,
    _active_components,
    _axis_index,
    _cell_bounds,
    _collocate,
    _crop,
    _crop_inverse,
    _crop_or_fill,
    _make_box,
    _material_seams,
    _register_fields,
    _scatter,
    _use_native,
    apply_tensor_operator,
    electric_losses,
    magnetic_losses,
    tensor_pole_weights,
)


def has_edot(poles) -> bool:
    """Whether any pole couples to the field's time derivative (``B != 0``)."""
    return any(getattr(p, "has_edot", False) for p in poles)


@dataclass(frozen=True)
class CCPRPolarizationRegion(TensorPolarizationRegion):
    """Tensor poles with an ``E'`` coupling (see the module docstring)."""

    edot_diagonal: tuple = ()  # per component (poles, *site): (dt / d) B weights
    edot_offdiagonal: jax.Array | None = None  # (poles, *shared, 3)
    static: tuple | None = None  # per component eps_inf + beta, or None (full kernel)


_register_fields(
    CCPRPolarizationRegion,
    tuple(f.name for f in fields(CCPRPolarizationRegion)),
)


# ---------------------------------------------------------------------------
# Compilation


def _edot_drive(poles, dt) -> np.ndarray:
    """Packed ``(dt / (1 + gamma dt / 2)) B`` per pole, shape (poles, 6)."""
    damping = np.array([p.damping for p in poles], dtype=np.float64)
    factor = dt / (1.0 + 0.5 * damping * dt)
    return np.stack([p.edot_packed for p in poles]) * factor[:, None]


def _edot_tensors(fraction, drive, box):
    """Packed ``K = sum_p drive_p`` at each site and at the shared support.

    The site weights are exactly those :func:`tensor_pole_weights` gives the
    ``B`` operator, so ``K`` restricted to the diagonals and the shared cross
    terms is that operator; the remaining entries are the medium's tensor at
    the same point, which the pointwise inverse needs.
    """
    coupled = bool(np.any(drive[:, 3:] != 0.0)) and len(box.components) > 1
    unit = np.array([[1.0, 1.0, 1.0] + [float(coupled)] * 3])
    sites, cross = tensor_pole_weights(fraction, unit, box)
    if cross is not None:
        shared = cross[0, ..., 0]
    elif box.family == "H":
        shared = fraction
    else:
        shared = _collocate(
            fraction, "Cell", "Node", box.shared_shape, box.widths, _material_seams(box)
        )
    total = drive.sum(axis=0)

    def spread(weight):
        weight = np.asarray(weight, dtype=np.float64)
        return total.reshape((6,) + (1,) * weight.ndim) * weight[None]

    return tuple(spread(site[0]) for site in sites), spread(shared)


def _static_sites(grid, box, dt) -> tuple:
    """``eps_inf + beta`` (or ``mu_inf + beta``) at each box site, float64."""
    out = []
    if box.family == "E":
        losses = electric_losses(grid, box.components, dt)
        for component, part, shape in zip(
            box.components, box.slices, box.shapes, strict=True
        ):
            eps = _crop_or_fill(
                getattr(grid, "eps_" + component[-1].lower()), part, shape
            )
            out.append(eps + _crop_or_fill(losses[component], part, shape))
        return tuple(out)
    tensors = grid.material_grid.magnetic_tensors
    losses = magnetic_losses(grid, box.components, dt)
    for component, part, shape in zip(
        box.components, box.slices, box.shapes, strict=True
    ):
        name = "mu_h" + component[-1].lower()
        mu = (
            expand_packed(tensors[name])[_axis_index(component)][part]
            if name in tensors
            else np.ones(shape)
        )
        out.append(mu + _crop_or_fill(losses[component], part, shape))
    return tuple(out)


def _require_positive_definite(tensors, what: str) -> None:
    """Reject packed ``(6, ...)`` tensors with a nonpositive eigenvalue."""
    from beamz.design.anisotropic import packed_to_matrix

    for tensor in tensors:
        matrix = packed_to_matrix(np.moveaxis(np.asarray(tensor), 0, -1))
        if np.linalg.eigvalsh(matrix.reshape(-1, 3, 3))[:, 0].min() <= 0.0:
            raise ValueError(
                f"The implicit CCPR update divides by {what} + sum_p dt/(1 + "
                "gamma_p dt/2) B_p, which is not positive definite here; the "
                "update is unstable. Reduce the timestep or refit the medium."
            )


def _diagonal_kernel_inverse(static, sites, node, box) -> InverseOperator:
    """``Inv(S + K)`` on the box for a diagonal kernel with statics ``S``."""
    seams = _material_seams(box)
    what = "eps_inf" if box.family == "E" else "mu_inf"
    axes = [_axis_index(c) for c in box.components]
    if len(axes) == 1 or not np.any(np.asarray(node)[3:] != 0.0):
        diagonal = [s + k[a] for s, k, a in zip(static, sites, axes, strict=True)]
        _require_positive_definite(
            [np.stack([d, d, d, *(np.zeros_like(d),) * 3]) for d in diagonal], what
        )
        return InverseOperator(
            tuple(jnp.asarray(1.0 / d, dtype=jnp.float32) for d in diagonal), None
        )

    def collocate(value, source, target, shape):
        return np.asarray(
            _collocate(jnp.asarray(value), source, target, shape, box.widths, seams),
            dtype=np.float64,
        )

    def packed(diagonals, shape):
        out = np.zeros((6, *shape))
        out[:3] = 1.0  # inactive 2D axes decouple
        for axis, value in zip(axes, diagonals, strict=True):
            out[axis] = value
        return out

    # A site's other diagonal entries come from its neighbors through the
    # shared support, like the sampled tensors of the full-tensor kernel.
    shared = [
        collocate(s, c, box.shared, box.shared_shape)
        for s, c in zip(static, box.components, strict=True)
    ]
    site_tensors = []
    for row, (component, shape, k) in enumerate(
        zip(box.components, box.shapes, sites, strict=True)
    ):
        values = [collocate(v, box.shared, component, shape) for v in shared]
        values[row] = static[row]
        site_tensors.append(packed(values, shape) + k)
    node_tensor = packed(shared, box.shared_shape) + node
    _require_positive_definite([*site_tensors, node_tensor], what)
    diagonal = tuple(
        jnp.asarray(inverse_permittivity_components(t)[..., a], dtype=jnp.float32)
        for t, a in zip(site_tensors, axes, strict=True)
    )
    cross = inverse_permittivity_components(node_tensor)[..., 3:]
    return InverseOperator(diagonal, jnp.asarray(cross, dtype=jnp.float32))


def _ccpr_media(material_grid, family="E"):
    from beamz.design.anisotropic import TensorPole

    for medium, fraction in material_grid.tensor_dispersion:
        poles = medium.electric_poles if family == "E" else medium.magnetic_poles
        tensor = [p for p in poles if isinstance(p, TensorPole)]
        if has_edot(tensor) and np.any(np.asarray(fraction) > 1e-7):
            yield tensor, np.asarray(fraction, dtype=np.float64)


def electric_kernel_tensors(grid, dt, periodic, base=None):
    """Full-tensor kernel permittivity with every CCPR ``K`` folded in, or None.

    ``periodic`` maps periodic storage axes to their seam phases, as in
    :func:`beamz.simulation.anisotropic.compile_anisotropic`. ``base`` gives
    the kernel's support tensors (default: the material grid's). Returns packed
    ``eps_x``, ``eps_y``, ``eps_z`` and ``eps_node`` arrays, or None when the
    grid has no off-diagonal permittivity or no CCPR ``B`` term.
    """
    material_grid = grid.material_grid
    if not material_grid.uses_full_permittivity:
        return None
    media = list(_ccpr_media(material_grid))
    if not media:
        return None
    if base is None:
        base = material_grid.yee_tensors
    tensors = {
        name: np.array(expand_packed(value), dtype=np.float64)
        for name, value in base.items()
    }
    components = _active_components("E", grid)
    for poles, fraction in media:
        box = _make_box(
            "E", components, _cell_bounds(fraction > 1e-7, periodic), grid, periodic
        )
        cropped = jnp.asarray(fraction[box.cell_slices], dtype=jnp.float32)
        sites, node = _edot_tensors(cropped, _edot_drive(poles, dt), box)
        for component, part, value in zip(components, box.slices, sites, strict=True):
            tensors["eps_" + component[-1].lower()][(slice(None), *part)] += value
        tensors["eps_node"][(slice(None), *box.shared_slices)] += node
    return tensors


def compile_ccpr_region(region, poles, fraction, grid, dt, periodic):
    """Extend a compiled tensor-pole region with its CCPR ``B`` coupling."""
    box = region.box
    drive = _edot_drive(poles, dt)
    if len(box.shared_shape) == 2 and np.any(drive[:, 4:] != 0.0):
        raise ValueError(
            "2D simulations require tensor poles without xz or yz coupling; "
            "use a 3D simulation for out-of-plane coupling."
        )
    material_grid = grid.material_grid
    if box.family == "H" and any(
        np.any(np.abs(expand_packed(v)[3:]) > 1e-10)
        for v in material_grid.magnetic_tensors.values()
    ):
        raise ValueError(
            "Magnetic CCPR poles with an E' (b != 0) term are not supported "
            "together with off-diagonal static permeability; use a diagonal "
            "mu_inf or Lorentz/Drude magnetic poles."
        )
    cropped = jnp.asarray(np.asarray(fraction)[box.cell_slices], dtype=jnp.float32)
    edot_diagonal, edot_offdiagonal = tensor_pole_weights(cropped, drive, box)
    if box.family == "E" and material_grid.uses_full_permittivity:
        tensors = electric_kernel_tensors(grid, dt, periodic)
        components = box.components
        support = {c: tensors["eps_" + c[-1].lower()] for c in components}
        _require_positive_definite(
            [
                *(
                    support[c][(slice(None), *s)]
                    for c, s in zip(components, box.slices, strict=True)
                ),
                tensors["eps_node"][(slice(None), *box.shared_slices)],
            ],
            "eps_inf",
        )
        diagonal, cross = lossy_tensor_inverse(
            support,
            tensors["eps_node"],
            electric_losses(grid, components, dt),
            "Node",
        )
        inverse, static = _crop_inverse(diagonal, cross, box), None
    else:
        sites, node = _edot_tensors(cropped, drive, box)
        values = _static_sites(grid, box, dt)
        inverse = _diagonal_kernel_inverse(values, sites, node, box)
        static = tuple(jnp.asarray(v, dtype=jnp.float32) for v in values)
    return CCPRPolarizationRegion(
        box=box,
        c1=region.c1,
        c2=region.c2,
        diagonal=region.diagonal,
        offdiagonal=region.offdiagonal,
        inverse=inverse,
        edot_diagonal=edot_diagonal,
        edot_offdiagonal=edot_offdiagonal,
        static=static,
    )


# ---------------------------------------------------------------------------
# Runtime


def apply_ccpr_region(region, fields, flat, state, native=False):
    """Advance one CCPR region from the pre-kernel box ``fields``.

    Returns ``(state, new pole state)``; see the module docstring.
    """
    box = region.box
    native = _use_native(native, box, state)
    if native:
        from beamz.simulation.cuda import tensor_media

        delta, advanced = tensor_media.advance_poles(region, fields, flat)
    else:
        delta, advanced = region.advance(fields, flat)
    if region.static is None:
        source, base, sign = delta, None, -1.0
    else:
        current = _crop(state, box)
        source = tuple(
            s * (c - o) - d
            for s, c, o, d in zip(region.static, current, fields, delta, strict=True)
        )
        base, sign = fields, 1.0
    if native:
        state = tensor_media.apply_inverse(
            state, box, region.inverse, source, base=base, sign=sign
        )
    else:
        mapped = apply_tensor_operator(
            source, region.inverse.diagonal, region.inverse.offdiagonal, box
        )
        start = _crop(state, box) if base is None else base
        state = state._replace(
            **_scatter(
                state,
                box,
                tuple(b + sign * m for b, m in zip(start, mapped, strict=True)),
            )
        )
    change = tuple(n - o for n, o in zip(_crop(state, box), fields, strict=True))
    if native:
        return state, tensor_media.add_edot(region, change, advanced)
    coupled = apply_tensor_operator(
        change, region.edot_diagonal, region.edot_offdiagonal, box
    )
    advanced = list(advanced)
    for k, value in enumerate(coupled):
        advanced[2 * k] = advanced[2 * k] + value.astype(advanced[2 * k].dtype)
    return state, tuple(advanced)
