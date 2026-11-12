"""Tensor permeability and tensor-pole dispersion on the single-device JAX path.

Both features run as exact corrections around the unchanged vacuum-permeability
H kernel and the existing (diagonal or full-tensor) E kernel:

* Static permeability. The kernel advances ``H' = H - dt/mu0 * curl E`` (with
  CPML stretching inside ``curl``). The physical update is the same increment
  mapped through the inverse permeability, ``H = H_old + mu^-1 (H' - H_old)``.
* Tensor poles. Each pole carries a symmetric tensor strength ``W``, so a
  normalized polarization ``p = P / eps0`` obeys
  ``p'' + gamma p' + omega0^2 p = W E``. Its central-difference recurrence is
  explicit in ``E`` (off-diagonal coupling mixes neighboring Yee sites, so an
  implicit local solve does not exist), and the field receives
  ``E -= eps_inf^-1 (p_new - p_old)``. Magnetic poles mirror this on H.

* Full static tensors. Non-symmetric eps or mu, off-diagonal electric and any
  magnetic conductivity use the semi-implicit (Cayley) update
  ``(T + B) F_new = (T - B) F_old + inc`` with ``B = sigma dt / (2 eps0)`` (or
  ``mu0``), i.e. ``F_new = F_old + (T + B)^-1 (inc - 2 B F_old)``, where
  ``inc`` is the vacuum kernel's increment (CPML terms included; sources inject
  afterwards). The E kernel sees vacuum coefficients inside that box (see
  :func:`vacuum_coefficient_grid`); the H kernel is always vacuum.

Every tensor operator keeps its diagonal on the component's own Yee support and
its cross terms on the shared support — primal nodes for E, cell centers for H —
reached with the reciprocal averaging of :func:`collocate_yee_component`. That
averaging is the volume-weighted adjoint of its reverse, so each discrete
operator is symmetric in the Yee energy inner product, which is what keeps the
explicit recurrences stable for mixed-sign (hyperbolic) responses.

All work happens on static bounding boxes around the media with a two-cell
halo, so the cost scales with the anisotropic volume, not the domain.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.constitutive import (
    _support_offsets,
    collocate_yee_component,
    expand_packed,
    lossy_tensor_inverse,
)
from beamz.design.material_views import port_cells
from beamz.lattice import component_axis_offsets_3d

_HALO = 2
_PAIR = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
_SHARED = {"E": "Node", "H": "Cell"}
_PACKED_IDENTITY = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _axis_index(component: str) -> int:
    return "xyz".index(component[-1].lower())


@dataclass(frozen=True)
class YeeBox:
    """A cropped, self-consistent Yee sub-lattice covering whole cells."""

    family: str
    components: tuple[str, ...]
    slices: tuple[tuple[slice, ...], ...]
    shapes: tuple[tuple[int, ...], ...]
    shared_shape: tuple[int, ...]
    shared_slices: tuple[slice, ...]
    cell_slices: tuple[slice, ...]
    widths: tuple | None
    seams: tuple = ()  # (public axis, phase) for fully spanned periodic axes

    @property
    def shared(self) -> str:
        return _SHARED[self.family]


def _material_seams(box: YeeBox) -> tuple:
    """Seams for material coefficients, which are periodic without Bloch phase."""
    return tuple((axis, 1.0) for axis, _ in box.seams)


@dataclass(frozen=True)
class InverseOperator:
    """Tensor operator restricted to a box (diagonal + cross terms).

    ``offdiagonal`` holds the symmetric cross terms ``(xy, xz, yz)`` on the
    shared support; ``skew`` the antisymmetric ones, so entry ``(r, c)`` is
    ``offdiagonal + skew`` above the diagonal and ``offdiagonal - skew``
    below it.
    """

    diagonal: tuple
    offdiagonal: jax.Array | None
    skew: jax.Array | None = None


@dataclass(frozen=True)
class StaticTensorPlan:
    """Static tensor update on one box: ``F + (T + B)^-1 (inc - 2 B F)``.

    ``loss`` is the diagonal ``beta`` the vacuum kernel already applied (an H
    sponge), used to recover its increment; ``damping`` is the full ``B`` when
    it differs from that diagonal.
    """

    box: YeeBox
    inverse: InverseOperator
    loss: tuple | None = None  # sponge beta = sigma_m dt / (2 mu0) per component
    damping: InverseOperator | None = None


MagneticPlan = StaticTensorPlan


@dataclass(frozen=True)
class TensorPolarizationRegion:
    """All poles of one medium on one field family, cropped to its box."""

    box: YeeBox
    c1: jax.Array
    c2: jax.Array
    diagonal: tuple  # drive strengths with dt^2 / (1 + gamma dt / 2) folded in
    offdiagonal: jax.Array | None
    inverse: InverseOperator

    @property
    def state_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Flat state layout: (current, previous) per component."""
        poles = int(np.shape(self.c1)[0])
        return tuple((poles, *shape) for shape in self.box.shapes for _ in range(2))

    def advance(self, fields, flat):
        """Advance from the previous box fields; return (site delta p, new state)."""
        driven = apply_tensor_operator(
            fields, self.diagonal, self.offdiagonal, self.box
        )
        new_flat, delta = [], []
        for k, drive in enumerate(driven):
            current, previous = flat[2 * k], flat[2 * k + 1]
            new = self.c1 * current + self.c2 * previous + drive.astype(current.dtype)
            new_flat += [new, current]
            delta.append(jnp.sum(new - current, axis=0))
        return tuple(delta), tuple(new_flat)


@dataclass(frozen=True)
class GyrotropicRegion:
    """Gyrotropic poles: local diagonal response, precession via shared support.

    Each polarization component lives on its own Yee site and is driven
    locally (``w W F``), exactly like a diagonal tensor pole, so the strongly
    negative response of a plasma stays exact at every wavevector. Only the
    precession ``gyration x p'`` couples components; it reaches the other
    components through the reciprocal shared-support collocation, scaled by
    the receiving site's weight. That operator is antisymmetric in the
    polarization energy, and Crank-Nicolson keeps it energy-neutral. The
    implicit rotation is solved by a fixed-point iteration contracting with
    |gyration| dt / 2.
    """

    box: YeeBox
    rotation: jax.Array  # (poles, n, n) dt [gyration]x on the active axes
    strength: jax.Array  # (poles,) dt^2 * strength
    resonance_squared: jax.Array  # (poles,) (resonance dt)^2
    damping: jax.Array  # (poles,) damping dt / 2
    weights: tuple  # medium fraction per component site
    iterations: int
    inverse: InverseOperator

    @property
    def state_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Flat state layout: (polarization p, scaled current dt p') per component.

        All coefficients are dimensionless: the state is p and dt p', so no
        factor of dt or dt^2 meets a 1/s^2 strength inside the float32 update.
        """
        poles = int(np.shape(self.rotation)[0])
        return tuple((poles, *shape) for shape in self.box.shapes for _ in range(2))

    def _precession(self, currents):
        box = self.box
        shared = [
            _collocate(j, c, box.shared, box.shared_shape, box.widths, box.seams)
            for j, c in zip(currents, box.components, strict=True)
        ]
        spatial = (None,) * len(box.shared_shape)
        out = []
        for row, (component, shape, weight) in enumerate(
            zip(box.components, box.shapes, self.weights, strict=True)
        ):
            cross = sum(
                (
                    self.rotation[(slice(None), row, col, *spatial)] * shared[col]
                    for col in range(len(shared))
                    if col != row
                ),
                jnp.zeros_like(shared[row]),
            )
            out.append(
                weight
                * _collocate(cross, box.shared, component, shape, box.widths, box.seams)
            )
        return out

    def advance(self, fields, flat):
        count = len(self.box.components)
        p = [flat[2 * k] for k in range(count)]
        j = [flat[2 * k + 1] for k in range(count)]
        spatial = (None,) * len(self.box.shared_shape)
        strength = self.strength[(slice(None), *spatial)]
        resonance = self.resonance_squared[(slice(None), *spatial)]
        half = self.damping[(slice(None), *spatial)]
        rotate = self._precession(j)
        rhs = [
            (1 - half) * j[k]
            + strength * w * f.astype(j[k].dtype)
            - resonance * p[k]
            + 0.5 * rotate[k]
            for k, (w, f) in enumerate(zip(self.weights, fields, strict=True))
        ]
        current = [r / (1 + half) for r in rhs]
        for _ in range(self.iterations):
            rotate = self._precession(current)
            current = [
                (r + 0.5 * g) / (1 + half) for r, g in zip(rhs, rotate, strict=True)
            ]
        new_flat, delta = [], []
        for k in range(count):
            p_new = p[k] + current[k]
            new_flat += [p_new, current[k]]
            delta.append(jnp.sum(current[k], axis=0))
        return tuple(delta), tuple(new_flat)


@dataclass(frozen=True)
class AnisotropicPlan:
    magnetic: MagneticPlan | None
    electric_regions: tuple[TensorPolarizationRegion, ...]
    magnetic_regions: tuple[TensorPolarizationRegion, ...]
    electric: StaticTensorPlan | None = None  # full static eps / sigma tensors

    @property
    def regions(self) -> tuple[TensorPolarizationRegion, ...]:
        return (*self.electric_regions, *self.magnetic_regions)


# ---------------------------------------------------------------------------
# Discrete operators


def _collocate(value, source, target, shape, widths, seams=()):
    """Collocate one spatial support, mapping over any leading pole axis."""

    if value.ndim > len(shape):
        return jax.vmap(lambda v: _collocate(v, source, target, shape, widths, seams))(
            value
        )
    return collocate_yee_component(
        value, source, target, shape, cell_widths=widths, seams=seams
    )


def apply_tensor_operator(values, diagonal, offdiagonal, box: YeeBox, skew=None):
    """Apply a packed symmetric tensor field to one vector on the Yee box.

    ``values`` holds one array per box component. ``diagonal`` and
    ``offdiagonal`` may carry a leading pole axis that broadcasts against them.
    ``skew`` adds antisymmetric cross terms (see :class:`InverseOperator`).
    """

    out = [d * v for d, v in zip(diagonal, values, strict=True)]
    if (offdiagonal is None and skew is None) or len(box.components) < 2:
        return tuple(out)
    shared = tuple(
        _collocate(v, c, box.shared, box.shared_shape, box.widths, box.seams)
        for v, c in zip(values, box.components, strict=True)
    )
    axes = tuple(_axis_index(c) for c in box.components)

    def entry(row, col):
        pair = _PAIR[(min(axes[row], axes[col]), max(axes[row], axes[col]))]
        value = 0.0 if offdiagonal is None else offdiagonal[..., pair]
        if skew is not None:
            sign = 1.0 if axes[row] < axes[col] else -1.0
            value = value + sign * skew[..., pair]
        return value

    for row, (component, shape) in enumerate(
        zip(box.components, box.shapes, strict=True)
    ):
        cross = sum(
            entry(row, col) * shared[col] for col in range(len(axes)) if col != row
        )
        out[row] = out[row] + _collocate(
            cross, box.shared, component, shape, box.widths, box.seams
        )
    return tuple(out)


def apply_operator(values, operator: InverseOperator, box: YeeBox):
    """Apply an :class:`InverseOperator` (including skew terms) on the box."""
    return apply_tensor_operator(
        values, operator.diagonal, operator.offdiagonal, box, operator.skew
    )


def _crop(state, box: YeeBox):
    return tuple(
        getattr(state, c.lower())[s]
        for c, s in zip(box.components, box.slices, strict=True)
    )


def _scatter(state, box: YeeBox, values):
    return {
        c.lower(): getattr(state, c.lower())
        .at[s]
        .set(v.astype(getattr(state, c.lower()).dtype))
        for c, s, v in zip(box.components, box.slices, values, strict=True)
    }


@dataclass(frozen=True)
class BoxFields:
    """Pre-update fields cropped to each correction box.

    Taking these crops before the vacuum kernel lets an in-place (CUDA) update
    reuse the field buffers instead of XLA copying whole arrays to keep the
    previous fields alive.
    """

    regions: tuple  # per region: one cropped array per box component
    magnetic: tuple | None = None  # static tensor-permeability box
    electric: tuple | None = None  # static full-tensor permittivity box


def snapshot_electric(state, plan: AnisotropicPlan) -> BoxFields:
    return BoxFields(
        tuple(_crop(state, r.box) for r in plan.electric_regions),
        electric=None if plan.electric is None else _crop(state, plan.electric.box),
    )


def snapshot_magnetic(state, plan: AnisotropicPlan) -> BoxFields:
    return BoxFields(
        tuple(_crop(state, r.box) for r in plan.magnetic_regions),
        None if plan.magnetic is None else _crop(state, plan.magnetic.box),
    )


def _use_native(native: bool, box: YeeBox, state) -> bool:
    """Whether ``box`` runs on the streamed-CUDA tensor-media kernels."""
    if not native:
        return False
    from beamz.simulation.cuda.tensor_media import supports

    return supports(box, state.ex.dtype)


def _apply_regions(regions, old_fields, state, flat, native=False):
    """Advance each region and subtract eps_inf^-1 (or mu_inf^-1) of its delta."""
    new_flat, index = [], 0
    for region, fields in zip(regions, old_fields, strict=True):
        count = len(region.state_shapes)
        part = flat[index : index + count]
        index += count
        if getattr(region, "edot_diagonal", None):
            from beamz.simulation.ccpr import apply_ccpr_region

            state, advanced = apply_ccpr_region(region, fields, part, state, native)
        elif isinstance(region, TensorPolarizationRegion) and _use_native(
            native, region.box, state
        ):
            from beamz.simulation.cuda import tensor_media

            delta, advanced = tensor_media.advance_poles(region, fields, part)
            state = tensor_media.apply_inverse(
                state, region.box, region.inverse, delta, sign=-1.0
            )
        else:
            delta, advanced = region.advance(fields, part)
            correction = apply_operator(delta, region.inverse, region.box)
            current = _crop(state, region.box)
            state = state._replace(
                **_scatter(
                    state,
                    region.box,
                    tuple(c - d for c, d in zip(current, correction, strict=True)),
                )
            )
        new_flat.extend(advanced)
    return state, tuple(new_flat)


def _state_count(regions) -> int:
    return sum(len(region.state_shapes) for region in regions)


def _static_increment(plan: StaticTensorPlan, before, after):
    """``inc - 2 B F_old`` from the box fields before and after the kernel."""
    if plan.loss is None:
        increment = tuple(a - b for a, b in zip(after, before, strict=True))
    else:
        # The lossy vacuum kernel gives F' = a F + b inc with
        # a = (1 - beta) / (1 + beta) and b = 1 / (1 + beta), so
        # inc = (F' - a F)(1 + beta).
        increment = tuple(
            (a - (1 - beta) / (1 + beta) * b) * (1 + beta)
            for a, b, beta in zip(after, before, plan.loss, strict=True)
        )
    # B F_old: the full damping operator when present, else the diagonal loss.
    if plan.damping is not None:
        damped = apply_operator(before, plan.damping, plan.box)
    elif plan.loss is not None:
        damped = tuple(beta * b for b, beta in zip(before, plan.loss, strict=True))
    else:
        return increment
    return tuple(i - 2 * d for i, d in zip(increment, damped, strict=True))


def _apply_static(plan: StaticTensorPlan, before, state, native):
    """Replace the box fields by ``F_old + (T + B)^-1 (inc - 2 B F_old)``."""
    box = plan.box
    increment = _static_increment(plan, before, _crop(state, box))
    if _use_native(native, box, state):
        from beamz.simulation.cuda.tensor_media import apply_inverse

        return apply_inverse(state, box, plan.inverse, increment, base=before, sign=1.0)
    mapped = apply_operator(increment, plan.inverse, box)
    return state._replace(
        **_scatter(
            state, box, tuple(b + m for b, m in zip(before, mapped, strict=True))
        )
    )


def update_magnetic(old, state, plan: AnisotropicPlan, tail, native=False):
    """Apply tensor permeability and magnetic poles after the vacuum H kernel.

    ``old`` is the pre-update state or its :func:`snapshot_magnetic`.
    ``native`` selects the streamed-CUDA kernels where a box supports them.
    """

    if not isinstance(old, BoxFields):
        old = snapshot_magnetic(old, plan)
    if plan.magnetic is not None:
        state = _apply_static(plan.magnetic, old.magnetic, state, native)
    if not plan.magnetic_regions:
        return state, tail
    split = _state_count(plan.electric_regions)
    state, magnetic = _apply_regions(
        plan.magnetic_regions, old.regions, state, tail[split:], native
    )
    return state, tuple(tail[:split]) + magnetic


def update_static_electric(old, state, plan: AnisotropicPlan, native=False):
    """Apply full static eps / sigma tensors right after the vacuum E kernel.

    Like the permeability mapping, this runs before E-phase sources inject, so
    sources see the same update as in any other medium. ``old`` is the
    pre-update state or its :func:`snapshot_electric`.
    """

    if plan.electric is None:
        return state
    if not isinstance(old, BoxFields):
        old = snapshot_electric(old, plan)
    return _apply_static(plan.electric, old.electric, state, native)


def update_electric(old, state, plan: AnisotropicPlan, tail, native=False):
    """Apply electric tensor poles after the E kernel and E-phase sources.

    ``old`` is the pre-update state or its :func:`snapshot_electric`.
    ``native`` selects the streamed-CUDA kernels where a box supports them.
    """

    if not plan.electric_regions:
        return state, tail
    if not isinstance(old, BoxFields):
        old = snapshot_electric(old, plan)
    split = _state_count(plan.electric_regions)
    state, electric = _apply_regions(
        plan.electric_regions, old.regions, state, tail[:split], native
    )
    return state, electric + tuple(tail[split:])


def initial_states(plan: AnisotropicPlan, dtype) -> tuple:
    return tuple(
        jnp.zeros(shape, dtype=dtype)
        for region in plan.regions
        for shape in region.state_shapes
    )


# ---------------------------------------------------------------------------
# Compilation


def _active_components(family: str, grid) -> tuple[str, ...]:
    if len(grid.material_grid.shape) == 3:
        return tuple(family + axis for axis in "xyz")
    polarization = grid.polarization_2d
    if family == "E":
        return ("Ez",) if polarization == "tm" else ("Ex", "Ey")
    return ("Hx", "Hy") if polarization == "tm" else ("Hz",)


def _make_box(family, components, cell_bounds, grid, periodic=None) -> YeeBox:
    ndim = len(cell_bounds)
    axes = ("z", "y", "x")[-ndim:]
    slices, shapes = [], []
    for component in components:
        offsets = component_axis_offsets_3d(component)
        part = tuple(
            slice(start, stop + int(offsets[axis] == 0.0))
            for axis, (start, stop) in zip(axes, cell_bounds, strict=True)
        )
        slices.append(part)
        shapes.append(tuple(s.stop - s.start for s in part))
    node = family == "E"
    shared_slices = tuple(slice(start, stop + int(node)) for start, stop in cell_bounds)
    widths = None
    if grid.material_grid.metric_kind != "isotropic_uniform":
        geometry = grid.geometry
        widths = tuple(
            jnp.asarray(
                np.asarray(geometry.cell_widths(axis))[
                    cell_bounds[axes.index(axis)][0] : cell_bounds[axes.index(axis)][1]
                ],
                dtype=jnp.float32,
            )
            if axis in axes
            else None
            for axis in "xyz"
        )
    counts = grid.material_grid.shape
    seams = tuple(
        (axes[axis], phase)
        for axis, phase in sorted((periodic or {}).items())
        if cell_bounds[axis] == (0, counts[axis])
    )
    return YeeBox(
        family,
        tuple(components),
        tuple(slices),
        tuple(shapes),
        tuple(s.stop - s.start for s in shared_slices),
        shared_slices,
        tuple(slice(start, stop) for start, stop in cell_bounds),
        widths,
        seams,
    )


def _cell_bounds(active_cells: np.ndarray, periodic):
    """Halo-padded bounding box; spans a periodic axis it reaches the edge of.

    Averaging near a periodic seam reaches the other side of the domain, so a
    box touching either end of a periodic axis covers the whole axis and that
    axis becomes a seam of the box.
    """
    bounds = []
    for axis in range(active_cells.ndim):
        other = tuple(i for i in range(active_cells.ndim) if i != axis)
        occupied = np.flatnonzero(np.any(active_cells, axis=other))
        count = active_cells.shape[axis]
        start = max(0, int(occupied[0]) - _HALO)
        stop = min(count, int(occupied[-1]) + 1 + _HALO)
        if axis in (periodic or {}) and (start == 0 or stop == count):
            start, stop = 0, count
        bounds.append((start, stop))
    return tuple(bounds)


def _support_to_cells(values: np.ndarray, component: str, cell_shape) -> np.ndarray:
    """Map nonzero entries of a Yee support onto the cells sharing it."""

    axes = ("z", "y", "x")[-len(cell_shape) :]
    offsets = _support_offsets(component)
    mask = np.asarray(values, dtype=bool)
    for index, axis in enumerate(axes):
        if offsets[axis] == 0.0:  # node-aligned: one more site than cells
            leading = (slice(None),) * index
            mask = mask[(*leading, slice(0, -1))] | mask[(*leading, slice(1, None))]
    return mask


def electric_losses(grid, components, dt) -> dict:
    """Semi-implicit electric loss ``sigma dt / (2 eps0)`` at each E support."""
    return {
        c: dt
        * np.asarray(getattr(grid, "sig_" + c[-1].lower()), dtype=np.float64)
        / (2.0 * EPS_0)
        for c in components
    }


def _crop_or_fill(value, part, shape) -> np.ndarray:
    """Crop a support array to a box component, or fill it with a scalar."""
    value = np.asarray(value, dtype=np.float64)
    return value[part] if value.ndim else np.full(shape, float(value))


def _crop_inverse(diagonal, cross, box: YeeBox) -> InverseOperator:
    cropped = tuple(
        jnp.asarray(diagonal[c][s], dtype=jnp.float32)
        for c, s in zip(box.components, box.slices, strict=True)
    )
    cross = cross[box.shared_slices]
    if len(box.components) < 2 or not np.any(np.abs(cross) > 1e-12):
        return InverseOperator(cropped, None)
    return InverseOperator(cropped, jnp.asarray(cross, dtype=jnp.float32))


def _electric_inverse(grid, box: YeeBox, dt) -> InverseOperator:
    material_grid = grid.material_grid
    losses = electric_losses(grid, box.components, dt)
    if material_grid.uses_full_permittivity:
        support = {
            c: material_grid.yee_tensors["eps_" + c[-1].lower()] for c in box.components
        }
        diagonal, cross = lossy_tensor_inverse(
            support, material_grid.yee_tensors["eps_node"], losses, "Node"
        )
        return _crop_inverse(diagonal, cross, box)
    diagonal = []
    for component, part, shape in zip(
        box.components, box.slices, box.shapes, strict=True
    ):
        eps = _crop_or_fill(getattr(grid, "eps_" + component[-1].lower()), part, shape)
        loss = _crop_or_fill(losses[component], part, shape)
        diagonal.append(jnp.asarray(1.0 / (eps + loss), dtype=jnp.float32))
    return InverseOperator(tuple(diagonal), None)


def magnetic_losses(grid, components, dt) -> dict:
    """Semi-implicit magnetic (sponge) loss ``sigma_m dt / (2 mu0)`` at H supports."""
    return {
        c: dt
        * np.asarray(getattr(grid, "sigma_m_h" + c[-1].lower(), 0.0), dtype=np.float64)
        / (2.0 * MU_0)
        for c in components
    }


def _magnetic_inverse(grid, box: YeeBox, dt) -> InverseOperator:
    tensors = grid.material_grid.magnetic_tensors
    losses = magnetic_losses(grid, box.components, dt)
    if not tensors:
        return InverseOperator(
            tuple(
                jnp.asarray(
                    1.0 / (1.0 + _crop_or_fill(losses[c], part, shape)),
                    dtype=jnp.float32,
                )
                for c, part, shape in zip(
                    box.components, box.slices, box.shapes, strict=True
                )
            ),
            None,
        )
    support = {c: tensors["mu_h" + c[-1].lower()] for c in box.components}
    diagonal, cross = lossy_tensor_inverse(support, tensors["mu_cell"], losses, "Cell")
    return _crop_inverse(diagonal, cross, box)


def _family_inverse(grid, box: YeeBox, dt) -> InverseOperator:
    media = static_media(grid, box.family)
    if media:
        return coupled_operators(grid, box, dt, media)[0]
    if box.family == "E":
        return _electric_inverse(grid, box, dt)
    return _magnetic_inverse(grid, box, dt)


def static_media(grid, family: str) -> list:
    """``(medium, fraction)`` pairs whose static terms need the box update.

    For E: non-symmetric permittivity or off-diagonal conductivity. For H:
    non-symmetric permeability or any magnetic conductivity.
    """
    two_d = len(grid.material_grid.shape) == 2
    out = []
    for medium, fraction in grid.material_grid.tensor_dispersion:
        tensor, loss = _static_extras(medium, family, 1.0)
        coupling = np.abs(tensor) + np.abs(loss)
        if not np.any(coupling != 0.0):
            continue
        if two_d and (np.any(coupling[2, :2] != 0.0) or np.any(coupling[:2, 2] != 0.0)):
            raise ValueError(
                "2D simulations require static tensors and conductivities "
                "without xz or yz coupling; use a 3D simulation."
            )
        out.append((medium, np.asarray(fraction, dtype=np.float64)))
    return out


def _static_extras(medium, family: str, dt):
    """Box-update terms beyond the raster: (T extra, B extra) as 3x3 matrices.

    The raster already holds the symmetric tensor and the diagonal electric
    conductivity; the box adds the antisymmetric tensor part, the off-diagonal
    electric conductivity, and the whole magnetic conductivity.
    """
    from beamz.design.anisotropic import skew_matrix, skew_values

    if family == "E":
        tensor = skew_matrix(skew_values(medium.permittivity_matrix))
        sigma = medium.conductivity_matrix
        loss = (sigma - np.diag(np.diag(sigma))) * dt / (2.0 * EPS_0)
    else:
        tensor = skew_matrix(skew_values(medium.permeability_matrix))
        loss = medium.magnetic_conductivity_matrix * dt / (2.0 * MU_0)
    return tensor, loss


def coupled_operators(grid, box: YeeBox, dt, media):
    """Pointwise ``(T + B)^-1`` and ``B`` of the full static tensors on a box.

    ``T`` is the raster's tensor plus each medium's antisymmetric part, and
    ``B = sigma dt / (2 eps0)`` (``mu0`` for H) the raster's diagonal loss plus
    each medium's remaining conductivity, both volume-averaged by the medium's
    cell fraction. Each support (component sites and the shared support)
    inverts its own full tensor; diagonals stay on the sites and the cross
    terms, split into symmetric and antisymmetric parts, on the shared
    support, exactly like the symmetric full-tensor update. Returns
    ``(inverse, damping)`` operators, ``damping`` None when ``B`` vanishes.
    """
    from beamz.design.anisotropic import packed_to_matrix

    materials = grid.material_grid
    components = box.components
    count = len(components)
    index = [_axis_index(c) for c in components]
    seams = _material_seams(box)
    supports = tuple(zip(components, box.slices, box.shapes, strict=True))
    site_parts = {c: (part, shape) for c, part, shape in supports}
    if count > 1:
        supports += ((box.shared, box.shared_slices, box.shared_shape),)

    def move(value, source, target, shape):
        if source == target:
            return np.asarray(value, dtype=np.float64)
        return np.asarray(
            collocate_yee_component(
                jnp.asarray(value),
                source,
                target,
                shape,
                cell_widths=box.widths,
                seams=seams,
            ),
            dtype=np.float64,
        )

    if box.family == "E":
        full = materials.uses_full_permittivity
        names = {c: "eps_" + c[-1].lower() for c in components}
        shared_name, packed = "eps_node", materials.yee_tensors
        loss_names = {c: "sig_" + c[-1].lower() for c in components}
        scale = dt / (2.0 * EPS_0)
    else:
        full = bool(materials.magnetic_tensors)
        names = {c: "mu_h" + c[-1].lower() for c in components}
        shared_name, packed = "mu_cell", materials.magnetic_tensors
        loss_names = {c: "sigma_m_h" + c[-1].lower() for c in components}
        scale = dt / (2.0 * MU_0)
    cropped = np.asarray(
        [np.asarray(f, dtype=np.float64)[box.cell_slices] for _, f in media]
    )
    extras = [_static_extras(medium, box.family, dt) for medium, _ in media]
    inverses, losses = {}, {}
    for support, part, shape in supports:
        tensor = np.zeros((*shape, count, count))
        loss = np.zeros((*shape, count, count))
        if full:
            name = names.get(support, shared_name)
            values = expand_packed(packed[name])[(slice(None), *part)]
            matrix = packed_to_matrix(np.moveaxis(values, 0, -1))
            tensor += matrix[..., index, :][..., :, index]
        for k, component in enumerate(components):
            own_part, own_shape = site_parts[component]
            if not full:
                if box.family == "E":
                    value = _crop_or_fill(
                        getattr(grid, names[component]), own_part, own_shape
                    )
                    tensor[..., k, k] += move(value, component, support, shape)
                else:
                    tensor[..., k, k] += 1.0
            sigma = _crop_or_fill(
                getattr(grid, loss_names[component], 0.0), own_part, own_shape
            )
            if np.any(sigma != 0.0):
                loss[..., k, k] += scale * move(sigma, component, support, shape)
        for fraction, (extra_tensor, extra_loss) in zip(cropped, extras, strict=True):
            weight = move(fraction, "Cell", support, shape)[..., None, None]
            tensor += weight * extra_tensor[np.ix_(index, index)]
            loss += weight * extra_loss[np.ix_(index, index)]
        inverses[support] = np.linalg.inv(tensor + loss)
        losses[support] = loss

    def operator(matrices, allow_empty):
        if allow_empty and not any(np.any(m != 0.0) for m in matrices.values()):
            return None
        diagonal = tuple(
            jnp.asarray(matrices[c][..., k, k], dtype=jnp.float32)
            for k, c in enumerate(components)
        )
        if count < 2:
            return InverseOperator(diagonal, None)
        shared = matrices[box.shared]
        symmetric = np.zeros((*box.shared_shape, 3))
        skew = np.zeros((*box.shared_shape, 3))
        for row in range(count):
            for col in range(row + 1, count):
                pair = _PAIR[(index[row], index[col])]
                upper, lower = shared[..., row, col], shared[..., col, row]
                symmetric[..., pair] = 0.5 * (upper + lower)
                skew[..., pair] = 0.5 * (upper - lower)
        # Drop cross terms that are roundoff relative to the diagonal.
        floor = 1e-12 * max(float(np.abs(d).max(initial=0.0)) for d in diagonal)

        def cross(values):
            if not np.any(np.abs(values) > floor):
                return None
            return jnp.asarray(values, dtype=jnp.float32)

        return InverseOperator(diagonal, cross(symmetric), cross(skew))

    return operator(inverses, False), operator(losses, True)


def static_electric_box(grid, periodic_axes=frozenset()) -> YeeBox | None:
    """The E box of the full static-tensor update, or None without such media."""
    media = static_media(grid, "E")
    if not media:
        return None
    active = np.zeros(tuple(grid.material_grid.shape), dtype=bool)
    for _, fraction in media:
        active |= fraction > 1e-7
    if not np.any(active):
        return None
    periodic = _seam_phases(grid, periodic_axes)
    return _make_box(
        "E",
        _active_components("E", grid),
        _cell_bounds(active, periodic),
        grid,
        periodic,
    )


def vacuum_coefficient_grid(grid, periodic_axes=frozenset()):
    """Grid and full-tensor supports whose E coefficients are vacuum on the box.

    The E kernel then returns ``E + inc`` there and the static plan applies
    the full ``(T + B)^-1``. Returns ``(grid, yee_tensors)``; both are the
    inputs unchanged without non-symmetric or off-diagonal-conductive media.
    """
    from dataclasses import replace

    materials = grid.material_grid
    box = static_electric_box(grid, periodic_axes)
    if box is None:
        return grid, materials.yee_tensors
    changes = {}
    for component, part in zip(box.components, box.slices, strict=True):
        axis = component[-1].lower()
        shape = tuple(grid.component_shapes[component])
        for prefix, value in (("eps_", 1.0), ("sig_", 0.0)):
            array = jnp.asarray(getattr(grid, prefix + axis))
            array = jnp.broadcast_to(array, shape).at[part].set(value)
            changes[prefix + axis] = array
            if prefix == "eps_":
                changes["eps_e" + axis] = array
    tensors = materials.yee_tensors
    if materials.uses_full_permittivity:
        node = expand_packed(tensors["eps_node"])[(slice(None), *box.shared_slices)]
        ring = np.zeros(box.shared_shape, dtype=bool)
        seam_axes = {axis for axis, _ in box.seams}
        for position, axis in enumerate(("z", "y", "x")[-len(box.shared_shape) :]):
            if axis in seam_axes:
                continue
            for end in (0, -1):
                edge = [slice(None)] * len(box.shared_shape)
                edge[position] = end
                ring[tuple(edge)] = True
        if np.any(np.abs(node[3:][:, ring]) > 1e-10):
            raise ValueError(
                "Non-symmetric or off-diagonal-conductive media must stay at "
                f"least {_HALO + 1} cells from other off-diagonal permittivity."
            )
        identity = _PACKED_IDENTITY.reshape((6,) + (1,) * len(box.shared_shape))
        tensors = dict(tensors)
        for component, part in zip(box.components, box.slices, strict=True):
            name = "eps_" + component[-1].lower()
            values = expand_packed(tensors[name]).copy()
            values[(slice(None), *part)] = identity
            tensors[name] = values
        values = expand_packed(tensors["eps_node"]).copy()
        values[(slice(None), *box.shared_slices)] = identity
        tensors["eps_node"] = values
    return replace(grid, **changes), tensors


def _cropped_losses(losses, box: YeeBox):
    """Per-component box arrays of a loss mapping, or None when lossless."""
    if not any(np.any(np.asarray(loss) != 0.0) for loss in losses.values()):
        return None
    return tuple(
        jnp.asarray(_crop_or_fill(losses[c], s, shape), dtype=jnp.float32)
        for c, s, shape in zip(box.components, box.slices, box.shapes, strict=True)
    )


def _compile_magnetic(grid, periodic_axes, dt) -> MagneticPlan | None:
    tensors = grid.material_grid.magnetic_tensors
    media = static_media(grid, "H")
    if not tensors and not media:
        return None
    if tensors:
        _check_static_tensor_stability(grid, dt)
    cell_shape = tuple(grid.material_grid.shape)
    components = _active_components("H", grid)
    active = np.zeros(cell_shape, dtype=bool)
    for name, component in (
        *((f"mu_h{c[-1].lower()}", c) for c in components),
        ("mu_cell", "Cell"),
    ):
        if name not in tensors:
            continue
        values = expand_packed(tensors[name])
        differs = np.any(
            np.abs(values.reshape(6, -1) - _PACKED_IDENTITY[:, None]) > 1e-10, axis=0
        ).reshape(values.shape[1:])
        active |= _support_to_cells(differs, component, cell_shape)
    for _, fraction in media:
        active |= fraction > 1e-7
    if not np.any(active):
        return None
    box = _make_box(
        "H", components, _cell_bounds(active, periodic_axes), grid, periodic_axes
    )
    loss = _cropped_losses(magnetic_losses(grid, components, dt), box)
    if media:
        inverse, damping = coupled_operators(grid, box, dt, media)
        return StaticTensorPlan(box, inverse, loss, damping)
    return StaticTensorPlan(box, _magnetic_inverse(grid, box, dt), loss)


def _compile_electric(grid, periodic_axes, dt) -> StaticTensorPlan | None:
    box = static_electric_box(grid, periodic_axes)
    if box is None:
        return None
    inverse, damping = coupled_operators(grid, box, dt, static_media(grid, "E"))
    return StaticTensorPlan(box, inverse, None, damping)


def _check_static_tensor_stability(grid, dt):
    """Bound the wave speed using the smallest constitutive eigenvalues."""
    from beamz.design.anisotropic import packed_to_matrix

    def minimum_eigenvalue(tensors):
        return min(
            float(
                np.linalg.eigvalsh(
                    packed_to_matrix(np.moveaxis(expand_packed(tensor), 0, -1))
                ).min()
            )
            for tensor in tensors
        )

    materials = grid.material_grid
    eps_min = minimum_eigenvalue(
        (materials.tensors["epsilon"], *materials.yee_tensors.values())
    )
    mu_min = minimum_eigenvalue(materials.magnetic_tensors.values())
    axes = ("x", "y", "z") if len(materials.shape) == 3 else ("x", "y")
    limit = grid.geometry.cfl_time_step(np.sqrt(eps_min * mu_min), active_axes=axes)
    if dt > limit * (1 + 1e-6):
        raise ValueError(
            f"Static tensor media exceed the CFL stability bound at dt={dt:.3e} s; "
            f"use dt <= {limit:.3e} s."
        )


def _pole_coefficients(poles, dt):
    resonance = np.array([p.resonance for p in poles])
    damping = np.array([p.damping for p in poles])
    denominator = 1.0 + 0.5 * damping * dt
    c1 = (2.0 - (resonance * dt) ** 2) / denominator
    c2 = -(1.0 - 0.5 * damping * dt) / denominator
    c3 = dt**2 / denominator
    return c1, c2, c3


def max_stable_timestep(medium, family, spacings, polarization=None) -> float:
    """Largest stable dt for a medium's poles on a grid with these ``spacings``.

    A von Neumann analysis of the implemented scheme (see
    :mod:`beamz.simulation.pole_stability`); with two spacings,
    ``polarization`` (``"te"`` or ``"tm"``) keeps only the simulated field
    components, both when None.
    """
    from beamz.simulation import pole_stability

    return pole_stability.max_stable_timestep(medium, family, spacings, polarization)


def _compile_regions(medium, fraction, family, grid, dt, periodic_axes) -> list:
    from beamz.design.anisotropic import GyrotropicPole
    from beamz.simulation import pole_stability

    poles = medium.electric_poles if family == "E" else medium.magnetic_poles
    fraction = np.asarray(fraction, dtype=np.float64)
    if not poles or not np.any(fraction > 1e-7):
        return []
    pole_stability.check_stability(medium, family, dt, grid)
    tensor = [p for p in poles if not isinstance(p, GyrotropicPole)]
    gyrotropic = [p for p in poles if isinstance(p, GyrotropicPole)]
    regions = []
    if tensor:
        region = _compile_tensor_region(
            tensor, fraction, family, grid, dt, periodic_axes
        )
        if any(getattr(p, "has_edot", False) for p in tensor):
            from beamz.simulation.ccpr import compile_ccpr_region

            region = compile_ccpr_region(
                region, tensor, fraction, grid, dt, periodic_axes
            )
        regions.append(region)
    if gyrotropic:
        regions.append(
            _compile_gyrotropic_region(
                gyrotropic, fraction, family, grid, dt, periodic_axes
            )
        )
    return regions


def _compile_gyrotropic_region(poles, fraction, family, grid, dt, periodic_axes):
    from beamz.design.anisotropic import _cross_matrix

    components = _active_components(family, grid)
    if len(grid.material_grid.shape) == 2 and any(
        abs(p.gyration[0]) > 0 or abs(p.gyration[1]) > 0 for p in poles
    ):
        raise ValueError(
            "2D simulations support gyration along z only (in-plane fields "
            "precess about the out-of-plane axis)."
        )
    box = _make_box(
        family,
        components,
        _cell_bounds(fraction > 1e-7, periodic_axes),
        grid,
        periodic_axes,
    )
    cropped = jnp.asarray(fraction[box.cell_slices], dtype=jnp.float32)
    weights = tuple(
        _collocate(cropped, "Cell", c, shape, box.widths, _material_seams(box))
        for c, shape in zip(components, box.shapes, strict=True)
    )
    index = [_axis_index(c) for c in components]
    rotation = np.stack(
        [_cross_matrix(p.gyration)[np.ix_(index, index)] for p in poles]
    )
    contraction = 0.5 * dt * max(float(np.linalg.norm(p.gyration)) for p in poles)
    if contraction >= 0.5:
        raise ValueError(
            f"Gyration is too fast for dt={dt:.3e} s: |gyration| dt / 2 = "
            f"{contraction:.3f} must stay below 0.5. Reduce the timestep."
        )
    if contraction == 0.0:
        iterations = 0
    else:
        iterations = int(np.clip(np.ceil(-9 / np.log10(contraction)), 1, 12))
    return GyrotropicRegion(
        box,
        jnp.asarray(rotation * dt, dtype=jnp.float32),
        jnp.asarray([p.strength * dt**2 for p in poles], dtype=jnp.float32),
        jnp.asarray([(p.resonance * dt) ** 2 for p in poles], dtype=jnp.float32),
        jnp.asarray([0.5 * p.damping * dt for p in poles], dtype=jnp.float32),
        weights,
        iterations,
        _family_inverse(grid, box, dt),
    )


def tensor_pole_weights(fraction, drive, box: YeeBox):
    """Per-site and shared-support pole strengths for a cropped cell fraction.

    ``fraction`` is the medium's volume fraction on ``box.cell_slices`` (JAX or
    NumPy; traced values are fine), ``drive`` the folded strengths (poles, 6).
    Returns ``(diagonal per component, cross terms or None)``.
    """
    seams = _material_seams(box)
    cell_ndim = len(box.shared_shape)
    reshape = (-1,) + (1,) * cell_ndim
    coupled = bool(np.any(drive[:, 3:] != 0.0)) and len(box.components) > 1
    # Cross terms live on the shared support. With coupling, the diagonal weight
    # of each site must be the shared weight averaged back to it: the operator
    # is then a sum of per-support forms w * avg(E)^T W avg(E) (Jensen), hence
    # positive semidefinite. Plain cell averages leave sites just outside a
    # tilted medium with cross coupling but no diagonal, an indefinite operator
    # that diverges for Drude (hyperbolic) poles.
    if box.family == "H":
        shared = fraction
    else:
        shared = _collocate(
            fraction, "Cell", "Node", box.shared_shape, box.widths, seams
        )
    if coupled:
        site_support, site_weight = box.shared, shared
    else:
        site_support, site_weight = "Cell", fraction
    diagonal = []
    for component, shape in zip(box.components, box.shapes, strict=True):
        site = _collocate(
            site_weight, site_support, component, shape, box.widths, seams
        )
        strength = drive[:, _axis_index(component)].reshape(reshape)
        diagonal.append(jnp.asarray(strength, dtype=jnp.float32) * site[None])
    offdiagonal = None
    if coupled:
        offdiagonal = (
            jnp.asarray(drive[:, 3:], dtype=jnp.float32)[
                (slice(None),) + (None,) * cell_ndim + (slice(None),)
            ]
            * shared[None, ..., None]
        )
    return tuple(diagonal), offdiagonal


def _compile_tensor_region(poles, fraction, family, grid, dt, periodic_axes):
    cell_shape = tuple(grid.material_grid.shape)
    weights = np.stack([np.asarray(p.weight) for p in poles])  # (poles, 6)
    if len(cell_shape) == 2 and np.any(weights[:, 4:] != 0.0):
        raise ValueError(
            "2D simulations require tensor poles without xz or yz coupling; "
            "use a 3D simulation for out-of-plane coupling."
        )
    c1, c2, c3 = _pole_coefficients(poles, dt)
    # Fold dt^2/(1 + gamma dt / 2) into the strengths in float64: separately
    # they are ~1e-31 and ~1e31, and reverse-mode cotangents multiplied by the
    # tiny factor first would underflow float32.
    drive = weights * c3[:, None]
    components = _active_components(family, grid)
    box = _make_box(
        family,
        components,
        _cell_bounds(fraction > 1e-7, periodic_axes),
        grid,
        periodic_axes,
    )
    reshape = (-1,) + (1,) * len(cell_shape)
    cropped = jnp.asarray(fraction[box.cell_slices], dtype=jnp.float32)
    diagonal, offdiagonal = tensor_pole_weights(cropped, drive, box)
    return TensorPolarizationRegion(
        box,
        jnp.asarray(c1.reshape(reshape), dtype=jnp.float32),
        jnp.asarray(c2.reshape(reshape), dtype=jnp.float32),
        diagonal,
        offdiagonal,
        _family_inverse(grid, box, dt),
    )


def _warn_on_absorber_overlap(grid) -> None:
    if not getattr(grid, "has_cpml", False):
        return  # loss-based sponges absorb backward waves too
    pml_data = getattr(grid, "pml_data", None) or {}
    mask = pml_data.get("mask")
    if mask is None:
        return
    mask = np.asarray(mask, dtype=bool)
    for medium, fraction in grid.material_grid.tensor_dispersion:
        fraction = np.asarray(fraction)
        if not medium.is_indefinite():
            continue
        if mask.shape == fraction.shape and np.any(mask & (fraction > 1e-7)):
            warnings.warn(
                "A hyperbolic (indefinite) tensor medium extends into a PML. Such media carry "
                "backward waves (phase and energy opposed along the absorber "
                "normal) that a PML amplifies, which shows up as late-time "
                "exponential growth inside the absorber. Terminate such media "
                "in a lossless background before the PML, or use bz.Absorber "
                "(a graded-loss sponge, which absorbs backward waves).",
                UserWarning,
                stacklevel=4,
            )
            return


def _seam_phases(grid, periodic_axes) -> dict:
    """Map periodic storage axes to their forward Bloch seam phase (1 if plain)."""
    if not periodic_axes:
        return {}
    from beamz.devices.boundaries import bloch_storage_phases

    is_3d = len(grid.material_grid.shape) == 3
    phases = bloch_storage_phases(
        grid.boundaries,
        grid.geometry,
        is_3d=is_3d,
        plane_2d=getattr(grid, "plane_2d", "xy"),
    )
    out = {}
    for axis in periodic_axes:
        phase = complex(phases[int(axis)])
        out[int(axis)] = phase.real if phase.imag == 0.0 else phase
    return out


def compile_anisotropic(grid, dt, periodic_axes=frozenset()) -> AnisotropicPlan | None:
    """Build static tensor-permeability and tensor-pole plans for a compiled grid."""

    material_grid = grid.material_grid
    if not material_grid.uses_magnetic_tensors and not material_grid.tensor_dispersion:
        return None
    periodic = _seam_phases(grid, periodic_axes)
    _warn_on_absorber_overlap(grid)
    electric, magnetic = [], []
    for medium, fraction in material_grid.tensor_dispersion:
        for family, bucket in (("E", electric), ("H", magnetic)):
            bucket.extend(
                _compile_regions(medium, fraction, family, grid, dt, periodic)
            )
    _reject_ccpr_static_overlap(grid, electric, "E")
    _reject_ccpr_static_overlap(grid, magnetic, "H")
    return AnisotropicPlan(
        _compile_magnetic(grid, periodic, dt),
        tuple(electric),
        tuple(magnetic),
        _compile_electric(grid, periodic_axes, dt),
    )


def _reject_ccpr_static_overlap(grid, regions, family: str) -> None:
    """CCPR ``E'`` terms and full static-tensor boxes cannot share cells yet.

    Both rewrite the same support's constitutive solve: the static box runs a
    vacuum kernel, while the CCPR term augments the kernel's own tensor.
    """
    from beamz.simulation.ccpr import CCPRPolarizationRegion

    ccpr = [r for r in regions if isinstance(r, CCPRPolarizationRegion)]
    if not ccpr:
        return
    media = static_media(grid, family)
    if not media:
        return
    static = np.zeros(grid.material_grid.shape, dtype=bool)
    for _medium, fraction in media:
        static |= np.asarray(fraction) > 1e-7
    for region in ccpr:
        reach = tuple(
            slice(max(0, s.start - _HALO), s.stop + _HALO)
            for s in region.box.cell_slices
        )
        if np.any(static[reach]):
            raise ValueError(
                "CCPR poles with an E' (b != 0) term cannot yet overlap media "
                "with non-symmetric tensors or off-diagonal/magnetic "
                f"conductivity (within {_HALO} cells)."
            )


def validate_mode_ports(material_grid, sources, monitors) -> None:
    """Reject mode ports whose modal solve cannot reproduce the propagation media.

    Mode sources and mode monitors inside tensor-pole media are solved at each
    of their frequencies (see
    :class:`beamz.design.material_views.DispersionFolding`), with electric
    poles folded into permittivity and magnetic poles into permeability.
    Diagonal and full symmetric permeability tensors are reproduced by the 3D
    port solve; a 2D (line) port supports diagonal permeability only. Ports
    must avoid gyrotropic media, whose poles cannot be frozen into static
    reciprocal tensors, and media needing the full static-tensor update
    (non-symmetric tensors, off-diagonal or magnetic conductivity).
    """
    from beamz.design.material_views import foldable

    # Only 2D TM line ports see in-plane (xy) magnetic coupling.
    tm_2d = len(material_grid.shape) == 2 and material_grid.polarization == "tm"
    in_plane_coupling = np.zeros(material_grid.shape, dtype=bool)
    if material_grid.magnetic_tensors and tm_2d:
        cell = expand_packed(material_grid.magnetic_tensors["mu_cell"])
        in_plane_coupling = np.abs(cell[3]) > 1e-10
    unfoldable = np.zeros(material_grid.shape, dtype=bool)
    static = np.zeros(material_grid.shape, dtype=bool)
    for medium, fraction in material_grid.tensor_dispersion:
        if medium.requires_static_correction:
            static |= np.asarray(fraction) > 1e-7
        if not foldable(medium):
            unfoldable |= np.asarray(fraction) > 1e-7
        elif tm_2d and any(
            p.weight[3] != 0
            or (getattr(p, "edot", None) is not None and p.edot[3] != 0)
            for p in medium.magnetic_poles
        ):
            in_plane_coupling |= np.asarray(fraction) > 1e-7
    for kind, ports in (("ModeSource", sources), ("ModeMonitor", monitors)):
        for port in ports:
            cells = port_cells(material_grid, port.center, port.size)
            if np.any(cells & in_plane_coupling):
                raise ValueError(
                    f"2D TM {kind} planes cannot yet overlap off-diagonal (xy) "
                    "permeability; use a diagonal permeability tensor at the "
                    "port or a 3D simulation."
                )
            if np.any(cells & unfoldable):
                raise ValueError(
                    f"{kind} planes cannot yet overlap gyrotropic media; "
                    "place the port outside them."
                )
            if np.any(cells & static):
                raise ValueError(
                    f"{kind} planes cannot yet overlap media with non-symmetric "
                    "tensors, off-diagonal conductivity or magnetic "
                    "conductivity; place the port outside them."
                )


# ---------------------------------------------------------------------------
# Pytree registration: plans can be rebuilt from traced design parameters and
# passed through jitted scans. Boxes are static metadata (slices are stored as
# (start, stop) pairs because slices are unhashable on older Pythons); every
# array is a leaf.


def _slice_key(part):
    return tuple((s.start, s.stop) for s in part)


def _slice_value(key):
    return tuple(slice(start, stop) for start, stop in key)


def _flatten_box(box: YeeBox):
    meta = (
        box.family,
        box.components,
        tuple(_slice_key(part) for part in box.slices),
        box.shapes,
        box.shared_shape,
        _slice_key(box.shared_slices),
        _slice_key(box.cell_slices),
        box.seams,
    )
    return (box.widths,), meta


def _unflatten_box(meta, children):
    family, components, slices, shapes, shared_shape, shared, cells, seams = meta
    return YeeBox(
        family,
        components,
        tuple(_slice_value(part) for part in slices),
        shapes,
        shared_shape,
        _slice_value(shared),
        _slice_value(cells),
        children[0],
        seams,
    )


def _register_fields(cls, data, meta=()):
    def flatten(value):
        children = tuple(getattr(value, f) for f in data)
        aux = tuple(getattr(value, f) for f in meta)
        return children, aux

    def unflatten(aux, children):
        return cls(
            **dict(zip(data, children, strict=True)),
            **dict(zip(meta, aux, strict=True)),
        )

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)


jax.tree_util.register_pytree_node(YeeBox, _flatten_box, _unflatten_box)
_register_fields(InverseOperator, ("diagonal", "offdiagonal", "skew"))
_register_fields(StaticTensorPlan, ("box", "inverse", "loss", "damping"))
_register_fields(
    TensorPolarizationRegion,
    ("box", "c1", "c2", "diagonal", "offdiagonal", "inverse"),
)
_register_fields(
    GyrotropicRegion,
    (
        "box",
        "rotation",
        "strength",
        "resonance_squared",
        "damping",
        "weights",
        "inverse",
    ),
    ("iterations",),
)
_register_fields(
    AnisotropicPlan, ("magnetic", "electric_regions", "magnetic_regions", "electric")
)
