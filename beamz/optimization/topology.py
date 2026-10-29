"""Immutable topology-optimization configuration, state, and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from beamz._cache_tokens import cache_token

# Defer imports to avoid circular dependencies if any,
# or import at top level if safe. design shouldn't depend on optimization.
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Structure
from beamz.devices._immutable import readonly_array as _readonly_array

from ._validation import fixed_density_masks
from .autodiff import compute_parameter_gradient_vjp, transform_density
from .projections import validate_projection_options


@dataclass(frozen=True, slots=True, eq=False)
class TopologyState:
    """Latent density, optimizer state, and completed objective values.

    ``density`` uses ``(y, x)`` order.
    """

    density: np.ndarray
    optimizer_state: Any = field(default=None, repr=False, compare=False)
    objective_history: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "density", _readonly_array(self.density, float))
        object.__setattr__(
            self, "objective_history", tuple(float(v) for v in self.objective_history)
        )

    def with_objective(self, value: float) -> "TopologyState":
        """Return state with one completed objective value appended."""
        return replace(self, objective_history=(*self.objective_history, float(value)))

    def canonical_spec(self):
        """Return evolving values included in state cache identity."""
        return self.density, self.objective_history

    def __eq__(self, other):
        if not isinstance(other, TopologyState):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))


@dataclass(frozen=True, slots=True, eq=False)
class TopologySpec:
    """Immutable density-based topology optimization.

    ``region_mask`` is ``(y, x)``. ``resolution`` is required and is metres.
    ``projection_type`` is ``"heaviside"``, ``"ssp"``, or ``"identity"``; use
    identity with zero filter radius for externally mapped densities.
    ``fixed_solid`` and ``fixed_void`` pin cells inside the mask to density 1
    and 0, including during export. Masks are copied read-only. Evolving values
    live in :class:`TopologyState`.
    """

    design: Design
    region_mask: np.ndarray
    optimizer: str = "adam"
    learning_rate: float = 0.1
    filter_radius: float = 0.0
    projection_eta: float = 0.5
    projection_type: str = "heaviside"
    ssp_smoothing_radius: float = 0.55
    beta_schedule: tuple[float, float] = (1.0, 20.0)
    eps_min: float = 1.0
    eps_max: float = 12.0
    resolution: float | None = None
    filter_type: str = "conic"
    morphology_operation: str = "openclose"
    morphology_smooth_tau: float = 0.01
    fixed_solid: Any = field(default=None, repr=False, compare=False)
    fixed_void: Any = field(default=None, repr=False, compare=False)
    fixed_structure_mask: np.ndarray = field(init=False, repr=False)
    filter_radius_cells: int = field(init=False)

    def __post_init__(self):
        if self.resolution is None:
            raise ValueError("TopologySpec requires explicit `resolution` (meters).")
        resolution = float(self.resolution)
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution!r}")
        optimizer = str(self.optimizer).lower()
        if optimizer not in {"adam", "sgd"}:
            raise ValueError(
                f"Unknown optimizer {self.optimizer!r}. Supported: 'adam', 'sgd'"
            )
        projection_type = str(self.projection_type).lower()
        ssp_smoothing_radius = float(self.ssp_smoothing_radius)
        validate_projection_options(projection_type, ssp_smoothing_radius)
        mask = _readonly_array(self.region_mask, bool)
        solid, void = fixed_density_masks(self.fixed_solid, self.fixed_void, mask.shape)
        for pinned in (solid, void):
            if pinned is not None and np.any(pinned & ~mask):
                raise ValueError("Fixed density cells must lie inside region_mask.")
        object.__setattr__(self, "fixed_solid", solid)
        object.__setattr__(self, "fixed_void", void)
        base_grid = self.design.rasterize(resolution)
        fixed = get_fixed_structure_mask(
            base_grid, float(self.eps_min), float(self.eps_max), mask
        )
        object.__setattr__(self, "region_mask", mask)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "projection_type", projection_type)
        object.__setattr__(self, "ssp_smoothing_radius", ssp_smoothing_radius)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(
            self, "beta_schedule", tuple(float(v) for v in self.beta_schedule)
        )
        object.__setattr__(self, "fixed_structure_mask", _readonly_array(fixed, bool))
        object.__setattr__(
            self,
            "filter_radius_cells",
            int(round(float(self.filter_radius) / resolution)),
        )

    def initial_state(self, density: np.ndarray | None = None) -> TopologyState:
        """Create state. Optimizable cells start at 0.5 when ``density`` is omitted."""
        if density is None:
            density = np.zeros(self.region_mask.shape, dtype=float)
            density[self.region_mask] = 0.5
        elif np.shape(density) != self.region_mask.shape:
            raise ValueError(
                f"density shape {np.shape(density)} does not match mask "
                f"shape {self.region_mask.shape}"
            )
        return TopologyState(density)

    def beta(self, step: int, total_steps: int) -> float:
        """Return the projection strength for a zero-based continuation step."""
        start, end = self.beta_schedule
        if total_steps <= 1:
            return end
        return start + step / (total_steps - 1) * (end - start)

    def physical_density(self, state: TopologyState, beta: float) -> np.ndarray:
        """Transform latent density into a detached physical density."""
        import jax.numpy as jnp

        density = transform_density(
            jnp.asarray(state.density), beta=beta, **self._transform_options
        )
        return _readonly_array(density, float)

    @property
    def _transform_options(self):
        """Use identical filter/projection settings for forward maps and VJPs."""
        import jax.numpy as jnp

        return dict(
            mask=jnp.asarray(self.region_mask),
            eta=self.projection_eta,
            radius=self.filter_radius_cells,
            filter_type=self.filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=jnp.asarray(self.fixed_structure_mask),
            projection_type=self.projection_type,
            ssp_smoothing_radius=self.ssp_smoothing_radius,
            fixed_solid=self.fixed_solid,
            fixed_void=self.fixed_void,
        )

    def density_for_step(
        self, state: TopologyState, step: int, total_steps: int
    ) -> tuple[float, np.ndarray]:
        """Return continuation beta and physical density for one iteration."""
        beta = self.beta(step, total_steps)
        return beta, self.physical_density(state, beta)

    def apply_gradient(
        self, state: TopologyState, grad_eps: np.ndarray, beta: float
    ) -> tuple[TopologyState, float]:
        """Advance one gradient-ascent step from a permittivity gradient."""
        import jax.numpy as jnp

        grad_eps = self.gradient_to_design_grid(grad_eps)
        grad_physical = grad_eps * (self.eps_max - self.eps_min)
        grad_param = compute_parameter_gradient_vjp(
            jnp.asarray(state.density),
            jnp.asarray(grad_physical),
            beta=beta,
            **self._transform_options,
        )
        return self.apply_parameter_gradient(state, grad_param)

    def apply_parameter_gradient(self, state: TopologyState, gradient):
        """Advance using a gradient already differentiated to latent density."""
        import jax.numpy as jnp

        grad_param = jnp.asarray(gradient)
        if grad_param.shape != state.density.shape:
            raise ValueError("Parameter gradient must match the density shape.")
        grad_param = jnp.where(self.region_mask, grad_param, 0.0)
        optimizer = self._optimizer()
        opt_state = state.optimizer_state
        if opt_state is None:
            opt_state = optimizer.init(jnp.asarray(state.density))
        updates, opt_state = optimizer.update(-grad_param, opt_state)
        update = np.asarray(updates)
        density = np.asarray(state.density).copy()
        density[self.region_mask] += update[self.region_mask]
        density = np.clip(density, 0.0, 1.0)
        return replace(state, density=density, optimizer_state=opt_state), float(
            np.max(np.abs(update))
        )

    def gradient_to_design_grid(self, grad_eps: np.ndarray) -> np.ndarray:
        """Collocate a field-grid permittivity gradient onto design cells."""
        return fold_high_side_yee_padding_to_shape(grad_eps, self.region_mask.shape)

    def _optimizer(self):
        import optax

        factory = optax.adam if self.optimizer == "adam" else optax.sgd
        return factory(learning_rate=self.learning_rate)


def compute_overlap_gradient(
    forward_fields_history,
    adjoint_fields_history,
    field_key="Ez",
    forward_start=0,
    adjoint_start=0,
):
    """
    Compute the gradient of the overlap integral with respect to epsilon.
    Gradient = Re(E_fwd * E_adj) integrated over time.
    """
    if len(forward_fields_history) == 0 or len(adjoint_fields_history) == 0:
        raise ValueError("Field histories must be non-empty.")

    forward_start = max(int(forward_start), 0)
    adjoint_start = max(int(adjoint_start), 0)
    fwd_hist = forward_fields_history[forward_start:]
    adj_hist = adjoint_fields_history[adjoint_start:]

    grad = np.zeros_like(forward_fields_history[0], dtype=float)

    n_steps = min(len(fwd_hist), len(adj_hist))
    if n_steps <= 0:
        return grad

    for i in range(n_steps):
        grad += fwd_hist[i] * adj_hist[n_steps - 1 - i]

    return grad


def fold_high_side_yee_padding_to_shape(
    grad: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Add a one-cell high-side Yee plane into the last material-grid index."""

    out = np.asarray(grad, dtype=float)
    target_shape = tuple(int(v) for v in target_shape)
    if out.shape == target_shape:
        return out
    if out.ndim != len(target_shape):
        raise ValueError(
            f"Gradient ndim {out.ndim} does not match target ndim {len(target_shape)}."
        )

    for axis, target in enumerate(target_shape):
        size = out.shape[axis]
        if size == target:
            continue
        if size != target + 1:
            raise ValueError(
                "Cannot align gradient shape "
                f"{out.shape} to target shape {target_shape}: axis {axis} has "
                f"size {size}, expected {target} or {target + 1}."
            )

        core: list[Any] = [slice(None)] * out.ndim
        core[axis] = slice(0, target)
        folded = out[tuple(core)].copy()

        last: list[Any] = [slice(None)] * out.ndim
        last[axis] = target - 1
        high: list[Any] = [slice(None)] * out.ndim
        high[axis] = target
        folded[tuple(last)] += out[tuple(high)]
        out = folded

    return out


def create_optimization_mask(grid, region_structure):
    """Rasterize a structure into a boolean mask aligned with the material grid."""
    # Create temp design to rasterize mask exactly as grid does
    temp_design = Design(
        width=grid.width, height=grid.height, material=Material(permittivity=1.0)
    )

    # Set to a distinct permittivity to detect it without mutating the input.
    marker = Material(permittivity=2.0)
    if not isinstance(region_structure, Structure):
        raise TypeError("Optimization regions must be canonical Structure objects.")
    temp_design += region_structure.with_material(marker)

    # Use the production Rust rasterizer so optimization masks follow the same
    # geometry classification as the solver material grid.
    temp_grid = temp_design.rasterize(resolution=grid.dx)

    # Mask is where permittivity > background
    # Use a safe threshold to include any partial fill
    mask = temp_grid.permittivity > 1.001

    return mask


def get_fixed_structure_mask(grid, eps_min, eps_max, design_mask):
    """Return solid cells outside the design region."""
    # Threshold for "solid" material. E.g. > 90% of core-clad difference
    threshold = eps_min + 0.9 * (eps_max - eps_min)

    # High permittivity regions
    high_eps = grid.permittivity >= threshold

    # Exclude design region (we only care about fixed structures outside)
    fixed_structures = high_eps & (~design_mask)

    return fixed_structures
