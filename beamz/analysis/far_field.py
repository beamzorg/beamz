"""Differentiable 3D radiation in a homogeneous, lossless nonmagnetic medium.

The phasor convention is Re(E exp(-i omega t)). Equivalent currents are
J = n x H and M = -n x E. Returned angular fields omit exp(i k r)/r:
E(r s) ~ electric(s) exp(i k r)/r, with SI units throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.monitors import FieldMonitor
from beamz.lattice import plane_axes_3d, yee_plane_coordinates_3d
from beamz.simulation.observe import normalization_from_result

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class FarFieldData(NamedTuple):
    """Angular E/H arrays (frequency, direction, xyz), and their SI metadata."""

    electric: jax.Array
    magnetic: jax.Array
    directions: jax.Array
    frequencies: jax.Array
    impedance: jax.Array

    @property
    def intensity(self):
        """Radiated power per solid angle, in W/sr, shape (frequency, direction)."""
        return 0.5 * jnp.real(
            jnp.sum(
                jnp.cross(self.electric, jnp.conj(self.magnetic))
                * self.directions[None],
                axis=-1,
            )
        )

    @property
    def polarization(self):
        """Complex (E_theta, E_phi) in the global z-polar spherical basis.

        At either pole phi is defined to be zero.
        """
        s = self.directions
        phi = jnp.arctan2(s[:, 1], s[:, 0])
        sin_theta = jnp.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
        theta_hat = jnp.stack(
            (s[:, 2] * jnp.cos(phi), s[:, 2] * jnp.sin(phi), -sin_theta), -1
        )
        phi_hat = jnp.stack((-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)), -1)
        return (
            jnp.sum(self.electric * theta_hat, axis=-1),
            jnp.sum(self.electric * phi_hat, axis=-1),
        )

    def power(self, solid_angle_weights):
        """Integrate W/sr using explicit solid-angle quadrature, returning W."""
        weights = jnp.asarray(solid_angle_weights)
        if weights.shape != (self.directions.shape[0],):
            raise ValueError("solid_angle_weights must have one entry per direction.")
        return jnp.sum(self.intensity * weights, axis=-1)

    def overlap_power(self, target, solid_angle_weights):
        """Power in a coherent transverse target beam, one value per frequency.

        ``target`` is a complex Cartesian angular E field (direction, 3) or
        (frequency, direction, 3). Its scale cancels. Longitudinal components
        are discarded. A zero-norm target returns NaN rather than false success.
        """
        target = jnp.asarray(target)
        if target.shape not in (self.electric.shape, self.electric.shape[1:]):
            raise ValueError("target must match the angular Cartesian field shape.")
        s = self.directions
        target = target - s * jnp.sum(target * s, axis=-1, keepdims=True)
        w = jnp.asarray(solid_angle_weights)
        if w.shape != (s.shape[0],):
            raise ValueError("solid_angle_weights must have one entry per direction.")
        overlap = jnp.sum(
            jnp.sum(jnp.conj(target) * self.electric, axis=-1) * w, axis=-1
        )
        norm = jnp.sum(jnp.sum(jnp.abs(target) ** 2, axis=-1) * w, axis=-1)
        return (
            jnp.abs(overlap) ** 2
            / (2 * self.impedance)
            / jnp.where(norm > 0, norm, jnp.nan)
        )


def near_to_far(
    electric,
    magnetic,
    points,
    normals,
    area_weights,
    frequencies,
    directions,
    *,
    refractive_index=1.0,
):
    """Pure JAX equivalent-current far-field projection in three dimensions.

    E/H have shape (F, P, 3), points/normals (P, 3), areas (P,), frequencies
    (F,) in Hz and *unit* directions (D, 3). Normals point out of the enclosing
    surface (or toward the output half-space for an aperture). All surfaces
    must share a homogeneous medium with real positive index and mu_r=1.
    Geometry/medium inputs may also be differentiated. The validated static
    ``FarFieldProjection`` interface is preferable for recorded simulation data.

    Directions are mapped sequentially to bound working memory by O(F P),
    rather than allocating a direction-by-frequency-by-surface phase tensor.
    """
    electric, magnetic = jnp.asarray(electric), jnp.asarray(magnetic)
    points, normals = jnp.asarray(points), jnp.asarray(normals)
    weights, frequencies = jnp.asarray(area_weights), jnp.asarray(frequencies)
    directions = jnp.asarray(directions)
    if (
        electric.ndim != 3
        or electric.shape[-1] != 3
        or magnetic.shape != electric.shape
        or points.shape != electric.shape[1:]
        or normals.shape != points.shape
        or weights.shape != (points.shape[0],)
        or frequencies.shape != (electric.shape[0],)
        or directions.ndim != 2
        or directions.shape[-1] != 3
        or not electric.shape[0]
        or not electric.shape[1]
        or not directions.shape[0]
    ):
        raise ValueError("Incompatible near-to-far field, geometry or angular shapes.")
    impedance = jnp.sqrt(MU_0 / EPS_0) / refractive_index
    k = 2 * jnp.pi * frequencies * refractive_index / LIGHT_SPEED
    current = jnp.cross(normals, magnetic)
    magnetic_current = -jnp.cross(normals, electric)

    def project(s):
        phase = jnp.exp(-1j * k[:, None] * (points @ s)[None]) * weights
        j = jnp.sum(current * phase[..., None], axis=1)
        m = jnp.sum(magnetic_current * phase[..., None], axis=1)
        return (1j * k[:, None] / (4 * jnp.pi)) * (
            impedance * (j - s * jnp.sum(j * s, axis=-1, keepdims=True))
            - jnp.cross(s, m)
        )

    # Recompute each direction's phase in reverse mode instead of retaining a
    # direction-by-frequency-by-point residual tensor for the gradient.
    angular_e = jax.lax.map(checkpoint(project), directions).transpose(1, 0, 2)
    angular_h = jnp.cross(directions, angular_e) / impedance
    return FarFieldData(
        angular_e, angular_h, directions, frequencies, jnp.asarray(impedance)
    )


def _vectors(value, name, *, unit=False):
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not len(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite (N, 3) array.")
    if unit:
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        if not np.allclose(norms, 1, atol=1e-6, rtol=1e-6):
            raise ValueError(f"{name} must contain unit vectors.")
        # Accept roundoff in float32 angular grids, then restore exact unit
        # vectors for transverse projection and the NA boundary calculation.
        array = array / norms
    return tuple(tuple(float(v) for v in row) for row in array)


def _frequencies(value):
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or not array.size
        or not np.isfinite(array).all()
        or np.any(array <= 0)
        or np.unique(array).size != array.size
    ):
        raise ValueError("frequencies must be distinct positive finite values in Hz.")
    return tuple(float(f) for f in array)


@dataclass(frozen=True)
class FarFieldSurface:
    """A named FieldMonitor and an outward normal sign along its plane normal."""

    monitor: str
    normal_sign: int = 1

    def __post_init__(self):
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("surface monitor must be a nonempty name.")
        if isinstance(self.normal_sign, bool) or self.normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be +1 or -1.")


@dataclass(frozen=True)
class SurfaceFieldData:
    """Trace-preserving, source-normalized and time-aligned planar DFT fields.

    Fields are collocated using the solver's Yee interpolation, never a
    propagation phase guessed from a particular outgoing direction.
    """

    electric: Any
    magnetic: Any
    points: Any
    area_weights: Any
    frequencies: tuple[float, ...]
    normal_axis: int
    components: tuple[str, ...]
    source_power: float = 1.0


@dataclass(frozen=True)
class FarFieldProjection:
    """Reusable static projection for ordinary results or inverse-design data.

    ``solid_angle_weights`` are optional for directional intensity; integrated
    power and beam overlap require positive quadrature weights in steradians.
    A finite plane is a truncated aperture approximation. Use an enclosing box
    to recover full-space emission. This first implementation supports 3D only.
    """

    surfaces: tuple[FarFieldSurface, ...]
    directions: tuple[tuple[float, float, float], ...]
    solid_angle_weights: tuple[float, ...] | None = None
    refractive_index: float = 1.0
    frequencies: tuple[float, ...] | None = None

    def __post_init__(self):
        surfaces = tuple(self.surfaces)
        if not surfaces or not all(isinstance(s, FarFieldSurface) for s in surfaces):
            raise ValueError("surfaces must contain FarFieldSurface specifications.")
        if len({s.monitor for s in surfaces}) != len(surfaces):
            raise ValueError("Surface monitor names must be unique.")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(
            self, "directions", _vectors(self.directions, "directions", unit=True)
        )
        if (
            isinstance(self.refractive_index, bool)
            or not isinstance(self.refractive_index, Real)
            or not np.isfinite(self.refractive_index)
            or self.refractive_index <= 0
        ):
            raise ValueError("refractive_index must be finite, real and positive.")
        object.__setattr__(self, "refractive_index", float(self.refractive_index))
        if self.frequencies is not None:
            object.__setattr__(self, "frequencies", _frequencies(self.frequencies))
        if self.solid_angle_weights is not None:
            weights = np.asarray(self.solid_angle_weights, dtype=float)
            if (
                weights.shape != (len(self.directions),)
                or not np.isfinite(weights).all()
                or np.any(weights <= 0)
            ):
                raise ValueError(
                    "solid_angle_weights must match directions and be finite and positive."
                )
            object.__setattr__(
                self, "solid_angle_weights", tuple(float(w) for w in weights)
            )

    def canonical_spec(self):
        # Callable specifications must retain all geometry in checkpoint identity.
        return (
            self.surfaces,
            self.directions,
            self.solid_angle_weights,
            self.refractive_index,
            self.frequencies,
        )

    def __call__(self, data):
        if not isinstance(data, dict) and hasattr(data, "metadata"):
            data = surface_data(data, [s.monitor for s in self.surfaces])
        inputs = self._inputs(data)
        return near_to_far(
            *inputs,
            jnp.asarray(self.directions),
            refractive_index=self.refractive_index,
        )

    def _inputs(self, data):
        electric, magnetic, points, normals, weights = [], [], [], [], []
        first = data[self.surfaces[0].monitor]
        if not isinstance(first, SurfaceFieldData):
            raise ValueError("Far-field projection requires planar FieldMonitor data.")
        frequencies = self.frequencies or _frequencies(first.frequencies)
        source_power = first.source_power
        for surface in self.surfaces:
            field = data[surface.monitor]
            if not isinstance(field, SurfaceFieldData):
                raise ValueError(
                    "Far-field projection requires planar FieldMonitor data."
                )
            if field.source_power != source_power:
                raise ValueError("Surface source powers must agree.")
            tangential = [i for i in range(3) if i != field.normal_axis]
            required = {prefix + "xyz"[i] for prefix in ("E", "H") for i in tangential}
            if not required <= set(field.components):
                raise ValueError(
                    "Every projection surface must record tangential E and H."
                )
            indices = _frequency_indices(field.frequencies, frequencies)
            electric.append(jnp.asarray(field.electric)[indices])
            magnetic.append(jnp.asarray(field.magnetic)[indices])
            points.append(jnp.asarray(field.points))
            normal = np.eye(3)[field.normal_axis] * surface.normal_sign
            normals.append(jnp.broadcast_to(normal, np.shape(field.points)))
            weights.append(jnp.asarray(field.area_weights))
        return (
            jnp.concatenate(electric, axis=1),
            jnp.concatenate(magnetic, axis=1),
            jnp.concatenate(points),
            jnp.concatenate(normals),
            jnp.concatenate(weights),
            jnp.asarray(frequencies),
        )


def _frequency_indices(recorded, requested):
    indices = []
    for frequency in requested:
        match = np.flatnonzero(np.isclose(recorded, frequency, rtol=1e-10, atol=0))
        if match.size != 1:
            raise ValueError(
                f"Surface must record frequency {frequency:g} Hz exactly once."
            )
        indices.append(int(match[0]))
    return jnp.asarray(indices)


def near_field_box(*, center, size, freqs, name="near_field"):
    """Return (six tangential FieldMonitors, six outward FarFieldSurfaces).

    Place the entire box outside scatterers and sources, inside the PML, in
    one homogeneous medium. Sources/scatterers may lie inside the box.
    """
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    if (
        center.shape != (3,)
        or size.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(size).all()
        or np.any(size <= 0)
    ):
        raise ValueError("Box center and positive size must be finite xyz triples.")
    monitors, surfaces = [], []
    for axis in range(3):
        extent = size.copy()
        extent[axis] = 0
        components = tuple(c for c in _COMPONENTS if c[-1] != "xyz"[axis])
        for sign in (-1, 1):
            position = center.copy()
            position[axis] += sign * size[axis] / 2
            label = f"{name}_{'xyz'[axis]}{'+' if sign > 0 else '-'}"
            monitors.append(
                FieldMonitor(
                    center=tuple(position),
                    size=tuple(extent),
                    freqs=freqs,
                    name=label,
                    fields=components,
                )
            )
            surfaces.append(FarFieldSurface(label, sign))
    return tuple(monitors), tuple(surfaces)


def _surface_metadata(results, name):
    result = results[name]
    monitor = result.monitor
    if (
        not results.metadata.is_3d
        or not isinstance(monitor, FieldMonitor)
        or np.count_nonzero(np.asarray(monitor.size) == 0) != 1
    ):
        raise ValueError(
            "Radiation projection requires 3D planar FieldMonitor outputs."
        )
    grid = results.metadata.grid
    grid = (
        grid
        if grid is not None
        and (grid.metric_kind != "isotropic_uniform" or grid.origin != (0.0, 0.0, 0.0))
        else None
    )
    if result.sample_region is None:
        raise ValueError("The field monitor is missing its sampling geometry.")
    axis = "xyz".index(monitor.plane_normal)
    c0, c1 = yee_plane_coordinates_3d(
        monitor.center,
        monitor.size,
        monitor.plane_normal,
        result.sample_region,
        grid=grid,
    )
    coords = np.meshgrid(c0, c1, indexing="ij")
    points = np.broadcast_to(monitor.center, (coords[0].size, 3)).copy()
    for coordinate, label in zip(
        coords, plane_axes_3d(monitor.plane_normal), strict=True
    ):
        points[:, "xyz".index(label)] = coordinate.ravel()
    weights = np.asarray(result.integration_weights)
    if weights.size != len(points):
        weights = np.full(len(points), result.power_scale)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Surface area weights must be finite and positive.")
    return points, weights, axis


def _normalization(results, name):
    normalization = normalization_from_result(results, results[name], source=0)
    if normalization is None:
        raise ValueError("The source has no usable spectrum for radiation projection.")
    norm = np.asarray(normalization.field_amplitude_norm)
    if not np.isfinite(norm).all() or np.any(np.abs(norm) < 1e-12):
        raise ValueError("The source spectrum is too small for radiation projection.")
    power = float(results.sources[0].power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError("The incident source power must be positive.")
    return norm, power


def surface_data(results, names):
    """Decode ordinary results into the same normalized surface-data contract."""
    data = {}
    for name in names:
        result = results[name]
        points, weights, axis = _surface_metadata(results, name)
        norm, power = _normalization(results, name)
        raw_fields = result._raw_dft_fields or {}
        values = np.stack(
            [
                # SimulationResults may already expose source-normalized views.
                # Always start from retained raw accumulators, then apply our
                # single explicit normalization, just as the bound state does.
                np.asarray(raw_fields[c]).reshape(len(norm), -1)
                * result.dft_amplitude_scale
                / np.maximum(result.dft_weight_sum[:, None], 1e-18)
                if c in raw_fields
                else np.zeros((len(norm), len(points)), complex)
                for c in _COMPONENTS
            ]
        )
        values[3:] *= np.exp(-1j * np.pi * result.dft_base_dt * result.dft_frequencies)[
            None, :, None
        ]
        values /= norm[None, :, None]
        data[name] = SurfaceFieldData(
            values[:3].transpose(1, 2, 0),
            values[3:].transpose(1, 2, 0),
            points,
            weights,
            tuple(float(f) for f in result.dft_frequencies),
            axis,
            tuple(result.dft_fields),
            power,
        )
    return data


def bind_surface_data(results, program, names):
    """Bind fixed geometry and normalization; retain derivatives through DFTs."""
    if names and program.boundary.periodic_axes:
        raise ValueError("Radiation surface outputs require nonperiodic boundaries.")
    plans = []
    for name in names:
        points, weights, axis = _surface_metadata(results, name)
        norm, power = _normalization(results, name)
        spec = next(s for s in program.monitors if s.name == name)
        if len(points) != spec.dft_point_count:
            raise ValueError("Surface coordinates and DFT sampling shapes differ.")
        plans.append((name, spec, points, weights, axis, jnp.asarray(norm), power))

    def data(state):
        output = {}
        for name, spec, points, weights, axis, norm, power in plans:
            start = spec.dft_value_offset
            count = 6 * spec.freq_count * spec.dft_point_count
            fields = (
                state.dft_vec_re[start : start + count]
                + 1j * state.dft_vec_im[start : start + count]
            ).reshape(6, spec.freq_count, spec.dft_point_count)
            fields = fields * spec.dft_component_mask[:, None, None]
            w = state.dft_weight_sum[
                spec.dft_weight_offset : spec.dft_weight_offset + spec.freq_count
            ]
            fields = fields * (2 / jnp.maximum(w, 1e-18) / norm)[None, :, None]
            frequencies = results[name].dft_frequencies
            phase = jnp.exp(-1j * jnp.pi * program.config.dt * jnp.asarray(frequencies))
            fields = fields.at[3:].multiply(phase[None, :, None])
            output[name] = SurfaceFieldData(
                fields[:3].transpose(1, 2, 0),
                fields[3:].transpose(1, 2, 0),
                points,
                weights,
                tuple(float(f) for f in frequencies),
                axis,
                tuple(results[name].dft_fields),
                power,
            )
        return output

    return data
