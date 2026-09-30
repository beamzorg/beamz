"""Experimental time-domain FIT with diagonal or coupled constitutive operators.

The first backend has full-domain PEC walls, a uniform orthogonal mesh,
positive nondispersive materials, dielectric interfaces, and impressed currents.
It does not yet use BEAMZ's FDTD source, monitor, or PML compilers.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.simulation.fit import topology
from beamz.simulation.fit.constitutive import CoupledElectricOperator
from beamz.simulation.fit.interfaces import (
    FITInterfaceMaterial,
    PlanarDielectricInterface,
)
from beamz.simulation.fit.mesh import UniformFITMesh


class FITState(NamedTuple):
    """Voltages e at n dt, magnetic fluxes b at (n - 1/2) dt."""

    e: dict
    b: dict
    step: jax.Array
    # Coupled interfaces evolve displacement; e is cached as K d.
    d: dict | None = None
    max_solve_residual: jax.Array | None = None


@dataclass(frozen=True)
class FITCurrentSource:
    """Impressed electric current density in A/m² at a native E sample.

    Position is in metres in (x, y) or (x, y, z) order. Signal entries are
    sampled at (n + 1/2) dt and are zero beyond the supplied waveform.
    A point source occupies one dual face; its integrated current therefore
    changes with mesh resolution. This is not a calibrated electrical port.
    """

    component: str
    position: tuple[float, ...]
    signal: object


@dataclass(frozen=True)
class FITResults:
    """Native E/H fields at the final step and optional field histories."""

    mesh: UniformFITMesh
    fields: dict
    history: dict
    times: jax.Array
    dt: float
    final_step: int

    def to_xarray(self):
        """Export native staggering and E/H sample times without colocation."""
        import xarray as xr

        axes = ("y", "x") if self.mesh.dimension == 2 else ("z", "y", "x")
        variables = {}
        for name, final in self.fields.items():
            dims = tuple(f"{axis}_{name}" for axis in axes)
            coords = dict(zip(dims, self.mesh.coordinates(name)))
            if name in self.history:
                time_dim = "t_e" if name.startswith("E") else "t_h"
                coords[time_dim] = np.asarray(self.times) - (
                    0.5 * self.dt if name.startswith("H") else 0.0
                )
                data, variable_dims = self.history[name], (time_dim, *dims)
            else:
                data, variable_dims = final, dims
            variables[name] = xr.DataArray(
                np.asarray(data),
                dims=variable_dims,
                coords=coords,
                attrs={"units": "V/m" if name.startswith("E") else "A/m"},
            )
        dataset = xr.Dataset(
            variables,
            attrs={"backend": "fit", "dt": self.dt, "final_step": self.final_step},
        )
        for coord in dataset.coords:
            dataset.coords[coord].attrs["units"] = (
                "s" if coord.startswith("t_") else "m"
            )
        return dataset


def _cell_to_native(values, shape):
    """Arithmetic averaging on nodal axes; nearest extension at walls.

    This is an initial diagonal material approximation, not an interface-aware
    or conformal constitutive law. Preserve that distinction in comparisons.
    """
    result = np.asarray(values)
    for axis, target in enumerate(shape):
        if target == result.shape[axis] + 1:
            pads = [(0, 0)] * result.ndim
            pads[axis] = (1, 1)
            padded = np.pad(result, pads, mode="edge")
            left, right = [slice(None)] * result.ndim, [slice(None)] * result.ndim
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            result = 0.5 * (padded[tuple(left)] + padded[tuple(right)])
    return result


class FITSimulation:
    """Independent FIT engine alongside Simulation, with all walls PEC.

    Permittivity/permeability are relative values, scalar or cell arrays of
    ``mesh.shape``. Conductivity is in S/m. State uses integrated SI quantities;
    ``fields`` exposes native E in V/m and H in A/m. The stable timestep uses
    BEAMZ's EPS_0/MU_0 constants, rather than a separately rounded wave speed.

    ``interface`` enables a lossless coupled electric operator and a matrix-free
    constitutive solve per step. In this mode ``m_epsilon`` is only the scalar
    reference; ``electric_operator`` defines the actual permittivity matrix.
    """

    def __init__(
        self,
        mesh: UniformFITMesh,
        *,
        permittivity=1.0,
        permeability=1.0,
        conductivity=0.0,
        interface=None,
        interface_solve_tolerance=None,
        interface_solve_maxiter=256,
        dt=None,
        courant=0.9,
        sources=(),
        precision="float32",
    ):
        if not isinstance(mesh, UniformFITMesh):
            raise TypeError("mesh must be a UniformFITMesh")
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")
        if precision == "float64" and not jax.config.jax_enable_x64:
            raise ValueError("Enable JAX x64 before requesting float64")
        if not np.isfinite(courant) or not 0 < courant <= 1:
            raise ValueError("courant must be in (0, 1]")
        self.mesh = mesh
        self.dtype = jnp.float64 if precision == "float64" else jnp.float32
        self.interface_solve_tolerance = float(
            interface_solve_tolerance
            if interface_solve_tolerance is not None
            else (1e-11 if precision == "float64" else 5e-7)
        )
        if (
            not np.isfinite(self.interface_solve_tolerance)
            or not 0 < self.interface_solve_tolerance < 1
        ):
            raise ValueError("interface_solve_tolerance must be in (0, 1)")
        if (
            isinstance(interface_solve_maxiter, bool)
            or not isinstance(interface_solve_maxiter, Integral)
            or interface_solve_maxiter < 1
        ):
            raise ValueError("interface_solve_maxiter must be a positive integer")
        self.interface_solve_maxiter = int(interface_solve_maxiter)
        if isinstance(interface, PlanarDielectricInterface):
            interface = interface.material(mesh)
        if interface is not None and not isinstance(interface, FITInterfaceMaterial):
            raise TypeError(
                "interface must be PlanarDielectricInterface or FITInterfaceMaterial"
            )
        self.electric_operator = None
        if interface is not None:
            if np.ndim(permittivity) != 0 or float(permittivity) != 1.0:
                raise ValueError(
                    "Specify permittivity through interface, not both arguments"
                )
            if np.any(np.asarray(conductivity) != 0):
                raise ValueError(
                    "Coupled interface materials currently require zero conductivity"
                )
            self.electric_operator = CoupledElectricOperator(
                mesh, interface, self.dtype
            )
            permittivity = self.electric_operator.parallel_permittivity
        self.interface_material = interface

        def cells(value, name, positive):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.full(mesh.shape, arr)
            if arr.shape != mesh.shape:
                raise ValueError(
                    f"{name} must be scalar or have cell shape {mesh.shape}"
                )
            if not np.all(np.isfinite(arr)) or np.any(
                arr <= 0 if positive else arr < 0
            ):
                qualifier = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {qualifier}")
            return arr

        eps = cells(permittivity, "permittivity", True)
        mu = cells(permeability, "permeability", True)
        sigma = cells(conductivity, "conductivity", False)
        self.max_dt = (
            mesh.spacing
            * np.sqrt(
                EPS_0
                * MU_0
                * (
                    eps.min()
                    if self.electric_operator is None
                    else self.electric_operator.epsilon_lower_bound
                )
                * mu.min()
            )
            / np.sqrt(mesh.dimension)
        )
        self.dt = float(courant * self.max_dt if dt is None else dt)
        if not np.isfinite(self.dt) or self.dt <= 0 or self.dt > self.max_dt:
            raise ValueError(f"dt must be positive and <= the CFL bound {self.max_dt}")
        self.m_epsilon, self.m_conductivity, self.e_masks = {}, {}, {}
        for name, shape in mesh.electric_shapes.items():
            metric = mesh.dual_area(name) / mesh.edge_length(name)
            self.m_epsilon[name] = jnp.asarray(
                EPS_0 * _cell_to_native(eps, shape) * metric, self.dtype
            )
            self.m_conductivity[name] = jnp.asarray(
                _cell_to_native(sigma, shape) * metric, self.dtype
            )
            self.e_masks[name] = jnp.asarray(mesh.pec_mask(name))
        self.m_reluctivity = {
            name: jnp.asarray(
                mesh.dual_edge_length(name)
                / (MU_0 * _cell_to_native(mu, shape) * mesh.face_area(name)),
                self.dtype,
            )
            for name, shape in mesh.magnetic_shapes.items()
        }
        self.state = FITState(
            {k: jnp.zeros(s, self.dtype) for k, s in mesh.electric_shapes.items()},
            {k: jnp.zeros(s, self.dtype) for k, s in mesh.magnetic_shapes.items()},
            jnp.asarray(0, dtype=jnp.int32),
            {k: jnp.zeros(s, self.dtype) for k, s in mesh.electric_shapes.items()}
            if self.electric_operator is not None
            else None,
            jnp.asarray(0, self.dtype) if self.electric_operator is not None else None,
        )
        self._sources = tuple(self._prepare_source(source) for source in sources)
        self._compiled_step = jax.jit(self.advance)
        self._compiled_run = jax.jit(
            self._scan, static_argnames=("num_steps", "record_fields")
        )

    @classmethod
    def from_design(cls, design, *, resolution, polarization="tm", **kwargs):
        """Rasterize a dimensioned Design into the initial diagonal FIT model.

        Geometry must fit an integer number of uniform cells. This adapter
        inherits the current scalar rasterization and makes no new interface
        accuracy claim. Domain origins and centered geometry are not translated.
        """
        if getattr(design, "_centered_coordinates", False):
            raise ValueError(
                "FIT from_design requires explicit domain dimensions and origin-zero geometry"
            )
        sizes = (design.height, design.width)
        if design.is_3d:
            sizes = (design.depth, *sizes)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        counts = np.asarray(sizes, dtype=float) / resolution
        rounded = np.rint(counts).astype(int)
        if not np.allclose(counts, rounded, rtol=0, atol=1e-8):
            raise ValueError(
                "Domain dimensions must be integer multiples of resolution"
            )
        mesh = UniformFITMesh(tuple(rounded), resolution, polarization)
        grid = design.rasterize(resolution=resolution)
        return cls(
            mesh,
            permittivity=grid.permittivity,
            permeability=grid.permeability,
            conductivity=grid.conductivity,
            **kwargs,
        )

    def _prepare_source(self, source):
        if not isinstance(source, FITCurrentSource):
            raise TypeError("FIT accepts FITCurrentSource specifications only")
        if source.component not in self.mesh.electric_shapes:
            raise ValueError(f"Inactive electric source component: {source.component}")
        position = np.asarray(source.position, dtype=float)
        if position.shape != (self.mesh.dimension,) or not np.all(
            np.isfinite(position)
        ):
            raise ValueError("Source position must contain finite x/y[/z] coordinates")
        if np.any(position < 0) or np.any(
            position[::-1] > np.asarray(self.mesh.shape) * self.mesh.spacing
        ):
            raise ValueError("Source position lies outside the domain")
        coords = self.mesh.coordinates(source.component)
        index = tuple(
            int(np.argmin(abs(c - p))) for c, p in zip(coords, position[::-1])
        )
        if self.mesh.pec_mask(source.component)[index]:
            raise ValueError("Source maps to an electric edge constrained by PEC")
        signal = np.asarray(source.signal, dtype=float)
        if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
            raise ValueError(
                "Source signal must be a nonempty finite one-dimensional waveform"
            )
        area = float(self.mesh.dual_area(source.component)[index])
        return source.component, index, jnp.asarray(signal * area, self.dtype)

    def _topology(self, function, values):
        return function(
            values, dimension=self.mesh.dimension, polarization=self.mesh.polarization
        )

    def advance(self, state, electric_current=None):
        """Pure leapfrog step; optional current contains integrated amperes.

        e and b enter at n and n-1/2, respectively. Update b to n+1/2,
        then e to n+1. Ohmic loss uses a time-centered electric voltage.
        PEC masking also enforces zero impressed current on constrained edges.
        """
        ce = self._topology(topology.curl, state.e)
        b = {k: state.b[k] - self.dt * ce[k] for k in state.b}
        h = {k: self.m_reluctivity[k] * b[k] for k in b}
        rhs = self._topology(topology.curl_transpose, h)
        current = {k: jnp.zeros_like(v) for k, v in state.e.items()}
        if electric_current is not None:
            unknown = set(electric_current) - set(current)
            if unknown:
                raise ValueError(f"Inactive current components: {sorted(unknown)}")
            for name, value in electric_current.items():
                value = jnp.asarray(value, self.dtype)
                if value.shape != current[name].shape:
                    raise ValueError(
                        f"Current {name} must have shape {current[name].shape}"
                    )
                current[name] = value
        for name, index, signal in self._sources:
            amplitude = jnp.where(
                state.step < signal.size,
                signal[jnp.minimum(state.step, signal.size - 1)],
                0.0,
            )
            current[name] = current[name].at[index].add(amplitude)
        if self.electric_operator is not None:
            if state.d is None:
                raise ValueError(
                    "Coupled stepping needs electric flux state; initialize with set_fields"
                )
            d = {
                name: jnp.where(
                    self.e_masks[name], 0.0, old + self.dt * (rhs[name] - current[name])
                )
                for name, old in state.d.items()
            }
            e, residual = self.electric_operator.solve(
                d,
                initial=state.e,
                tolerance=self.interface_solve_tolerance,
                maxiter=self.interface_solve_maxiter,
            )
            return FITState(
                e, b, state.step + 1, d, jnp.maximum(state.max_solve_residual, residual)
            )
        e = {}
        for name, old in state.e.items():
            m, g = self.m_epsilon[name], self.m_conductivity[name]
            updated = (
                (m - 0.5 * self.dt * g) * old + self.dt * (rhs[name] - current[name])
            ) / (m + 0.5 * self.dt * g)
            e[name] = jnp.where(self.e_masks[name], 0.0, updated)
        return FITState(e, b, state.step + 1)

    def fields_from_state(self, state):
        """Convert integrated voltages/fluxes into native E/H field values."""
        fields = {k: v / self.mesh.edge_length(k) for k, v in state.e.items()}
        fields.update(
            {
                "H" + k[-1]: self.m_reluctivity[k]
                * v
                / jnp.asarray(self.mesh.dual_edge_length(k), self.dtype)
                for k, v in state.b.items()
            }
        )
        return fields

    @property
    def fields(self):
        return self.fields_from_state(self.state)

    def set_fields(self, **fields):
        """Set initial native fields; PEC tangential E/normal B are zeroed.

        Unspecified components retain their values. H is the half-step field
        at (current_step - 1/2) dt, not the electric timestamp.
        """
        e, b = dict(self.state.e), dict(self.state.b)
        for name, value in fields.items():
            if name not in self.fields:
                raise ValueError(f"Inactive field component: {name}")
            arr = np.asarray(value, dtype=float)
            if arr.shape != self.fields[name].shape or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} must be finite and have shape {self.fields[name].shape}"
                )
            if name.startswith("E"):
                e[name] = jnp.where(
                    self.e_masks[name],
                    0.0,
                    jnp.asarray(arr * self.mesh.edge_length(name), self.dtype),
                )
            else:
                key = "B" + name[-1]
                if self.mesh.dimension == 3 or key != "Bz":
                    axis = self.mesh._axis(key)
                    low, high = [slice(None)] * arr.ndim, [slice(None)] * arr.ndim
                    low[axis], high[axis] = 0, -1
                    arr = arr.copy()
                    arr[tuple(low)] = arr[tuple(high)] = 0.0
                b[key] = (
                    jnp.asarray(arr, self.dtype)
                    * jnp.asarray(self.mesh.dual_edge_length(key), self.dtype)
                    / self.m_reluctivity[key]
                )
        d = self.state.d
        if self.electric_operator is not None and any(
            name.startswith("E") for name in fields
        ):
            d = self.electric_operator.apply_permittivity(e)
        self.state = FITState(e, b, self.state.step, d, self.state.max_solve_residual)

    def _check_constitutive_solve(self):
        if self.electric_operator is not None:
            residual = float(self.state.max_solve_residual)
            if (
                not np.isfinite(residual)
                or residual > 20 * self.interface_solve_tolerance
            ):
                raise RuntimeError(
                    f"Constitutive CG did not converge: maximum relative residual={residual}; increase interface_solve_maxiter or adjust tolerance"
                )

    def step(self):
        self.state = self._compiled_step(self.state)
        self._check_constitutive_solve()
        return self.state

    def _scan(self, state, *, num_steps, record_fields):
        def body(carry, unused):
            updated = self.advance(carry)
            fields = self.fields_from_state(updated)
            return updated, {name: fields[name] for name in record_fields}

        return jax.lax.scan(body, state, xs=None, length=num_steps)

    def run(self, num_steps, *, record_fields=()):
        """Advance additional steps; recording is opt-in and stores every step."""
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, Integral)
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a nonnegative integer")
        if isinstance(record_fields, str):
            raise TypeError("record_fields must be a sequence of component names")
        record_fields = tuple(record_fields)
        if len(set(record_fields)) != len(record_fields) or set(record_fields) - set(
            self.fields
        ):
            raise ValueError("record_fields must contain unique active E/H names")
        first_step = int(self.state.step)
        self.state, history = self._compiled_run(
            self.state, num_steps=int(num_steps), record_fields=record_fields
        )
        self._check_constitutive_solve()
        times = (jnp.arange(num_steps, dtype=self.dtype) + first_step + 1) * self.dt
        return FITResults(
            self.mesh, self.fields, history, times, self.dt, first_step + num_steps
        )

    def energy(self, state=None, *, conserved=False):
        """Energy in J (2D: J/m); optional leapfrog conserved quadratic.

        With no sources/loss and below CFL, the conserved expression is
        1/2 eᵀ Mε e + 1/2 bᵀ Mν b - dt/2 (C e)ᵀ Mν b.
        Physical staggered energy oscillates; this modified energy is invariant.
        """
        state = self.state if state is None else state
        # Multiply by the material operator before squaring: SI magnetic fluxes
        # at photonic scales can underflow when squared first in float32.
        electric = sum(
            jnp.sum(v * (self.m_epsilon[k] * v if state.d is None else state.d[k]))
            for k, v in state.e.items()
        )
        magnetic = sum(
            jnp.sum(v * (self.m_reluctivity[k] * v)) for k, v in state.b.items()
        )
        result = 0.5 * (electric + magnetic)
        if conserved:
            ce = self._topology(topology.curl, state.e)
            result -= (
                0.5
                * self.dt
                * sum(
                    jnp.sum(ce[k] * self.m_reluctivity[k] * state.b[k]) for k in state.b
                )
            )
        return result

    def magnetic_divergence(self, state=None):
        """Integrated net magnetic flux per primal cell."""
        state = self.state if state is None else state
        return self._topology(topology.divergence, state.b)

    def electric_charge(self, state=None):
        """Integrated dual-node electric charge, including PEC surface charge."""
        state = self.state if state is None else state
        d = (
            {k: self.m_epsilon[k] * v for k, v in state.e.items()}
            if state.d is None
            else state.d
        )
        return self._topology(topology.dual_divergence, d)
