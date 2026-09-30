"""SPD electric constitutive assembly and matrix-free JAX inversion.

Gather native edge voltages at each cell corner, apply an SPD dielectric tensor,
then scatter with the exact gather transpose. The global permittivity is a sum
of positive local energy forms. CG inverts it without assembling BCOO/CSR.
"""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from beamz.const import EPS_0


class CoupledElectricOperator:
    """Corner-quadrature permittivity, positive on free PEC degrees.

    The explicit Maxwell step uses a matrix-free constitutive CG solve, not an
    implicit time integrator. Coupling improves the interface approximation but
    costs more than diagonal division. No universal accuracy order is assumed.
    """

    def __init__(self, mesh, material, dtype):
        self.mesh, self.dtype = mesh, dtype
        self.names = tuple(mesh.electric_shapes)
        self.masks = {name: jnp.asarray(mesh.pec_mask(name)) for name in self.names}
        tensors, self.parallel_permittivity = material.tensors(mesh)
        self.epsilon_lower_bound = min(
            material.permittivity_minus, material.permittivity_plus
        )
        self.metric_scale = (
            mesh.spacing**2
            if mesh.dimension == 2 and mesh.polarization == "tm"
            else mesh.spacing ** (mesh.dimension - 2)
        )
        weight = mesh.spacing**mesh.dimension / 2**mesh.dimension
        self._corners = []
        diagonal = {
            name: np.zeros(shape) for name, shape in mesh.electric_shapes.items()
        }
        for corner in product((0, 1), repeat=mesh.dimension):
            regions, lengths, free = [], [], []
            for name in self.names:
                own_axis = mesh._axis(name)
                region = tuple(
                    slice(0, n)
                    if axis == own_axis and not (mesh.dimension == 2 and name == "Ez")
                    else slice(side, n + side)
                    for axis, (n, side) in enumerate(zip(mesh.shape, corner))
                )
                regions.append(region)
                lengths.append(mesh.edge_length(name))
                free.append(~mesh.pec_mask(name)[region])
            free = np.stack(free, axis=-1)
            lengths = np.asarray(lengths)
            blocks = (
                tensors
                * free[..., :, None]
                * free[..., None, :]
                * weight
                / (lengths[:, None] * lengths[None, :])
            )
            for axis, name in enumerate(self.names):
                diagonal[name][regions[axis]] += blocks[..., axis, axis]
            self._corners.append(
                (tuple(regions), jnp.asarray(blocks / self.metric_scale, dtype))
            )
        self.diagonal = {
            name: jnp.asarray(value / self.metric_scale, dtype)
            for name, value in diagonal.items()
        }

    def _apply_normalized(self, e):
        result = {name: jnp.zeros_like(e[name]) for name in self.names}
        for regions, blocks in self._corners:
            gathered = jnp.stack(
                [e[name][region] for name, region in zip(self.names, regions)], axis=-1
            )
            applied = jnp.einsum("...ij,...j->...i", blocks, gathered)
            for axis, (name, region) in enumerate(zip(self.names, regions)):
                result[name] = result[name].at[region].add(applied[..., axis])
        return result

    def apply_relative_permittivity(self, e):
        """Apply relative dielectric operator, including geometric metrics."""
        return {
            name: value * self.metric_scale
            for name, value in self._apply_normalized(e).items()
        }

    def apply_permittivity(self, e):
        """Convert edge voltages into integrated electric displacement fluxes."""
        return {
            name: EPS_0 * value
            for name, value in self.apply_relative_permittivity(e).items()
        }

    def solve(self, d, *, initial=None, tolerance=5e-7, maxiter=256):
        """Recover voltages from SI displacement fluxes with preconditioned CG.

        Normalize geometry and RHS magnitude to prevent SI-scale FP32 squared
        residuals from underflowing on CUDA. Return a checked relative residual;
        JAX's CG info alone does not report convergence.
        """
        rhs = {name: value / (EPS_0 * self.metric_scale) for name, value in d.items()}
        scale = jnp.max(jnp.stack([jnp.max(jnp.abs(value)) for value in rhs.values()]))

        def nonzero(unused):
            normalized = {name: value / scale for name, value in rhs.items()}
            x0 = (
                None
                if initial is None
                else {name: value / scale for name, value in initial.items()}
            )

            def operator(e):
                result = self._apply_normalized(e)
                return {
                    name: result[name] + jnp.where(self.masks[name], e[name], 0.0)
                    for name in self.names
                }

            def preconditioner(e):
                return {
                    name: e[name]
                    / jnp.where(self.masks[name], 1.0, self.diagonal[name])
                    for name in self.names
                }

            voltage, _ = cg(
                operator,
                normalized,
                x0=x0,
                M=preconditioner,
                tol=tolerance,
                maxiter=maxiter,
            )
            actual = operator(voltage)
            residual = sum(
                jnp.sum((actual[name] - normalized[name]) ** 2) for name in self.names
            )
            norm = sum(jnp.sum(value**2) for value in normalized.values())
            return {name: value * scale for name, value in voltage.items()}, jnp.sqrt(
                residual / norm
            )

        def zero(unused):
            return {
                name: jnp.zeros_like(value) for name, value in d.items()
            }, jnp.asarray(0, self.dtype)

        return jax.lax.cond(scale > 0, nonzero, zero, operand=None)

    def apply_relative(self, d):
        """Apply inverse relative dielectric operator (validation convenience)."""
        return self.solve(
            {name: EPS_0 * value for name, value in d.items()},
            tolerance=1e-11 if self.dtype == jnp.float64 else 5e-7,
        )[0]

    def apply(self, d):
        """Apply inverse SI dielectric operator to integrated displacement."""
        return self.solve(d, tolerance=1e-11 if self.dtype == jnp.float64 else 5e-7)[0]
