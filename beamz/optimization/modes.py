"""Fixed modal projections for objectives built from differentiable DFT fields."""

import jax
import jax.numpy as jnp
import numpy as np

from beamz.analysis.data import analysis_data
from beamz.analysis.modal_projection.geometry import (
    _monitor_analysis_plane_3d,
)
from beamz.analysis.mode_projection import _build_port_projection
from beamz.devices.modes.fields import _modal_overlap
from beamz.devices.modes.specs import ModeSpec
from beamz.devices.monitors.monitors import ModeMonitor
from beamz.devices.ports import Port


def build_mode_projector(results, monitor):
    """Freeze a 3D port basis and return a differentiable modal projection.

    Build once from ordinary, unnormalized ``SimulationResults`` containing a
    ``ModeMonitor``. The returned function accepts common-time phasors from
    ``DifferentiableSimulation.monitor_fields`` with shape ``(6, nf, points)``
    and returns amplitudes with shape ``(nf, 2, num_modes)``. Direction indices
    0 and 1 mean propagation along the positive and negative Cartesian axis,
    respectively, matching ``results.mode(name).amps``.

    Mode solving, normalization, transverse interpolation, and the coupled
    forward/backward overlap system use the same contract as ordinary analysis.
    Their linear weights are frozen; only the measured fields are differentiated.
    The monitor geometry, frequencies, grid, and port materials must therefore
    remain unchanged. This helper currently requires a 3D mode monitor.
    """
    data = analysis_data(results, monitor)
    geometry = data.monitor_geometry
    if not data.is_3d or not isinstance(geometry, ModeMonitor):
        raise ValueError("A fixed modal projector requires a 3D ModeMonitor")
    count = int(geometry.mode_spec.num_modes or 1)
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    points = data.field("Ex").shape[-1]
    matrices = []
    cache = {}
    for frequency in data.frequencies:
        projections = [
            _build_port_projection(
                data,
                Port(
                    center=geometry.center,
                    size=geometry.size,
                    name=f"{monitor}_m{index}",
                    monitor_name=monitor,
                    direction="+",
                    mode_spec=ModeSpec(
                        mode_index=index,
                        num_modes=count,
                        polarization=geometry.mode_spec.polarization or "te",
                        target_neff=geometry.mode_spec.target_neff,
                    ),
                ),
                geometry,
                float(frequency),
                cache,
            )
            for index in range(count)
        ]
        projection = projections[0]
        axis = projection["axis"]
        measure = projection["integration_weights"]
        basis = [
            item[key]
            for item in projections
            for key in ("mode_components", "mode_components_bwd")
        ]
        overlap = np.array(
            [[_modal_overlap(a, b, axis, measure) for b in basis] for a in basis]
        ).T
        rhs_weights = np.zeros((2 * count, 6, points), dtype=np.complex128)
        paired = {
            "x": (("Ey", "Hz", 1), ("Ez", "Hy", -1)),
            "y": (("Ez", "Hx", 1), ("Ex", "Hz", -1)),
            "z": (("Ex", "Hy", 1), ("Ey", "Hx", -1)),
        }[axis]
        dst0, dst1 = projection["analysis_coords0"], projection["analysis_coords1"]
        plane_shape = (len(dst0), len(dst1))
        for name in projection["components"]:
            src0, src1 = _monitor_analysis_plane_3d(data, geometry, axis)
            if len(src0) * len(src1) != points:
                raise ValueError("Mode projector and DFT sampling geometry disagree")
            # Adjoint of the separable interpolation used by ordinary analysis.
            interp0 = np.array(
                [np.interp(dst0, src0, row) for row in np.eye(len(src0))]
            )
            interp1 = np.array(
                [np.interp(dst1, src1, row) for row in np.eye(len(src1))]
            )
            for index, mode in enumerate(basis):
                for electric, magnetic, sign in paired:
                    partner = magnetic if name == electric else electric
                    if name not in (electric, magnetic):
                        continue
                    weights = (
                        0.25 * sign * measure * np.conjugate(mode[partner]).reshape(-1)
                    ).reshape(plane_shape)
                    rhs_weights[index, components.index(name)] = (
                        interp0 @ weights @ interp1.T
                    ).reshape(-1)
        if np.linalg.cond(overlap) < 1e8:
            weights = np.linalg.solve(overlap, rhs_weights.reshape(2 * count, -1))
        else:
            weights = np.linalg.pinv(overlap) @ rhs_weights.reshape(2 * count, -1)
        weights = weights.reshape(count, 2, 6, points)
        # The discrete x/y basis stores its +axis branch second.
        if axis in {"x", "y"}:
            weights = weights[:, ::-1]
        matrices.append(weights.transpose(1, 0, 2, 3))
    weights = jnp.asarray(np.stack(matrices))
    expected = (6, len(data.frequencies), points)

    @jax.jit
    def project(fields):
        if fields.shape != expected:
            raise ValueError(f"Modal fields must have shape {expected}")
        return jnp.einsum("fdmcp,cfp->fdm", weights, fields)

    return project
