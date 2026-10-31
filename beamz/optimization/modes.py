"""Fixed modal projections for objectives built from differentiable DFT fields."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from beamz.analysis.data import analysis_data
from beamz.analysis.modal_projection.geometry import (
    _monitor_analysis_plane_3d,
)
from beamz.analysis.mode_projection import (
    _build_port_projection,
    _modal_coefficient_rows_3d,
)
from beamz.devices.monitors.monitors import ModeMonitor
from beamz.devices.ports import Port


def _monitor_port(monitor, mode_index, polarization=None):
    """Adapt a monitor without dropping solver, symmetry, or refinement options."""
    return Port(
        center=monitor.center,
        size=monitor.size,
        name=f"{monitor.name}_mode_{mode_index}",
        monitor_name=monitor.name,
        direction="+",
        mode_spec=replace(
            monitor.mode_spec,
            mode_index=mode_index,
            polarization=monitor.mode_spec.polarization or polarization,
        ),
    )


def _coupled_projection(data, port, monitor, frequency, cache):
    """Bind all modes together, preserving the monitor's complete ModeSpec."""
    key = ("coupled_rows", monitor.name, float(frequency))
    if key not in cache:
        mode_cache = cache.setdefault(("mode_projections", monitor.name), {})
        projections = [
            _build_port_projection(
                data,
                port.updated_copy(
                    name=f"{monitor.name}_mode_{index}",
                    mode_spec=replace(port.mode_spec, mode_index=index),
                ),
                monitor,
                float(frequency),
                mode_cache,
            )
            for index in range(monitor.mode_spec.num_modes)
        ]
        cache[key] = projections, _modal_coefficient_rows_3d(projections)
    return cache[key]


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
        projections, coefficient_rows = _coupled_projection(
            data,
            _monitor_port(geometry, 0, "te"),
            geometry,
            frequency,
            cache,
        )
        projection = projections[0]
        axis = projection["axis"]
        src0, src1 = _monitor_analysis_plane_3d(data, geometry, axis)
        if len(src0) * len(src1) != points:
            raise ValueError("Mode projector and DFT sampling geometry disagree")
        dst0, dst1 = projection["analysis_coords0"], projection["analysis_coords1"]
        # Pull the shared coefficient rows back through analysis interpolation.
        interp0 = np.array([np.interp(dst0, src0, row) for row in np.eye(len(src0))])
        interp1 = np.array([np.interp(dst1, src1, row) for row in np.eye(len(src1))])
        rows = coefficient_rows.reshape(
            2 * count, len(projection["components"]), len(dst0), len(dst1)
        )
        weights = np.zeros((2 * count, 6, points), dtype=np.complex128)
        for index, name in enumerate(projection["components"]):
            weights[:, components.index(name)] = (
                interp0 @ rows[:, index] @ interp1.T
            ).reshape(2 * count, points)
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
