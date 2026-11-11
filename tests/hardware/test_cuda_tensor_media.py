"""Hardware parity of tensor permeability and tensor-pole media on streamed CUDA.

Run explicitly on a CUDA host with the matching native component:
JAX_PLATFORMS=cuda PYTHONPATH=. pytest -q tests/hardware/test_cuda_tensor_media.py
"""

from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
from beamz.design.raster import RasterOptions
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_step_context
from beamz.simulation.kernels import select_update_kernel

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available,
    reason=STATUS.reason or "CUDA backend unavailable",
)

FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")


def _tensor_media_simulation(steps=200):
    um = bz.um
    wavelength = 1.5 * um
    frequency = bz.LIGHT_SPEED / wavelength
    omega = 2 * np.pi * frequency
    dx = wavelength / 12
    length = 3 * um
    pole = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.TensorPole.lorentz(
                (1.5, 0.5, 0.2), resonance=2 * omega, damping=0.02 * omega
            )
        ],
    ).rotated((0.2, 0.3, 0.4))
    magnetic = bz.AnisotropicMedium(1.5, (2.0, 1.4, 1.2)).rotated((0.1, 0.5, 0.2))
    design = bz.Design(width=length, height=length, depth=length)
    design += bz.Box(
        center=(1.3 * um, 1.5 * um, 1.3 * um),
        size=(1.0 * um, 1.4 * um, 1.0 * um),
        material=pole,
    )
    design += bz.Box(
        center=(2.2 * um, 1.2 * um, 1.4 * um),
        size=(0.6 * um, 0.6 * um, 0.8 * um),
        material=magnetic,
    )
    dt = 0.5 * dx / (bz.LIGHT_SPEED * np.sqrt(3))
    time = np.arange(steps + 1) * dt
    signal = bz.ramped_cosine(
        time,
        amplitude=1,
        frequency=frequency,
        ramp_duration=3 * wavelength / bz.LIGHT_SPEED,
        t_max=time[-1],
    )
    return bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=(1.5 * um, 2.4 * um, 1.5 * um),
                width=wavelength / 6,
                signal=signal,
            )
        ],
        boundaries=[bz.PML(edges="all", thickness=0.4 * um, formulation="cpml")],
        time=time,
        resolution=dx,
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )


def test_tensor_media_streamed_cuda_matches_jax_and_reexecutes():
    sim = _tensor_media_simulation()
    program = sim.compile(backend="cuda_streamed")
    assert program.config.backend == "cuda_streamed"
    assert select_update_kernel(build_step_context(program)).kind == "cuda_streamed"

    reference = sim.advance(num_steps=200, backend="jax", performance=False).state
    first = sim.advance(num_steps=200, backend="cuda_streamed", performance=False).state
    # Re-executing the cached executable used to crash inside XLA's CUDA-graph
    # update once per-step FFI phases were captured into command buffers.
    second = sim.advance(num_steps=200, backend="cuda_streamed", performance=False).state
    for name in FIELDS:
        expected = np.asarray(getattr(reference, name))
        scale = np.abs(expected).max()
        assert scale > 0
        for state in (first, second):
            actual = np.asarray(getattr(state, name))
            assert np.abs(actual - expected).max() < 1e-4 * scale, name
