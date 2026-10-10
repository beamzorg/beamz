"""Hardware parity gates for the optional Futhark-generated FFI library."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid
from beamz.devices.boundaries import PEC
from beamz.simulation.execute import build_scan, initial_program_state
from beamz.simulation.futhark import futhark_backend_status

REASON = futhark_backend_status()
pytestmark = pytest.mark.skipif(REASON is not None, reason=REASON or "")

DX = 80e-9
STATE_FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
DFT_FIELDS = ("dft_vec_re", "dft_vec_im", "dft_weight_sum")


def _simulation(boundaries, *, monitors: bool, lossy: bool, steps: int = 90):
    shape = (32, 36, 40)
    size = np.array(shape[::-1]) * DX
    dt = 0.95 * DX / (bz.LIGHT_SPEED * np.sqrt(3))
    eps = np.full(shape, 2.0, np.float32)
    eps[10:20, 12:24, :] = 12.0
    sigma = np.where(eps > 3, np.float32(50), np.float32(0)) if lossy else np.float32(0)
    grid = MaterialGrid(
        permittivity=eps,
        conductivity=sigma,
        permeability=np.float32(1),
        resolution=DX,
        shape=shape,
    )
    freq = bz.LIGHT_SPEED / 1.55e-6
    n = np.arange(steps)
    signal = np.exp(-(((n - 24) / 8) ** 2)) * np.sin(2 * np.pi * freq * n * dt)
    observed = [
        bz.FieldMonitor(
            center=(0.6 * size[0], size[1] / 2, size[2] / 2),
            size=(0, size[1] / 2, size[2] / 2),
            freqs=[0.98 * freq, freq],
            fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            name="plane",
        )
    ]
    return bz.Simulation(
        material_grid=grid,
        size=tuple(size),
        sources=[
            bz.GaussianSource(
                position=(0.3 * size[0], size[1] / 2, size[2] / 2),
                width=3 * DX,
                signal=signal.astype(np.float32),
            )
        ],
        monitors=observed if monitors else [],
        boundaries=boundaries,
        time=n * dt,
    )


def _run(simulation, backend, steps):
    program = simulation.compile(num_steps=steps, backend=backend)
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    return jax.block_until_ready(
        build_scan(program, donate_state=False)(state, program.coefficients)
    )


def _assert_close(reference, candidate, names):
    for name in names:
        expected = np.asarray(getattr(reference, name))
        actual = np.asarray(getattr(candidate, name))
        scale = max(float(np.abs(expected).max()), 1e-30)
        np.testing.assert_array_less(np.abs(actual - expected).max() / scale, 1e-5)


PML = bz.PML(edges="all", thickness=6 * DX, formulation="cpml")
MIXED = (
    PEC(edges=["top", "bottom"]),
    bz.PML(
        edges=["left", "right", "front", "back"], thickness=6 * DX, formulation="cpml"
    ),
)


@pytest.mark.parametrize(
    "boundaries, monitors, lossy",
    [
        ([PEC(edges="all")], False, True),
        (list(MIXED), True, False),
        ([PML], True, True),
    ],
    ids=["pec-lossy", "pec-cpml-dft", "cpml-lossy-dft"],
)
def test_futhark_program_matches_jax(boundaries, monitors, lossy):
    simulation = _simulation(boundaries, monitors=monitors, lossy=lossy)
    reference = _run(simulation, "jax", 90)
    candidate = _run(simulation, "futhark", 90)
    _assert_close(reference, candidate, STATE_FIELDS)
    if monitors:
        _assert_close(reference, candidate, DFT_FIELDS)
    assert int(candidate.current_step) == int(reference.current_step) == 90
    assert float(candidate.t) == float(reference.t)


def test_futhark_continuation_matches_one_call():
    simulation = _simulation([PML], monitors=True, lossy=False)
    whole = simulation.advance(num_steps=90, backend="futhark").state
    half = simulation.advance(num_steps=45, backend="futhark").state
    split = simulation.advance(state=half, num_steps=45, backend="futhark").state
    _assert_close(whole, split, STATE_FIELDS + DFT_FIELDS)
