"""Seeded state preservation and exact candidate validation for CUDA tuning."""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from beamz.simulation import execute
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.cuda import tuning
from scripts.benchmark_cuda_realistic import build_simulation

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def _seed_state(state):
    rng = np.random.default_rng(20260918)
    return state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        **{
            name: tuple(
                rng.normal(size=x.shape).astype(np.float32) * 100
                for x in getattr(state, name)
            )
            for name in ("cpml_psi_h_terms", "cpml_psi_e_terms")
        },
        dft_vec_re=rng.normal(size=state.dft_vec_re.shape).astype(np.float32),
        dft_vec_im=rng.normal(size=state.dft_vec_im.shape).astype(np.float32),
    )


@pytest.mark.parametrize("setup_device", ["default", "cpu"])
@pytest.mark.parametrize("steps", [33, 519])
def test_seeded_candidate_parity_and_persistent_lookup(
    monkeypatch, tmp_path, setup_device, steps
):
    monkeypatch.setenv("BEAMZ_CUDA_AUTOTUNE", "off")
    sim = build_simulation(
        SimpleNamespace(
            shape=(37, 49, 65),
            steps=steps,
            pml=12,
            monitors=2,
            frequencies=3,
            material="smooth",
            source="mode",
            monitor_type="mode",
        )
    )
    sim = sim.updated_copy(setup_device=setup_device)
    program = sim.compile(num_steps=steps, backend="cuda_streamed")
    original_initial = execute.initial_program_state

    def seeded_initial(*args, **kwargs):
        return _seed_state(original_initial(*args, **kwargs))

    @dataclass
    class TestIdentity:
        t0: float = 0.0
        profile: str = "seeded-smooth-37-49-65-33"

    monkeypatch.setattr(execute, "initial_program_state", seeded_initial)
    # Exercise calibration at unit-test scale; production retains its large-grid gate.
    monkeypatch.setattr(tuning, "_eligible", lambda _: True)
    policy = ("calibrate", str(tmp_path), ())
    selected = tuning.select_program(program, TestIdentity(), policy)
    report = tuning.tuning_report(selected)
    assert report and report["validated"] and not report["cache_hit"]
    assert report["calibration_steps"] == min(steps, 256)
    assert selected.config.num_steps == steps
    assert report["rejected"] == {}
    assert len(report["samples_s"]) == 6
    cached = tuning.select_program(program, TestIdentity(), policy)
    assert tuning.tuning_report(cached)["cache_hit"]
    assert cached.config == selected.config

    state = seeded_initial(program, t=0, current_step=0, monitor_steps=steps)
    before = tuning._state_hashes(state)
    reference = execute.run_program(program, state)
    actual = execute.run_program(selected, state)
    for expected, observed in zip(
        jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    assert tuning._state_hashes(state) == before


@pytest.mark.parametrize(
    "shape, axes, steps, setup_device",
    [
        ((145, 49, 65), (1, 2, 0), 33, "default"),
        ((49, 145, 65), (2, 0, 1), 519, "cpu"),
    ],
)
def test_predictive_selection_preserves_seeded_complete_state(
    monkeypatch, shape, axes, steps, setup_device
):
    monkeypatch.setenv("BEAMZ_CUDA_AUTOTUNE", "off")
    sim = build_simulation(
        SimpleNamespace(
            shape=shape,
            steps=steps,
            pml=12,
            monitors=2,
            frequencies=3,
            material="smooth",
            source="mode",
            monitor_type="mode",
        )
    ).updated_copy(setup_device=setup_device)
    program = sim.compile(num_steps=steps, backend="cuda_streamed")
    # Numerical regression at small scale; no performance conclusions from this test.
    monkeypatch.setattr(tuning, "_eligible", lambda _: True)

    def forbidden(*args, **kwargs):
        pytest.fail("Prediction attempted calibration")

    monkeypatch.setattr(tuning, "_select_program", forbidden)
    monkeypatch.setattr(tuning, "_gpu_status", forbidden)
    selected = tuning.select_program(program, None, ("auto", "", ()))
    assert selected.config.cuda_storage_axes == axes
    assert selected.config.num_steps == steps
    assert tuning.tuning_report(selected)["calibration_s"] == 0
    state = _seed_state(
        execute.initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    )
    before = tuning._state_hashes(state)
    reference = execute.run_program(program, state)
    actual = execute.run_program(selected, state)
    assert int(actual.current_step) == steps
    for expected, observed in zip(
        jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    assert tuning._state_hashes(state) == before
