"""Public state checkpoints exercise the same solver as native complete runs."""

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine._contracts import _StateContract
from beamz.simulation.cuda.engine._runtime import _SESSIONS

STATUS = cuda_backend_status()
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
    ),
]


@pytest.fixture
def simulation():
    module = _load_extension()
    if not hasattr(getattr(module, "NativeSession", None), "restore_state"):
        pytest.skip("Native continuation component required")
    steps, dt, t0 = 113, 3e-17, 7e-15
    frequency = bz.LIGHT_SPEED / 700e-9
    scene = bz.Simulation(
        domain=(1.28e-6,) * 3,
        resolution=20e-9,
        sources=[
            bz.PlaneWaveSource(
                center=(-0.28e-6, 0.0, 0.0),
                size=(0, 1.28e-6, 1.28e-6),
                direction="+x",
                source_time=bz.SampledSignal(
                    np.ones(steps),
                    dt,
                    quadrature=np.zeros(steps),
                    start_time=t0,
                    freq0=frequency,
                ),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(35e-9, 0.0, 0.0),
                size=(0.0, 0.4e-6, 0.4e-6),
                freqs=[2e14, 4e14],
                name="field",
            )
        ],
        boundaries=[bz.PML(formulation="cpml")],
        time=t0 + np.arange(steps) * dt,
    )
    try:
        yield scene
    finally:
        scene.clear_compiled_cache()


def _compare(actual, expected, *, exact):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            scale = np.max(np.abs(b), initial=0)
            np.testing.assert_allclose(a, b, rtol=5e-5, atol=scale * 1e-6)


@pytest.mark.parametrize("backend", ["cuda_streamed", "cuda_mstep"])
def test_native_segments_preserve_branchable_state_and_dft_horizon(simulation, backend):
    first = simulation.advance(num_steps=37, backend=backend, performance=False)
    frozen = jax.tree_util.tree_map(
        lambda value: np.array(value, copy=True), first.state
    )
    a = simulation.advance(state=first.state, backend=backend, performance=False)
    b = simulation.advance(
        state=first.state, backend=backend, progress=True, performance=False
    )
    whole = simulation.advance(backend=backend, performance=False)
    for run in (first, a, b, whole):
        assert run.results.cuda_plan["binding"] == "native", run.results.cuda_plan[
            "reason"
        ]
        assert run.results.cuda_plan["schedule"] == (
            "temporal" if backend == "cuda_mstep" else "ordinary"
        )
    assert a.results.cuda_plan["reused_resident_checkpoint"]
    assert not b.results.cuda_plan["reused_resident_checkpoint"]
    _compare(first.state, frozen, exact=True)
    _compare(a.state, b.state, exact=True)
    _compare(a.state, whole.state, exact=True)
    assert int(a.state.current_step) == simulation.num_steps
    assert np.max(np.abs(a.state.dft_vec_re)) > 0.0
    assert any(np.max(np.abs(v), initial=0) > 0 for v in a.state.cpml_psi_e_terms)
    np.testing.assert_array_equal(a.state.dft_weight_sum, [simulation.num_steps] * 2)


def test_native_step_imports_legacy_checkpoint_with_off_grid_time(simulation):
    # Receive a standalone legacy CUDA checkpoint by using the compatibility owner.
    legacy = simulation._advance_interop(
        contract=_StateContract(simulation.num_steps, 0, 39),
        backend="cuda_streamed",
        performance=False,
    )
    state = legacy.state._replace(t=legacy.state.t + np.float32(8e-16))
    expected = simulation._advance_interop(
        contract=_StateContract(simulation.num_steps, 39, 1, "preserved"),
        state=state,
        backend="cuda_streamed",
        performance=False,
    ).state
    actual = simulation.step(state, backend="cuda_streamed")
    _compare(actual, expected, exact=False)
    assert int(actual.current_step) == 40
    assert np.float32(actual.t) == np.float32(state.t) + np.float32(simulation.dt)
    run = simulation.advance(
        state=actual,
        num_steps=1,
        donate_state=True,
        backend="cuda_streamed",
        performance=False,
    )
    assert run.results.cuda_plan["binding"] == "native", run.results.cuda_plan["reason"]
    assert run.results.cuda_plan["state_ownership"] == "donated"
    assert int(run.state.current_step) == 41


def test_native_device_step_checkpoint_outlives_session_and_budget(simulation):
    if not hasattr(_load_extension().NativeSession, "download_device_state"):
        pytest.skip("CUDA checkpoint extension is unavailable")
    warm = simulation.advance(num_steps=25, backend="cuda_streamed", performance=False)
    state = simulation.step(warm.state, backend="cuda_streamed")
    owner = _SESSIONS[simulation].native
    assert state.ex.device.platform == "gpu"
    assert state.current_step.device.platform == "cpu"
    assert not state.t.committed
    assert (state.ex + state.t).device.platform == "gpu"
    assert owner.metadata()["checkpoint_device_bytes"] > 0
    assert owner.download_device_state(0) is None
    saved = jax.tree_util.tree_map(lambda v: np.array(v, copy=True), state)
    continued = simulation.step(state, backend="cuda_streamed")
    simulation.clear_compiled_cache()
    assert owner.metadata()["closed"]
    _compare(state, saved, exact=True)
    branch = simulation.step(state, backend="cuda_streamed")
    _compare(branch, continued, exact=True)

    # Retained JAX buffers own their detached allocation independently. Deleting
    # all leaves releases it even after the session closed.
    for leaf in jax.tree_util.tree_leaves((state, continued)):
        leaf.delete()
    jax.effects_barrier()
    assert owner.metadata()["checkpoint_device_bytes"] == 0
