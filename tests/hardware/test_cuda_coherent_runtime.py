"""Public native ownership, explicit overrides and interoperable fallback."""

import numpy as np
import pytest

import beamz as bz
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine._runtime import _SESSIONS

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def test_public_native_reset_aliases_and_readonly_diagnostics():
    if not getattr(_load_extension(), "coherent_backend_version", 0):
        pytest.skip("Coherent component required")
    frequency = bz.LIGHT_SPEED / 700e-9
    simulation = bz.Simulation(
        domain=(1.28e-6,) * 3,
        resolution=20e-9,
        sources=[
            bz.PlaneWaveSource(
                center=(0.36e-6, 0.64e-6, 0.64e-6),
                size=(0, 1.28e-6, 1.28e-6),
                direction="+x",
                source_time=bz.GaussianPulse(frequency, frequency * 0.4),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(0.92e-6, 0.64e-6, 0.64e-6),
                size=(0, 0.4e-6, 0.4e-6),
                freqs=[frequency],
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(1025) * 3e-17,
    )
    try:
        ordinary = simulation.run(backend="cuda_streamed", performance=False)
        assert (
            ordinary.performance is None
            and ordinary.cuda_plan["schedule"] == "ordinary"
        )
        temporal = simulation.run(backend="cuda_mstep", performance=False)
        assert (
            temporal.cuda_plan["depth"] == 16
            and temporal.cuda_plan["raw_slots"] == 4
            and temporal.cuda_plan["binding"] == "native"
        )
        again = simulation.run(backend="cuda_mstep", progress=True, performance=False)
        a = ordinary.renormalize(None)["field"].dft_fields
        b = temporal.renormalize(None)["field"].dft_fields
        c = again.renormalize(None)["field"].dft_fields
        assert any(np.max(np.abs(v), initial=0) > 1e-12 for v in a.values())
        for field in a:
            assert np.array_equal(
                np.ascontiguousarray(a[field]).view(np.uint8),
                np.ascontiguousarray(b[field]).view(np.uint8),
            )
            assert np.array_equal(
                np.ascontiguousarray(b[field]).view(np.uint8),
                np.ascontiguousarray(c[field]).view(np.uint8),
            )
        assert dict(temporal.renormalize(None).cuda_plan) == dict(temporal.cuda_plan)
        with pytest.raises(TypeError):
            temporal.cuda_plan["depth"] = 4
        assert _SESSIONS[simulation].native.metadata()["runs"] >= 2
    finally:
        simulation.clear_compiled_cache()
    assert simulation not in _SESSIONS


def test_public_reduced_dimension_preserves_cuda_capability_error():
    from beamz.simulation.backend import CudaBackendUnavailable

    simulation = bz.Simulation(
        domain=(0.4e-6, 0.3e-6), resolution=0.1e-6, time=np.arange(5) * 1e-17
    )
    with pytest.raises(CudaBackendUnavailable, match="3D"):
        simulation.run(backend="cuda", performance=False)
    result = simulation.run(backend="jax", performance=False)
    assert result.cuda_plan["binding"] == "jax" and result.cuda_plan["backend"] == "jax"


def test_native_run_and_continuation_share_the_resident_owner():
    if not getattr(_load_extension(), "coherent_backend_version", 0):
        pytest.skip("Coherent component required")
    if not hasattr(_load_extension().NativeSession, "restore_state"):
        pytest.skip("Native continuation component required")
    sim = bz.Simulation(
        domain=(1.28e-6,) * 3,
        resolution=20e-9,
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(33) * 3e-17,
    )
    try:
        result = sim.run(backend="cuda_streamed", performance=False)
        native = _SESSIONS[sim].native
        segment = sim.advance(num_steps=17, backend="cuda_streamed", performance=False)
        assert _SESSIONS[sim].native is native and not native.metadata()["closed"]
        assert result.cuda_plan["state_ownership"] == "private"
        assert segment.results.cuda_plan["state_ownership"] == "private"
        continued = sim.advance(
            state=segment.state,
            num_steps=15,
            backend="cuda_streamed",
            performance=False,
        )
        assert continued.results.cuda_plan["binding"] == "native"
        assert continued.results.cuda_plan["state_ownership"] == "preserved"
        assert continued.results.cuda_plan["reused_resident_checkpoint"]
        assert int(continued.state.current_step) == 32
    finally:
        sim.clear_compiled_cache()
