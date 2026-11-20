"""A/B/A complete public solves retain detached observations and one device owner."""

import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine import _runtime as runtime

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def test_temporal_cross_scene_ownership_preserves_results_and_fresh_clocks():
    if not getattr(_load_extension(), "coherent_backend_version", 0):
        pytest.skip("Coherent component required")
    runtime._clear()
    dx, dt, frequency = 20e-9, 2e-17, bz.LIGHT_SPEED / 700e-9
    shape = (40, 43, 46)
    simulations = []
    for variant in range(2):
        epsilon = np.ones(shape, np.float32)
        if variant:
            epsilon[18:22, 18:25, 19:27] = np.float32(2.25)
        simulations.append(
            bz.Simulation(
                material_grid=MaterialGrid(
                    epsilon, np.float32(0), np.float32(1), dx, shape
                ),
                sources=[
                    bz.PlaneWaveSource(
                        center=(16 * dx, 21 * dx, 23 * dx),
                        size=(0, 8 * dx, 8 * dx),
                        direction="+x",
                        source_time=bz.GaussianPulse(
                            frequency,
                            frequency * 0.8,
                            amplitude=1 + variant,
                            offset=0.1,
                        ),
                    )
                ],
                monitors=[
                    bz.FieldMonitor(
                        center=(23 * dx, 21 * dx, 23 * dx),
                        size=(0, 6 * dx, 6 * dx),
                        freqs=np.asarray([frequency * 0.9, frequency]),
                        fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                        name="field",
                    )
                ],
                boundaries=[bz.PML(formulation="cpml")],
                time=variant * 7e-17 + np.arange(49 + 16 * variant) * dt,
            )
        )
    a, b = simulations
    try:
        first = a.run(backend="cuda_mstep", performance=False)
        first_owner = runtime._SESSIONS[a]
        first_clocks = first_owner.native.clocks()
        assert (
            int(first_clocks["elapsed_steps"])
            == int(first_clocks["current_step"])
            == 49
        )
        first_fields = first.renormalize(None)["field"].dft_fields
        first_snapshot = {
            name: np.array(v, copy=True) for name, v in first_fields.items()
        }
        second = b.run(backend="cuda_mstep", performance=False)
        second_owner = runtime._SESSIONS[b]
        second_clocks = second_owner.native.clocks()
        assert (
            int(second_clocks["elapsed_steps"])
            == int(second_clocks["current_step"])
            == 65
        )
        second_fields = second.renormalize(None)["field"].dft_fields
        second_snapshot = {
            name: np.array(v, copy=True) for name, v in second_fields.items()
        }
        assert first_owner.native.metadata()["closed"]
        assert a not in runtime._SESSIONS and list(runtime._SESSIONS) == [b]
        again = a.run(backend="cuda_mstep", performance=False)
        final_owner = runtime._SESSIONS[a]
        assert second_owner.native.metadata()["closed"]
        assert final_owner is not first_owner and final_owner is not second_owner
        assert list(runtime._SESSIONS) == [a]
        fields_again = again.renormalize(None)["field"].dft_fields
        for result, steps in ((first, 49), (second, 65), (again, 49)):
            plan = result.cuda_plan
            assert plan["binding"] == "native" and plan["schedule"] == "temporal"
            assert plan["depth"] == 16 and plan["initial_step"] == 0
            assert plan["final_step"] == steps
        clocks = final_owner.native.clocks()
        assert int(clocks["elapsed_steps"]) == 49
        assert int(clocks["current_step"]) == 49
        assert clocks == first_clocks
        for owner, origin, steps, observed in (
            (first_owner, a.time[0], 49, first_clocks),
            (second_owner, b.time[0], 65, second_clocks),
        ):
            expected = np.float32(
                np.float32(origin)
                + np.float32(steps) * np.float32(owner.packet.attributes["dt"])
            )
            assert np.float32(observed["time"]) == expected
        metadata = final_owner.native.metadata()
        assert metadata["moving_ready"] and metadata["moving_depth"] == 16
        assert metadata["runs"] == 1 and metadata["last_nsteps"] == 49
        assert first_fields.keys() == second_fields.keys() == fields_again.keys()
        for name in first_fields:
            np.testing.assert_array_equal(
                np.ascontiguousarray(first_fields[name]).view(np.uint8),
                np.ascontiguousarray(first_snapshot[name]).view(np.uint8),
            )
            np.testing.assert_array_equal(
                np.ascontiguousarray(second_fields[name]).view(np.uint8),
                np.ascontiguousarray(second_snapshot[name]).view(np.uint8),
            )
            np.testing.assert_array_equal(
                np.ascontiguousarray(fields_again[name]).view(np.uint8),
                np.ascontiguousarray(first_snapshot[name]).view(np.uint8),
            )
            assert not np.shares_memory(first_fields[name], fields_again[name])
            assert not np.shares_memory(first_fields[name], second_fields[name])
            assert (
                np.isfinite(first_fields[name]).all()
                and np.isfinite(second_fields[name]).all()
            )
        assert any(np.max(np.abs(v), initial=0) > 0 for v in first_snapshot.values())
        assert any(np.max(np.abs(v), initial=0) > 0 for v in second_snapshot.values())
        assert any(
            not np.array_equal(first_snapshot[n], second_snapshot[n])
            for n in first_snapshot
        )
    finally:
        for simulation in simulations:
            simulation.clear_compiled_cache()
        runtime._clear()
