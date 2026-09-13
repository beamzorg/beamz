"""Vector gradients, ordinary modal parity, and the 3D user workflow."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from beamz.optimization import AdamOptimizer
from tests.topology_3d_case import make_3d_design, mode_power


@pytest.mark.parametrize(
    "axis,polarization,direction",
    [("x", "te", "+"), ("x", "tm", "+"), ("y", "te", "-"), ("z", "te", "+")],
)
def test_vector_gradients(axis, polarization, direction):
    ad = make_3d_design(axis=axis, polarization=polarization, direction=direction)
    adj = ad.updated_copy(gradient_backend="adjoint")
    params = ad.design_region.initial_parameters + np.random.default_rng(3).uniform(
        -0.1, 0.1, ad.design_region.params_shape
    )

    def objective(data):
        amp = data["output"].amps.sel(direction=direction, mode_index=0).values[0]
        return abs(amp) ** 2 + 0.01 * amp.real

    fun = ad.make_objective_fn(objective)
    va, ga = jax.value_and_grad(fun)(params)
    vb, gb = jax.value_and_grad(adj.make_objective_fn(objective))(params)
    assert float(vb) == pytest.approx(float(va), rel=2e-5)
    assert np.linalg.norm(ga) > 1e-6
    assert np.linalg.norm(ga - gb) / np.linalg.norm(ga) < 2e-3
    vector = np.array(ga) / np.linalg.norm(ga)
    fd = float((fun(params + 0.01 * vector) - fun(params - 0.01 * vector)) / 0.02)
    assert fd == pytest.approx(float(np.sum(ga * vector)), rel=3e-3, abs=1e-7)
    ordinary = (
        ad.to_simulation(params).run(backend="jax", performance=False).mode("output")
    )
    differentiated = ad.to_simulation_data(params)
    for branch in ("+", "-"):
        np.testing.assert_allclose(
            differentiated["output"].amps.sel(direction=branch, mode_index=0).values,
            ordinary.amps.sel(direction=branch, mode_index=0).values,
            rtol=3e-5,
            atol=1e-7,
        )
    diag = adj.gradient_diagnostics
    assert diag["terminal_field_ratio"] < 1e-4
    assert max(diag["adjoint_terminal_field_ratios"]) < 1e-4


def test_ordinary_modal_analysis_and_resume(tmp_path):
    design = make_3d_design("adjoint")
    params = design.design_region.initial_parameters
    data = design.to_simulation_data(params)
    sim = design.to_simulation(params)
    results = sim.run()
    for direction in ("+", "-"):
        actual = data["output"].amps.sel(direction=direction, mode_index=0).values
        expected = (
            results.mode("output").amps.sel(direction=direction, mode_index=0).values
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
    optimizer = AdamOptimizer(
        design=design,
        learning_rate=0.05,
    )
    full = optimizer.run(mode_power, steps=2)
    first = optimizer.run(mode_power, steps=1, checkpoint=tmp_path / "history.npz")
    restored = optimizer.load_result(tmp_path / "history.npz", mode_power)
    resumed = optimizer.run(mode_power, steps=2, resume=restored)
    np.testing.assert_allclose(full.params[-1], resumed.params[-1], rtol=0, atol=1e-7)
    assert float(mode_power(full.simulation_data())) > float(mode_power(data))
    assert len(first.params) == 1
    assert first.completed_steps == 1
    exported = design.export_design(full.params[-1])
    assert exported.depth == design.simulation.design.depth
    assert all(s.depth == pytest.approx(0.4e-6) for s in exported.structures[-2:])
    # The native material-grid view contains the same pattern throughout z.
    patch = np.asarray(full.to_simulation()._material_grid().permittivity)[
        design._slices
    ]
    np.testing.assert_allclose(
        patch, np.broadcast_to(patch[0], patch.shape), rtol=0, atol=0
    )


@pytest.mark.parametrize("source_mode", [0, 1])
def test_multimode_data_matches_ordinary_projection(source_mode):
    design = make_3d_design()
    monitor = design.simulation.monitors[0]
    monitor = monitor.updated_copy(mode_spec=replace(monitor.mode_spec, num_modes=2))
    source = design.simulation.sources[0]
    source = source.updated_copy(
        mode_spec=replace(source.mode_spec, num_modes=2, mode_index=source_mode)
    )
    design = design.updated_copy(
        simulation=design.simulation.updated_copy(sources=[source], monitors=[monitor])
    )
    params = design.design_region.initial_parameters
    # Break symmetry so the higher-order mode also contributes to the check.
    params += np.random.default_rng(42).uniform(-0.2, 0.2, params.shape)
    actual = design.to_simulation_data(params)["output"].amps.values
    expected = (
        design.to_simulation(params)
        .run()
        .mode("output")
        .amps.transpose("direction", "f", "mode_index")
        .values
    )
    # Excite each mode in turn; both must pass with substantial transmitted power.
    # Nearly dark modes also incur cancellation error from float32 projection.
    assert np.linalg.norm(expected[:, :, source_mode]) > 0.1
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
