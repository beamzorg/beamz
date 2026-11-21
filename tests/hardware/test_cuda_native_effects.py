"""CUDA effects on the resident ordinary schedule, including public checkpoints."""

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.lattice import component_shape_3d
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine._runtime import _SESSIONS

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def _scene(steps, grid_shape=(32, 35, 38), *, source_z=15):
    spacing = 22e-9
    shape = component_shape_3d("Ez", grid_shape)
    target = (np.asarray([source_z, source_z, source_z + 1]), slice(15, 20, 2), 18)
    wave = np.sin(np.arange(steps) * 0.19).astype(np.float32)
    coefficients = np.arange(1, 10, dtype=np.float32).reshape(3, 3) * 1e-4
    source = bz.CustomSource("Ez", "e", target, coefficients, wave, shape)
    slab = bz.CustomSource(
        "Ez", "e", (source_z, slice(15, 20), 18), np.full(5, 3e-4), wave, shape
    )
    return bz.Simulation(
        domain=tuple(x * spacing for x in grid_shape[::-1]),
        resolution=spacing,
        time=2e-15 + np.arange(steps) * 3.4e-17,
        boundaries=(bz.PML(formulation="cpml"),),
        sources=(source, slab),
        monitors=(
            bz.FieldRecorder(("Ez",), interval=17, name="volume"),
            bz.FieldRecorder(
                ("Ez", "Hy"),
                interval=9,
                name="slice",
                center=(0, 0, 0),
                size=(360e-9, 300e-9, 0),
            ),
            bz.FieldMonitor(
                center=(0, 0, 0),
                size=(360e-9, 300e-9, 0),
                freqs=(3.8e14, 4e14),
                fields=("Ez", "Hy"),
                name="dft",
            ),
        ),
        setup_device="cpu",
        normalize_source=None,
    )


@pytest.fixture
def scene():
    if not hasattr(_load_extension().NativeSession, "configure_effects"):
        pytest.skip("Native effects build required")
    simulation = _scene(539)
    yield simulation
    simulation.clear_compiled_cache()


def _assert_same(actual, expected, atol_ratio=2e-6):
    for name in ("volume", "slice"):
        a, b = actual[name], expected[name]
        np.testing.assert_array_equal(a.field_steps, b.field_steps)
        # CPU XLA may contract the FP32 clock multiply-add. CUDA preserves
        # the two roundings of its native checkpoint clock.
        np.testing.assert_array_max_ulp(
            a.field_times.astype(np.float32), b.field_times.astype(np.float32), maxulp=1
        )
        for component in a.fields:
            x, y = a.fields[component], b.fields[component]
            peak = float(np.max(np.abs(y)))
            assert peak > 0.0
            np.testing.assert_allclose(x, y, rtol=5e-5, atol=atol_ratio * peak)
        assert not a.fields["Ez"].flags.writeable
    for component in actual["dft"].dft_fields:
        a, b = (
            actual["dft"].dft_fields[component],
            expected["dft"].dft_fields[component],
        )
        np.testing.assert_allclose(a, b, rtol=6e-5, atol=atol_ratio * np.max(np.abs(b)))
    np.testing.assert_array_equal(
        actual["dft"].dft_weight_sum, expected["dft"].dft_weight_sum
    )


def test_native_recorders_and_irregular_sources_share_exact_chunk_clock(scene):
    with jax.default_device(jax.devices("cpu")[0]):
        expected = scene.run(backend="jax", performance=False)
    got = scene.run(backend="cuda", performance=False)
    assert got.cuda_plan["binding"] == "native"
    assert got.cuda_plan["schedule"] == "ordinary"
    _assert_same(got, expected)

    session = _SESSIONS[scene]
    allocated = session.native.metadata()["effects_gpu_bytes"]
    public_history_bytes = sum(
        array.nbytes for array in session.initial.recorded_fields
    )
    assert allocated < public_history_bytes
    repeated = scene.run(backend="cuda", performance=False)
    _assert_same(repeated, got)


@pytest.mark.parametrize(
    "source_z, phase_z", [(15, 32), (63, 64)], ids=["separate", "two-planes"]
)
def test_rolling_cuda_replays_seeded_cpml_sources_and_recorders(
    monkeypatch, source_z, phase_z
):
    from dataclasses import replace

    from beamz.simulation.cuda.engine import _runtime

    if not getattr(_load_extension(), "compiled_rolling_tiles", False):
        pytest.skip("Rolling CUDA build required")
    shape = (67, 57, 71)
    scene = _scene(38, shape, source_z=source_z)
    # Scatter across a band, including two updates to the same cell.
    additions = tuple(
        bz.CustomSource(
            component,
            phase,
            (np.array([phase_z - 1, phase_z, phase_z]), 28, 36),
            [0.3, -0.1, 0.4],
            np.sin(np.arange(38) * 0.17),
            component_shape_3d(component, shape),
        )
        for component, phase in (("Hy", "h"), ("Ex", "pre_e"))
    )
    scene = scene.updated_copy(sources=(*scene.sources, *additions))
    try:
        initial = scene.advance(num_steps=1, backend="cuda_streamed").state
        rng = np.random.default_rng(43)
        initial = initial._replace(
            **{
                name: tuple(
                    (rng.standard_normal(a.shape) * 1e-4).astype(np.float32)
                    for a in getattr(initial, name)
                )
                for name in ("cpml_psi_h_terms", "cpml_psi_e_terms")
            }
        )
        states = []
        select = _runtime._choose
        for band in (0, 32):
            # Fix placement for this small correctness fixture. Automatic
            # scheduling reserves rolling for sufficiently large workloads.
            monkeypatch.setattr(
                _runtime,
                "_choose",
                lambda *a, value=band, **kw: replace(
                    select(*a, **kw), rolling_band=value
                ),
            )
            scene.clear_compiled_cache()
            first = scene.advance(
                state=initial, num_steps=19, backend="cuda_mstep", performance=False
            )
            scene.clear_compiled_cache()
            run = scene.advance(
                state=first.state, backend="cuda_mstep", performance=False
            )
            owner = _SESSIONS[scene].native
            assert owner.metadata()["rolling_band"] == band
            assert run.results.cuda_plan["temporal_blocks"] > 0
            states.append((first.state, run.state))
        for a, b in zip(*map(jax.tree_util.tree_leaves, states), strict=True):
            np.testing.assert_array_equal(a, b)
        assert np.any(states[1][-1].dft_vec_re)
        assert all(np.any(a) for a in states[1][-1].recorded_fields)
    finally:
        scene.clear_compiled_cache()


def test_native_recorder_tail_continuation_and_reuse_of_retained_checkpoint(scene):
    first = scene.advance(num_steps=117, backend="cuda", performance=False)
    branch = scene.advance(
        state=first.state, num_steps=81, backend="cuda", performance=False
    )
    tail = scene.advance(state=first.state, backend="cuda", performance=False)
    full = scene.run(backend="cuda", performance=False)
    assert (
        first.results.cuda_plan["binding"]
        == branch.results.cuda_plan["binding"]
        == tail.results.cuda_plan["binding"]
        == "native"
    )
    assert int(first.state.current_step) == 117
    assert int(branch.state.current_step) == 198
    _assert_same(tail.results, full)
    for name, interval in (("volume", 17), ("slice", 9)):
        prefix = branch.results[name]
        stop = 198 // interval
        np.testing.assert_array_equal(prefix.field_steps, full[name].field_steps[:stop])
        for component in prefix.fields:
            np.testing.assert_array_equal(
                prefix.fields[component], full[name].fields[component][:stop]
            )


def test_native_adaptive_recording_retains_the_exact_acquired_prefix(scene):
    policy = bz.AutoTermination(chunk_steps=49, max_wall_time=1e-12)
    recorded = scene.run(backend="cuda", termination=policy, performance=False)
    expected = scene.advance(num_steps=49, backend="cuda", performance=False).results
    assert recorded.cuda_plan["binding"] == "native"
    assert recorded.termination.reason == "wall_time_limit"
    assert recorded.termination.steps == 49
    _assert_same(recorded, expected)


@pytest.mark.parametrize(
    "interval,cohort,weighted",
    [(17, False, False), (32, True, False), (32, True, True)],
)
def test_boundary_recording_preserves_staggered_frames_across_reset_resume_and_termination(
    monkeypatch, interval, cohort, weighted
):
    from beamz.simulation.cuda.engine import _plan, _runtime

    module = _load_extension()
    if not getattr(module, "compiled_rolling_volume_windows", False):
        pytest.skip("Rolling volume recording build required")
    if weighted and not getattr(module, "compiled_boundary_acquisition", False):
        pytest.skip("Boundary gather recording build required")
    region = (
        dict(center=(397e-9, 355e-9, 333e-9), size=(221e-9, 239e-9, 0))
        if weighted
        else {}
    )
    scene = _scene(4 * interval + 1, (67, 57, 71))
    components = ("Hx", "Ex") if cohort else ("Hx",)
    extra = (
        (bz.FieldRecorder(("Ez",), interval=2 * interval, name="second", **region),)
        if cohort
        else ()
    )
    scene = scene.updated_copy(
        sources=(scene.sources[1],),
        monitors=(
            bz.FieldRecorder(components, interval=interval, name="volume", **region),
            *extra,
            scene.monitors[-1],
        ),
    )
    # Exercise the measured placement on a small grid with staggered Hx extents.
    monkeypatch.setattr(_plan, "_single_rolling_band", lambda *args: 32)
    monkeypatch.setattr(
        _runtime,
        "_choose",
        lambda inputs, gpu, info, **kw: _plan._choose(
            inputs, gpu, dict(info, name="NVIDIA GeForce RTX 3090"), **kw
        ),
    )
    try:
        ref = scene.advance(backend="cuda_streamed", performance=False)
        first = scene.advance(
            num_steps=interval + 5, backend="cuda_mstep", performance=False
        )
        scene.clear_compiled_cache()
        final = scene.advance(
            state=first.state, backend="cuda_mstep", performance=False
        )
        assert final.results.cuda_plan["direct_volume_recording"] == (not weighted)
        assert bool(final.results.cuda_plan.get("boundary_gather_bytes")) == weighted
        for a, b in zip(
            jax.tree_util.tree_leaves(final.state),
            jax.tree_util.tree_leaves(ref.state),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        assert np.any(final.results["volume"].fields["Hx"])
        expected = ref.results["volume"]
        acquired = scene.run(
            backend="cuda_mstep",
            performance=False,
            termination=bz.AutoTermination(
                chunk_steps=2 * interval + 1, max_wall_time=1e-12
            ),
        )
        if cohort:
            np.testing.assert_array_equal(
                acquired["second"].fields["Ez"], ref.results["second"].fields["Ez"][:1]
            )
        for actual, count in ((first.results["volume"], 1), (acquired["volume"], 2)):
            np.testing.assert_array_equal(
                actual.field_steps, expected.field_steps[:count]
            )
            np.testing.assert_array_equal(
                actual.field_times, expected.field_times[:count]
            )
            np.testing.assert_array_equal(
                actual.fields["Hx"], expected.fields["Hx"][:count]
            )
        retained = scene.run(backend="cuda_mstep", performance=False)["volume"].fields[
            "Hx"
        ]
        replay = scene.run(backend="cuda_mstep", performance=False)["volume"].fields[
            "Hx"
        ]
        np.testing.assert_array_equal(replay, expected.fields["Hx"])
        assert not np.shares_memory(replay, retained)
        np.testing.assert_array_equal(retained, expected.fields["Hx"])
    finally:
        scene.clear_compiled_cache()


@pytest.mark.parametrize("volume_interval", [17, 128, 131])
def test_temporal_recorders_and_mixed_source_phases_survive_restart(
    scene, volume_interval
):
    module = _load_extension()
    if not getattr(module, "compiled_temporal_recorders", False):
        pytest.skip("Temporal recording build required")
    grid_shape = scene.grid.shape_zyx
    steps = scene.num_steps
    wave = np.sin(np.arange(steps) * 0.11).astype(np.float32) * 1e-3
    additions = (
        bz.CustomSource(
            "Ez", "pre_e", (14, 14, 17), 0.4, wave, component_shape_3d("Ez", grid_shape)
        ),
        bz.CustomSource(
            "Hy", "h", (14, 15, 18), 0.2, wave, component_shape_3d("Hy", grid_shape)
        ),
        bz.CustomSource(
            "Ez",
            "e",
            (15, slice(16, 21), 18),
            np.full(5, 2e-4),
            wave,
            component_shape_3d("Ez", grid_shape),
        ),
    )
    scene = scene.updated_copy(
        boundaries=(
            bz.PML(edges=("left",), thickness=5 * 22e-9, formulation="cpml"),
            bz.PML(edges=("right", "bottom"), thickness=8 * 22e-9, formulation="cpml"),
            bz.PEC(edges=("top",)),
        ),
        sources=(*scene.sources, *additions),
        monitors=(
            bz.FieldRecorder(("Ez",), interval=volume_interval, name="volume"),
            *scene.monitors[1:],
        ),
    )
    try:
        expected = scene.advance(backend="cuda", schedule="streamed", performance=False)
        first = scene.advance(
            num_steps=117, backend="cuda", schedule="mstep", performance=False
        )
        final = scene.advance(
            state=first.state, backend="cuda", schedule="mstep", performance=False
        )
        assert (
            first.results.cuda_plan["schedule"]
            == final.results.cuda_plan["schedule"]
            == "temporal"
        )
        assert final.results.cuda_plan["binding"] == "native"
        assert first.results.cuda_plan["temporal_blocks"] > 0
        assert final.results.cuda_plan["temporal_blocks"] > 0
        assert int(final.state.current_step) == steps
        _assert_same(final.results, expected.results, atol_ratio=2e-5)
        # Ordinary CUDA atomically adds overlapping noncoincident slabs.
        # Compare against each field family's physical scale after cancellation.
        for family in ("e", "h"):
            reference = [np.asarray(getattr(expected.state, family + c)) for c in "xyz"]
            scale = max(float(np.max(np.abs(value))) for value in reference)
            for c, b in zip("xyz", reference, strict=True):
                a = np.asarray(getattr(final.state, family + c))
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5 * scale)
        for name in ("volume", "slice"):
            stop = int(first.state.current_step) // (
                volume_interval if name == "volume" else 9
            )
            for key, values in first.results[name].fields.items():
                np.testing.assert_array_equal(
                    values, final.results[name].fields[key][:stop]
                )
    finally:
        scene.clear_compiled_cache()


def test_adaptive_tensor_diagnostic_includes_weighted_cross_energy():
    from beamz.design.discretization import MaterialGrid
    from beamz.design.grid import RectilinearGrid
    from beamz.design.raster import RasterOptions, Scene, rasterize
    from beamz.simulation import execute

    module = _load_extension()
    if not getattr(module, "compiled_tensor_diagnostics", False):
        pytest.skip("Native tensor diagnostics build required")
    widths = 25e-9 * (1 + 0.15 * np.sin(np.arange(33) * 0.2))
    edges = np.r_[0.0, np.cumsum(widths)]
    grid = RectilinearGrid(edges, edges * 1.1, edges * 1.2)
    material = bz.Material(permittivity=(2.5, 3.0, 3.3, 0.38, 0.14, 0.10))
    materials = MaterialGrid.from_raster_result(
        rasterize(
            Scene((material,)), grid, options=RasterOptions(smoothing="farjadpour_full")
        ),
        dimensions=3,
    )
    steps = 83
    sources = tuple(
        bz.CustomSource(
            component,
            "e",
            (16, 16, 16),
            2e-3,
            np.sin(np.arange(steps) * 0.22),
            component_shape_3d(component, grid.shape_zyx),
        )
        for component in ("Ex", "Ey", "Ez")
    )
    scene = bz.Simulation(
        material_grid=materials,
        sources=sources,
        normalize_source=None,
        boundaries=(bz.PML(formulation="cpml"),),
        time=np.arange(steps) * grid.cfl_time_step(0.55),
        setup_device="cpu",
    )
    try:
        policy = bz.AutoTermination(chunk_steps=41, max_wall_time=1e-12)
        result = scene.run(backend="cuda", termination=policy, performance=False)
        assert result.cuda_plan["binding"] == "native"
        assert result.termination.steps == 41
        program = _SESSIONS[scene].program
        checkpoint = scene.advance(
            num_steps=41, backend="cuda", performance=False
        ).state
        with jax.default_device(jax.devices("cpu")[0]):
            terms, cross, volume, primal = execute._energy_terms(program)
            exact = execute._field_diagnostics(
                checkpoint, (terms, cross, volume, primal)
            )
            diagonal = execute._field_diagnostics(
                checkpoint, (terms, (), volume, primal)
            )
        assert abs(exact[0] - diagonal[0]) > 1e-3 * exact[0]
        assert result.termination.energy == pytest.approx(exact[0], rel=2e-6)
        assert result.termination.max_field == exact[1]
    finally:
        scene.clear_compiled_cache()
