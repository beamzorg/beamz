"""Owned fields, CPML and DFT gathers across the native CUDA shard interface."""

from dataclasses import replace

import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.design.raster import Scene, rasterize
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.runtime.native_sharding import _run_shards

pytestmark = pytest.mark.hardware


def _scene(graded, pec, *, cells=(45, 48, 51), thickness=None, use_recorders=False, boundaries=None, use_scatter=False):
    edges = [
        np.r_[
            0.0,
            np.cumsum(
                spacing
                * (1 + 0.09 * np.sin(np.arange(n) * 0.31) if graded else np.ones(n))
            ),
        ]
        for n, spacing in zip(cells, (20e-9, 20e-9, 20e-9), strict=True)
    ]
    grid = RectilinearGrid(*edges)
    background = bz.Material(permittivity=2.25)
    materials = MaterialGrid.from_raster_result(
        rasterize(Scene((background,)), grid), dimensions=3
    )
    center = tuple(edge[-1] / 2 for edge in edges)
    freq = bz.LIGHT_SPEED / 720e-9
    steps = 113
    monitor = bz.FieldMonitor(
        center=tuple(x + 3e-9 for x in center),
        size=(200e-9, 220e-9, 0),
        freqs=(freq, 1.07 * freq),
        fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        interval=3,
        name="field",
    )
    recorders = (
        bz.FieldRecorder(("Ez",), interval=17, name="volume"),
        bz.FieldRecorder(
            ("Ez", "Hy"), interval=7, name="slice", center=center,
            size=(360e-9, 300e-9, 0),
        ),
    ) if use_recorders else ()
    scatter = ()
    if use_scatter:
        shape = materials.grid.shape_zyx
        z, y, x = (n // 2 for n in shape)
        scatter = (bz.CustomSource(
            "Ez", "e", (np.array([z, z, z + 1]), y, np.array([x - 1, x, x + 1])),
            np.asarray([0.001, 0.002, 0.003], np.float32),
            np.sin(np.arange(steps) * 0.19), (shape[0], shape[1] + 1, shape[2] + 1),
        ),)
    return bz.Simulation(
        material_grid=materials,
        sources=[
            bz.GaussianSource(
                position=center, width=35e-9, signal=np.sin(np.arange(steps) * 0.29)
            ),
            *scatter,
        ],
        boundaries=boundaries if boundaries is not None else [
            bz.PML(formulation="cpml", thickness=thickness),
            *([bz.PEC(edges=("left",))] if pec else []),
        ],
        monitors=(monitor, *recorders),
        time=1.2e-16 + np.arange(steps) * grid.cfl_time_step(0.55),
        normalize_source=None,
        setup_device="cpu",
    )


def _module():
    if not cuda_backend_status().available:
        pytest.skip("CUDA unavailable")
    module = _load_extension()
    if not hasattr(module.NativeSession, "_run_shards"):
        pytest.skip("Native sharding component unavailable")
    return module


def _compare(candidate, reference, steps=113, *, atol_fraction=2e-6):
    assert candidate.cuda_plan["binding"] == "native"
    assert (
        candidate.cuda_plan["final_step"] == reference.cuda_plan["final_step"] == steps
    )
    np.testing.assert_array_equal(candidate.metadata.time, reference.metadata.time)
    a, b = candidate["field"], reference["field"]
    np.testing.assert_array_equal(a.dft_weight_sum, b.dft_weight_sum)
    assert max(np.abs(value).max() for value in b.dft_fields.values()) > 1e-12
    for component, expected in b.dft_fields.items():
        np.testing.assert_allclose(
            a.dft_fields[component],
            expected,
            rtol=3e-5,
            atol=max(1e-18, np.abs(expected).max()) * atol_fraction,
        )


@pytest.mark.parametrize(
    "axis, graded, pec, partitions",
    [
        (0, False, False, 2),
        (1, False, False, 2),
        (2, False, False, 2),
        (0, True, False, 5),
        (1, True, True, 3),
        (2, True, True, 3),
    ],
)
def test_logical_shards_match_unsharded_on_one_gpu(axis, graded, pec, partitions):
    module = _module()
    # Every shard spans aligned full warps and a nonempty tail in x.
    x_cells = 83 * (partitions if axis == 2 else 1) + 3
    sim = _scene(graded, pec, cells=(x_cells, 48, 51))
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = _run_shards(sim, module, (0,) * partitions, axis, performance=False)
        assert result.cuda_plan["shard_axis"] == "zyx"[axis]
        assert len(result.cuda_plan["shard_payload_bytes"]) == partitions
        assert (
            max(result.cuda_plan["shard_payload_bytes"])
            < reference.cuda_plan["payload_bytes"]
        )
        _compare(result, reference)
    finally:
        sim.clear_compiled_cache()


def test_sharded_adaptive_checks_use_owned_yee_volumes():
    module = _module()
    sim = _scene(graded=True, pec=False)
    termination = bz.AutoTermination(chunk_steps=37, max_wall_time=1e-12)
    try:
        ordinary = sim.run(
            backend="cuda_streamed", termination=termination, performance=False
        )
        sharded = _run_shards(
            sim, module, (0, 0, 0), 1, termination=termination, performance=False
        )
        _compare(sharded, ordinary, steps=37)
        assert (
            sharded.termination.reason
            == ordinary.termination.reason
            == "wall_time_limit"
        )
        assert sharded.termination.time == ordinary.termination.time
        assert ordinary.termination.energy > 0
        assert sharded.termination.energy == pytest.approx(
            ordinary.termination.energy, rel=3e-6
        )
        assert sharded.termination.max_field == ordinary.termination.max_field
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_public_sharding_across_two_physical_gpus(axis):
    module = _module()
    if module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    sim = _scene(graded=True, pec=False)
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = sim.run(
            backend="cuda", sharding={"axis": axis, "num_devices": 2}, performance=False
        )
        assert result.cuda_plan["shard_devices"] == (0, 1)
        _compare(result, reference)
    finally:
        sim.clear_compiled_cache()


def test_resident_partitions_reset_reuse_and_release(monkeypatch):
    from beamz.simulation.cuda.engine import _runtime
    from beamz.simulation.cuda.runtime import native_sharding

    module = _module()
    sim = _scene(graded=False, pec=False)
    calls = []
    prepare = native_sharding._local_packet

    def counted(*args, **kwargs):
        calls.append((args[4], args[5], args[6]))
        return prepare(*args, **kwargs)

    monkeypatch.setattr(native_sharding, "_local_packet", counted)
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        for axis, expected_count in ((1, 2), (1, 2), (2, 4)):
            run = _run_shards(sim, module, (0, 0), axis, performance=False)
            _compare(run, reference)
            assert len(calls) == expected_count
        # Transition back onto a rank's device through the normal public dispatch.
        _compare(sim.run(backend="cuda_streamed", performance=False), reference)
        assert _runtime._SESSIONS[sim].key[0] != "sharded"
    finally:
        sim.clear_compiled_cache()
    assert sim not in _runtime._SESSIONS


@pytest.mark.parametrize(
    "bands",
    [(0, 0), (32, 32), (0, 32), (32, 0)],
    ids=["wavefront", "rolling", "rolling-right", "rolling-left"],
)
def test_temporal_shards_exchange_fields_and_finish_tail(monkeypatch, bands):
    from beamz.simulation.cuda.runtime import native_sharding
    from beamz.simulation.cuda.runtime.native_sharding import _halo_guard

    module = _module()
    if not hasattr(module.NativeSession, "_run_temporal_partitions"):
        pytest.skip("Temporal shard execution is unavailable")
    if any(bands) and not getattr(module, "compiled_rolling_tiles", False):
        pytest.skip("Rolling CUDA build required")
    if getattr(module, "compiled_rolling_tiles", False):
        selected = iter(bands)
        # Keep the requested 32-step communication window. Only local packets
        # with an imported halo receive the forced rank-specific placement.
        monkeypatch.setattr(
            native_sharding,
            "_rolling_band",
            lambda shape, *_: (
                next(selected) if shape[0] > sim.grid.shape_zyx[0] // 2 else 0
            ),
        )
        monkeypatch.setattr(
            native_sharding,
            "_admit_rolling",
            lambda resource, cells, width, available, **kwargs: (
                width,
                resource["rolling_extra_bytes"],
            ),
        )
    axis = 0 if any(bands) else 2
    cells = (57, 55, 130) if any(bands) else (130, 55, 57)
    sim = _scene(False, False, cells=cells, thickness=12 * 20e-9)
    try:
        full = sim.advance(backend="cuda_streamed", performance=False)
        reference = full.results
        for _ in range(2):
            result = _run_shards(
                sim, module, (0, 0), axis, performance=False, temporal=(32, 16, 6)
            )
            assert result.cuda_plan["schedule"] == "mstep"
            assert result.cuda_plan["exchange_window"] == 32
            _compare(result, reference)

        from beamz.simulation.cuda.engine._contracts import _DFT
        from beamz.simulation.cuda.engine._runtime import _SESSIONS

        stepper = _SESSIONS[sim]
        if bands == (32, 32) and getattr(module, "compiled_temporal_packet_links", False):
            spans, axis, window, depth, slots, links = stepper.temporal
            assert links == (False,)  # Two logical ranks share one physical device.
            with pytest.raises(ValueError, match="one entry per temporal interface"):
                module.NativeSession._run_temporal_partitions(
                    [shard[1] for shard in stepper.shards], spans, axis, 1, window, depth, slots, True, [True, True]
                )
        paired = stepper._snapshot()
        stepper.reset()
        for steps in (9, 37, 67):
            stepper._advance(steps, snapshot=False)
        resumed = stepper._snapshot()
        for key in _DFT:
            np.testing.assert_array_equal(getattr(resumed, key), getattr(paired, key))
        # A shard's inner face is artificial. Only its outer half of a
        # normal CPML history belongs to the global physical state.
        terms = (*full.state.cpml_psi_h_terms, *full.state.cpml_psi_e_terms)
        for rank, (_, owner, _, _) in enumerate(stepper.shards):
            assert owner.metadata().get("rolling_band", 0) == bands[rank]
            # Compare the actual owned fields on both sides of the cut.
            # The native rank keeps an additional causal halo at the inner face.
            start = stepper.edges[rank]
            halo = 32 + _halo_guard(module)
            local = slice(halo, None) if rank else slice(None, stepper.edges[1])
            owned = slice(start, None) if rank else slice(None, stepper.edges[1])
            for name, values in zip(
                ("hx", "hy", "hz", "ex", "ey", "ez"), owner.download()[:6], strict=True
            ):
                np.testing.assert_array_equal(
                    np.moveaxis(values, axis, 0)[local],
                    np.moveaxis(np.asarray(getattr(full.state, name)), axis, 0)[owned],
                )
            side = slice(0, 12) if rank == 0 else slice(12, 24)
            for term in (1, 2, 7, 8) if axis == 0 else (3, 4, 9, 10):
                expected = np.asarray(terms[term])
                actual = np.empty_like(expected)
                owner.download_into(6 + term, actual)
                np.testing.assert_array_equal(
                    np.moveaxis(actual, axis, 0)[side].view(np.uint32),
                    np.moveaxis(expected, axis, 0)[side].view(np.uint32),
                )
    finally:
        sim.clear_compiled_cache()


def test_temporal_restore_preserves_transparent_physical_cpml(monkeypatch):
    from beamz.simulation import execute

    if not hasattr(_module().NativeSession, "_run_temporal_partitions"):
        pytest.skip("Temporal shard execution is unavailable")
    prepare = execute.build_step_context

    def transparent_lower_face(term):
        if term.axis != 0:
            return term
        profiles = {}
        for name in ("a", "b", "inv_kappa"):
            values = np.asarray(getattr(term, name)).copy()
            values.reshape(-1)[: term.slab.low] = float(name == "inv_kappa")
            profiles[name] = values
        return replace(term, **profiles)

    def prepare_with_transparent_face(*args, **kwargs):
        ctx = prepare(*args, **kwargs)
        cpml = ctx.boundary.cpml
        cpml = replace(
            cpml,
            h_terms=tuple(map(transparent_lower_face, cpml.h_terms)),
            e_terms=tuple(map(transparent_lower_face, cpml.e_terms)),
        )
        return replace(ctx, boundary=replace(ctx.boundary, cpml=cpml))

    monkeypatch.setattr(execute, "build_step_context", prepare_with_transparent_face)
    sim = _scene(False, False, cells=(131, 96, 105))
    try:
        state = sim.advance(num_steps=1, backend="cuda_streamed").state
        groups = ("cpml_psi_h_terms", "cpml_psi_e_terms")
        state = state._replace(
            **{
                key: tuple(np.full_like(a, 0.25) for a in getattr(state, key))
                for key in groups
            }
        )
        ordinary = sim.advance(state=state, num_steps=16, backend="cuda_streamed")
        temporal = sim.advance(state=state, num_steps=16, backend="cuda_mstep")
        assert temporal.results.cuda_plan["schedule"] == "temporal"
        for key in groups:
            for actual, expected in zip(
                getattr(temporal.state, key), getattr(ordinary.state, key), strict=True
            ):
                np.testing.assert_array_equal(
                    np.asarray(actual).view(np.uint32),
                    np.asarray(expected).view(np.uint32),
                )
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("axis", ["x", "z"])
def test_temporal_shards_on_two_physical_gpus(axis):
    module = _module()
    if module._device_count() < 2 or not hasattr(
        module.NativeSession, "_run_temporal_partitions"
    ):
        pytest.skip("Two physical CUDA GPUs with temporal sharding required")
    cells = (210, 55, 57) if axis == "x" else (57, 55, 210)
    sim = _scene(False, False, cells=cells, thickness=12 * 20e-9)
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = sim.run(
            backend="cuda",
            schedule="mstep",
            sharding={"axis": axis, "num_devices": 2},
            performance=False,
        )
        assert result.cuda_plan["shard_devices"] == (0, 1)
        assert result.cuda_plan["schedule"] == "mstep"
        _compare(result, reference)
    finally:
        sim.clear_compiled_cache()


def test_public_two_temporal_partitions_share_one_gpu(monkeypatch):
    from beamz.simulation.cuda.runtime import native_sharding as sharding

    module = _module()
    if not hasattr(module.NativeSession, "_run_temporal_partitions"):
        pytest.skip("Temporal shard execution is unavailable")
    # Exercise the public placement on a small grid, including on a developer GPU.
    monkeypatch.setattr(
        sharding, "_temporal_shard_choice", lambda *a, **kw: (32, 16, 4)
    )
    monkeypatch.setattr(sharding, "_split_underfilled_frontier", lambda *a: True)
    sim = _scene(False, False, cells=(210, 55, 57), thickness=12 * 20e-9)
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = sim.run(
            backend="cuda", sharding={"axis": "x", "num_devices": 1}, performance=False
        )
        assert result.cuda_plan["shard_devices"] == (0,)
        assert result.cuda_plan["partition_devices"] == (0, 0)
        assert result.cuda_plan["schedule"] == "mstep"
        _compare(result, reference)

        from beamz.simulation.cuda.engine._runtime import _SESSIONS

        owners = [owner for _, owner, _, _ in _SESSIONS[sim].shards]
        planned = sum(result.cuda_plan["shard_payload_bytes"])
        assert all(owner.metadata()["moving_macro_slab_count"] == 2 for owner in owners)
        assert planned == sum(
            owner.metadata()["allocated_payload_bytes"] for owner in owners
        )
        board_limit = planned + result.cuda_plan["reserve_bytes"]
        sim.clear_compiled_cache()
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(board_limit - 1))
        with pytest.raises(MemoryError, match="shard 1"):
            _run_shards(sim, module, (0, 0), 2, performance=False, temporal=(32, 16, 4))
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(board_limit))
        _compare(
            _run_shards(
                sim, module, (0, 0), 2, performance=False, temporal=(32, 16, 4)
            ),
            reference,
        )
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("physical", [False, True])
def test_general_temporal_shards_with_adaptive_rectilinear_cpml(axis, physical):
    module = _module()
    if physical and module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    cells = [35, 38, 41]
    cells[2 - axis] = 84
    sim = _scene(True, True, cells=cells, thickness=7 * 20e-9)
    termination = bz.AutoTermination(chunk_steps=39, max_wall_time=1e-12)
    try:
        reference = sim.run(
            backend="cuda_streamed", termination=termination, performance=False
        )
        result = _run_shards(
            sim, module, (0, int(physical)), axis, performance=False,
            temporal=(16, 8, 4), termination=termination,
        )
        assert result.cuda_plan["schedule"] == "mstep"
        assert result.cuda_plan["temporal_windows"] == 2
        _compare(result, reference, steps=39)
        assert result.termination.reason == reference.termination.reason == "wall_time_limit"
        assert result.termination.energy == pytest.approx(reference.termination.energy, rel=3e-6)
        assert result.termination.max_field == pytest.approx(reference.termination.max_field, rel=3e-6)
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_temporal_shards_record_interior_steps_and_tail(physical, adaptive):
    module = _module()
    if physical and module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    sim = _scene(True, False, cells=(84, 35, 38), thickness=7 * 20e-9, use_recorders=True, use_scatter=True)
    termination = bz.AutoTermination(chunk_steps=39, max_wall_time=1e-12) if adaptive else None
    try:
        reference = sim.run(backend="cuda_streamed", performance=False, termination=termination)
        result = _run_shards(
            sim, module, (0, int(physical)), 2, performance=False, temporal=(16, 8, 4), termination=termination,
        )
        assert result.cuda_plan["schedule"] == "mstep"
        _compare(result, reference, steps=39 if adaptive else 113)
        for name in ("volume", "slice"):
            actual, expected = result[name], reference[name]
            np.testing.assert_array_equal(actual.field_steps, expected.field_steps)
            np.testing.assert_array_max_ulp(actual.field_times, expected.field_times, maxulp=1)
            for component, values in actual.fields.items():
                expected_values = expected.fields[component]
                assert np.any(expected_values != 0)
                np.testing.assert_allclose(
                    values, expected_values, rtol=5e-5, atol=2e-6 * np.max(np.abs(expected_values)),
                )
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("absorber_crosses_cut", [False, True])
def test_temporal_asymmetric_cpml_across_internal_cut(absorber_crosses_cut):
    module = _module()
    sim = _scene(
        True, False, cells=(84, 35, 38),
        boundaries=(
            bz.PML(
                edges=("left",), formulation="cpml",
                thickness=(48 if absorber_crosses_cut else 7) * 20e-9,
            ),
            bz.PEC(edges=("right",)),
        ),
    )
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = _run_shards(sim, module, (0, 0), 2, performance=False, temporal=(16, 8, 4))
        _compare(result, reference)
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("sponge", [False, True])
def test_temporal_shards_without_cpml(sponge):
    module = _module()
    boundaries = (bz.PML(formulation="sponge", thickness=7 * 20e-9),) if sponge else ()
    sim = _scene(True, False, cells=(84, 35, 38), boundaries=boundaries)
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = _run_shards(sim, module, (0, 0), 2, performance=False, temporal=(16, 8, 4))
        assert result.cuda_plan["temporal_windows"] == 7
        _compare(result, reference)
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_temporal_shards_full_tensor_rectilinear(physical, adaptive):
    from beamz.design.raster import RasterOptions

    module = _module()
    if physical and module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    sim = _scene(True, False, cells=(84, 35, 38), thickness=7 * 20e-9)
    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene((bz.Material(permittivity=(2.25, 2.5, 2.75, 0.25, 0.1, 0.15)),)),
            sim.grid, options=RasterOptions(smoothing="farjadpour_full"),
        ),
        dimensions=3,
    )
    sim = sim.updated_copy(material_grid=material)
    termination = bz.AutoTermination(chunk_steps=39, max_wall_time=1e-12) if adaptive else None
    try:
        reference = sim.run(backend="cuda_streamed", performance=False, termination=termination)
        result = _run_shards(
            sim, module, (0, int(physical)), 2, performance=False, temporal=(8, 4, 6),
            termination=termination,
        )
        steps = 39 if adaptive else 113
        assert result.cuda_plan["temporal_windows"] == steps // 8
        _compare(result, reference, steps=steps, atol_fraction=5e-5)
        if adaptive:
            assert result.termination.reason == reference.termination.reason
            assert result.termination.energy == pytest.approx(reference.termination.energy, rel=3e-6)
            assert result.termination.max_field == pytest.approx(reference.termination.max_field, rel=3e-6)
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("physical", [False, True])
def test_temporal_shards_bf16_cpml(monkeypatch, physical):
    module = _module()
    if physical and module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "bf16")
    sim = _scene(True, False, cells=(84, 35, 38), thickness=7 * 20e-9)
    try:
        reference = sim.run(backend="cuda_streamed", performance=False)
        result = _run_shards(
            sim, module, (0, int(physical)), 2, performance=False, temporal=(16, 8, 4)
        )
        assert result.cuda_plan["temporal_windows"] == 7
        _compare(result, reference)
    finally:
        sim.clear_compiled_cache()


def test_public_temporal_adaptive_recorders_on_two_physical_gpus():
    module = _module()
    if module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    sim = _scene(True, True, cells=(108, 35, 38), thickness=7 * 20e-9, use_recorders=True)
    termination = bz.AutoTermination(chunk_steps=39, max_wall_time=1e-12)
    try:
        reference = sim.run(backend="cuda_streamed", termination=termination, performance=False)
        result = sim.run(
            backend="cuda", schedule="mstep", sharding={"axis": "x", "num_devices": 2},
            termination=termination, performance=False,
        )
        assert result.cuda_plan["shard_devices"] == (0, 1)
        assert result.cuda_plan["temporal_windows"] > 0
        _compare(result, reference, steps=39)
        np.testing.assert_array_equal(result["slice"].field_steps, reference["slice"].field_steps)
        assert result.termination.steps == reference.termination.steps
        assert result.termination.energy == pytest.approx(reference.termination.energy, rel=3e-6)
    finally:
        sim.clear_compiled_cache()


def test_temporal_shards_converge_at_the_reference_check():
    module = _module()
    sim = _scene(True, True, cells=(84, 35, 38), thickness=7 * 20e-9)
    termination = bz.AutoTermination(
        chunk_steps=19, min_steps=20, field_decay=1.0, source_decay=1.0,
        monitor_change=None, consecutive_checks=2,
    )
    try:
        reference = sim.run(backend="cuda_streamed", termination=termination, performance=False)
        result = _run_shards(
            sim, module, (0, 0), 2, performance=False, temporal=(16, 8, 4),
            termination=termination,
        )
        assert reference.termination.reason == result.termination.reason == "converged"
        assert reference.termination.steps == result.termination.steps
        assert result.cuda_plan["temporal_windows"] == result.termination.steps // 19
        assert result.termination.energy == pytest.approx(reference.termination.energy, rel=3e-6)
        for name, expected in reference["field"].dft_fields.items():
            peak = np.max(np.abs(expected))
            np.testing.assert_allclose(result["field"].dft_fields[name], expected, rtol=3e-5, atol=2e-6 * peak)
    finally:
        sim.clear_compiled_cache()


@pytest.mark.parametrize("physical", [False, True])
def test_temporal_shards_preserve_public_checkpoint_branches(physical, monkeypatch):
    from beamz.simulation.cuda.runtime import native_sharding

    module = _module()
    if physical and module._device_count() < 2:
        pytest.skip("Two physical CUDA GPUs required")
    scene = _scene(True, True, cells=(108, 35, 38), thickness=7 * 20e-9,
                   use_recorders=True, use_scatter=True)
    count = 2 if physical else 1
    if not physical:
        monkeypatch.setattr(native_sharding, "_minimum_temporal_slabs", lambda *a: 2)
    config = {"axis": "x", "num_devices": count}
    try:
        reference = scene.advance(backend="cuda_streamed", performance=False)
        first = scene.advance(
            num_steps=39, backend="cuda", schedule="mstep", sharding=config, performance=False,
        )
        assert first.results.cuda_plan["temporal_windows"] > 0
        initial_field = np.copy(first.state.ez)
        for _ in range(2):
            resumed = scene.advance(
                state=first.state, backend="cuda", schedule="mstep", sharding=config, performance=False,
            )
            assert resumed.results.cuda_plan["temporal_windows"] > 0
            assert resumed.results.cuda_plan["initial_step"] == 39
            assert resumed.results.cuda_plan["final_step"] == 113
            _compare(resumed.results, reference.results)
            for field in ("ex", "ey", "ez", "hx", "hy", "hz"):
                expected = np.asarray(getattr(reference.state, field))
                np.testing.assert_allclose(
                    getattr(resumed.state, field), expected,
                    rtol=3e-5, atol=2e-6 * np.max(np.abs(expected)),
                )
            for name in ("cpml_psi_h_terms", "cpml_psi_e_terms"):
                for actual, expected in zip(getattr(resumed.state, name), getattr(reference.state, name), strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=4e-5,
                                               atol=2e-6 * np.max(np.abs(expected), initial=0.0))
            for name in ("volume", "slice"):
                recording = resumed.results[name]
                expected = reference.results[name]
                np.testing.assert_array_equal(recording.field_steps, expected.field_steps)
                for component, values in expected.fields.items():
                    np.testing.assert_allclose(
                        recording.fields[component], values, rtol=3e-5,
                        atol=2e-6 * np.max(np.abs(values)),
                    )
            np.testing.assert_array_equal(first.state.ez, initial_field)
    finally:
        scene.clear_compiled_cache()
