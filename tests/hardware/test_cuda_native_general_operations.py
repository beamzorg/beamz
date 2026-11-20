"""Exercise general ordinary effects and bounded native coefficient conversion."""

import numpy as np
import pytest

import beamz as bz
from beamz.lattice import component_shape_3d
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine._runtime import _SESSIONS, _clear, _Session

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def test_electric_table_conversion_preserves_bits_across_bounded_tiles():
    module = _load_extension()
    epsilon = (
        np.random.default_rng(20261116)
        .uniform(0.2, 12, (1 << 22) + 17)
        .astype(np.float32)
    )
    actual = module.electric_coefficient_table(
        epsilon, np.float32(1e-17), np.float32(bz.EPS_0)
    )
    split = len(epsilon) - 31
    expected = np.concatenate(
        [
            module.electric_coefficient_table(
                values, np.float32(1e-17), np.float32(bz.EPS_0)
            )
            for values in (epsilon[:split], epsilon[split:])
        ]
    )
    assert module.compiled_coefficient_staging_bytes == 32 << 20
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("schedule", ["streamed", "mstep"])
def test_ragged_monitor_gathers_match_dense_with_ordinary_tail(monkeypatch, schedule):
    module = _load_extension()
    if not getattr(module, "compiled_ragged_dft_gather", False):
        pytest.skip("installed module predates ragged monitor gathers")
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        time=np.arange(49) * 1e-17,
        boundaries=(bz.PML(formulation="cpml"),),
        sources=(
            bz.CustomSource(
                "Ez",
                "e",
                (20, slice(19, 22), 20),
                np.ones(3, np.float32),
                np.ones(49, np.float32),
                component_shape_3d("Ez", (40, 40, 40)),
            ),
        ),
        monitors=(
            bz.FieldMonitor(
                center=(0.4e-6,) * 3,
                size=(0.5e-6, 0.5e-6, 0),
                freqs=[4e14],
                fields=("Ez",),
                name="plane",
            ),
            bz.FluxMonitor(
                center=(0.45e-6, 0.4e-6, 0.4e-6),
                size=(0, 0.2e-6, 0.2e-6),
                freqs=np.linspace(2e14, 5e14, 9),
                name="port",
            ),
        ),
    )
    states, gather_bytes = [], []
    try:
        for packed in (False, True):
            monkeypatch.setattr(module, "compiled_ragged_dft_gather", packed)
            simulation.run(backend="cuda", schedule=schedule, performance=False)
            session = _SESSIONS[simulation]
            start = session.packet._layout.monitor
            indices, weights = session.packet.arguments[start : start + 2]
            assert indices.ndim == (2 if packed else 4)
            gather_bytes.append(indices.nbytes + weights.nbytes)
            snapshot = session._snapshot(session.native.clocks(), fields=True)
            states.append(
                (
                    snapshot.hx,
                    snapshot.hy,
                    snapshot.hz,
                    snapshot.ex,
                    snapshot.ey,
                    snapshot.ez,
                    *snapshot.cpml_psi_h_terms,
                    *snapshot.cpml_psi_e_terms,
                    snapshot.dft_vec_re,
                    snapshot.dft_vec_im,
                    snapshot.dft_weight_sum,
                )
            )
            _clear(simulation)
    finally:
        _clear(simulation)
    assert gather_bytes[1] < gather_bytes[0] / 2
    assert np.max(np.abs(states[0][5])) > 0
    for expected, actual in zip(*states, strict=True):
        np.testing.assert_array_equal(
            np.asarray(expected).view(np.uint8), np.asarray(actual).view(np.uint8)
        )


@pytest.mark.parametrize("pec", [False, True])
def test_public_ordinary_sources_and_boundaries_match_complete_program(pec):
    module = _load_extension()
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        time=np.arange(33) * 1e-17,
        boundaries=[
            bz.PML(formulation="cpml"),
            *([bz.PEC(edges=("left",))] if pec else []),
        ],
        sources=[
            bz.GaussianSource(position=(0.4e-6,) * 3, width=40e-9, signal=np.ones(33))
        ],
        monitors=[
            bz.FieldMonitor(
                center=(0.4e-6,) * 3,
                size=(0, 0.2e-6, 0.2e-6),
                freqs=[4e14],
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
    )
    try:
        result = simulation.run(backend="cuda", performance=False)
        assert (
            result.cuda_plan["binding"] == "native"
            and result.cuda_plan["schedule"] == "ordinary"
        )
        session = _SESSIONS[simulation]
        oracle = module.NativeSession(
            session.packet.arguments,
            dict(session.packet.attributes),
            preserve_inputs=False,
            kind="program",
            initialize_scratch=False,
        )
        try:
            oracle.run_schedule([simulation.num_steps])
            actual, expected = session.native.download(), oracle.download()
            for index in (*range(18), 30, 31, 32):
                np.testing.assert_array_equal(
                    np.asarray(actual[index]).view(np.uint8),
                    np.asarray(expected[index]).view(np.uint8),
                )
            assert session.native.clocks() == oracle.clocks()
            assert any(
                np.max(np.abs(actual[index]), initial=0) > 0 for index in range(6)
            )
        finally:
            oracle.close()
        if not pec:
            values, attributes = (
                list(session.packet.arguments),
                dict(session.packet.attributes),
            )
            group = next(g for g in range(9) if np.asarray(values[86 + 3 * g]).shape[0])
            offset = 86 + 3 * group
            for index in range(3):
                values[offset + index] = np.concatenate(
                    [values[offset + index]] * 2, axis=0
                )
            values[offset + 2][1, 2] += 1
            attributes["coincident_source_group_mask"] |= 1 << group
            attributes["disjoint_source_group_mask"] &= ~(1 << group)
            for kind in ("generated", "program", "mstep"):
                with pytest.raises(ValueError, match="coincident source starts differ"):
                    module.NativeSession(
                        tuple(values),
                        attributes,
                        preserve_inputs=False,
                        kind=kind,
                        initialize_scratch=False,
                    )
    finally:
        simulation.clear_compiled_cache()


def test_ordinary_dense_h_coefficients_use_physical_coordinates():
    module = _load_extension()
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        time=np.arange(2) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")],
    )
    prepared = _Session(simulation, module, requested="ordinary")
    try:
        values = [
            np.zeros(tuple(v.shape), np.dtype(v.dtype))
            if isinstance(v, module.ZeroBuffer)
            else np.asarray(v).copy()
            for v in prepared.packet.arguments
        ]
        for component in range(3):
            values[9 + component] = np.full(
                values[component].shape, values[9 + component], np.float32
            )
            values[9 + component][..., 20:] *= np.float32(0.5)
        values[3][:] = np.arange(values[3].shape[1], dtype=np.float32)[None, :, None]
        solver = module.NativeSession(
            tuple(values),
            dict(prepared.packet.attributes),
            preserve_inputs=False,
            kind="generated",
            initialize_scratch=False,
        )
        try:
            solver.run_schedule([1])
            hz = solver.download()[2]
            for x in (17, 22):
                expected = values[11][20, 20, x] * np.float32(1 / 20e-9)
                np.testing.assert_array_equal(
                    hz[20, 20, x].view(np.uint32), expected.view(np.uint32)
                )
            assert hz[20, 20, 17] == np.float32(2) * hz[20, 20, 22]
        finally:
            solver.close()
    finally:
        prepared._close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("initialize_scratch", [False, True])
def test_immutable_palette_aliases_share_owned_snapshots_and_never_mutable_state(
    initialize_scratch,
):
    module = _load_extension()
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        time=np.arange(33) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")],
        monitors=[
            bz.FieldMonitor(
                center=(0.4e-6,) * 3,
                size=(0, 0.2e-6, 0.2e-6),
                freqs=[4e14],
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
    )
    prepared = _Session(simulation, module, requested="ordinary")
    solvers = []
    try:
        values = [
            np.zeros(tuple(v.shape), np.dtype(v.dtype))
            if isinstance(v, module.ZeroBuffer)
            else np.asarray(v).copy()
            for v in prepared.packet.arguments
        ]
        block_counts = [(values[3 + c].size + 31) // 32 for c in range(3)]
        palette = (
            np.random.default_rng(20261116)
            .uniform(1e-7, 1e-6, sum(block_counts) * 32)
            .astype(np.float32)
        )
        offset = 0
        for component, blocks in enumerate(block_counts):
            values[37 + component] = palette
            values[40 + component] = (offset + 32 * np.arange(blocks, dtype=np.int32))[
                None, :
            ]
            offset += blocks * 32
        values[3][:] = np.sin(
            np.arange(values[3].shape[1], dtype=np.float32) * np.float32(0.2)
        )[None, :, None] * np.float32(1e-4)
        values[121] = values[120]
        values[123] = values[124] = values[125] = np.zeros_like(values[123])
        distinct = list(values)
        for index in (*range(37, 40), 120, 121, 123, 124, 125):
            distinct[index] = values[index].copy()
        for packet, kind in (
            (values, "generated"),
            (distinct, "generated"),
            (values, "mstep"),
        ):
            solvers.append(
                module.NativeSession(
                    tuple(packet),
                    dict(prepared.packet.attributes),
                    preserve_inputs=False,
                    kind=kind,
                    initialize_scratch=initialize_scratch,
                )
            )
        shared, separate, moving = solvers
        rolling = None
        if getattr(module, "compiled_rolling_material_trace", False):
            attrs = dict(prepared.packet.attributes, rolling_band=32)
            rolling = module.NativeSession(
                tuple(values), attrs, preserve_inputs=False, kind="mstep"
            )
            solvers.append(rolling)
            for component, invalid in enumerate((2**31 - 1, -2**31, palette.size)):
                malformed = list(values)
                headers = values[40 + component].copy()
                headers[0, -1] = invalid
                malformed[40 + component] = headers
                with pytest.raises(ValueError, match="coefficient pool"):
                    module.NativeSession(tuple(malformed), attrs, preserve_inputs=False, kind="mstep")
        assert moving.metadata()["initialize_scratch"] is False
        assert moving.metadata()["scratch_zero_bytes_per_run"] == 0
        moving_resident = moving.metadata()["resident_bytes"]
        with pytest.raises(ValueError, match="require run_moving"):
            moving.run_schedule([1])
        assert moving.metadata()["resident_bytes"] == moving_resident
        assert moving.clocks()["current_step"] == 0
        a, b = shared.metadata(), separate.metadata()
        assert module.compiled_immutable_aliases
        assert a["immutable_input_alias_count"] - b["immutable_input_alias_count"] == 2
        assert b["resident_bytes"] - a["resident_bytes"] == 2 * palette.nbytes
        assert (
            a["immutable_gpu_alias_bytes"] - b["immutable_gpu_alias_bytes"]
            == 2 * palette.nbytes
        )
        # Ordinary execution only needs the device copy. Temporal planning
        # snapshots this palette once even though all three components share it.
        assert a["immutable_host_alias_bytes"] == b["immutable_host_alias_bytes"] == 0
        assert moving.metadata()["immutable_host_alias_bytes"] >= 2 * palette.nbytes
        # A failed admission must retain snapshots until a valid plan exists.
        assert moving.metadata()["retired_coefficient_host_bytes"] == 0
        with pytest.raises(ValueError, match="depth must"):
            moving.run_moving([16], 7, 4)
        assert moving.metadata()["retired_coefficient_host_bytes"] == 0
        # First certification consumes snapshots; resets reuse its immutable GPU proof.
        palette.fill(-7)
        for c in range(3):
            values[40 + c].fill(np.iinfo(np.int32).max)
        expected_retired = palette.nbytes + sum(values[40 + c].nbytes for c in range(3))
        for _repeat in range(2):
            for solver in solvers:
                solver.reset()
            if rolling is not None:
                assert rolling.metadata()["retired_coefficient_host_bytes"] == expected_retired
                rolling.run_moving([32, 1], 16, 4)
            separate.run_schedule([33])
            shared.run_schedule([33])
            moving_timing = moving.run_moving([32, 1], 16, 4)
            moving_metadata = moving.metadata()
            assert moving_metadata["retired_coefficient_host_bytes"] == expected_retired
            with pytest.raises(ValueError, match="fixed per session"):
                moving.run_moving([1], 4, 4)
            with pytest.raises(ValueError, match="fixed per session"):
                moving.run_moving([1], 16, 3)
            assert moving_timing["initialize_scratch"] is False
            assert moving_timing["scratch_zero_bytes"] == 0
            assert moving_metadata["scratch_zero_bytes_per_run"] == 0
            assert moving_metadata["allocated_scratch_bytes"] == 0
            assert (
                moving_metadata["workspace_policy"]
                == "unused_workspace_deferred_no_scratch_fill"
            )
            expected = separate.download()
            for solver in (shared, moving, *([rolling] if rolling is not None else [])):
                actual = solver.download()
                for index in (*range(18), 30, 31, 32):
                    np.testing.assert_array_equal(
                        np.asarray(actual[index]).view(np.uint8),
                        np.asarray(expected[index]).view(np.uint8),
                    )
                assert solver.clocks() == separate.clocks()
        assert any(np.max(np.abs(expected[index]), initial=0) > 0 for index in range(6))
    finally:
        for solver in solvers:
            solver.close()
        prepared._close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("cpml", [False, True])
@pytest.mark.parametrize("kind", ["generated", "program"])
@pytest.mark.parametrize("preserve", [False, True])
def test_ordinary_unused_workspace_stays_deferred_and_diagnostics_owned(
    cpml, kind, preserve
):
    module = _load_extension()
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        time=np.arange(33) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")] if cpml else [],
    )
    prepared = _Session(simulation, module, requested="ordinary")
    solver = None
    try:
        values = list(prepared.packet.arguments)
        graph_count, state_count = (74, 18) if cpml else (24, 6)
        expected = []
        for component in range(6):
            initial = np.full(values[component].shape, component + 0.25, np.float32)
            values[graph_count + component] = initial
            expected.append(initial.copy())
        solver = module.NativeSession(
            tuple(values),
            dict(prepared.packet.attributes),
            preserve_inputs=preserve,
            kind=kind,
            initialize_scratch=True,
        )
        metadata = solver.metadata()
        assert metadata["allocated_final_workspace_bytes"] == 0
        assert metadata["allocated_scratch_bytes"] == 0
        assert metadata["deferred_final_workspace_bytes"] == sum(
            value.nbytes for value in expected
        )
        for component in range(6):
            values[graph_count + component].fill(-99)
        for chunks in ([1, 16, 16], [33]):
            solver.reset()
            timing = solver.run_schedule(chunks)
            downloaded = solver.download()
            for component, initial in enumerate(expected):
                np.testing.assert_array_equal(
                    downloaded[state_count + component], initial
                )
                target = np.empty_like(initial)
                solver.download_into(state_count + component, target)
                np.testing.assert_array_equal(target, initial)
            actual_metadata = solver.metadata()
            assert actual_metadata["resident_bytes"] == metadata["resident_bytes"]
            assert (
                actual_metadata["preserving_copy_bytes_per_run"]
                == timing["preserving_copy_bytes"]
            )
            assert actual_metadata["initialize_scratch"] is False
            assert actual_metadata["scratch_zero_bytes_per_run"] == 0
            assert solver.clocks()["current_step"] == 33
    finally:
        if solver is not None:
            solver.close()
        prepared._close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("cpml", [False, True])
@pytest.mark.parametrize("kind", ["generated", "program"])
@pytest.mark.parametrize("preserve", [False, True])
def test_ordinary_material_capture_retirement_preserves_owned_gpu_state(
    cpml, kind, preserve
):
    module = _load_extension()
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        time=np.arange(65) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")] if cpml else [],
        sources=[
            bz.GaussianSource(
                position=(0.4e-6,) * 3,
                width=40e-9,
                signal=np.sin(np.arange(65) * 0.17).astype(np.float32),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(0.4e-6,) * 3,
                size=(0, 0.2e-6, 0.2e-6),
                freqs=[4e14],
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
    )
    prepared = _Session(simulation, module, requested="ordinary")
    solvers = []
    try:
        layout = prepared.packet._layout
        values = [
            np.zeros(tuple(v.shape), np.dtype(v.dtype))
            if isinstance(v, module.ZeroBuffer)
            else np.asarray(v).copy()
            for v in prepared.packet.arguments
        ]
        electric = layout.inputs["e_decay_x"]
        blocks = [(values[3 + c].size + 31) // 32 for c in range(3)]
        rng = np.random.default_rng(20261125)
        palette = rng.uniform(1e-7, 1e-6, sum(blocks) * 32).astype(np.float32)
        offset = 0
        for c, count in enumerate(blocks):
            values[electric + c] = palette
            values[electric + 3 + c] = (offset + 32 * np.arange(count, dtype=np.int32))[
                None, :
            ]
            offset += count * 32
        mutable = [
            *range(6),
            *(layout.inputs[name] for name in ("dft_vec_re", "dft_vec_im")),
        ]
        if cpml:
            mutable.extend(
                layout.inputs[f"{phase}_psi_{i}"]
                for phase in ("h", "e")
                for i in range(6)
            )
        for i in mutable:
            values[i][...] = rng.normal(size=values[i].shape).astype(np.float32) * 1e-4
        assert any(values[layout.source + 3 * g].shape[0] for g in range(3))
        for selected in ("program", kind):
            solvers.append(
                module.NativeSession(
                    tuple(values),
                    dict(prepared.packet.attributes),
                    preserve_inputs=preserve,
                    kind=selected,
                    initialize_scratch=False,
                )
            )
        reference, actual = solvers
        # Ordinary uploads retire the caller's storage without allocating
        # private coefficient snapshots. Every device-side palette remains owned.
        expected_retired = 0
        assert actual.metadata()["retired_coefficient_host_bytes"] == expected_retired
        assert actual.metadata()["retired_upload_host_bytes"] >= palette.nbytes
        # The caller can change both the palette and its addressing after upload.
        palette.fill(-7)
        for c in range(3):
            values[electric + 3 + c].fill(np.iinfo(np.int32).max)
        outputs = [
            *range(18 if cpml else 6),
            *(
                layout.outputs[name]
                for name in ("dft_vec_re", "dft_vec_im", "dft_weight_sum")
            ),
        ]

        def compare():
            expected, observed = reference.download(), actual.download()
            for index in outputs:
                assert (
                    np.isfinite(expected[index]).all()
                    and np.isfinite(observed[index]).all()
                )
                np.testing.assert_array_equal(
                    np.asarray(expected[index]).view(np.uint8),
                    np.asarray(observed[index]).view(np.uint8),
                )
            assert reference.clocks() == actual.clocks()
            assert (
                actual.metadata()["retired_coefficient_host_bytes"] == expected_retired
            )

        reference.run_schedule([33])
        actual.run_schedule([16, 17])
        compare()
        for _ in range(2):
            reference.reset()
            actual.reset()
            reference.run_schedule([33])
            actual.run_schedule([33])
            compare()
        if not preserve:
            reference.reset()
            actual.reset()
            reference.run_schedule([33])
            actual.run_schedule([16])
            actual.run_schedule([17])
            compare()
    finally:
        for solver in solvers:
            solver.close()
        prepared._close()
        simulation.clear_compiled_cache()


def test_rolling_material_trace_checks_each_staggered_plane_and_its_halo():
    from beamz.simulation.cuda.storage.material_blocks import pack_material_blocks

    module = _load_extension()
    if not getattr(module, "compiled_rolling_tiles", False):
        pytest.skip("Rolling CUDA build required")
    dx = 20e-9
    sim = bz.Simulation(
        domain=tuple(n * dx for n in (94, 89, 83)),
        resolution=dx,
        time=np.arange(35) * 1e-17,
        boundaries=[bz.PML(formulation="cpml", thickness=12 * dx)],
        monitors=[
            bz.FieldMonitor(
                center=(0.3e-6,) * 3, size=(0, 0.16e-6, 0.16e-6),
                freqs=[4e14], fields=("Ex", "Hy"), name="probe",
            )
        ],
    )
    prepared = _Session(sim, module, requested="ordinary")
    owners = []
    try:
        args = list(prepared.packet.arguments)
        rng = np.random.default_rng(20261119)
        for field in range(6):
            args[field] = rng.uniform(-1e-5, 1e-5, args[field].shape).astype(np.float32)
        profiles = []
        for component, field in enumerate(args[3:6]):
            z, y, x = np.ogrid[tuple(slice(n) for n in field.shape)]
            # One region is uniform on every XY plane, another is uniform along Z.
            values = np.where(x < 54, 1 + (z // 5 + component) % 3, 1 + (y // 7) % 2)
            values = np.ascontiguousarray(values, dtype=np.float32) * np.float32(2e-7)
            # A lone unequal value near a tile seam must invalidate the affected proofs.
            values[39 + component, 40, 35] *= 3
            profiles.append(values)
        pools, codes = pack_material_blocks(profiles, max_bytes=np.inf, host=True)
        args[37:40], args[40:43] = pools, codes
        for kind, band in (("generated", 0), ("mstep", 32)):
            owners.append(
                module.NativeSession(
                    tuple(args), dict(prepared.packet.attributes, rolling_band=band),
                    preserve_inputs=False, kind=kind, initialize_scratch=False,
                )
            )
        ordinary, rolling = owners
        ordinary.run_schedule([35])
        rolling.run_moving([32, 3], 16, 4)
        expected, actual = ordinary.download(), rolling.download()
        for field in (*range(18), 30, 31, 32):
            np.testing.assert_array_equal(actual[field].view(np.uint8), expected[field].view(np.uint8))
        assert ordinary.clocks() == rolling.clocks()
    finally:
        for owner in owners:
            owner.close()
        prepared._close()
        sim.clear_compiled_cache()
