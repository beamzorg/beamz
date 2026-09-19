"""Hardware parity gates for the optional CUDA FFI wheel."""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

import beamz as bz
import beamz.simulation.execute as execute_runtime
from beamz.design import MaterialGrid
from beamz.design.raster import Grid, Material, Scene, rasterize
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_scan, initial_program_state
from tests.performance.h100_workloads import H100Workload

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available,
    reason=STATUS.reason or "CUDA backend unavailable",
)


def _simulation_and_seed(
    *,
    cpml: bool,
    source: bool = True,
    monitor: bool = True,
    heterogeneous: bool = True,
    timesteps: int = 32,
):
    workload = H100Workload(
        name="cuda_hardware_parity",
        shape_zyx=(18, 20, 35),
        # Cross the Gaussian source peak at step 24 and accumulate multiple DFT
        # phases rather than validating only the near-zero leading envelope.
        timesteps=timesteps,
        resolution=80e-9,
        pml_cells=3,
        heterogeneous=heterogeneous,
        cpml=cpml,
        source=source,
        monitor=monitor,
    )
    simulation = workload.build()
    program = simulation.compile(backend="jax")
    state = initial_program_state(
        program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=workload.timesteps,
    )
    rng = np.random.default_rng(20260803)
    fields = {
        name: rng.normal(size=np.asarray(getattr(state, name)).shape).astype(np.float32)
        * 1e-5
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    return simulation, state._replace(**fields)


def _copy_state(state):
    return jax.tree_util.tree_map(lambda value: np.array(value, copy=True), state)


def _nonuniform_simulation(*, metric_kind: str, cpml: bool):
    if metric_kind == "axis_uniform":
        grid = Grid.from_spacing((18, 14, 12), (80e-9, 100e-9, 120e-9))
    else:

        def graded_edges(count, base, growth):
            widths = base * np.linspace(1.0, growth, count, dtype=np.float64)
            return np.concatenate(([0.0], np.cumsum(widths)))

        grid = Grid(
            graded_edges(18, 72e-9, 1.22),
            graded_edges(14, 88e-9, 1.18),
            graded_edges(12, 104e-9, 1.15),
        )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(Scene((Material(epsilon_r=2.25),)), grid), dimensions=3
    )
    boundaries = (
        [bz.PML(thickness=240e-9, formulation="cpml")]
        if cpml
        else [bz.PEC(edges="all")]
    )
    simulation = bz.Simulation(
        material_grid=material_grid,
        boundaries=boundaries,
        time=np.arange(16, dtype=np.float64) * 4e-17,
    )
    state = simulation.initial_state()
    rng = np.random.default_rng(20260812)
    return simulation, state._replace(
        **{
            name: rng.normal(size=np.asarray(getattr(state, name)).shape).astype(
                np.float32
            )
            * 1e-5
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )


def _assert_state_close(reference, actual, *, dynamic_atol_scale=1e-6):
    reference_leaves = jax.tree_util.tree_leaves(reference)
    actual_leaves = jax.tree_util.tree_leaves(actual)
    assert len(reference_leaves) == len(actual_leaves)
    for expected, observed in zip(reference_leaves, actual_leaves, strict=True):
        expected = np.asarray(expected)
        observed = np.asarray(observed)
        if np.issubdtype(expected.dtype, np.inexact):
            # CPML memories span roughly five orders of magnitude. Near-zero
            # elements can differ by a few float32 ULPs even when the complete
            # recurrence agrees to sub-ppm scale, so keep the strict relative
            # check and derive an absolute floor from the leaf's dynamic range.
            scale = float(np.max(np.abs(expected), initial=0.0))
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=3e-5,
                atol=max(3e-6, dynamic_atol_scale * scale),
            )
        else:
            np.testing.assert_array_equal(observed, expected)


def _feature_simulation(profile: str):
    shape = (16, 18, 24)
    resolution = 80e-9
    timesteps = 24
    dt = 0.9 * resolution / (bz.LIGHT_SPEED * np.sqrt(3.0))
    permittivity = np.ones(shape, dtype=np.float32)
    permittivity[5:11, 6:12, :] = np.float32(2.25)
    conductivity = np.float32(0.0)
    boundaries = [bz.PEC(edges="all")]
    if profile == "conductive":
        conductivity_grid = np.zeros(shape, dtype=np.float32)
        conductivity_grid[4:12, 5:13, :] = np.float32(2.5e3)
        conductivity = conductivity_grid
    elif profile == "sponge":
        boundaries = [
            bz.PML(edges="all", thickness=3 * resolution, formulation="sponge")
        ]
    elif profile == "mixed_faces":
        boundaries = [
            bz.PML(
                edges=("left", "right"),
                thickness=3 * resolution,
                formulation="sponge",
            ),
            bz.PEC(edges=("front", "back", "bottom", "top")),
        ]
    elif profile in {"asymmetric_cpml", "cpml_multiple_monitors"}:
        boundaries = [
            bz.PML(
                edges=("back", "top", "left")
                if profile == "asymmetric_cpml"
                else "all",
                thickness=3 * resolution,
                formulation="cpml",
            ),
        ]
        if profile == "asymmetric_cpml":
            boundaries.append(bz.PEC(edges=("front", "bottom", "right")))
    material_grid = MaterialGrid(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=np.float32(1.0),
        resolution=resolution,
        shape=shape,
    )
    time = np.arange(timesteps, dtype=np.float64) * dt
    size_xyz = (shape[2] * resolution, shape[1] * resolution, shape[0] * resolution)
    waveform = np.sin(np.linspace(0.0, 3.0 * np.pi, timesteps)).astype(np.float32)
    sources = []
    if profile in {
        "multiple_sources",
        "overlapping_sources",
        "multiple_monitors",
        "cpml_multiple_monitors",
        "scheduled_windowed_monitor",
    }:
        sources = [
            bz.GaussianSource(
                position=(fraction * size_xyz[0], 0.5 * size_xyz[1], 0.5 * size_xyz[2]),
                width=2.5 * resolution,
                signal=waveform if index == 0 else 0.6 * waveform,
            )
            for index, fraction in enumerate(
                (0.38, 0.38) if profile == "overlapping_sources" else (0.3, 0.45)
            )
        ]
    elif profile == "h_source":
        probe = bz.Simulation(material_grid=material_grid, time=time)
        target_shape = tuple(probe.initial_state().hz.shape)
        source_index = (
            slice(target_shape[0] // 2, target_shape[0] // 2 + 1),
            slice(3, target_shape[1] - 3),
            slice(4, target_shape[2] - 4),
        )
        source_shape = tuple(key.stop - key.start for key in source_index)
        sources = [
            bz.CustomSource(
                component="Hz",
                timing="h",
                index=source_index,
                coeff=np.full(source_shape, 1e-4, dtype=np.float32),
                waveform=waveform,
                target_shape=target_shape,
            )
        ]
    elif profile == "gaussian_beam":
        carrier = 200e12
        sources = [
            bz.GaussianBeamSource(
                center=(
                    0.5 * size_xyz[0],
                    0.5 * size_xyz[1],
                    0.75 * size_xyz[2],
                ),
                size=(8 * resolution, 8 * resolution, 0.0),
                source_time=bz.GaussianPulse(
                    freq0=carrier,
                    fwidth=carrier,
                    offset=0.5,
                    remove_dc_component=False,
                ),
                direction="-z",
                angle_theta=np.deg2rad(14.5),
                angle_phi=np.pi,
                pol_angle=np.pi / 2.0,
                waist_radius=2.5 * resolution,
                wavelength=bz.LIGHT_SPEED / carrier,
            )
        ]
    monitors = []
    if profile in {
        "multiple_monitors",
        "cpml_multiple_monitors",
        "scheduled_windowed_monitor",
    }:
        positions = (
            (0.65, 0.78)
            if profile in {"multiple_monitors", "cpml_multiple_monitors"}
            else (0.72,)
        )
        monitors = [
            bz.FieldMonitor(
                center=(fraction * size_xyz[0], 0.5 * size_xyz[1], 0.5 * size_xyz[2]),
                size=(0.0, 0.5 * size_xyz[1], 0.5 * size_xyz[2]),
                freqs=np.asarray((190e12, 195e12)),
                fields=("Ey", "Ez", "Hy", "Hz"),
                interval=3 if profile == "scheduled_windowed_monitor" else 1,
                name=f"plane_{index}",
            )
            for index, fraction in enumerate(positions)
        ]
    return bz.Simulation(
        material_grid=material_grid,
        boundaries=boundaries,
        sources=sources,
        monitors=monitors,
        time=time,
    )


def _feature_program_state(simulation, backend: str, profile: str, state):
    program = simulation.compile(num_steps=simulation.num_steps, backend=backend)
    if profile == "scheduled_windowed_monitor":
        monitor = replace(
            program.monitors[0],
            dft_t_start=float(simulation.time[3]),
            dft_t_end=float(simulation.time[-4]),
            dft_window_code=1,
        )
        program = replace(program, monitors=(monitor,))
    return build_scan(program)(state, program.coefficients)


@pytest.mark.parametrize(
    "profile",
    [
        "conductive",
        "sponge",
        "mixed_faces",
        "asymmetric_cpml",
        "multiple_sources",
        "overlapping_sources",
        "h_source",
        "gaussian_beam",
        "multiple_monitors",
        "cpml_multiple_monitors",
        "scheduled_windowed_monitor",
    ],
)
def test_streamed_cuda_matches_jax_for_extended_feature_envelope(profile):
    simulation = _feature_simulation(profile)
    reference_program = simulation.compile(backend="jax")
    state = initial_program_state(
        reference_program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=simulation.num_steps,
    )
    rng = np.random.default_rng(20260813)
    state = state._replace(
        **{
            name: rng.normal(size=np.asarray(getattr(state, name)).shape).astype(
                np.float32
            )
            * 1e-6
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )

    reference = _feature_program_state(simulation, "jax", profile, _copy_state(state))
    actual = _feature_program_state(
        simulation, "cuda_streamed", profile, _copy_state(state)
    )

    # Mixed CPML/PEC intersections accumulate a slightly larger, still sub-2-ppm
    # absolute error in packed recurrence leaves after the field is constrained.
    # This remains tight enough to catch a skipped edge recurrence (>8 ppm here).
    _assert_state_close(
        reference,
        actual,
        dynamic_atol_scale=1.5e-6 if profile == "asymmetric_cpml" else 1e-6,
    )


@pytest.mark.parametrize("cpml", [False, True], ids=["pec", "cpml"])
def test_streamed_cuda_matches_jax_complete_state(cpml):
    simulation, state = _simulation_and_seed(cpml=cpml)
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state

    _assert_state_close(reference, actual)


@pytest.mark.parametrize("axis", ["z", "y", "x"])
@pytest.mark.parametrize("num_devices", [2, 4])
@pytest.mark.parametrize("cpml", [False, True])
def test_sharded_streamed_cuda_matches_jax_and_continuation(axis, num_devices, cpml):
    """Real FFI gate: uneven Yee supports, material interfaces, source and DFT."""
    if len(jax.devices("gpu")) < num_devices:
        pytest.skip(f"requires {num_devices} CUDA devices")
    simulation, state = _simulation_and_seed(cpml=cpml)
    sharding = dict(axis=axis, num_devices=num_devices, backend="gpu")
    reference = simulation.advance(
        state=_copy_state(state), num_steps=32, backend="jax", progress=False
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=32,
        backend="cuda_streamed",
        sharding=sharding,
        progress=False,
    ).state
    first = simulation.advance(
        state=_copy_state(state),
        num_steps=16,
        backend="cuda_streamed",
        sharding=sharding,
        progress=False,
    ).state
    continued = simulation.advance(
        state=first,
        num_steps=16,
        backend="cuda_streamed",
        sharding=sharding,
        progress=False,
    ).state
    _assert_state_close(reference, actual)
    _assert_state_close(actual, continued)


@pytest.mark.parametrize("axis", ["z", "y", "x"])
@pytest.mark.parametrize(
    "profile", ["asymmetric_cpml", "rectilinear", "mode", "tensor"]
)
def test_sharded_streamed_extended_features(axis, profile):
    if len(jax.devices("gpu")) < 2:
        pytest.skip("requires two CUDA devices")
    if profile == "rectilinear":
        simulation, state = _nonuniform_simulation(metric_kind="rectilinear", cpml=True)
    else:
        if profile == "mode":
            from tests.unit.test_cuda_sharded_features import mode_simulation

            simulation = mode_simulation()
        elif profile == "tensor":
            from tests.unit.test_execution_backend import _full_tensor_3d_simulation

            base = _full_tensor_3d_simulation()
            simulation = bz.Simulation(
                material_grid=base._material_grid(),
                time=np.arange(12) * 1e-10,
                boundaries=[bz.PML(thickness=0.05, formulation="cpml")],
            )
        else:
            simulation = _feature_simulation(profile)
        from tests.unit.test_cuda_sharding import seed_state

        state = seed_state(simulation)
    reference = simulation.advance(
        state=_copy_state(state),
        backend="jax",
        progress=False,
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        backend="cuda_streamed",
        progress=False,
        sharding=dict(axis=axis, num_devices=2, backend="gpu"),
    ).state
    _assert_state_close(reference, actual)


def test_bf16_cpml_program_matches_jax_application_state(monkeypatch):
    simulation, seeded = _simulation_and_seed(
        cpml=True,
        source=True,
        monitor=True,
        heterogeneous=False,
        timesteps=24,
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "bf16")
    reference = simulation.advance(
        state=_copy_state(seeded), num_steps=24, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(seeded), num_steps=24, backend="cuda_streamed"
    ).state

    assert all(str(value.dtype) == "bfloat16" for value in actual.cpml_psi_h_terms)
    assert all(str(value.dtype) == "bfloat16" for value in actual.cpml_psi_e_terms)
    # BF16 is confined to CPML recurrence storage. The complete application
    # state—including FP32 fields, source evolution, and DFT accumulators—must
    # remain within a 2% per-leaf dynamic-range envelope of the JAX reference.
    # The worst leaf in this seeded 24-step case is 1.61% from the FP32 result.
    _assert_state_close(reference, actual, dynamic_atol_scale=2e-2)


@pytest.mark.parametrize("cpml", [False, True], ids=["pec_graph", "cpml_graph"])
def test_streamed_cuda_owns_source_free_constraints(cpml):
    simulation, state = _simulation_and_seed(
        cpml=cpml,
        source=False,
        monitor=False,
        heterogeneous=False,
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state

    _assert_state_close(reference, actual)


@pytest.mark.parametrize("num_steps", [4, 5, 6, 7, 8])
def test_streamed_cuda_fused_workspace_and_tails_match_jax(num_steps):
    simulation, state = _simulation_and_seed(
        cpml=False,
        source=False,
        monitor=False,
        heterogeneous=False,
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state), num_steps=num_steps, backend="cuda_streamed"
    ).state

    _assert_state_close(reference, actual)


def test_streamed_cuda_graphs_one_packed_cpml_source():
    simulation, state = _simulation_and_seed(
        cpml=True,
        source=True,
        monitor=False,
        heterogeneous=True,
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state

    _assert_state_close(reference, actual)


def test_streamed_cuda_source_graph_continues_from_nonzero_step():
    simulation, state = _simulation_and_seed(
        cpml=True,
        source=True,
        monitor=False,
        heterogeneous=True,
    )
    prefix = simulation.advance(
        state=_copy_state(state), num_steps=7, backend="jax"
    ).state
    reference = simulation.advance(
        state=_copy_state(prefix), num_steps=25, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(prefix), num_steps=25, backend="cuda_streamed"
    ).state

    _assert_state_close(reference, actual)


@pytest.mark.parametrize(
    ("cpml", "source", "monitor"),
    [(False, False, False), (True, True, True)],
    ids=["temporal_pec", "cpml_source_dft"],
)
def test_streamed_cuda_bounded_graph_replay_preserves_native_result(
    monkeypatch, cpml, source, monitor
):
    # Two full native chunks plus a tail exercise graph reuse as well as absolute
    # source timing, DFT phase/window timing, and clock advancement across boundaries.
    timesteps = 519
    simulation, state = _simulation_and_seed(
        cpml=cpml,
        source=source,
        monitor=monitor,
        heterogeneous=cpml,
        timesteps=timesteps,
    )
    program = simulation.compile(num_steps=timesteps, backend="cuda_streamed")
    monkeypatch.setattr(execute_runtime, "CUDA_GRAPH_MAX_STEPS", timesteps)
    reference = build_scan(program)(_copy_state(state), program.coefficients)
    monkeypatch.setattr(execute_runtime, "CUDA_GRAPH_MAX_STEPS", 256)
    actual = build_scan(program)(_copy_state(state), program.coefficients)

    _assert_state_close(reference, actual)


@pytest.mark.parametrize("metric_kind", ["axis_uniform", "rectilinear"])
@pytest.mark.parametrize("cpml", [False, True], ids=["pec", "cpml"])
def test_streamed_cuda_matches_jax_on_nonuniform_grids(metric_kind, cpml):
    simulation, state = _nonuniform_simulation(metric_kind=metric_kind, cpml=cpml)
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state

    assert simulation.compile(backend="cuda_streamed").config.metric_kind == metric_kind
    _assert_state_close(reference, actual)


@pytest.mark.skipif(
    not STATUS.compute_capabilities
    or any(capability < 90 for capability in STATUS.compute_capabilities),
    reason="Hopper tiled target requires SM90+",
)
@pytest.mark.parametrize("cpml", [False, True], ids=["pec", "cpml"])
def test_hopper_cuda_matches_streamed_complete_state(cpml):
    simulation, state = _simulation_and_seed(cpml=cpml)
    reference = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_hopper",
    ).state

    _assert_state_close(reference, actual)
