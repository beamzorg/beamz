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


@pytest.mark.parametrize("padding", ["32", "64", "32x8", "64x8"])
@pytest.mark.parametrize("fusion", ["0", "1"])
@pytest.mark.parametrize("material", ["binary", "smooth"])
@pytest.mark.parametrize("tile", ["32x8x8", "64x4x8", "32x4x8"])
def test_streamed_padded_fields_preserve_logical_state(
    padding, fusion, material, tile, monkeypatch
):
    """Odd extents, 12-cell CPML, source/DFT and odd/even continuation parity."""
    from argparse import Namespace

    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", padding)
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "1")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", fusion)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TILE", tile)
    monkeypatch.setenv(
        "BEAMZ_CUDA_CPML_SHELL_TILE",
        {"32x8x8": "64x4", "64x4x8": "32x8", "32x4x8": "32x4"}[tile],
    )
    simulation = build_simulation(
        Namespace(
            shape=(37, 49, 65),
            steps=33,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="field",
        )
    )
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=33)
    rng = np.random.default_rng(20260918)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=33, backend="jax"
    ).state
    first = simulation.advance(
        state=_copy_state(state), num_steps=17, backend="cuda_streamed"
    ).state
    actual = simulation.advance(
        state=first, num_steps=16, backend="cuda_streamed"
    ).state
    _assert_state_close(reference, actual)
    assert tuple(actual.ex.shape) == tuple(state.ex.shape)


@pytest.mark.parametrize("steps", [2, 3, 4, 5, 33])
@pytest.mark.parametrize(
    "padding,material,tile",
    [
        ("none", "binary", "32x8x8"),
        ("32", "binary", "64x4x8"),
        ("64x8", "smooth", "32x4x8"),
    ],
)
def test_streamed_temporal_pair_full_state_and_continuation(
    steps, padding, material, tile, monkeypatch
):
    from argparse import Namespace

    from beamz.simulation.cuda import runtime
    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", padding)
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "2")
    monkeypatch.setenv(
        "BEAMZ_CUDA_PAIR_TILE",
        {
            "32x8x8": "16x8x16",
            "64x4x8": "32x4x16",
            "32x4x8": "32x8x16",
        }[tile],
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TILE", tile)
    monkeypatch.setenv(
        "BEAMZ_CUDA_CPML_SHELL_TILE",
        {"32x8x8": "64x4", "64x4x8": "32x8", "32x4x8": "32x4"}[tile],
    )
    plans = []
    choose = runtime._native_schedule_plan

    def record(*args, **kwargs):
        plan = choose(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(runtime, "_native_schedule_plan", record)
    simulation = build_simulation(
        Namespace(
            shape=(61, 73, 97),
            steps=33,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="field",
        )
    )
    if steps == 5:
        # Coincident mode slabs straddle the 12-cell CPML/core coupling band.
        source = simulation.sources[0]
        edge = source.updated_copy(center=(13.2 * 80e-9, *source.center[1:]))
        simulation = simulation.updated_copy(sources=(edge, edge))
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=33)
    rng = np.random.default_rng(20260919)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state), num_steps=steps, backend="cuda_streamed"
    ).state
    _assert_state_close(reference, actual)
    assert any(plan.temporal_steps == 2 for plan in plans)
    if steps < 33:
        reference = simulation.advance(
            state=reference, num_steps=33 - steps, backend="jax"
        ).state
        actual = simulation.advance(
            state=actual, num_steps=33 - steps, backend="cuda_streamed"
        ).state
        _assert_state_close(reference, actual)


@pytest.mark.parametrize("io", ["none", "source", "mode_monitor"])
@pytest.mark.parametrize("pair_tile", ["16x8x16", "single"])
def test_streamed_temporal_pair_io_variants(io, pair_tile, monkeypatch):
    """Exercise both FFI arities and six-component mode-monitor gathering."""
    from argparse import Namespace

    from beamz.simulation.cuda import runtime
    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", "32x8")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "2")
    monkeypatch.setenv("BEAMZ_CUDA_PAIR_TILE", pair_tile)
    plans = []
    choose = runtime._native_schedule_plan

    def record(*args, **kwargs):
        plan = choose(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(runtime, "_native_schedule_plan", record)
    simulation = build_simulation(
        Namespace(
            shape=(48, 56, 80),
            steps=6,
            pml=12,
            monitors=2,
            frequencies=3,
            material="binary",
            source="mode",
            monitor_type="mode",
        )
    )
    if io != "mode_monitor":
        simulation = simulation.updated_copy(monitors=())
    if io == "none":
        simulation = simulation.updated_copy(sources=())
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=6)
    rng = np.random.default_rng(20260920)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=6, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state), num_steps=6, backend="cuda_streamed"
    ).state
    _assert_state_close(reference, actual)
    assert any(plan.temporal_steps == 2 for plan in plans)


@pytest.mark.parametrize("shell_tile", ["64x4", "32x8", "32x4"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("fusion,temporal_steps", [("0", "1"), ("1", "1"), ("1", "2")])
@pytest.mark.parametrize("material", ["binary", "smooth"])
def test_cpml_shell_tiles_rotated_sources(
    shell_tile, axis, fusion, temporal_steps, material, monkeypatch
):
    """Odd tails, all source normals, shell intersections and continuation."""
    from argparse import Namespace

    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", "none")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", temporal_steps)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", fusion)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TILE", "auto")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SHELL_TILE", shell_tile)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp32")
    simulation = build_simulation(
        Namespace(
            shape=(37, 49, 65),
            steps=65,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="mode" if material == "binary" else "field",
        )
    )
    order = {"x": (0, 1, 2), "y": (2, 0, 1), "z": (1, 2, 0)}[axis]

    def rotate(values):
        return tuple(values[i] for i in order)

    eps = np.transpose(
        np.asarray(simulation.material_grid.permittivity),
        tuple(2 - order[2 - i] for i in range(3)),
    ).copy()
    grid = MaterialGrid(
        permittivity=eps,
        conductivity=np.float32(0),
        permeability=np.float32(1),
        resolution=80e-9,
        shape=eps.shape,
    )
    source = simulation.sources[0]
    source = source.updated_copy(
        center=rotate((13.2 * 80e-9, *source.center[1:])), size=rotate(source.size)
    )
    monitors = []
    for monitor in simulation.monitors:
        changes = dict(center=rotate(monitor.center), size=rotate(monitor.size))
        if material == "smooth":
            changes["fields"] = tuple(
                field[0] + "xyz"[order.index("xyz".index(field[1].lower()))]
                for field in monitor.fields
            )
        monitors.append(monitor.updated_copy(**changes))
    simulation = simulation.updated_copy(
        material_grid=grid,
        sources=(source, source),
        monitors=tuple(monitors),
    )
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=65)
    rng = np.random.default_rng(20260921)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=65, backend="jax"
    ).state
    first = simulation.advance(
        state=_copy_state(state), num_steps=32, backend="cuda_streamed"
    ).state
    actual = simulation.advance(
        state=first, num_steps=33, backend="cuda_streamed"
    ).state
    # With 65 steps and coincident sources, the original tile also differs from
    # JAX by about two ppm of a leaf's peak near cancellation zeros. Retain a
    # three-ppm absolute scale here, and separately require exact tile parity.
    _assert_state_close(reference, actual, dynamic_atol_scale=3e-6)
    for expected, observed in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(actual),
        strict=True,
    ):
        expected, observed = np.asarray(expected), np.asarray(observed)
        if np.issubdtype(expected.dtype, np.inexact):
            peak = np.max(np.abs(expected), initial=0)
            assert np.max(np.abs(expected - observed), initial=0) <= 3e-6 * peak + 3e-6
    if shell_tile != "64x4":
        monkeypatch.setenv("BEAMZ_CUDA_CPML_SHELL_TILE", "64x4")
        baseline = simulation.advance(
            state=_copy_state(state), num_steps=32, backend="cuda_streamed"
        ).state
        baseline = simulation.advance(
            state=baseline, num_steps=33, backend="cuda_streamed"
        ).state
        for expected, observed in zip(
            jax.tree_util.tree_leaves(baseline),
            jax.tree_util.tree_leaves(actual),
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)


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
        if expected.dtype.name == "bfloat16":
            # ml_dtypes BF16 is not classified as np.inexact by NumPy.
            expected = expected.astype(np.float32)
            observed = observed.astype(np.float32)
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


@pytest.mark.parametrize("material", ["binary", "smooth", "scalar", "lossy"])
@pytest.mark.parametrize("fusion", ["0", "1"])
def test_streamed_cuda_realistic_sources_and_cpml_match_jax(
    material, fusion, monkeypatch
):
    """Exercise dense/scalar/codebook CPML, batched H/E sources and large DFTs."""
    from argparse import Namespace

    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", fusion)
    simulation = build_simulation(
        Namespace(
            shape=(32, 192, 192)
            if material == "binary" and fusion == "0"
            else (24, 32, 48),
            steps=33,
            pml=4,
            monitors=2,
            frequencies=5,
            material=material,
            source="gaussian" if material in {"scalar", "lossy"} else "mode",
        )
    )
    from types import SimpleNamespace

    from beamz.simulation.cuda import runtime as cuda_runtime

    cuda_program = simulation.compile(backend="cuda_streamed")
    context = SimpleNamespace(
        config=cuda_program.config, boundary=cuda_program.boundary
    )
    assert cuda_runtime._temporal_cpml_fields_supported(
        context, cuda_program.coefficients, simulation.num_steps
    ) == (material != "lossy")
    reference_program = simulation.compile(backend="jax")
    state = initial_program_state(
        reference_program,
        t=0,
        current_step=0,
        monitor_steps=simulation.num_steps,
    )
    rng = np.random.default_rng(20260917)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-6
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = simulation.advance(
        state=_copy_state(state),
        num_steps=33,
        backend="jax",
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=33,
        backend="cuda_streamed",
    ).state
    _assert_state_close(reference, actual)


@pytest.mark.parametrize("fusion", ["0", "1"])
def test_streamed_cuda_overlapping_h_sources_cross_cpml_core_boundary(
    fusion, monkeypatch
):
    """Halo recomputation must include each overlapping source exactly once."""
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", fusion)
    simulation, state = _simulation_and_seed(cpml=True, source=False)
    waveform = np.sin(np.arange(simulation.num_steps) * 0.3).astype(np.float32)
    sources = []
    for component in ("Hx", "Hy", "Hz"):
        for shift in (0, 1, 1):
            index = (slice(1, 8), slice(2, 10), slice(4 + shift, 16 + shift))
            sources.append(
                bz.CustomSource(
                    component=component,
                    timing="h",
                    index=index,
                    coeff=np.full((7, 8, 12), 1e-5, dtype=np.float32),
                    waveform=waveform,
                    target_shape=getattr(state, component.lower()).shape,
                )
            )
    simulation = simulation.updated_copy(sources=sources)
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
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


@pytest.mark.parametrize("interval,normalization", [(1, 0), (2, 1), (3, 1)])
@pytest.mark.parametrize("pair_tile", ["16x8x16", "single"])
def test_temporal_pair_dft_windows_and_intervals(
    interval, normalization, pair_tile, monkeypatch
):
    """Two observations, masked components, inactive windows and an odd tail."""
    from argparse import Namespace

    from beamz.simulation import _cuda_abi as abi
    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "2")
    monkeypatch.setenv("BEAMZ_CUDA_PAIR_TILE", pair_tile)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", "1")
    simulation = build_simulation(
        Namespace(
            shape=(61, 73, 97),
            steps=65,
            pml=12,
            monitors=2,
            frequencies=3,
            material="binary",
            source="mode",
            monitor_type="field",
        )
    )
    first, second = simulation.monitors
    second = second.updated_copy(
        size=(first.size[1] * 0.5, 0, first.size[2] * 0.75),
        freqs=np.asarray(second.freqs)[:2],
        interval=interval + 1,
    )
    inactive = first.updated_copy(
        freqs=np.asarray(first.freqs)[:1],
        fields=("Ex",),
        name="inactive",
    )
    simulation = simulation.updated_copy(
        monitors=(
            first.updated_copy(interval=interval),
            second,
            inactive,
        )
    )
    program = simulation.compile(num_steps=65, backend="cuda_streamed")
    monitors = tuple(
        replace(
            m,
            dft_window_code=1,
            dft_normalization_code=normalization,
            dft_t_start=float(
                simulation.time[4]
                if i == 0
                else simulation.time[1]
                if i == 1
                else simulation.time[-1] * 2
            ),
            dft_t_end=float(
                simulation.time[-5]
                if i == 0
                else simulation.time[-2]
                if i == 1
                else simulation.time[-1] * 3
            ),
        )
        for i, m in enumerate(program.monitors)
    )
    program = replace(program, monitors=monitors)
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=65)
    baseline = replace(
        program,
        config=replace(
            program.config,
            cuda_flags=program.config.cuda_flags & ~abi.CUDA_TEMPORAL_PAIR,
        ),
    )
    expected = build_scan(baseline, donate_state=False)(state, baseline.coefficients)
    actual = build_scan(program, donate_state=False)(state, program.coefficients)
    for ref, result in zip(
        jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(actual),
        strict=True,
    ):
        np.testing.assert_array_equal(result, ref)


@pytest.mark.parametrize("steps", [2, 3, 4, 5, 33])
@pytest.mark.parametrize(
    "shape,padding,material",
    [
        ((37, 41, 61), "none", "binary"),
        ((61, 37, 49), "32", "smooth"),
        ((49, 61, 37), "64x8", "binary"),
    ],
)
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize(
    "cpml_tile", ["16x8x16", "16x8x4", "oriented", "spatial", "spatial_single"]
)
def test_temporally_blocked_cpml_seeded_state(
    steps, shape, padding, material, cpml_tile, precision, monkeypatch
):
    """Frozen-psi halos, staggered edges, bank parity and continued runs."""
    from argparse import Namespace

    from scripts.benchmark_cuda_realistic import build_simulation

    spatial = cpml_tile.startswith("spatial")
    depth = "1" if cpml_tile == "spatial_single" else "2"
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", precision)
    tolerance = 2e-2 if precision == "bf16" else 3e-6
    monkeypatch.setenv(
        "BEAMZ_CUDA_CPML_PAIR_TILE", "oriented" if spatial else cpml_tile
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SPATIAL", "1" if spatial else "0")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", depth)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TEMPORAL", "0" if spatial else "1")
    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", padding)
    simulation = build_simulation(
        Namespace(
            shape=shape,
            steps=35,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="field",
        )
    )
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=35)
    rng = np.random.default_rng(20260918)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        **{
            name: tuple(
                rng.normal(size=p.shape).astype(np.float32) * 1e2
                for p in getattr(state, name)
            )
            for name in ("cpml_psi_h_terms", "cpml_psi_e_terms")
        },
    )
    expected = simulation.advance(
        state=_copy_state(state), num_steps=steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state), num_steps=steps, backend="cuda_streamed"
    ).state
    assert all(
        str(p.dtype) == ("bfloat16" if precision == "bf16" else "float32")
        for p in (*actual.cpml_psi_h_terms, *actual.cpml_psi_e_terms)
    )
    _assert_state_close(expected, actual, dynamic_atol_scale=tolerance)
    # CUDA schedules share explicit derivative rounding and multiply/FMA order.
    # Unlike the independent JAX comparison, schedule parity must be exact,
    # including BF16 recurrence rounding and chronological monitor accumulation.
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TEMPORAL", "0")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SPATIAL", "0")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "1")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", "0")
    ordinary = simulation.advance(
        state=_copy_state(state), num_steps=steps, backend="cuda_streamed"
    ).state
    for ref, result in zip(
        jax.tree_util.tree_leaves(ordinary),
        jax.tree_util.tree_leaves(actual),
        strict=True,
    ):
        np.testing.assert_array_equal(result, ref)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TEMPORAL", "0" if spatial else "1")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SPATIAL", "1" if spatial else "0")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", depth)
    expected = simulation.advance(
        state=expected, num_steps=35 - steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=actual, num_steps=35 - steps, backend="cuda_streamed"
    ).state
    _assert_state_close(expected, actual, dynamic_atol_scale=tolerance)


@pytest.mark.parametrize("shell_tile", ["64x4", "32x8", "32x4"])
@pytest.mark.parametrize(
    "width,material", [(63, "binary"), (64, "smooth"), (65, "binary")]
)
def test_narrow_cpml_queue_matches_fused_state(
    width, material, shell_tile, monkeypatch
):
    """Short contiguous interiors and partial thread rows retain exact ownership."""
    from argparse import Namespace

    from scripts.benchmark_cuda_realistic import build_simulation

    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp32")
    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", "none")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "1")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TEMPORAL", "0")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SPATIAL", "0")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SHELL_TILE", shell_tile)
    simulation = build_simulation(
        Namespace(
            shape=(37, 49, width),
            steps=35,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="field",
        )
    )
    program = simulation.compile(backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=35)
    rng = np.random.default_rng(20260918)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        **{
            name: tuple(
                rng.normal(size=p.shape).astype(np.float32) * 1e2
                for p in getattr(state, name)
            )
            for name in ("cpml_psi_h_terms", "cpml_psi_e_terms")
        },
    )
    expected = simulation.advance(
        state=_copy_state(state), num_steps=33, backend="jax"
    ).state
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", "0")
    actual = simulation.advance(
        state=_copy_state(state), num_steps=33, backend="cuda_streamed"
    ).state
    _assert_state_close(expected, actual, dynamic_atol_scale=3e-6)
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", "1")
    fused = simulation.advance(
        state=_copy_state(state), num_steps=33, backend="cuda_streamed"
    ).state
    for ref, value in zip(jax.tree.leaves(fused), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(value, ref)
    fused = simulation.advance(state=fused, num_steps=2, backend="cuda_streamed").state
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", "0")
    actual = simulation.advance(
        state=actual, num_steps=2, backend="cuda_streamed"
    ).state
    for ref, value in zip(jax.tree.leaves(fused), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(value, ref)


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
