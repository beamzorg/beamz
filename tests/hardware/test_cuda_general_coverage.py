"""Public CUDA feature coverage and independent one-cell physics checks."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

import beamz as bz
from beamz._region_array import TILE_BYTES, RegionArray
from beamz.const import EPS_0, MU_0
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.design.raster import Object, RasterOptions, Scene, Sphere, rasterize
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine._contracts import _CPML
from beamz.simulation.cuda.engine._plan import _choose
from beamz.simulation.cuda.engine._prepare import _compile
from beamz.simulation.cuda.engine._runtime import _SESSIONS, _Session

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def _simulation(
    metric: str = "uniform",
    *,
    gaussian: bool = False,
    mixed: bool = False,
    conductivity: float = 0,
    steps: int = 129,
    cells: tuple[int, int, int] = (46, 43, 40),
) -> bz.Simulation:
    edges = []
    for count, spacing in zip(cells, (20e-9, 23e-9, 26e-9), strict=True):
        q = np.arange(count, dtype=np.float64)
        widths = np.full(count, 20e-9 if metric == "uniform" else spacing)
        if metric == "smooth":
            widths *= 1 + 0.2 * np.sin(q * 0.19)
        elif metric == "sharp":
            widths *= np.where(q < count // 2, 1, 1.8)
        edges.append(np.r_[0.0, np.cumsum(widths)])
    grid = RectilinearGrid(*edges)
    center = (float(edges[0][-1]) / 2, float(edges[1][-1]) / 2, float(edges[2][-1]) / 2)
    scene = Scene(
        (
            bz.Material(permittivity=2.25, conductivity=conductivity),
            bz.Material(permittivity=4, conductivity=conductivity),
        ),
        (Object(Sphere(center, float(min(e[-1] for e in edges)) * 0.12), 1, id=1),),
    )
    materials = MaterialGrid.from_raster_result(rasterize(scene, grid), dimensions=3)
    dt = grid.cfl_time_step(0.6)
    frequency = bz.LIGHT_SPEED / 700e-9
    # Offset the Gaussian from the sphere so Hz carries physical scattering signal.
    source = (
        bz.GaussianSource(
            position=tuple(
                v + d for v, d in zip(center, (80e-9, 60e-9, 20e-9), strict=True)
            ),
            width=40e-9,
            signal=np.sin(np.arange(steps) * 0.17).astype(np.float32),
        )
        if gaussian
        else bz.PlaneWaveSource(
            center=(float(edges[0][13]), center[1], center[2]),
            size=(0, float(edges[1][-1]), float(edges[2][-1])),
            direction="+x",
            background_index=1.5,
            source_time=bz.GaussianPulse(frequency, frequency * 0.4, offset=0.1),
        )
    )
    boundaries: list[bz.PML | bz.PEC] = [bz.PML(formulation="cpml")]
    if mixed:
        boundaries.append(bz.PEC(edges=("left",)))
    return bz.Simulation(
        material_grid=materials,
        sources=[source],
        boundaries=boundaries,
        monitors=[
            bz.FieldMonitor(
                center=(float(edges[0][24]), center[1], center[2]),
                size=(0, 100e-9, 100e-9),
                freqs=np.asarray([frequency, frequency * 1.1]),
                interval=3,
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
        time=1.25e-16 + np.arange(steps) * dt,
    )


def _assert_close(expected: Sequence[Any] = (), actual: Sequence[Any] = ()) -> None:
    assert len(expected) == len(actual)
    for reference, observed in zip(expected, actual, strict=True):
        reference, observed = np.asarray(reference), np.asarray(observed)
        assert reference.shape == observed.shape and reference.dtype == observed.dtype
        if np.issubdtype(reference.dtype, np.inexact):
            assert np.isfinite(reference).all() and np.isfinite(observed).all()
            scale = float(np.max(np.abs(reference), initial=0))
            np.testing.assert_allclose(
                observed, reference, rtol=2e-5, atol=5e-6 * scale
            )
        else:
            np.testing.assert_array_equal(observed, reference)


def _packet(
    simulation: bz.Simulation | str = "", *, requested: str = "ordinary"
) -> tuple[Any, list[np.ndarray], dict[str, Any]]:
    module = _load_extension()
    prepared = _Session(simulation, module, requested=requested)
    try:
        values = [
            np.zeros(tuple(v.shape), np.dtype(v.dtype))
            if isinstance(v, module.ZeroBuffer)
            else np.asarray(v).copy()
            for v in prepared.packet.arguments
        ]
        return module, values, dict(prepared.packet.attributes)
    finally:
        prepared._close()


@pytest.mark.parametrize("metric", ["uniform", "sharp"])
def test_all_compiled_temporal_depths_match_ordinary_across_continuation(metric):
    module = _load_extension()
    lengths = (module.moving_macro_slab_count * max(module.compiled_depths) + 3, 67)
    simulation = _simulation(metric, steps=sum(lengths) + 6)
    module, values, attributes = _packet(simulation, requested="temporal")
    values[_CPML.monitor + 4][:, 2] = 8
    values[_CPML.monitor + 14][...] = 5
    values[_CPML.monitor + 15][...] = 5

    def make(kind):
        return module.NativeSession(
            tuple(values),
            attributes,
            preserve_inputs=False,
            kind=kind,
            initialize_scratch=False,
        )

    reference = make("program")
    try:
        reference.run_schedule(list(lengths))
        expected = reference.download()
        assert np.any(expected[30]) or np.any(expected[31])
        for depth in module.compiled_depths:
            actual = make("mstep")
            try:
                for length in lengths:
                    timing = actual.run_moving([length], depth, 2)
                    assert timing["moving_macro_count"] == length // depth
                observed = actual.download()
                for i in (*range(18), 30, 31, 32):
                    np.testing.assert_array_equal(
                        expected[i].view(np.uint8),
                        observed[i].view(np.uint8),
                        err_msg=f"metric={metric}, depth={depth}, array={i}",
                    )
                assert actual.clocks() == reference.clocks()
            finally:
                actual.close()
    finally:
        reference.close()
        simulation.clear_compiled_cache()


def _vacuum_wave(metric: str = "sharp", *, cells: int = 192) -> bz.Simulation:
    index = np.arange(cells)
    grading = (
        np.where(index < 64, 0, 0.5)
        if metric == "sharp"
        else 0.5 / (1 + np.exp(-(index - 64) / 6))
    )
    x = np.r_[0.0, np.cumsum(20e-9 * (1 + grading))]
    grid = RectilinearGrid(x, np.arange(37) * 20e-9, np.arange(41) * 20e-9)
    material = MaterialGrid.from_raster_result(
        rasterize(Scene((bz.Material(permittivity=1),)), grid), dimensions=3
    )
    center = (grid.y_edges[-1] / 2, grid.z_edges[-1] / 2)
    source = bz.PlaneWaveSource(
        center=(x[52], *center),
        size=(0, grid.y_edges[-1], grid.z_edges[-1]),
        direction="+x",
        source_time=bz.GaussianPulse(
            bz.LIGHT_SPEED / 800e-9, bz.LIGHT_SPEED / 1e-6, offset=0.4
        ),
    )
    monitors: list[bz.FieldRecorder | bz.FieldMonitor] = [
        bz.FieldRecorder(
            ("Ey", "Hz"), interval=1, center=(x[i], *center), size=(0, 0, 0), name=name
        )
        for name, i in (("near", 60), ("far", 70), ("reflect", 130))
    ]
    monitors.extend(
        [
            bz.FieldRecorder(
                ("Ey", "Hz"),
                interval=8,
                center=(x[-1] / 2, *center),
                size=(x[-1], 0, 0),
                name="line",
            ),
            bz.FieldRecorder(
                ("Ey",),
                interval=13,
                center=(x[70], *center),
                size=(0, grid.y_edges[-1], grid.z_edges[-1]),
                name="transverse",
            ),
            *[
                bz.FieldMonitor(
                    center=(x[i], *center),
                    size=(0, 0, 0),
                    freqs=np.asarray([bz.LIGHT_SPEED / 800e-9]),
                    fields=("Ey", "Hz"),
                    name=name,
                )
                for name, i in (("phase_near", 60), ("phase_far", 70))
            ],
        ]
    )
    return bz.Simulation(
        material_grid=material,
        sources=[source],
        monitors=monitors,
        boundaries=[bz.PML(edges=("left", "right"), formulation="cpml")],
        time=np.arange(1801) * grid.cfl_time_step(0.8),
        normalize_source=None,
        setup_device="cpu",
    )


@pytest.mark.parametrize(
    "gaussian,mixed,conductivity",
    [
        (False, False, 0),
        (True, False, 0),
        (True, True, 0),
        (True, False, 2e4),
    ],
)
@pytest.mark.parametrize("metric", ["uniform", "axis_uniform", "smooth", "sharp"])
def test_public_sources_boundaries_and_loss_keep_cuda_coverage(
    metric: str,
    gaussian: bool,
    mixed: bool,
    conductivity: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(
        metric, gaussian=gaussian, mixed=mixed, conductivity=conductivity
    )
    try:
        with monkeypatch.context() as context:
            context.setattr(_load_extension(), "coherent_backend_version", 0)
            reference = simulation.run(backend="cuda_streamed", performance=False)
        actual = simulation.run(backend="cuda", performance=False)
        assert reference.cuda_plan["binding"] == "ffi"
        assert actual.cuda_plan["backend"] == "cuda"
        assert actual.cuda_plan["binding"] == "native"
        a = reference.renormalize(None)["field"].dft_fields
        b = actual.renormalize(None)["field"].dft_fields
        assert a.keys() == b.keys()
        assert max(float(np.max(np.abs(v), initial=0)) for v in a.values()) > 1e-12
        _assert_close(tuple(a.values()), tuple(b.values()))
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("metric", ["uniform", "axis_uniform", "smooth", "sharp"])
def test_rectangular_metric_classes_share_native_ordinary_and_temporal(
    metric: str,
) -> None:
    simulation = _simulation(metric)
    try:
        ordinary = simulation.run(backend="cuda_streamed", performance=False)
        temporal = simulation.run(backend="cuda_mstep", performance=False)
        assert (
            ordinary.cuda_plan["binding"] == temporal.cuda_plan["binding"] == "native"
        )
        assert temporal.cuda_plan["schedule"] == "temporal" and temporal.cuda_plan[
            "depth"
        ] in {4, 16}
        a = ordinary.renormalize(None)["field"].dft_fields
        b = temporal.renormalize(None)["field"].dft_fields
        assert max(float(np.max(np.abs(v), initial=0)) for v in a.values()) > 1e-12
        for field in a:
            assert np.isfinite(a[field]).all() and np.isfinite(b[field]).all()
            np.testing.assert_array_equal(
                np.ascontiguousarray(a[field]).view(np.uint8),
                np.ascontiguousarray(b[field]).view(np.uint8),
            )
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("metric", ["uniform", "smooth", "sharp"])
@pytest.mark.parametrize("donate", [False, True])
def test_odd_continuation_preserves_full_physics_and_observation_clocks(
    metric: str,
    donate: bool,
) -> None:
    simulation = _simulation(metric)
    try:
        initial = simulation.initial_state()
        rng = np.random.default_rng(20261116)
        initial = initial._replace(
            **{
                field: rng.normal(size=getattr(initial, field).shape).astype(np.float32)
                * 1e-4
                for field in ("ex", "ey", "ez", "hx", "hy", "hz")
            }
        )
        full = simulation.advance(
            state=jax.tree_util.tree_map(lambda v: np.array(v, copy=True), initial),
            num_steps=129,
            backend="cuda",
            performance=False,
        )
        first = simulation.advance(
            state=jax.tree_util.tree_map(lambda v: np.array(v, copy=True), initial),
            num_steps=49,
            backend="cuda",
            performance=False,
        )
        preserved = np.array(first.state.ex, copy=True)
        second = simulation.advance(
            state=first.state,
            num_steps=80,
            backend="cuda",
            donate_state=donate,
            performance=False,
        )
        assert (
            full.results.cuda_plan["backend"]
            == second.results.cuda_plan["backend"]
            == "cuda"
        )
        assert second.results.cuda_plan["final_step"] == 129
        _assert_close(
            jax.tree_util.tree_leaves(full.state),
            jax.tree_util.tree_leaves(second.state),
        )
        if not donate:
            np.testing.assert_array_equal(first.state.ex, preserved)
        assert int(second.state.current_step) == 129
        expected_time = np.float32(
            np.float32(simulation.time[0]) + np.float32(129) * np.float32(simulation.dt)
        )
        np.testing.assert_allclose(second.state.t, expected_time, rtol=3e-7, atol=0)
        values = second.results.renormalize(None)["field"].dft_fields
        assert max(float(np.max(np.abs(v), initial=0)) for v in values.values()) > 1e-12
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("permeability", ["one", "two", "interface"])
def test_public_magnetic_material_has_the_analytic_one_step_h_update(
    permeability: str,
) -> None:
    shape = (40, 40, 40)
    mu = np.ones(shape, np.float32)
    if permeability == "two":
        mu[:] = 2
    elif permeability == "interface":
        mu[20:] = 2
    dt, dx = 1e-17, 20e-9
    materials = MaterialGrid(np.ones(shape, np.float32), np.float32(0), mu, dx, shape)
    simulation = bz.Simulation(
        material_grid=materials,
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(4) * dt,
    )
    try:
        initial = simulation.initial_state()
        ex = np.broadcast_to(
            np.arange(initial.ex.shape[1], dtype=np.float32)[None, :, None],
            initial.ex.shape,
        ).copy()
        actual = simulation.step(state=initial._replace(ex=ex), backend="cuda")
        for z in (14, 26):
            # curl(E)_z=-1/dx; the H update is -dt*curl(E)/(MU_0*mu_r).
            expected = dt / (MU_0 * float(mu[z, 20, 20]) * dx)
            np.testing.assert_allclose(
                np.asarray(actual.hz)[z, 20, 20], expected, rtol=3e-6, atol=0
            )
        module, values, attributes = _packet(simulation)
        values[3][:] = ex
        native = module.NativeSession(
            tuple(values),
            attributes,
            preserve_inputs=False,
            kind="generated",
            initialize_scratch=False,
        )
        try:
            native.run_schedule([1])
            hz = native.download()[2]
            for z in (14, 26):
                expected = dt / (MU_0 * float(mu[z, 20, 20]) * dx)
                np.testing.assert_allclose(hz[z, 20, 20], expected, rtol=3e-6, atol=0)
        finally:
            native.close()
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("metric", ["uniform", "axis_uniform", "smooth", "sharp"])
def test_public_electric_conductivity_has_the_analytic_one_step_decay(
    scalar: bool,
    metric: str,
) -> None:
    geometry = _simulation(metric).material_grid
    assert geometry is not None and geometry.grid is not None
    grid = geometry.grid
    shape, dt, epsilon, conductivity = grid.shape_zyx, 1e-17, 2.25, 2e4
    materials = MaterialGrid(
        np.float32(epsilon) if scalar else np.full(shape, epsilon, np.float32),
        np.float32(conductivity),
        np.float32(1),
        grid.minimum_spacing,
        shape,
        grid=grid,
    )
    simulation = bz.Simulation(
        material_grid=materials,
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(4) * dt,
    )
    try:
        initial = simulation.initial_state()
        actual = simulation.step(
            state=initial._replace(
                ex=np.ones(initial.ex.shape, np.float32),
                hz=np.ones(initial.hz.shape, np.float32),
            ),
            backend="cuda",
        )
        beta = conductivity * dt / (2 * EPS_0 * epsilon)
        expected = (1 - beta) / (1 + beta)
        np.testing.assert_allclose(
            np.asarray(actual.ex)[20, 20, 20], expected, rtol=3e-6, atol=0
        )
        np.testing.assert_array_equal(np.asarray(actual.hz)[20, 20, 20], 1)
        module, values, attributes = _packet(simulation)
        values[2][:] = 1
        values[3][:] = 1
        native = module.NativeSession(
            tuple(values),
            attributes,
            preserve_inputs=False,
            kind="generated",
            initialize_scratch=False,
        )
        try:
            native.run_schedule([1])
            output = native.download()
            np.testing.assert_allclose(
                output[3][20, 20, 20], expected, rtol=3e-6, atol=0
            )
            np.testing.assert_array_equal(output[2][20, 20, 20], 1)
        finally:
            native.close()
    finally:
        simulation.clear_compiled_cache()


def test_public_pec_boundary_enforces_zero_tangential_electric_field() -> None:
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3,
        resolution=20e-9,
        boundaries=[bz.PML(formulation="cpml"), bz.PEC(edges=("left",))],
        time=np.arange(4) * 1e-17,
    )
    try:
        initial = simulation.initial_state()
        actual = simulation.step(
            state=initial._replace(
                ey=np.ones(initial.ey.shape, np.float32),
                ez=np.ones(initial.ez.shape, np.float32),
            ),
            backend="cuda",
        )
        np.testing.assert_array_equal(np.asarray(actual.ey)[..., 0], 0)
        np.testing.assert_array_equal(np.asarray(actual.ez)[..., 0], 0)
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("metric", ["uniform", "axis_uniform", "smooth", "sharp"])
@pytest.mark.parametrize("phase", ["h", "e"])
@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_native_metric_update_has_the_physical_linear_gradient(
    metric: str,
    phase: str,
    mu: float,
) -> None:
    simulation = _simulation(metric)
    assert simulation.material_grid is not None
    grid = simulation.material_grid.grid
    assert grid is not None
    epsilon, scale = 2.25, 20e-9
    simulation = simulation.updated_copy(
        material_grid=MaterialGrid(
            np.full(grid.shape_zyx, epsilon, np.float32),
            np.float32(0),
            np.float32(mu),
            grid.minimum_spacing,
            grid.shape_zyx,
            grid=grid,
        ),
        sources=(),
        monitors=(),
    )
    module, values, attributes = _packet(simulation)
    # Ex is sampled at y edges; Hx at y centers. Both have d(field)/dy=1/scale.
    index = 3 if phase == "h" else 0
    coordinates = grid.y_edges if phase == "h" else grid.centers("y")
    values[index][:] = (np.asarray(coordinates, np.float32) / np.float32(scale))[
        None, :, None
    ]
    native = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="generated",
        initialize_scratch=False,
    )
    try:
        native.run_schedule([1])
        field = native.download()[2 if phase == "h" else 5]
        expected = (
            simulation.dt / (MU_0 * mu * scale)
            if phase == "h"
            else -simulation.dt / (EPS_0 * epsilon * scale)
        )
        # Points on both sides of the sharp y-spacing transition, outside CPML.
        for y in (16, 24):
            np.testing.assert_allclose(field[20, y, 26], expected, rtol=5e-6, atol=0)
    finally:
        native.close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0])
def test_temporal_magnetic_material_matches_linear_growth_before_boundary_arrival(
    mu: float,
) -> None:
    simulation = _simulation(cells=(104, 103, 100), steps=33)
    assert simulation.material_grid is not None
    grid = simulation.material_grid.grid
    assert grid is not None
    simulation = simulation.updated_copy(
        material_grid=MaterialGrid(
            np.full(grid.shape_zyx, 2.25, np.float32),
            np.float32(0),
            np.float32(mu),
            grid.minimum_spacing,
            grid.shape_zyx,
            grid=grid,
        ),
        sources=(),
        monitors=(),
    )
    module, values, attributes = _packet(simulation, requested="temporal")
    values[3][:] = np.arange(values[3].shape[1], dtype=np.float32)[None, :, None]
    ordinary = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="generated",
        initialize_scratch=False,
    )
    temporal = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="mstep",
        initialize_scratch=False,
    )
    try:
        ordinary.run_schedule([33])
        result = temporal.run_moving([33], 16, 4)
        assert result["moving_macro_count"] > 0
        a, b = ordinary.download(), temporal.download()
        for index in (*range(18), 30, 31, 32):
            assert np.isfinite(a[index]).all() and np.isfinite(b[index]).all()
            np.testing.assert_array_equal(
                a[index].view(np.uint8), b[index].view(np.uint8)
            )
        assert ordinary.clocks() == temporal.clocks()
        # The observation is >33 cells from CPML. A linear Ex(y) keeps curl(H)=0
        # there, so Hz grows by the same -curl(E)*dt/mu at every physical step.
        expected = 33 * simulation.dt / (MU_0 * mu * 20e-9)
        np.testing.assert_allclose(b[2][50, 50, 50], expected, rtol=3e-6, atol=0)
    finally:
        ordinary.close()
        temporal.close()
        simulation.clear_compiled_cache()


def test_float64_scalar_material_preparation_keeps_bounded_fp32_recipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (254, 255, 256)
    material = MaterialGrid(
        np.float64(2.25), np.float64(0), np.float64(1), 20e-9, shape
    )
    simulation = bz.Simulation(
        material_grid=material,
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(4) * 1e-17,
    )
    sizes: list[int] = []
    original = RegionArray.__getitem__

    def read(self: RegionArray, key: Any = ()) -> Any:
        value = original(self, key)
        sizes.append(np.asarray(value).nbytes)
        return value

    def reject_volume(
        self: RegionArray, dtype: Any = "", copy: bool = False
    ) -> np.ndarray:
        raise AssertionError(
            "Native material preparation materialized a complete Yee recipe"
        )

    monkeypatch.setattr(RegionArray, "__getitem__", read)
    monkeypatch.setattr(RegionArray, "__array__", reject_volume)
    try:
        program = _compile(simulation, _load_extension(), steps=4)
        values = program.grid.permittivity
        assert values.shape == shape and values.dtype == np.dtype("float32")
        assert all(stride == 0 for stride in values.strides)
        owner_bytes = int(getattr(values.base, "nbytes", values.nbytes))
        assert owner_bytes <= values.dtype.itemsize
        for axis in "xyz":
            value = getattr(program.grid, "eps_" + axis)
            if not isinstance(value, RegionArray):
                value = np.asarray(value)
                assert value.ndim == 0 or not any(value.strides)
        assert not sizes or max(sizes) <= TILE_BYTES
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("metric", ["uniform", "smooth"])
@pytest.mark.parametrize("boundary", ["pec", "sponge", "partial_cpml"])
def test_public_boundary_envelopes_keep_nonzero_native_dft(
    metric: str,
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(metric, gaussian=True)
    if boundary == "pec":
        boundaries = [bz.PEC()]
    elif boundary == "sponge":
        boundaries = [bz.PML(formulation="sponge")]
    else:
        boundaries = [
            bz.PML(
                formulation="cpml", edges=("right", "top", "bottom", "front", "back")
            ),
            bz.PEC(edges=("left",)),
        ]
    simulation = simulation.updated_copy(boundaries=boundaries)
    try:
        with monkeypatch.context() as context:
            context.setattr(_load_extension(), "coherent_backend_version", 0)
            reference = simulation.run(
                backend="cuda_streamed", schedule="streamed", performance=False
            )
        assert reference.cuda_plan["binding"] == "ffi"
        actual = simulation.run(backend="cuda", schedule="streamed", performance=False)
        assert (
            actual.cuda_plan["backend"] == "cuda"
            and actual.cuda_plan["binding"] == "native"
        )
        assert actual.cuda_plan["schedule"] == "ordinary"
        a, b = (r.renormalize(None)["field"].dft_fields for r in (reference, actual))
        assert a.keys() == b.keys()
        assert max(float(np.max(np.abs(v), initial=0)) for v in a.values()) > 1e-12
        _assert_close(tuple(a.values()), tuple(b.values()))
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("metric", ["uniform", "smooth"])
@pytest.mark.parametrize("spatial_mu", [False, True])
def test_nonzero_cpml_history_and_magnetic_interfaces_match_same_packet_program(
    metric: str,
    spatial_mu: bool,
) -> None:
    simulation = _simulation(metric, steps=33)
    assert (
        simulation.material_grid is not None
        and simulation.material_grid.grid is not None
    )
    grid = simulation.material_grid.grid
    center = (
        float(grid.x_edges[-1]) / 2,
        float(grid.y_edges[-1]) / 2,
        float(grid.z_edges[-1]) / 2,
    )
    materials = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (
                    bz.Material(permittivity=2.25, permeability=1 if spatial_mu else 2),
                    bz.Material(permittivity=2.25, permeability=2),
                ),
                (Object(Sphere(center, min(center) * 0.4), 1, id=1),)
                if spatial_mu
                else (),
            ),
            grid,
        ),
        dimensions=3,
    )
    simulation = simulation.updated_copy(
        material_grid=materials, sources=(), monitors=()
    )
    module, values, attributes = _packet(simulation)
    rng = np.random.default_rng(20261117)
    for index in range(6):
        values[index][:] = rng.standard_normal(
            values[index].shape, dtype=np.float32
        ) * np.float32(1e-4)
    for offset in (abi.CPML_H_PSI_INPUT_OFFSET, abi.CPML_E_PSI_INPUT_OFFSET):
        for index in range(offset, offset + abi.CPML_TERM_COUNT):
            values[index][:] = rng.standard_normal(
                values[index].shape, dtype=np.float32
            ) * np.float32(1e3)
    reference = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="program",
        initialize_scratch=False,
    )
    actual = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="generated" if spatial_mu else "mstep",
        initialize_scratch=False,
    )
    try:
        reference.run_schedule([17, 16])
        if spatial_mu:
            actual.run_schedule([17, 16])
        else:
            result = actual.run_moving([17, 16], 16, 4)
            assert result["moving_macro_count"] > 0
        a, b = reference.download(), actual.download()
        for index in range(18):
            assert np.isfinite(a[index]).all() and np.isfinite(b[index]).all()
            np.testing.assert_array_equal(
                a[index].view(np.uint8), b[index].view(np.uint8)
            )
        assert reference.clocks() == actual.clocks()
        assert any(np.max(np.abs(v), initial=0) > 0 for v in b[6:18])
    finally:
        reference.close()
        actual.close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize(
    "metric,widths,pec",
    [
        ("uniform", (2, 5, 3, 6, 4, 7), ()),
        ("smooth", (0, 4, 0, 0, 0, 0), ("top", "back")),
        ("uniform", (0, 0, 0, 0, 0, 0), ("left", "right")),
        ("uniform", (22, 24, 0, 0, 0, 0), ()),
    ],
)
def test_native_moving_independent_cpml_faces_and_pec(metric, widths, pec):
    module = _load_extension()
    if not getattr(module, "compiled_temporal_boundary_widths", False):
        pytest.skip("Native temporal boundary descriptors required")
    simulation = _simulation(metric, steps=273)
    faces = ("left", "right", "bottom", "top", "front", "back")
    boundaries = [
        bz.PML(edges=(face,), thickness=width * 20e-9, formulation="cpml")
        for face, width in zip(faces, widths, strict=True)
        if width
    ]
    if pec:
        boundaries.append(bz.PEC(edges=pec))
    # Seeded fields exercise the nearly full-width slabs without requiring
    # a homogeneous source sheet inside their extruded material.
    sources = () if widths[:2] == (22, 24) else simulation.sources
    simulation = simulation.updated_copy(boundaries=boundaries, sources=sources)
    module, values, attributes = _packet(simulation, requested="temporal")
    state_fields = abi.CPML_GRAPH_OUTPUT_COUNT if any(widths) else abi.FIELD_COUNT
    rng = np.random.default_rng(34)
    state_indices = list(range(6))
    if any(widths):
        for start in (abi.CPML_H_PSI_INPUT_OFFSET, abi.CPML_E_PSI_INPUT_OFFSET):
            state_indices.extend(range(start, start + abi.CPML_TERM_COUNT))
    for i in state_indices:
        scale = np.float32(1e-4 if i < 6 else 1e2)
        values[i][:] = rng.standard_normal(values[i].shape, dtype=np.float32) * scale

    reference = module.NativeSession(tuple(values), attributes, False, "program", False)
    actual = module.NativeSession(tuple(values), attributes, False, "mstep", False)
    try:
        reference.run_schedule([256, 17])
        measured = actual.run_moving([256, 17], 16, 4)
        assert measured["moving_macro_count"] == 17
        assert measured["ordinary_tail_steps"] == 1
        got, expected = actual.download(), reference.download()
        for i in (*range(state_fields), *range(state_fields + 12, state_fields + 15)):
            np.testing.assert_array_equal(got[i], expected[i])
        assert actual.clocks() == reference.clocks()
        public = simulation.run(backend="cuda", schedule="mstep", performance=False)
        assert public.cuda_plan["binding"] == "native"
        assert public.cuda_plan["schedule"] == "temporal"
        assert public.cuda_plan["temporal_blocks"] == 17
    finally:
        reference.close()
        actual.close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("boundary", ["cpml", "sponge"])
def test_native_temporal_dense_loss_and_magnetic_coefficients(boundary):
    if not getattr(_load_extension(), "compiled_temporal_diagonal_materials", False):
        pytest.skip("Temporal diagonal material build required")
    simulation = _simulation("smooth", conductivity=25.0, steps=65)
    grid = simulation.grid
    materials = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (bz.Material(permittivity=2.7, permeability=1.3, conductivity=25.0),)
            ),
            grid,
        ),
        dimensions=3,
    )
    simulation = simulation.updated_copy(
        material_grid=materials, boundaries=(bz.PML(formulation=boundary),), sources=()
    )
    module, values, attributes = _packet(simulation, requested="temporal")
    electric_offset = abi.CPML_E_INPUT_OFFSET if attributes["cpml_enabled"] else 12
    assert values[electric_offset + 3].dtype == np.float32
    # This compact broadcast shape must be addressed by the physical coordinates.
    for i in range(9, 12):
        shape = values[i - 9].shape
        values[i] = np.linspace(
            float(np.ravel(values[i])[0]),
            0.8 * float(np.ravel(values[i])[0]),
            shape[0],
            dtype=np.float32,
        ).reshape(-1, 1, 1)
    rng = np.random.default_rng(25)
    for i in range(6):
        values[i][:] = np.float32(1e-4) * rng.standard_normal(
            values[i].shape, dtype=np.float32
        )
    reference = module.NativeSession(tuple(values), attributes, False, "program", False)
    actual = module.NativeSession(tuple(values), attributes, False, "mstep", False)
    try:
        reference.run_schedule([65])
        assert actual.run_moving([65], 16, 4)["moving_macro_count"] == 4
        got, expected = actual.download(), reference.download()
        count = abi.CPML_GRAPH_OUTPUT_COUNT if attributes["cpml_enabled"] else 6
        for i in (*range(count), *range(count + 12, count + 15)):
            np.testing.assert_array_equal(got[i], expected[i])
    finally:
        actual.close()
        reference.close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("metric", ["uniform", "smooth"])
def test_temporal_full_tensor_preserves_phase_and_mixed_macro_tails(metric):
    if not getattr(_load_extension(), "compiled_temporal_tensor", False):
        pytest.skip("Native temporal tensor component required")
    simulation = _simulation(metric, gaussian=True, steps=71)
    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene((bz.Material(permittivity=(2.25, 2.5, 2.75, 0.25, 0.1, 0.15)),)),
            simulation.grid,
            options=RasterOptions(smoothing="farjadpour_full"),
        ),
        dimensions=3,
    )
    simulation = simulation.updated_copy(material_grid=material)
    monitor = simulation.monitors[0]
    simulation = simulation.updated_copy(
        monitors=(
            monitor,
            bz.FieldRecorder(
                ("Ex", "Hz"),
                interval=3,
                center=monitor.center,
                size=monitor.size,
                name="plane",
            ),
            bz.FieldRecorder(("Ey",), interval=4, name="volume"),
        )
    )
    try:
        reference = simulation.advance(
            backend="cuda", schedule="streamed", performance=False
        )
        first = simulation.advance(
            num_steps=5, backend="cuda", schedule="mstep", performance=False
        )
        actual = simulation.advance(
            state=first.state, backend="cuda", schedule="mstep", performance=False
        )
        assert first.results.cuda_plan["temporal_blocks"] == 1
        assert actual.results.cuda_plan["depth"] == 4
        assert actual.results.cuda_plan["temporal_blocks"] == 16
        assert actual.results.cuda_plan["ordinary_tail_steps"] == 2
        assert actual.results.cuda_plan["binding"] == "native"
        assert int(actual.state.current_step) == 71
        for names in (("ex", "ey", "ez"), ("hx", "hy", "hz")):
            a = tuple(np.asarray(getattr(actual.state, name)) for name in names)
            b = tuple(np.asarray(getattr(reference.state, name)) for name in names)
            scale = max(np.max(np.abs(value)) for value in b)
            for x, y in zip(a, b, strict=True):
                np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6 * scale)
        _assert_close(
            (actual.state.dft_vec_re, actual.state.dft_vec_im),
            (reference.state.dft_vec_re, reference.state.dft_vec_im),
        )
        for name in ("plane", "volume"):
            baseline = reference.results[name]
            frames = actual.results[name]
            prefix = first.results[name]
            _assert_close(
                (baseline.field_steps, baseline.field_times, *baseline.fields.values()),
                (
                    frames.field_steps,
                    frames.field_times,
                    *(frames.fields[c] for c in baseline.fields),
                ),
            )
            np.testing.assert_array_equal(
                prefix.field_steps, baseline.field_steps[: len(prefix.field_steps)]
            )
    finally:
        simulation.clear_compiled_cache()


def test_public_incompressible_epsilon_shares_palette_and_matches_resident_accounting() -> (
    None
):
    shape = (40, 43, 46)
    epsilon = (
        np.random.default_rng(20261118).uniform(1.5, 4.5, shape).astype(np.float32)
    )
    material = MaterialGrid(epsilon, np.float32(0), np.float32(1), 20e-9, shape)
    simulation = bz.Simulation(
        material_grid=material,
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(4) * 1e-17,
    )
    try:
        result = simulation.run(backend="cuda", schedule="streamed", performance=False)
        assert result.cuda_plan["binding"] == "native"
        session = _SESSIONS[simulation]
        arguments = session.packet.arguments
        assert arguments[37] is arguments[38] is arguments[39]
        palette = arguments[37]
        assert palette.nbytes > epsilon.nbytes
        metadata = session.native.metadata()
        assert metadata["immutable_gpu_alias_bytes"] >= 2 * palette.nbytes
        assert session.plan.payload_bytes == metadata["resident_bytes"]
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize(
    "target,source,axis,component",
    [
        ("hx", "ey", 0, 0),
        ("hy", "ez", 2, 1),
        ("hz", "ex", 1, 2),
    ],
)
@pytest.mark.parametrize("yee_only", [False, True])
def test_raster_diagonal_magnetic_tensor_preserves_component_coefficients(
    target: str,
    source: str,
    axis: int,
    component: int,
    yee_only: bool,
) -> None:
    mu, dx, dt = (1.4, 2.1, 2.8), 20e-9, 1e-17
    grid = RectilinearGrid.from_spacing((40, 40, 40), dx)
    raster = rasterize(Scene((bz.Material(permittivity=2.25, permeability=mu),)), grid)
    tensors = dict(raster.tensors)
    if yee_only:
        tensors["mu"] = tensors["mu"].copy()
        tensors["mu"][:3] = 1
    material = MaterialGrid.from_raster_result(
        SimpleNamespace(
            tensors=tensors,
            yee_tensors=raster.yee_tensors,
            grid_edges=raster.grid_edges,
            smoothing=raster.smoothing,
        ),
        dimensions=3,
    )
    simulation = bz.Simulation(
        material_grid=material,
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(4) * dt,
    )
    try:
        initial = simulation.initial_state()
        shape = getattr(initial, source).shape
        coordinate_shape = tuple(shape[a] if a == axis else 1 for a in range(3))
        seed = np.broadcast_to(
            np.arange(shape[axis], dtype=np.float32).reshape(coordinate_shape), shape
        ).copy()
        actual = simulation.step(
            state=initial._replace(**{source: seed}), backend="cuda"
        )
        expected = dt / (MU_0 * mu[component] * dx)
        np.testing.assert_allclose(
            np.asarray(getattr(actual, target))[20, 20, 20], expected, rtol=3e-6, atol=0
        )
        module, values, attributes = _packet(simulation)
        values[{"ex": 3, "ey": 4, "ez": 5}[source]][:] = seed
        native = module.NativeSession(
            tuple(values),
            attributes,
            preserve_inputs=False,
            kind="generated",
            initialize_scratch=False,
        )
        try:
            native.run_schedule([1])
            np.testing.assert_allclose(
                native.download()[component][20, 20, 20], expected, rtol=3e-6, atol=0
            )
        finally:
            native.close()
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("location", ["cell", "yee"])
def test_cuda_rejects_off_diagonal_magnetic_coupling(location: str) -> None:
    from beamz.simulation.backend import CudaBackendUnavailable

    grid = RectilinearGrid.from_spacing((30, 31, 32), 20e-9)
    raster = rasterize(
        Scene(
            (bz.Material(permittivity=2.25, permeability=(1.4, 2.1, 2.8, 0.1, 0, 0)),)
        ),
        grid,
        options=RasterOptions(smoothing="volume"),
    )
    tensors, supports = dict(raster.tensors), dict(raster.yee_tensors)
    if location == "cell":
        for name in ("mu_hx", "mu_hy", "mu_hz"):
            supports[name] = supports[name].copy()
            supports[name][3:] = 0
    else:
        tensors["mu"] = tensors["mu"].copy()
        tensors["mu"][3:] = 0
    # CUDA must reject coupling even if only one representation carries it.
    result = SimpleNamespace(
        tensors=tensors,
        yee_tensors=supports,
        grid_edges=raster.grid_edges,
        smoothing=raster.smoothing,
    )
    simulation = bz.Simulation(
        material_grid=MaterialGrid.from_raster_result(result, dimensions=3),
        time=np.arange(4) * 1e-17,
    )
    with pytest.raises(CudaBackendUnavailable, match="Off-diagonal magnetic"):
        simulation.compile(backend="cuda")


@pytest.mark.parametrize("location", ["direct_yee", "raster_yee", "raster_cell"])
@pytest.mark.parametrize(
    "invalid",
    [0.0, -1.0, pytest.param(np.nan, id="nan"), pytest.param(np.inf, id="inf")],
)
def test_invalid_component_mu_cannot_hide_behind_positive_cell_mean(
    location: str, invalid: float
) -> None:
    grid = RectilinearGrid.from_spacing((5, 6, 7), 20e-9)
    raster = rasterize(
        Scene((bz.Material(permittivity=2.25, permeability=(1, 2, 3)),)), grid
    )
    tensors, supports = dict(raster.tensors), dict(raster.yee_tensors)
    supports["mu_hz"] = supports["mu_hz"].copy()
    if location != "raster_cell":
        supports["mu_hz"][2, 2, 2, 2] = invalid
    else:
        tensors["mu"] = tensors["mu"].copy()
        tensors["mu"][2, 2, 2, 2] = invalid
    with pytest.raises(ValueError, match="(?i)(mu|permeability).*(positive|finite)"):
        if location == "direct_yee":
            MaterialGrid(
                np.float32(2.25),
                np.float32(0),
                np.float32(1),
                grid.minimum_spacing,
                grid.shape_zyx,
                yee_materials={"mu_hz": supports["mu_hz"][2]},
                grid=grid,
            )
        else:
            MaterialGrid.from_raster_result(
                SimpleNamespace(
                    tensors=tensors,
                    yee_tensors=supports,
                    grid_edges=raster.grid_edges,
                    smoothing=raster.smoothing,
                ),
                dimensions=3,
            )


@pytest.mark.parametrize("feature", ["tensor", "recorder", "mode"])
def test_public_interoperability_features_preserve_results_and_report_actual_binding(
    feature: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = 1 << 30
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(budget))
    simulation = _simulation(gaussian=True, steps=65)
    assert simulation.material_grid is not None
    grid = simulation.material_grid.grid
    assert grid is not None
    if feature == "tensor":
        material = MaterialGrid.from_raster_result(
            rasterize(
                Scene((bz.Material(permittivity=(2.25, 2.5, 2.75, 0.1, 0.05, 0)),)),
                grid,
                options=RasterOptions(smoothing="farjadpour_full"),
            ),
            dimensions=3,
        )
        simulation = simulation.updated_copy(material_grid=material)
    elif feature == "recorder":
        monitor = bz.FieldRecorder(
            ("Ex", "Hz"),
            interval=3,
            center=simulation.monitors[0].center,
            size=simulation.monitors[0].size,
            name="field",
        )
        simulation = simulation.updated_copy(monitors=(monitor,))
    else:
        monitor = bz.ModeMonitor(
            center=simulation.monitors[0].center,
            size=(0, 300e-9, 300e-9),
            mode_spec=bz.ModeSpec(polarization="te"),
            freqs=simulation.monitors[0].freqs,
            name="field",
        )
        simulation = simulation.updated_copy(monitors=(monitor,))
    try:
        reference = simulation.run(backend="jax", performance=False)
        actual = simulation.run(backend="cuda", schedule="streamed", performance=False)
        assert reference.cuda_plan["binding"] == "jax"
        has_effects = hasattr(_load_extension().NativeSession, "configure_effects")
        expected_binding = "native" if has_effects or feature == "mode" else "ffi"
        assert actual.cuda_plan["backend"] == "cuda"
        assert actual.cuda_plan["binding"] == expected_binding
        assert actual.cuda_plan["memory_budget_bytes"] == budget
        if feature == "recorder":
            a, b = (r.monitor("field") for r in (reference, actual))
            np.testing.assert_array_equal(a.field_steps, b.field_steps)
            np.testing.assert_array_equal(a.field_times, b.field_times)
            _assert_close(tuple(a.fields.values()), tuple(b.fields.values()))
            assert max(np.max(np.abs(v)) for v in a.fields.values()) > 1e-12
        else:
            a, b = (
                r.renormalize(None)["field"].dft_fields for r in (reference, actual)
            )
            _assert_close(tuple(a.values()), tuple(b.values()))
            assert max(np.max(np.abs(v)) for v in a.values()) > 1e-12
            if feature == "mode":
                a, b = (
                    np.asarray(r.mode("field").amps.values) for r in (reference, actual)
                )
                _assert_close((a,), (b,))
                assert np.max(np.abs(a)) > 0
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("yee_only", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_subunit_mu_requires_a_material_aware_cfl_step(
    graded: bool, yee_only: bool, safe: bool
) -> None:
    widths = 20e-9 * (
        1 + (0.2 * np.sin(np.arange(18) * 0.3) if graded else np.zeros(18))
    )
    edges = np.r_[0.0, np.cumsum(widths)]
    grid = RectilinearGrid(edges, np.arange(17) * 20e-9, np.arange(16) * 20e-9)
    raster = rasterize(Scene((bz.Material(permittivity=1, permeability=0.5),)), grid)
    tensors = dict(raster.tensors)
    if yee_only:
        tensors["mu"] = np.ones_like(tensors["mu"])
        tensors["epsilon"] = 2 * np.ones_like(tensors["epsilon"])
    material = MaterialGrid.from_raster_result(
        SimpleNamespace(
            tensors=tensors,
            yee_tensors=raster.yee_tensors,
            grid_edges=raster.grid_edges,
            smoothing=raster.smoothing,
        ),
        dimensions=3,
    )
    # Independent Cartesian Yee bound using actual physical minimum widths.
    minima = [float(np.min(np.diff(axis))) for axis in grid.edges]
    bound = np.sqrt(0.5) / (bz.LIGHT_SPEED * np.sqrt(sum(1 / d**2 for d in minima)))
    simulation = bz.Simulation(
        material_grid=material,
        setup_device="cpu",
        time=np.arange(4) * bound * 0.9 if safe else None,
        run_time=None if safe else bound * 4,
    )
    try:
        if safe:
            program = simulation.compile(backend="jax")
            assert program.config.dt <= bound
        else:
            assert simulation.time[1] - simulation.time[0] > bound
            with pytest.raises(ValueError, match="(?i)(CFL|stability)"):
                simulation.compile(backend="jax")
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("feature", ["periodic", "dispersive"])
@pytest.mark.parametrize(
    "backend,budget,policy",
    [
        ("auto", 0, "auto"),
        ("auto", 128 << 20, "auto"),
        ("auto", 0, "invalid_cuda_policy"),
        ("cuda", 0, "auto"),
    ],
)
def test_auto_preserves_jax_only_physics_and_explicit_cuda_rejects_it(
    feature: str,
    backend: str,
    budget: int,
    policy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from beamz.simulation.backend import CudaBackendUnavailable

    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(budget))
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", policy)
    material = (
        bz.PoleResidue.lorentz(
            2.0,
            strength=0.2,
            resonance=6e15,
            damping=8e14,
            frequency_range=(2e14, 8e14),
        )
        if feature == "dispersive"
        else bz.Material(1)
    )
    simulation = bz.Simulation(
        domain=(0.4e-6,) * 3,
        background=material,
        resolution=20e-9,
        # Size-based domains use centered public coordinates.
        sources=[
            bz.GaussianSource(
                position=(0.0,) * 3, width=40e-9, signal=np.sin(np.arange(9) * 0.3)
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(0.02e-6, 0.0, 0.0),
                size=(0, 0, 0),
                freqs=np.asarray([4e14]),
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
        boundaries=[bz.Periodic(axes=("x", "y", "z"))]
        if feature == "periodic"
        else [bz.PEC()],
        time=np.arange(9) * 1e-17,
        setup_device="cpu",
        normalize_source=None,
    )
    try:
        if backend == "cuda":
            with pytest.raises(CudaBackendUnavailable, match="(?i)" + feature):
                simulation.run(backend=backend, performance=False)
        else:
            actual = simulation.run(backend=backend, performance=False)
            reference = simulation.run(backend="jax", performance=False)
            assert actual.cuda_plan["backend"] == actual.cuda_plan["binding"] == "jax"
            a, b = (
                r.renormalize(None)["field"].dft_fields for r in (reference, actual)
            )
            _assert_close(tuple(a.values()), tuple(b.values()))
            assert max(np.max(np.abs(v)) for v in a.values()) > 1e-12
    finally:
        simulation.clear_compiled_cache()


def test_native_memory_policy_changes_replace_owner_without_changing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation("smooth", gaussian=True, steps=33)
    previous, baseline = None, None
    try:
        for policy in ("auto", "capacity", "speed"):
            monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", policy)
            actual = simulation.run(backend="cuda", performance=False)
            session = _SESSIONS[simulation]
            assert actual.cuda_plan["binding"] == "native"
            assert actual.cuda_plan["memory_policy"] == policy
            assert actual.cuda_plan["schedule"] == "ordinary"
            if previous is not None:
                assert session is not previous and previous.native.metadata()["closed"]
            again = simulation.run(backend="cuda", performance=False)
            assert _SESSIONS[simulation] is session
            fields = actual.renormalize(None)["field"].dft_fields
            assert max(np.max(np.abs(v)) for v in fields.values()) > 1e-12
            if baseline is None:
                baseline = fields
            for candidate in (fields, again.renormalize(None)["field"].dft_fields):
                for name in baseline:
                    np.testing.assert_array_equal(
                        np.ascontiguousarray(baseline[name]).view(np.uint8),
                        np.ascontiguousarray(candidate[name]).view(np.uint8),
                    )
            previous = session
    finally:
        simulation.clear_compiled_cache()
    assert previous is not None and previous.native.metadata()["closed"]


@pytest.mark.parametrize("metric", ["sharp", "smooth"])
def test_long_graded_vacuum_wave_has_physical_delay_phase_and_small_cpml_reflection(
    metric: str,
) -> None:
    short, longer = _vacuum_wave(metric), _vacuum_wave(metric, cells=288)
    native = short.updated_copy(
        monitors=tuple(m for m in short.monitors if isinstance(m, bz.FieldMonitor))
    )
    try:
        actual = short.run(backend="cuda", performance=False)
        with jax.default_device(jax.devices("cpu")[0]):
            reference = longer.run(backend="jax", performance=False)
        phase_result = native.run(
            backend="cuda", schedule="streamed", performance=False
        )
        expected = (
            "native"
            if hasattr(_load_extension().NativeSession, "configure_effects")
            else "ffi"
        )
        assert actual.cuda_plan["binding"] == expected
        assert reference.cuda_plan["binding"] == "jax"
        assert phase_result.cuda_plan["binding"] == "native"
        x, xr = short.grid.x_edges, longer.grid.x_edges
        times = actual.monitor("near").field_times
        incoming = (
            times < (x[70] - x[52]) / bz.LIGHT_SPEED + 4 * 800e-9 / bz.LIGHT_SPEED
        )
        centroids = []
        for name in ("near", "far"):
            electric = np.asarray(actual.monitor(name).fields["Ey"]).reshape(-1)
            weight = electric[incoming].astype(float) ** 2
            centroids.append(float(np.sum(times[incoming] * weight) / np.sum(weight)))
        delay = (x[70] - x[60]) / bz.LIGHT_SPEED
        travel_error = abs((centroids[1] - centroids[0]) / delay - 1)
        assert travel_error < 0.02

        plane = np.asarray(actual.monitor("transverse").fields["Ey"])
        plane = plane[np.unravel_index(np.argmax(np.abs(plane)), plane.shape)[0]]
        np.testing.assert_allclose(
            plane, np.full_like(plane, plane.flat[0]), rtol=1e-6, atol=0
        )
        line = actual.monitor("line")
        ey, hz = (
            np.asarray(line.fields[c]).reshape(len(line.field_times), -1).astype(float)
            for c in ("Ey", "Hz")
        )
        energy = np.sum((EPS_0 * ey**2 + MU_0 * hz**2) * np.diff(x)[None, :], axis=1)
        assert np.isfinite(energy).all() and np.min(energy) >= 0 and np.max(energy) > 0
        plateau = (line.field_times > 8e-15) & (line.field_times < 10e-15)
        assert np.max(energy) < 1.05 * np.mean(energy[plateau])
        assert energy[-1] / np.max(energy) < 5e-4

        start = (2 * x[-13] - x[52] - x[130]) / bz.LIGHT_SPEED
        end = (2 * xr[-13] - x[52] - x[130]) / bz.LIGHT_SPEED - 2 * short.dt
        reflected = (times >= start) & (times < end)
        a, b = (
            np.asarray(r.monitor("reflect").fields["Ey"]).reshape(-1)
            for r in (actual, reference)
        )
        peak = float(np.max(np.abs(b)))
        np.testing.assert_allclose(
            a[times < start], b[times < start], rtol=2e-5, atol=5e-6 * peak
        )
        reflection = float(np.max(np.abs(a[reflected] - b[reflected])) / peak)
        assert reflection < 1e-3

        near, far = (
            phase_result.renormalize(None)[name].dft_fields["Ey"].item()
            for name in ("phase_near", "phase_far")
        )
        phase_error = abs(
            float(np.angle(far / near * np.exp(-2j * np.pi * (x[70] - x[60]) / 800e-9)))
        )
        assert phase_error < 0.01
    finally:
        for simulation in (short, longer, native):
            simulation.clear_compiled_cache()


@pytest.mark.parametrize("wide_frequencies", [False, True])
@pytest.mark.parametrize("normalization", [0, 1])
@pytest.mark.parametrize("zero_row", ["none", "np", "nf"])
def test_compact_temporal_indices_preserve_ragged_dft_and_frozen_counts(
    wide_frequencies: bool,
    normalization: int,
    zero_row: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(cells=(104, 103, 100), steps=129)
    center = simulation.monitors[0].center
    frequency = bz.LIGHT_SPEED / 700e-9
    monitors = tuple(
        bz.FieldMonitor(
            center=(center[0] + i * 37e-9, center[1] + 3e-9, center[2] - i * 2e-9),
            size=size,
            freqs=frequency * np.linspace(0.9, 1.1, count),
            fields=fields,
            interval=interval,
            name=f"ragged_{i}",
        )
        for i, (count, size, fields, interval) in enumerate(
            zip(
                (1, 17, 33, 2) if wide_frequencies else (1, 2, 3, 1),
                ((0, 120e-9, 80e-9), (0, 80e-9, 40e-9), (0, 0, 0), (0, 100e-9, 60e-9)),
                (("Ey", "Hz"), ("Ey", "Ez", "Hy", "Hz"), ("Ex",), ("Hz",)),
                (1, 3, 2, 2),
                strict=True,
            )
        )
    )
    simulation = simulation.updated_copy(monitors=monitors)
    # This exercises the rectangular ABI with synthetic empty rows. Packed
    # gathers have their own parity test and require positive row counts.
    with monkeypatch.context() as capture:
        capture.setattr(
            _load_extension(), "compiled_ragged_dft_gather", False, raising=False
        )
        module, values, attributes = _packet(simulation)
    assert module.compiled_compact_temporal_indices
    monitor = _CPML.monitor
    rng = np.random.default_rng(2026111604)
    for i in (
        *range(6),
        *range(31, 37),
        *range(62, 68),
        *range(monitor + 7, monitor + 10),
    ):
        values[i][:] = rng.standard_normal(
            values[i].shape, dtype=np.float32
        ) * np.float32(1e-4)
    counts = values[monitor + abi.MONITOR_COUNTS_INPUT]
    if zero_row != "none":
        counts[1, 1 if zero_row == "np" else 0] = 0
    codes = values[monitor + abi.MONITOR_CODES_INPUT]
    codes[:, 0], codes[:, 1] = 1, normalization
    windows = values[monitor + abi.MONITOR_WINDOWS_INPUT]
    windows[:, 0], windows[:, 1], windows[:, 2] = (
        simulation.time[9],
        simulation.time[110],
        0.9e-6,
    )
    windows[-1, :2] = simulation.time[-1] * np.asarray([2, 3])
    values[monitor + abi.MONITOR_TIME_INPUT][...] = simulation.time[5]
    values[monitor + abi.MONITOR_CURRENT_STEP_INPUT][...] = 5
    resource = module.moving_resource_plan(tuple(values), depth=16, raw_slots=4)
    ordinary = _choose(
        tuple(values), module, module.device_info(), steps=113, requested="ordinary"
    )
    indices = values[monitor + abi.MONITOR_INDICES_INPUT]
    key_count = resource["observation_keys"]
    width = 4
    if indices.shape[3] == 8 and key_count <= 0xFFFFFF:
        width = 2 if key_count <= 0xFFFF else 3
    expected_bytes = (
        int(np.sum(counts[:, 1], dtype=np.int64)) * 6 * indices.shape[3] * width
    )
    expected_blocks = 0
    if wide_frequencies:
        expected_blocks = 6 * sum(
            ((int(nf) + 7) // 8) * ((int(points) + 127) // 128)
            for nf, points in counts[:, :2]
        )
    assert expected_bytes < indices.nbytes
    assert resource["monitor_index_bytes"] == expected_bytes
    assert resource["monitor_index_offsets_bytes"] == (len(counts) + 1) * 4
    reference = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="program",
        initialize_scratch=False,
    )
    actual = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="mstep",
        initialize_scratch=False,
    )
    try:
        assert actual.metadata()["resident_bytes"] == ordinary.payload_bytes
        # Planning is lazy; caller mutation must not change the constructor's packet.
        counts[:, 1] = indices.shape[2] + 1
        reference.run_schedule([49, 64])
        first = actual.run_moving([49], 16, 4)
        second = actual.run_moving([64], 16, 4)
        assert first["moving_macro_count"] > 0 and second["moving_macro_count"] > 0
        a, b = reference.download(), actual.download()
        for i in (*range(18), 30, 31, 32):
            assert np.isfinite(a[i]).all() and np.isfinite(b[i]).all()
            np.testing.assert_array_equal(a[i].view(np.uint8), b[i].view(np.uint8))
        assert reference.clocks() == actual.clocks()
        metadata = actual.metadata()
        assert metadata["moving_dft_frequency_blocks"] == expected_blocks
        assert metadata["moving_monitor_index_bytes"] == expected_bytes
        assert (
            metadata["moving_monitor_index_offsets_bytes"]
            == resource["monitor_index_offsets_bytes"]
        )
        assert (
            metadata["resident_bytes"]
            == ordinary.payload_bytes + resource["extra_gpu_bytes"]
        )
    finally:
        reference.close()
        actual.close()
        simulation.clear_compiled_cache()


@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "shape",
        "negative_np",
        "high_np",
        "virtual_counts",
        "virtual_indices",
        "virtual_starts",
        "starts_dtype",
        "starts_shape",
    ],
)
def test_moving_resource_plan_rejects_unsafe_host_metadata(invalid: str) -> None:
    simulation = _simulation(steps=33)
    module, values, _attributes = _packet(simulation)
    assert module.compiled_compact_temporal_indices
    monitor = _CPML.monitor
    counts = values[monitor + abi.MONITOR_COUNTS_INPUT]
    if invalid == "dtype":
        values[monitor + abi.MONITOR_COUNTS_INPUT] = counts.astype(np.float32)
    elif invalid == "shape":
        values[monitor + abi.MONITOR_COUNTS_INPUT] = counts[:, :4].copy()
    elif invalid in {"negative_np", "high_np"}:
        counts[0, 1] = -1 if invalid == "negative_np" else values[monitor].shape[2] + 1
    else:
        index = (
            monitor + abi.MONITOR_COUNTS_INPUT
            if invalid == "virtual_counts"
            else monitor
        )
        if invalid in {"virtual_starts", "starts_dtype", "starts_shape"}:
            index = next(
                i + 2 for i in range(_CPML.source, monitor, 3) if values[i].shape[0]
            )
        if invalid == "starts_dtype":
            values[index] = values[index].astype(np.float32)
        elif invalid == "starts_shape":
            values[index] = values[index][:, :1].copy()
        else:
            values[index] = module.ZeroBuffer(
                list(values[index].shape), values[index].dtype.name
            )
    try:
        with pytest.raises(
            ValueError, match="(?i)(monitor|counts|point|host storage|source)"
        ):
            module.moving_resource_plan(tuple(values), depth=16, raw_slots=4)
    finally:
        simulation.clear_compiled_cache()


def test_native_one_bank_budget_bypasses_preserved_ffi_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from beamz.simulation.memory import _cuda_working_set

    simulation = _simulation(cells=(136, 136, 136), steps=17)
    monitor = simulation.monitors[0]
    simulation = simulation.updated_copy(
        monitors=(
            monitor.updated_copy(
                center=(simulation.grid.x_edges[15], *monitor.center[1:])
            ),
        )
    )
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", "0")
    try:
        reference = simulation.run(
            backend="cuda", schedule="streamed", performance=False
        )
        plan = _SESSIONS[simulation].plan
        budget = plan.payload_bytes + plan.reserve_bytes + (1 << 20)
        simulation.clear_compiled_cache()
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(budget))
        actual = simulation.run(backend="cuda", schedule="streamed", performance=False)
        session = _SESSIONS[simulation]
        assert actual.cuda_plan["binding"] == "native"
        assert actual.cuda_plan["memory_budget_bytes"] == budget
        assert session.native.metadata()["resident_bytes"] == session.plan.payload_bytes
        assert session.plan.payload_bytes + session.plan.reserve_bytes <= budget
        estimate = _cuda_working_set(
            session.program, donate_state=False, low_memory=False
        )
        assert estimate > budget
        with pytest.raises(MemoryError, match="estimated per-run storage"):
            _cuda_working_set(
                session.program, donate_state=False, low_memory=False, enforce=True
            )
        a, b = (r.renormalize(None)["field"].dft_fields for r in (reference, actual))
        assert max(np.max(np.abs(v)) for v in a.values()) > 1e-12
        for name in a:
            np.testing.assert_array_equal(
                np.ascontiguousarray(a[name]).view(np.uint8),
                np.ascontiguousarray(b[name]).view(np.uint8),
            )
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("backend", ["cuda_streamed", "cuda_mstep"])
def test_volume_diagonal_mu_keeps_public_source_and_dft_physics(backend: str) -> None:
    from beamz.lattice import component_shapes

    grid = RectilinearGrid.from_spacing((104, 103, 100), 20e-9)
    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene((bz.Material(permittivity=2.25, permeability=(1, 2, 3)),)),
            grid,
            options=RasterOptions(smoothing="volume"),
        ),
        dimensions=3,
    )
    assert material.uses_direct_yee_materials
    steps = 129
    source = bz.CustomSource(
        component="Ex",
        timing="e",
        index=(slice(48, 52),) * 3,
        coeff=np.ones((4, 4, 4), np.float32),
        waveform=(
            np.sin(np.arange(steps) * 0.2)
            * np.exp(-(((np.arange(steps) - 30) / 10) ** 2))
        ).astype(np.float32),
        target_shape=component_shapes(material.shape)["Ex"],
    )
    simulation = bz.Simulation(
        material_grid=material,
        sources=[source],
        boundaries=[bz.PML(formulation="cpml")],
        monitors=[
            bz.FieldMonitor(
                center=(grid.x_edges[55], grid.y_edges[50], grid.z_edges[50]),
                size=(0, 120e-9, 100e-9),
                freqs=np.asarray([4e14, 4.4e14]),
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
        time=np.arange(steps) * grid.cfl_time_step(0.6),
        normalize_source=None,
    )
    try:
        reference = simulation.run(backend="jax", performance=False)
        actual = simulation.run(backend=backend, performance=False)
        assert actual.cuda_plan["binding"] == "native"
        assert actual.cuda_plan["schedule"] == (
            "temporal" if backend == "cuda_mstep" else "ordinary"
        )
        if backend == "cuda_mstep":
            assert _SESSIONS[simulation].native.metadata()["moving_ready"]
        a, b = (r.renormalize(None)["field"].dft_fields for r in (reference, actual))
        assert max(np.max(np.abs(v)) for v in a.values()) > 1e-12
        _assert_close(tuple(a.values()), tuple(b.values()))
    finally:
        simulation.clear_compiled_cache()


def test_high_entropy_epsilon_keeps_exact_nonuniform_temporal_physics() -> None:
    from beamz.lattice import component_shapes

    simulation = _simulation(cells=(104, 103, 100), steps=33)
    assert simulation.material_grid is not None
    grid = simulation.material_grid.grid
    assert grid is not None
    epsilon = np.random.default_rng(20261121).uniform(1.5, 4.5, grid.shape_zyx)
    simulation = simulation.updated_copy(
        material_grid=MaterialGrid(
            epsilon.astype(np.float32),
            np.float32(0),
            np.float32(1),
            grid.minimum_spacing,
            grid.shape_zyx,
            grid=grid,
        ),
        sources=(
            bz.CustomSource(
                component="Ex",
                timing="e",
                index=(slice(48, 52),) * 3,
                coeff=np.ones((4, 4, 4), np.float32),
                waveform=np.sin(np.arange(33) * 0.17).astype(np.float32),
                target_shape=component_shapes(grid.shape_zyx)["Ex"],
            ),
        ),
        normalize_source=None,
    )
    module, values, attributes = _packet(simulation, requested="temporal")
    assert all(np.unique(values[i]).size > 4096 for i in (37, 38, 39))
    rng = np.random.default_rng(20261122)
    for name in (
        "hx",
        "hy",
        "hz",
        "ex",
        "ey",
        "ez",
        *(f"{phase}_psi_{i}" for phase in ("h", "e") for i in range(6)),
        "dft_vec_re",
        "dft_vec_im",
    ):
        index = _CPML.inputs[name]
        values[index][...] = (
            rng.normal(size=values[index].shape).astype(np.float32) * 1e-3
        )
    reference = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="program",
        initialize_scratch=False,
    )
    actual = module.NativeSession(
        tuple(values),
        attributes,
        preserve_inputs=False,
        kind="mstep",
        initialize_scratch=False,
    )
    try:
        reference.run_schedule([33])
        report = actual.run_moving([33], 16, 4)
        assert report["moving_macro_count"] > 0
        metadata = actual.metadata()
        assert metadata["moving_uniform_bulk_tiles"] == 0
        expected, observed = reference.download(), actual.download()
        for index in (*range(18), 30, 31, 32):
            assert np.isfinite(expected[index]).all()
            assert np.isfinite(observed[index]).all()
            np.testing.assert_array_equal(
                expected[index].view(np.uint8), observed[index].view(np.uint8)
            )
        assert reference.clocks() == actual.clocks()
        assert max(np.max(np.abs(v)) for v in observed[:6]) > 1e-6
        assert np.max(np.abs(observed[30])) > 1e-6
    finally:
        reference.close()
        actual.close()
        simulation.clear_compiled_cache()
