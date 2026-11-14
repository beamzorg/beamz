"""Full-state gates for whole-column two-step CUDA execution."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.compile.backend import cuda_backend_status
from beamz.simulation.cuda.native import abi
from beamz.simulation.cuda.native.mstep import build_mstep_plan
from beamz.simulation.execute import build_scan, initial_program_state

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available or abi.CUDA_MSTEP_TARGET not in STATUS.targets,
    reason=STATUS.reason or "cuda_mstep extension unavailable",
)


def _simulation(
    material="binary", boundary="all", shape=(37, 65, 97), steps=33, source="gaussian"
):
    from scripts.benchmark_cuda_realistic import build_simulation

    sim = build_simulation(
        SimpleNamespace(
            shape=shape,
            steps=max(2, steps),
            pml=4,
            monitors=2,
            frequencies=3,
            material=material,
            source=source,
            monitor_type="field",
        )
    )
    if boundary == "all":
        return sim
    boundaries = (
        [bz.PEC(edges="all")]
        if boundary == "pec"
        else [bz.PML(edges="all", thickness=4 * 80e-9, formulation="sponge")]
        if boundary == "sponge"
        else [
            bz.PML(edges=["right"], thickness=4 * 80e-9, formulation="cpml"),
            bz.PEC(edges=["left", "top", "bottom", "front", "back"]),
        ]
    )
    return bz.Simulation(
        material_grid=sim._material_grid(progress=False),
        sources=sim.sources,
        monitors=sim.monitors,
        boundaries=boundaries,
        time=sim.time,
    )


def _seed(program):
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=33)
    rng = np.random.default_rng(308)

    def noise(a):
        return rng.normal(size=a.shape).astype(np.float32) * 1e-4

    return state._replace(
        **{
            name: noise(getattr(state, name))
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        cpml_psi_h_terms=tuple(noise(a) for a in state.cpml_psi_h_terms),
        cpml_psi_e_terms=tuple(noise(a) for a in state.cpml_psi_e_terms),
    )


def _run(sim, state, backend, steps, *, source_timing=None):
    program = sim.compile(backend=backend, num_steps=steps)
    if source_timing is not None:
        program = replace(
            program,
            sources=tuple(
                replace(
                    s,
                    timing=source_timing,
                    component="Hx" if source_timing == "h" else "Ex",
                )
                for s in program.sources
            ),
        )
    copy = jax.tree_util.tree_map(lambda a: np.array(a, copy=True), state)
    return jax.block_until_ready(build_scan(program)(copy, program.coefficients))


def _compare(a, b, tolerance=1e-5):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        expected, actual = np.asarray(expected), np.asarray(actual)
        if np.issubdtype(expected.dtype, np.inexact):
            peak = np.max(np.abs(expected), initial=0)
            assert (
                np.max(np.abs(expected - actual), initial=0) <= tolerance * peak + 1e-30
            )
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tile", ["16x32", "8x32", "8x16", "16x16"])
@pytest.mark.parametrize("segment", ["whole", "32", "64", "128", "256"])
@pytest.mark.parametrize(
    "material,boundary", [("binary", "all"), ("smooth", "pec"), ("lossy", "right")]
)
def test_mstep_launch_variants_preserve_complete_state(
    monkeypatch, tile, segment, material, boundary
):
    sim = _simulation(material, boundary, shape=(103, 65, 97), steps=5)
    state = _seed(sim.compile(backend="jax", num_steps=5))
    reference = _run(sim, state, "cuda_streamed", 5)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", tile)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", segment)
    actual = _run(sim, state, "cuda_mstep", 5)
    _compare(reference, actual)


@pytest.mark.parametrize("tile", ["16x32", "8x32", "8x16", "16x16"])
@pytest.mark.parametrize("segment", ["128", "256"])
def test_mstep_long_column_segments_and_tail(monkeypatch, tile, segment):
    sim = _simulation(shape=(301, 49, 81), steps=5)
    state = _seed(sim.compile(backend="jax", num_steps=5))
    reference = _run(sim, state, "cuda_streamed", 5)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", tile)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", segment)
    _compare(reference, _run(sim, state, "cuda_mstep", 5))


def test_cuda_launch_choices_share_material_buffers(monkeypatch):
    sim = _simulation(steps=3)
    base = sim.compile(backend="cuda_streamed", num_steps=3)
    state = _seed(base)
    reference = _run(sim, state, "cuda_streamed", 3)
    for backend, env in (
        (
            "cuda_mstep",
            {"BEAMZ_CUDA_MSTEP_TILE": "8x32", "BEAMZ_CUDA_MSTEP_SEGMENT": "32"},
        ),
        ("cuda_streamed", {"BEAMZ_CUDA_STORAGE_AXES": "120"}),
        ("cuda_streamed", {"BEAMZ_CUDA_TEMPORAL_STEPS": "2"}),
        ("cuda_streamed", {"BEAMZ_CUDA_MEMORY_POLICY": "capacity"}),
    ):
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            program = sim.compile(backend=backend, num_steps=3)
            assert program.coefficients is base.coefficients
            assert program.grid is base.grid
            _compare(reference, _run(sim, state, backend, 3))


@pytest.mark.parametrize("axes", ["120", "201"])
@pytest.mark.parametrize("tile,segment", [("16x32", "whole"), ("8x32", "32")])
@pytest.mark.parametrize(
    "material,boundary", [("binary", "all"), ("smooth", "pec"), ("lossy", "right")]
)
def test_mstep_cyclic_storage_preserves_sources_and_monitors(
    monkeypatch, axes, tile, segment, material, boundary
):
    sim = _simulation(material, boundary, shape=(103, 65, 97), steps=5)
    state = _seed(sim.compile(backend="jax", num_steps=5))
    reference = _run(sim, state, "cuda_streamed", 5)
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", axes)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", tile)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", segment)
    _compare(reference, _run(sim, state, "cuda_mstep", 5))


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 17])
@pytest.mark.parametrize(
    "material,boundary",
    [("binary", "all"), ("binary", "right"), ("smooth", "pec"), ("lossy", "right")],
)
def test_mstep_complete_state_matches_cuda_and_jax(steps, material, boundary):
    sim = _simulation(material, boundary)
    state = _seed(sim.compile(backend="jax", num_steps=steps))
    actual = _run(sim, state, "cuda_mstep", steps)
    _compare(_run(sim, state, "cuda", steps), actual)
    _compare(_run(sim, state, "jax", steps), actual, 3e-5)


@pytest.mark.parametrize("timing", ["pre_e", "h", "e"])
def test_mstep_source_phases_and_continuation(timing):
    sim = _simulation()
    state = _seed(sim.compile(backend="jax", num_steps=7))
    actual = _run(sim, state, "cuda_mstep", 7, source_timing=timing)
    reference = _run(sim, state, "cuda", 7, source_timing=timing)
    _compare(reference, actual)
    # Public continuation also checks absolute source and DFT clocks.
    whole = sim.advance(state=state, num_steps=7, backend="cuda_mstep").state
    first = sim.advance(state=state, num_steps=3, backend="cuda_mstep").state
    split = sim.advance(state=first, num_steps=4, backend="cuda_mstep").state
    _compare(whole, split)


def test_mstep_small_domain_and_bounded_long_run():
    sim = _simulation(shape=(15, 17, 19), steps=259)
    state = _seed(sim.compile(backend="jax", num_steps=259))
    _compare(
        sim.advance(state=state, num_steps=259, backend="cuda").state,
        sim.advance(state=state, num_steps=259, backend="cuda_mstep").state,
        3e-5,
    )


def test_mstep_reports_a_real_core_and_rejects_recorders():
    sim = _simulation()
    program = sim.compile(backend="cuda_mstep", num_steps=2)
    from beamz.simulation.execute import compiled_source_batches

    batches = compiled_source_batches(program.sources)
    groups = tuple(
        batches[(timing, c)][0]
        for timing, components in (
            ("pre_e", ("Ex", "Ey", "Ez")),
            ("h", ("Hx", "Hy", "Hz")),
            ("e", ("Ex", "Ey", "Ez")),
        )
        for c in components
    )
    shapes = tuple(
        program.boundary.logical_component_shapes[c]
        for c in ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
    )
    assert build_mstep_plan(shapes, program.boundary.cpml, groups).core_cells > 0
    recorder = bz.FieldRecorder(center=(1e-6, 1e-6, 1e-6), size=(0, 0, 0))
    unsupported = bz.Simulation(
        material_grid=sim._material_grid(progress=False),
        monitors=[recorder],
        boundaries=sim.boundaries,
        time=sim.time,
    )
    from beamz.simulation.compile.backend import CudaBackendUnavailable

    with pytest.raises(CudaBackendUnavailable, match="recorders"):
        unsupported.advance(num_steps=2, backend="cuda_mstep")


def test_mstep_without_cpml_sources_or_monitors():
    sim = _simulation()
    bare = bz.Simulation(
        material_grid=sim._material_grid(progress=False), time=sim.time
    )
    state = _seed(bare.compile(backend="jax", num_steps=5))
    _compare(_run(bare, state, "cuda", 5), _run(bare, state, "cuda_mstep", 5))


def test_mstep_bf16_recurrence_matches_cuda(monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "bf16")
    sim = _simulation()
    program = sim.compile(backend="cuda_mstep", num_steps=5)
    state = _seed(program)
    state = state._replace(
        cpml_psi_h_terms=tuple(
            a.astype(jax.numpy.bfloat16) for a in state.cpml_psi_h_terms
        ),
        cpml_psi_e_terms=tuple(
            a.astype(jax.numpy.bfloat16) for a in state.cpml_psi_e_terms
        ),
    )
    _compare(_run(sim, state, "cuda", 5), _run(sim, state, "cuda_mstep", 5))


def test_mstep_mode_source_and_windowed_interval_dft():
    sim = _simulation(source="mode")
    state = _seed(sim.compile(backend="jax", num_steps=17))
    results = []
    for backend in ("cuda", "cuda_mstep", "jax"):
        program = sim.compile(backend=backend, num_steps=17)
        program = replace(
            program,
            monitors=tuple(
                replace(
                    m,
                    dft_window_code=1,
                    dft_normalization_code=1,
                    dft_record_interval=3,
                    dft_t_start=float(sim.time[2]),
                    dft_t_end=float(sim.time[15]),
                )
                for m in program.monitors
            ),
        )
        copied = jax.tree_util.tree_map(lambda a: np.array(a, copy=True), state)
        results.append(
            jax.block_until_ready(build_scan(program)(copied, program.coefficients))
        )
    _compare(results[0], results[1])
    _compare(results[2], results[1], 3e-5)


@pytest.mark.parametrize("axes", ["012", "120", "201"])
def test_mstep_state_donation_preserves_results(monkeypatch, axes):
    sim = _simulation()
    state = _seed(sim.compile(backend="jax", num_steps=5))
    state = state._replace(
        dft_vec_re=jax.numpy.arange(state.dft_vec_re.size, dtype=jax.numpy.float32)
        * 1e-5,
        dft_vec_im=-jax.numpy.arange(state.dft_vec_im.size, dtype=jax.numpy.float32)
        * 1e-5,
    )
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", axes)
    expected = sim.advance(state=state, num_steps=5, backend="cuda_mstep").state
    copied = jax.tree_util.tree_map(lambda a: jax.numpy.array(a), state)
    actual = sim.advance(
        state=copied, num_steps=5, backend="cuda_mstep", donate_state=True
    ).state
    _compare(expected, actual)


def test_mstep_dense_h_coefficients_with_variable_permeability():
    # Sponge damping produces dense H banks; electric loss alone does not.
    sim = _simulation(material="lossy", boundary="sponge")
    grid = sim._material_grid(progress=False)
    permeability = np.broadcast_to(
        np.linspace(1, 1.2, grid.shape[2], dtype=np.float32), grid.shape
    )
    sim = sim.updated_copy(material_grid=replace(grid, permeability=permeability))
    program = sim.compile(backend="cuda_mstep", num_steps=5)
    assert program.coefficients.h_source_x.ndim == 3
    state = _seed(program)
    actual = _run(sim, state, "cuda_mstep", 5)
    _compare(_run(sim, state, "cuda", 5), actual)
    _compare(_run(sim, state, "jax", 5), actual, 3e-5)


@pytest.mark.parametrize(
    "axes,tile,segment",
    [("012", "16x32", "whole"), ("012", "8x16", "32"), ("201", "8x16", "32")],
)
def test_mstep_verified_dense_conductive_coefficients(monkeypatch, axes, tile, segment):
    sim = _simulation(
        material="dense_conductive", boundary="sponge", shape=(103, 65, 97), steps=5
    )
    # JAX computes these banks during stepping; verify the native representation.
    program = sim.compile(backend="cuda_streamed", num_steps=5)
    for axis in "xyz":
        coefficient = getattr(program.coefficients, f"h_source_{axis}")
        assert coefficient.ndim == 3 and all(coefficient.shape)
    state = _seed(program)
    reference = _run(sim, state, "cuda_streamed", 5)
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", axes)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", tile)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", segment)
    actual = _run(sim, state, "cuda_mstep", 5)
    _compare(reference, actual)
    _compare(_run(sim, state, "jax", 5), actual, 3e-5)


def test_mstep_short_column_sanitizer():
    """Keep a real K=2 core and ring wraparound cheap to instrument."""
    sim = _simulation(shape=(12, 25, 55))
    sim = bz.Simulation(
        material_grid=sim._material_grid(progress=False),
        boundaries=[bz.PEC(edges="all")],
        time=sim.time,
    )
    program = sim.compile(backend="cuda_mstep", num_steps=2)
    shapes = tuple(
        program.boundary.logical_component_shapes[c]
        for c in ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
    )
    assert build_mstep_plan(shapes, program.boundary.cpml, (None,) * 9).core_cells > 0
    state = _seed(program)
    actual = _run(sim, state, "cuda_mstep", 2)
    _compare(_run(sim, state, "cuda", 2), actual)
    _compare(_run(sim, state, "jax", 2), actual, 3e-5)


def test_mstep_segmented_rotated_sanitizer(monkeypatch):
    """Cross multiple column segments with cyclic storage and seeded fields."""
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", "8x16")
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", "32")
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", "120")
    sim = _simulation(shape=(73, 49, 65))
    sim = bz.Simulation(
        material_grid=sim._material_grid(progress=False),
        boundaries=[bz.PEC(edges="all")],
        time=sim.time,
    )
    program = sim.compile(backend="cuda_mstep", num_steps=2)
    state = _seed(program)
    actual = _run(sim, state, "cuda_mstep", 2)
    _compare(_run(sim, state, "cuda", 2), actual)
    _compare(_run(sim, state, "jax", 2), actual, 3e-5)


def test_mstep_three_material_codebook_matches_cuda_and_jax():
    """Exercise a packed codebook with more than the common two dielectrics."""
    sim = _simulation(shape=(21, 41, 73))
    grid = sim._material_grid(progress=False)
    eps = np.asarray(grid.permittivity).copy()
    eps[:, :10, :] = np.float32(2.0**2)
    from beamz.design import MaterialGrid

    sim = bz.Simulation(
        material_grid=MaterialGrid(
            permittivity=eps,
            conductivity=np.float32(0),
            permeability=np.float32(1),
            resolution=grid.resolution,
            shape=eps.shape,
        ),
        sources=sim.sources,
        monitors=sim.monitors,
        boundaries=sim.boundaries,
        time=sim.time,
    )
    program = sim.compile(backend="cuda_mstep", num_steps=3)
    assert program.coefficients.e_source_x.dtype == np.int32
    assert program.coefficients.e_decay_x.shape[0] > 2
    state = _seed(program)
    actual = _run(sim, state, "cuda_mstep", 3)
    _compare(_run(sim, state, "cuda", 3), actual)
    _compare(_run(sim, state, "jax", 3), actual, 3e-5)


@pytest.mark.parametrize("steps", [2, 5, 17])
def test_mstep_unequal_six_face_cpml_matches_cuda_and_jax(steps):
    """Exercise dependency bounds with distinct nonzero low/high absorbers."""
    sim = _simulation()
    boundaries = [
        bz.PML(edges=[face], thickness=thickness * 80e-9, formulation="cpml")
        for face, thickness in zip(
            ("left", "right", "top", "bottom", "front", "back"),
            (2, 5, 3, 6, 4, 7),
            strict=True,
        )
    ]
    sim = bz.Simulation(
        material_grid=sim._material_grid(progress=False),
        sources=sim.sources,
        monitors=sim.monitors,
        boundaries=boundaries,
        time=sim.time,
    )
    state = _seed(sim.compile(backend="jax", num_steps=steps))
    actual = _run(sim, state, "cuda_mstep", steps)
    _compare(_run(sim, state, "cuda", steps), actual)
    _compare(_run(sim, state, "jax", steps), actual, 3e-5)


def test_unified_cuda_schedule_matches_compatibility_alias():
    sim = _simulation()
    state = _seed(sim.compile(backend="jax", num_steps=5))
    program = sim.compile(backend="cuda", schedule="mstep", num_steps=5)
    report = program.cuda_schedule_report()
    assert report["selected"] == "mstep"
    report["selected"] = "changed"
    assert program.cuda_schedule_report()["selected"] == "mstep"
    actual = sim.advance(
        state=state, num_steps=5, backend="cuda", schedule="mstep", performance=False
    ).state
    _compare(_run(sim, state, "cuda_mstep", 5), actual)


@pytest.mark.parametrize(
    "variant_flags", [0, abi.CUDA_MSTEP_TILE8X16 | abi.CUDA_MSTEP_SEGMENT32]
)
def test_automatic_schedule_dispatch_preserves_chunk_clocks(monkeypatch, variant_flags):
    """Exercise automatic dispatch on a small core; model admission is tested on host."""
    from beamz.simulation.compile import clear_program_cache
    from beamz.simulation.cuda import schedule
    from beamz.simulation.execute import clear_execution_cache

    # Each mocked policy is a different frozen decision for an equivalent case.
    clear_program_cache()
    clear_execution_cache()

    monkeypatch.setattr(schedule, "cuda_device_kind", lambda: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(
        schedule,
        "automatic_candidate",
        lambda program, geometry, **kw: (
            True,
            "parity probe",
            {
                **geometry.summary(),
                "mstep_cuda_flags": abi.CUDA_GRAPH_CACHE | variant_flags,
            },
        ),
    )
    sim = _simulation(steps=257)
    state = _seed(sim.compile(backend="jax", num_steps=257))
    program = sim.compile(backend="cuda", schedule="auto", num_steps=257)
    report = program.cuda_schedule_report()
    assert report["selected"] == "mstep"
    assert report["tile"] == ((8, 16) if variant_flags else (16, 32))
    assert report["column_segment"] == (32 if variant_flags else 0)
    actual = sim.advance(
        state=state, num_steps=257, backend="cuda", schedule="auto", performance=False
    ).state
    _compare(_run(sim, state, "cuda_streamed", 257), actual, 3e-5)
