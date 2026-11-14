"""Hardware parity of streamed tensor CUDA, including graph and phase paths.

Run explicitly on a CUDA host with the matching native component:
JAX_PLATFORMS=cuda PYTHONPATH=. pytest -q tests/performance/test_cuda_tensor.py
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    Grid,
    Material,
    Object,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.simulation.compile.backend import cuda_backend_status
from beamz.simulation.compile.material_blocks import TensorRegion
from beamz.simulation.compile.physics import select_update_kernel
from beamz.simulation.execute import (
    build_program_scan,
    build_step_context,
    forward_step,
    initial_program_state,
)
from beamz.simulation.execute.sharding import place_tree
from tests.jax_compat import enable_x64


def tensor_simulation(
    size=18,
    steps=12,
    grid_kind="graded",
    location="interior",
    boundary="cpml",
    monitors=True,
):
    length = size * 100e-9
    u = np.linspace(0, 1, size + 1)
    edges = [length * u] * 3
    if grid_kind == "graded":
        edges = [length * (u + a * np.sin(2 * np.pi * u)) for a in (0.04, 0.06, 0.08)]
    elif grid_kind == "axis":
        edges = [length * u * a for a in (1, 1.2, 0.9)]
    extent = np.array([e[-1] for e in edges])
    spans = {
        "interior": (0.35, 0.65),
        "lower": (0, 0.3),
        "upper": (0.7, 1),
        "bulk": (0, 1),
    }
    lo, hi = spans[location]
    epsilon = ((6.0, 0.2, 0.1), (0.2, 5.0, 0.3), (0.1, 0.3, 4.0))
    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (Material(), Material(epsilon_r=epsilon)),
                (Object(Box(tuple(lo * extent), tuple(hi * extent)), 1),),
            ),
            Grid(*edges),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    boundaries = (
        [bz.PEC(edges="all")]
        if boundary == "pec"
        else [
            bz.PML(
                edges="right" if boundary == "mixed" else "all",
                thickness=300e-9,
                formulation="cpml",
            )
        ]
    )
    return bz.Simulation(
        material_grid=material,
        time=np.arange(steps + 1) * material.grid.cfl_time_step(0.5),
        boundaries=boundaries,
        setup_device="cpu",
        monitors=[
            bz.FluxMonitor(
                center=tuple(extent * (0.5, 0.5, 0.6)),
                size=tuple(extent * (0.6, 0.6, 0)),
                freqs=[bz.LIGHT_SPEED / 1.55e-6],
                name="flux",
            ),
        ]
        if monitors
        else [],
    )


def seeded_state(program):
    state = initial_program_state(program, t=0.0, current_step=0)
    rng = np.random.default_rng(361)
    fields = {
        name: jnp.asarray(
            rng.normal(size=getattr(state, name).shape), dtype=jnp.float32
        )
        * (1e-3 if name[0] == "e" else 1e-6)
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    for phase in ("h", "e"):
        name = f"cpml_psi_{phase}_terms"
        fields[name] = tuple(
            jnp.asarray(rng.normal(size=v.shape), dtype=jnp.float32) * 0.01
            for v in getattr(state, name)
        )
    return state._replace(**fields)


def compare_states(reference, actual):
    errors = {}
    for name in (
        "ex",
        "ey",
        "ez",
        "hx",
        "hy",
        "hz",
        "cpml_psi_h_terms",
        "cpml_psi_e_terms",
        "dft_vec_re",
        "dft_vec_im",
    ):
        a = np.concatenate(
            [np.asarray(x).ravel() for x in jax.tree.leaves(getattr(reference, name))]
            or [np.empty(0)]
        )
        b = np.concatenate(
            [np.asarray(x).ravel() for x in jax.tree.leaves(getattr(actual, name))]
            or [np.empty(0)]
        )
        assert np.isfinite(b).all(), name
        error = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        errors[name] = error
        np.testing.assert_allclose(
            b,
            a,
            rtol=5e-5,
            atol=max(1e-12, 5e-6 * np.max(np.abs(a), initial=0)),
            err_msg=name,
        )
        assert error < 1e-5, (name, error)
    return errors


@pytest.fixture(autouse=True)
def require_cuda():
    status = cuda_backend_status()
    if not status.available:
        pytest.skip(status.reason)
    with enable_x64(False):
        yield


@pytest.mark.parametrize("grid_kind", ["uniform", "axis", "graded"])
@pytest.mark.parametrize("location", ["interior", "lower", "upper", "bulk"])
@pytest.mark.parametrize("boundary", ["pec", "cpml", "mixed"])
def test_tensor_cuda_graph_matches_jax(grid_kind, location, boundary):
    sim = tensor_simulation(grid_kind=grid_kind, location=location, boundary=boundary)
    plans = [sim.compile(num_steps=12, backend=b) for b in ("jax", "cuda_streamed")]
    state = seeded_state(plans[0])
    assert isinstance(plans[1].coefficients.e_inverse_offdiagonal, TensorRegion) == (
        location != "bulk"
    )
    results = [
        jax.block_until_ready(
            build_program_scan(p)(state, place_tree(p, p.coefficients))
        )
        for p in plans
    ]
    compare_states(*results)


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_tensor_cuda_phase_matches_graph_without_monitors(boundary):
    sim = tensor_simulation(boundary=boundary, monitors=False)
    p = sim.compile(num_steps=3, backend="cuda_streamed")
    state, coeffs = seeded_state(p), place_tree(p, p.coefficients)
    graph = build_program_scan(p)(state, coeffs)
    context = build_step_context(p)
    kernel = select_update_kernel(context)

    def steps(state, coeffs):
        for _ in range(3):
            state = forward_step(
                state, program=p, ctx=context, coeffs=coeffs, update_kernel=kernel
            )
        return state

    phase = jax.jit(steps)(state, coeffs)
    compare_states(jax.block_until_ready(graph), jax.block_until_ready(phase))


def test_tensor_cuda_rejects_unimplemented_adjoint():
    from beamz.simulation.api.differentiation import build_differentiable_scan
    from beamz.simulation.cuda.native.phase import payload

    p = tensor_simulation().compile(num_steps=1, backend="cuda_streamed")
    with pytest.raises(ValueError, match="autodiff.*jax"):
        payload(replace(build_step_context(p), differentiable=True), p.coefficients)
    with pytest.raises(ValueError, match="autodiff.*jax"):
        build_differentiable_scan(p)


def test_tensor_cuda_graph_continuation_preserves_dft_clock():
    sim = tensor_simulation()
    whole, part = [sim.compile(num_steps=n, backend="cuda_streamed") for n in (12, 6)]
    state = seeded_state(whole)
    coeffs = place_tree(whole, whole.coefficients)
    expected = build_program_scan(whole)(state, coeffs)
    advance = build_program_scan(part)
    actual = advance(advance(state, coeffs), coeffs)
    compare_states(jax.block_until_ready(expected), jax.block_until_ready(actual))
    assert int(actual.current_step) == int(expected.current_step)


@pytest.mark.parametrize("factor", [0.0, -1.0, 0.5])
def test_tensor_cuda_coefficients_remain_dynamic(factor):
    sim = tensor_simulation()
    plans = [sim.compile(num_steps=4, backend=b) for b in ("jax", "cuda_streamed")]
    state = seeded_state(plans[0])
    outputs = []
    for p in plans:
        coeffs = place_tree(p, p.coefficients)
        f = build_program_scan(p)
        # Warm the same graph before changing its coefficient values.
        jax.block_until_ready(f(state, coeffs))
        cross = coeffs.e_inverse_offdiagonal
        coeffs = coeffs._replace(
            e_inverse_offdiagonal=replace(cross, values=factor * cross.values)
        )
        outputs.append(jax.block_until_ready(f(state, coeffs)))
    compare_states(*outputs)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_tensor_cuda_modal_launch_and_projection(axis, direction):
    from tests.validation.regression.test_tensor_port_workflow import run

    reference, *_ = run(axis, direction, rectilinear=True)
    actual, *_ = run(axis, direction, rectilinear=True, backend="cuda_streamed")
    for name in ("near", "far", "shifted"):
        ref, value = reference["ports"][name], actual["ports"][name]
        assert value["forward_power"] == pytest.approx(abs(value["flux"]), rel=5e-5)
        if name != "shifted":
            assert value["forward_power"] == pytest.approx(1.0, rel=1e-3)
            assert abs(value["independent_flux"]) == pytest.approx(1.0, rel=1e-3)
        assert value["backward_power"] / value["forward_power"] < 1e-6
        np.testing.assert_allclose(
            value["forward_amplitude"],
            ref["forward_amplitude"],
            rtol=2e-4,
            atol=2e-6,
        )
        assert value["sampled_modal_flux"] == pytest.approx(value["flux"], rel=5e-5)
        if name != "shifted":
            assert value["sampled_modal_flux"] == pytest.approx(
                value["independent_flux"], rel=5e-5
            )


def test_tensor_cuda_smoothed_interface_and_archive(tmp_path):
    from beamz.simulation.results import SimulationResults
    from tests.validation.regression.test_tensor_port_workflow import run

    report, result, *_ = run(
        axis="z", rectilinear=True, smoothed=True, backend="cuda_streamed"
    )
    near = report["ports"]["near"]
    assert near["backward_power"] / near["forward_power"] < 1e-6
    assert near["sampled_modal_flux"] == pytest.approx(
        near["independent_flux"], rel=1e-4
    )
    path = tmp_path / "cuda-tensor.npz"
    result.save(path)
    restored = SimulationResults.load(path)
    np.testing.assert_allclose(
        restored.mode("near").amps, result.mode("near").amps, rtol=1e-10, atol=1e-10
    )
