"""PEC mask compression preserves physics and bounds captured JAX constants."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from beamz import PEC, PML, Design, Material, Simulation
from beamz.simulation import execute
from beamz.simulation.boundary_masks import compact_mask
from beamz.simulation.kernels import apply_zero_mask


def _captured_boolean_bytes(closed_jaxpr):
    total = sum(
        np.asarray(value).nbytes
        for value in getattr(closed_jaxpr, "consts", ())
        if np.asarray(value).dtype == np.bool_
    )
    jaxpr = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)
    for equation in getattr(jaxpr, "eqns", ()):
        for value in equation.params.values():
            if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                total += _captured_boolean_bytes(value)
    return total


@pytest.mark.parametrize("shape", [(9, 11), (7, 9, 11)])
@pytest.mark.parametrize("kind", ["empty", "full", "faces", "irregular"])
def test_compact_masks_match_dense_masks(shape, kind):
    mask = np.zeros(shape, dtype=bool)
    if kind == "full":
        mask[:] = True
    elif kind == "faces":
        mask[0] = True
        mask[..., -1] = True
    elif kind == "irregular":
        mask[tuple(2 for _ in shape)] = True
    values = np.random.default_rng(27).normal(size=shape).astype(np.float32)
    compiled_mask = compact_mask(mask)
    actual = jax.jit(lambda field: apply_zero_mask(field, compiled_mask))(values)
    np.testing.assert_array_equal(actual, np.where(mask, 0.0, values))
    if kind == "irregular":
        assert compiled_mask is mask


def test_captured_boundary_constants_scale_with_axis_lengths():
    shape = (31, 35, 41)
    mask = np.zeros(shape, dtype=bool)
    mask[0] = True
    mask[:, -1] = True
    mask[..., 0] = True
    compact = compact_mask(mask)
    traced = jax.make_jaxpr(lambda field: apply_zero_mask(field, compact))(
        np.ones(shape, dtype=np.float32)
    )
    assert sum(np.asarray(value).nbytes for value in traced.consts) <= sum(shape)


@pytest.mark.parametrize(
    "boundaries", [[PEC()], [PEC(), PML(edges=("left", "right"), thickness=1.0)]]
)
def test_execution_preserves_fields_and_canonical_masks(monkeypatch, boundaries):
    simulation = Simulation(
        design=Design(
            width=7.0, height=9.0, depth=11.0, material=Material(permittivity=1.0)
        ),
        time=np.arange(4, dtype=float) * 1e-18,
        resolution=1.0,
        boundaries=boundaries,
        sources=[],
        monitors=[],
    )
    program = simulation.compile(num_steps=3, backend="jax")
    state = simulation.initial_state()
    random = np.random.default_rng(23)
    names = ("ex", "ey", "ez", "hx", "hy", "hz")
    state = state._replace(
        **{
            name: random.normal(size=getattr(state, name).shape).astype(np.float32)
            * 1e-5
            for name in names
        }
    )
    original_masks = {
        name: getattr(program.boundary.metallic, name)
        for name in ("ex_mask", "ey_mask", "ez_mask", "hx_mask", "hy_mask", "hz_mask")
    }
    prepared = execute.runtime_inputs(program, state, monitor_steps=3)
    traced = jax.make_jaxpr(execute.build_scan(program))(prepared, program.coefficients)
    boolean_bytes = _captured_boolean_bytes(traced)
    assert boolean_bytes < 1000
    optimized, _ = execute._run_program_state_timed(
        program, state, monitor_steps=3, donate_state=False
    )
    execute.clear_execution_cache()
    monkeypatch.setattr(execute, "compact_boundary_masks", lambda boundary: boundary)
    reference, _ = execute._run_program_state_timed(
        program, state, monitor_steps=3, donate_state=False
    )
    for actual, expected in zip(
        jax.tree.leaves(optimized), jax.tree.leaves(reference), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for name, mask in original_masks.items():
        assert getattr(program.boundary.metallic, name) is mask
