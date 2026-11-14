"""Exact support cropping, including boundary halos and JAX derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.const import EPS_0
from beamz.lattice import component_shape_3d
from beamz.simulation.compile.kernels import (
    advance_e_centered_tensor,
    collocate_yee_component,
)
from beamz.simulation.compile.material_blocks import TensorRegion, compact_tensor_region
from tests.jax_compat import enable_x64


def reference_dense_tensor(
    fields,
    curls,
    inverse_diagonals,
    inverse_offdiagonal,
    components,
    dt,
    *,
    cell_widths=None,
):
    """Advance E with diagonal terms on E sites and cross terms on dual centers."""

    scale = jnp.asarray(dt, dtype=fields[0].dtype) / jnp.asarray(
        EPS_0, dtype=fields[0].dtype
    )
    node_shape = inverse_offdiagonal.shape[:-1]
    centered_curls = tuple(
        collocate_yee_component(
            curl, source, "Node", node_shape, cell_widths=cell_widths
        )
        for curl, source in zip(curls, components, strict=True)
    )
    cross_component = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    updated = []
    for row, (field, diagonal, target) in enumerate(
        zip(fields, inverse_diagonals, components, strict=True)
    ):
        coupled = diagonal * curls[row]
        for column, centered_curl in enumerate(centered_curls):
            if row == column:
                continue
            pair = (min(row, column), max(row, column))
            centered = inverse_offdiagonal[..., cross_component[pair]] * centered_curl
            coupled = coupled + collocate_yee_component(
                centered, "Node", target, field.shape, cell_widths=cell_widths
            )
        updated.append(field + scale * coupled)
    return tuple(updated)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "location", ["interior", "lower", "upper", "single", "separated"]
)
@pytest.mark.parametrize("complex_fields", [False, True])
def test_compact_tensor_matches_dense_update_on_graded_mesh(
    ndim, location, complex_fields
):
    rng = np.random.default_rng(361)
    shape = (12, 13, 14)[-ndim:]
    names = ("Ex", "Ey", "Ez")[:ndim]
    node_shape = tuple(n + 1 for n in shape)
    cross = np.zeros((*node_shape, 3))
    span = {
        "interior": slice(4, 7),
        "lower": slice(0, 3),
        "upper": slice(-3, None),
        "single": slice(5, 6),
        "separated": slice(1, 3),
    }[location]
    cross[(span,) * ndim] = rng.normal(size=3) * 0.1
    if location == "separated":
        cross[(slice(-3, None),) * ndim] = 0.05
    packed = compact_tensor_region(cross)
    assert isinstance(packed, TensorRegion) == (location != "separated")
    shapes = [
        component_shape_3d(k, (1, *shape))[-2:]
        if ndim == 2
        else component_shape_3d(k, shape)
        for k in names
    ]
    fields, curls = [
        tuple(
            rng.normal(size=s) + (1j * rng.normal(size=s) if complex_fields else 0)
            for s in shapes
        )
        for _ in range(2)
    ]
    diagonals = tuple(rng.uniform(0.5, 1, s) for s in shapes)
    widths = tuple(rng.uniform(0.2, 2, n) for n in shape[::-1]) + (
        () if ndim == 3 else (np.ones(1),)
    )
    with enable_x64(True):

        def apply(update, coefficients):
            return update(
                tuple(map(jnp.asarray, fields)),
                tuple(map(jnp.asarray, curls)),
                tuple(map(jnp.asarray, diagonals)),
                coefficients,
                names,
                EPS_0,
                cell_widths=widths,
            )

        expected = apply(reference_dense_tensor, jnp.asarray(cross))
        actual = jax.jit(
            lambda coefficients: apply(advance_e_centered_tensor, coefficients)
        )(jax.tree.map(jnp.asarray, packed))
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), atol=3e-14, rtol=3e-14
            )


def test_compact_tensor_retains_small_nonzero_coefficients():
    cross = np.zeros((20, 20, 20, 3))
    cross[10, 10, 10] = 1e-30
    packed = compact_tensor_region(cross)
    assert isinstance(packed, TensorRegion)
    np.testing.assert_array_equal(packed.values[0, 0, 0], cross[10, 10, 10])


def test_compiled_memory_report_counts_packed_tensor_storage():
    from beamz.simulation.compile import _add_memory_arrays

    cross = np.zeros((20, 20, 20, 3), dtype=np.float32)
    cross[8:12, 8:12, :] = 0.1
    packed = compact_tensor_region(cross)
    assert isinstance(packed, TensorRegion)
    entries = []
    _add_memory_arrays(entries, "tensor", packed, "coefficients")
    assert len(entries) == 1
    assert entries[0]["bytes"] == 4 * 4 * 20 * 3 * 4
    assert entries[0]["bytes"] < cross.nbytes / 10


def test_compact_tensor_gradients_match_dense_fixed_support():
    rng = np.random.default_rng(362)
    shape = (9, 10, 11)
    names = ("Ex", "Ey", "Ez")
    cross = np.zeros((10, 11, 12, 3))
    cross[3:5, 4:6, 5:7] = (0.1, 0.2, 0.3)
    region = compact_tensor_region(cross)
    assert isinstance(region, TensorRegion)
    with enable_x64(True):
        curls = tuple(
            jnp.asarray(rng.normal(size=component_shape_3d(k, shape))) for k in names
        )
        fields = tuple(jnp.zeros_like(v) for v in curls)
        diagonal = tuple(jnp.ones_like(v) for v in curls)
        widths = tuple(jnp.asarray(rng.uniform(0.5, 1.5, n)) for n in shape[::-1])

        def objective(values, curls, compact):
            if compact:
                coeff = TensorRegion(values, region.bounds, region.node_shape)
            else:
                coeff = jnp.pad(
                    values,
                    tuple(
                        (lo, n - hi)
                        for (lo, hi), n in zip(
                            region.bounds, region.node_shape, strict=True
                        )
                    )
                    + ((0, 0),),
                )
            result = advance_e_centered_tensor(
                fields, curls, diagonal, coeff, names, EPS_0, cell_widths=widths
            )
            return sum(jnp.sum(v * v) for v in result)

        grads = [
            jax.jit(
                jax.grad(lambda v, c, flag=flag: objective(v, c, flag), argnums=(0, 1))
            )(jnp.asarray(region.values), curls)
            for flag in (False, True)
        ]
        for a, b in zip(
            jax.tree.leaves(grads[0]), jax.tree.leaves(grads[1]), strict=True
        ):
            np.testing.assert_allclose(a, b, atol=3e-13, rtol=3e-13)
