"""Actual compiled coefficient-value checks for rotating constant-block storage."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beamz.simulation.cuda.material_blocks import pack_material_blocks

from beamz.simulation.cuda.storage import wrap_program_call
from beamz.simulation.memory import _rotated_block_workspace_bytes
from tests.unit.test_cuda_runtime_contract import _program_and_state


@pytest.mark.parametrize("axes", [(1, 2, 0), (2, 0, 1)])
def test_constant_block_rotation_preserves_compiled_physical_coefficients(axes):
    program, state, context = _program_and_state(cpml=True, shape=(8, 10, 12))
    values = []
    for index, c in enumerate("xyz"):
        shape = getattr(state, "e" + c).shape
        value = np.full(shape, index + 1, np.float32)
        # Include both constant pages and mixed interfaces in every component.
        value[shape[0] // 2 :, shape[1] // 2 :, :] += np.float32(0.25)
        values.append(value)
    packed = pack_material_blocks(tuple(values), max_bytes=float("inf"))
    assert packed is not None
    pools, headers = packed
    assert any(np.any(np.asarray(header) < 0) for header in headers)
    assert any(np.any(np.asarray(header) >= 0) for header in headers)
    changes = {}
    for c, pool, header in zip("xyz", pools, headers, strict=True):
        changes["e_decay_" + c] = pool
        changes["e_source_" + c] = header
    coeffs = program.coefficients._replace(**changes)

    def native(rotated, ctx, material, groups, monitors, nsteps):
        # Dense fallback must stop interpreting the codebook as field decay.
        for c in "xyz":
            assert getattr(material, "e_decay_" + c).ndim == 0
            assert getattr(material, "e_source_" + c).dtype == jnp.float32
        return rotated._replace(
            **{
                "e" + c: getattr(rotated, "e" + c) + getattr(material, "e_source_" + c)
                for c in "xyz"
            }
        )

    wrapped = wrap_program_call(native, axes)
    result = jax.jit(
        lambda material: wrapped(state, context, material, (None,) * 9, None, 2)
    )(coeffs)
    for c, value in zip("xyz", values, strict=True):
        np.testing.assert_array_equal(
            getattr(result, "e" + c), np.asarray(getattr(state, "e" + c)) + value
        )


@pytest.mark.parametrize("axes", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_rotated_header_admission_budgets_dense_values_and_index_workspace(axes):
    program, state, _ = _program_and_state(cpml=True, shape=(8, 10, 12))
    coeffs = program.coefficients._replace(
        e_source_x=jnp.zeros((1, 8), dtype=jnp.int32),
        e_source_y=jnp.zeros((8,), dtype=jnp.int32),
        e_source_z=jnp.zeros((1, 8), dtype=jnp.int32),
    )
    program = replace(
        program,
        coefficients=coeffs,
        config=replace(program.config, cuda_storage_axes=axes),
    )
    # Only the two header-encoded components expand; byte IDs retain repacking.
    dense_bytes = (state.ex.size + state.ez.size) * np.dtype(np.float32).itemsize
    expected = 0 if axes == (0, 1, 2) else 2 * dense_bytes
    assert _rotated_block_workspace_bytes(program) == expected
