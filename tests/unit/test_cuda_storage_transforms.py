"""CPU contract tests for storage geometry at the native-call boundary."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.cuda.storage import (
    _permute_monitor_arena,
    mstep_plan_inputs,
    wrap_native_calls,
    wrap_program_call,
)
from tests.unit.test_cuda_runtime_contract import _program_and_state


@dataclass
class SourceGroup:
    coeffs: object
    starts: object
    starts_tuple: tuple
    max_sizes: tuple


@pytest.mark.parametrize("order", [(2, 0, 1, 5, 3, 4), (1, 2, 0, 4, 5, 3)])
def test_monitor_block_permutation_matches_index_reference_with_ragged_gaps(order):
    values = jnp.arange(81, dtype=jnp.float32)
    counts = np.array([[1, 5, 0, 43, 0], [2, 3, 0, 2, 0]], dtype=np.int32)
    indices = np.arange(values.size)
    for frequencies, points, _, offset, _ in counts:
        end = offset + 6 * frequencies * points
        indices[offset:end] = indices[offset:end].reshape(6, -1)[list(order)].ravel()
    transformed = jax.jit(lambda v: _permute_monitor_arena(v, counts, order))(values)
    np.testing.assert_array_equal(transformed, np.asarray(values)[indices])
    inverse = tuple(order.index(i) for i in range(6))
    np.testing.assert_array_equal(
        _permute_monitor_arena(transformed, counts, inverse), values
    )


def test_empty_monitor_metadata_preserves_arena():
    values = jnp.arange(7, dtype=jnp.float32)
    assert (
        _permute_monitor_arena(values, np.empty((0, 5)), (2, 0, 1, 5, 3, 4)) is values
    )


def test_canonical_wrapper_preserves_clock_arguments():
    def native(state, ctx, coeffs, groups, monitors, nsteps, **clock):
        assert nsteps == 7
        assert clock == {"observation_origin": 0.5, "observation_step_offset": 19}
        return state

    state = object()
    result = wrap_program_call(native, (0, 1, 2))(
        state,
        None,
        None,
        None,
        None,
        7,
        observation_origin=0.5,
        observation_step_offset=19,
    )
    assert result is state


@pytest.mark.parametrize(
    "axes,components", [((1, 2, 0), (2, 0, 1)), ((2, 0, 1), (1, 2, 0))]
)
@pytest.mark.parametrize("packed", [False, True])
def test_rotated_native_boundary_preserves_physical_samples(axes, components, packed):
    program, state, context = _program_and_state(cpml=True)
    names = ("ex", "ey", "ez", "hx", "hy", "hz")
    # Distinguishable Yee components and coordinates reveal component swaps.
    fields = {
        name: jnp.arange(getattr(state, name).size, dtype=jnp.float32).reshape(
            getattr(state, name).shape
        )
        + 10000 * i
        for i, name in enumerate(names)
    }
    state = state._replace(
        **fields,
        cpml_psi_h_terms=tuple(v + i + 1 for i, v in enumerate(state.cpml_psi_h_terms)),
        cpml_psi_e_terms=tuple(
            v + i + 11 for i, v in enumerate(state.cpml_psi_e_terms)
        ),
        dft_vec_re=jnp.arange(42, dtype=jnp.float32),
        dft_vec_im=-jnp.arange(42, dtype=jnp.float32),
    )
    context = replace(
        context,
        boundary=replace(
            context.boundary,
            cpml=replace(
                context.boundary.cpml,
                metallic_edges=frozenset({"front", "right"}),
                magnetic_edges=frozenset({"back", "bottom"}),
            ),
        ),
    )
    coeffs = program.coefficients
    ids = {}
    if packed:
        updates = {}
        for i, component in enumerate("xyz"):
            shape = getattr(state, "e" + component).shape
            values = (np.arange(np.prod(shape)) % 251).astype(np.uint8).reshape(shape)
            ids[component] = values
            # Deliberately poison unused tail bytes; the rotated buffer must zero them.
            padded = np.pad(values.ravel(), (0, -values.size % 4), constant_values=255)
            updates["e_source_" + component] = jnp.asarray(padded.view("<i4"))
            updates["e_decay_" + component] = jnp.arange(251, dtype=jnp.float32) + i
        coeffs = coeffs._replace(**updates)
    group = SourceGroup(
        jnp.arange(30).reshape(1, 2, 3, 5),
        jnp.array([[1, 2, 3]]),
        ((1, 2, 3),),
        (2, 3, 5),
    )
    groups = (group, None, group, None, group, None, group, None, group)
    indices = jnp.array([[0] * 6, [7] * 6, [-1] * 6, [999999] * 6])
    weights = jnp.arange(24).reshape(4, 6)
    # Two ragged monitor arenas: 2 points x 2 frequencies and 3 points x 1.
    monitors = (
        indices,
        weights,
        jnp.array([5]),
        weights + 100,
        np.array([[2, 2, 0, 0, 0], [1, 3, 0, 24, 0]]),
    )
    lanes = (*components, *(3 + c for c in components))
    calls = []
    plan_program = replace(
        program,
        config=replace(program.config, cuda_storage_axes=axes),
        boundary=context.boundary,
    )
    planned_shapes, planned_cpml, planned_groups, planned_indices = mstep_plan_inputs(
        plan_program, groups, monitors
    )

    def native(rotated, ctx, material, sources, samples, nsteps):
        calls.append(nsteps)
        assert planned_shapes == tuple(
            getattr(rotated, name).shape
            for name in ("hx", "hy", "hz", "ex", "ey", "ez")
        )
        np.testing.assert_array_equal(planned_indices, samples[0])
        for planned, actual in zip(planned_groups, sources, strict=True):
            if actual is None:
                assert planned is None
            else:
                assert planned.coeffs.shape == actual.coeffs.shape
                np.testing.assert_array_equal(planned.starts, actual.starts)
        for phase in "he":
            for planned, actual in zip(
                getattr(planned_cpml, phase + "_terms"),
                getattr(ctx.boundary.cpml, phase + "_terms"),
                strict=True,
            ):
                assert (planned.component, planned.axis, planned.slab) == (
                    actual.component,
                    actual.axis,
                    actual.slab,
                )
        for new, old in enumerate(lanes):
            original = np.asarray(getattr(state, names[old]))
            transformed = np.asarray(getattr(rotated, names[new]))
            assert transformed.shape == tuple(original.shape[a] for a in axes)
            for point in [(0, 0, 0), tuple(n - 1 for n in original.shape)]:
                assert transformed[tuple(point[a] for a in axes)] == original[point]
            for row in (0, 1):
                coordinate = np.unravel_index(int(indices[row, old]), original.shape)
                expected = np.ravel_multi_index(
                    tuple(coordinate[a] for a in axes), transformed.shape
                )
                assert samples[0][row, new] == expected
        np.testing.assert_array_equal(samples[0][2:], -np.ones((2, 6)))
        np.testing.assert_array_equal(samples[1], np.asarray(weights)[:, lanes])
        np.testing.assert_array_equal(samples[3], np.asarray(weights + 100)[:, lanes])
        assert samples[2] is monitors[2]
        for offset, width in ((0, 4), (24, 3)):
            expected = np.asarray(state.dft_vec_re)[
                offset : offset + 6 * width
            ].reshape(6, width)[list(lanes)]
            np.testing.assert_array_equal(
                rotated.dft_vec_re[offset : offset + 6 * width], expected.ravel()
            )
            np.testing.assert_array_equal(
                rotated.dft_vec_im[offset : offset + 6 * width], -expected.ravel()
            )
        faces = (("front", "back"), ("bottom", "top"), ("left", "right"))
        assert ctx.boundary.cpml.metallic_edges == {
            faces[axes.index(0)][0],
            faces[axes.index(2)][1],
        }
        assert ctx.boundary.cpml.magnetic_edges == {
            faces[axes.index(0)][1],
            faces[axes.index(1)][0],
        }
        for phase in "he":
            old_terms = getattr(context.boundary.cpml, phase + "_terms")
            new_terms = getattr(ctx.boundary.cpml, phase + "_terms")
            for c, old in enumerate(components):
                for k in range(2):
                    term, original = new_terms[2 * c + k], old_terms[2 * old + k]
                    assert term.component == phase.upper() + "xyz"[c]
                    assert axes[term.axis] == original.axis
                    assert term.slab.shape == tuple(
                        original.slab.shape[a] for a in axes
                    )
                    np.testing.assert_array_equal(
                        term.a, np.asarray(original.a).transpose(axes)
                    )
        for phase in range(3):
            for c, old in enumerate(components):
                transformed = sources[3 * phase + c]
                if groups[3 * phase + old] is None:
                    assert transformed is None
                else:
                    assert transformed.starts_tuple == (
                        tuple((1, 2, 3)[a] for a in axes),
                    )
                    np.testing.assert_array_equal(
                        transformed.starts, transformed.starts_tuple
                    )
                    assert transformed.max_sizes == tuple((2, 3, 5)[a] for a in axes)
                    np.testing.assert_array_equal(
                        transformed.coeffs,
                        np.asarray(group.coeffs).transpose(0, *(a + 1 for a in axes)),
                    )
        if packed:
            for c, old in enumerate(components):
                actual = np.asarray(getattr(material, "e_source_" + "xyz"[c])).view(
                    np.uint8
                )
                expected = ids["xyz"[old]].transpose(axes).ravel()
                np.testing.assert_array_equal(actual[: expected.size], expected)
                assert np.all(actual[expected.size :] == 0)
                np.testing.assert_array_equal(
                    getattr(material, "e_decay_" + "xyz"[c]),
                    getattr(coeffs, "e_decay_" + "xyz"[old]),
                )
        # Distinct native updates must return to their original physical lanes.
        return rotated._replace(
            **{name: getattr(rotated, name) + 7 for name in names},
            dft_vec_re=rotated.dft_vec_re + 3,
        )

    result = wrap_program_call(native, axes)(
        state, context, coeffs, groups, monitors, 3
    )
    assert calls == [3]
    for name in names:
        np.testing.assert_array_equal(getattr(result, name), getattr(state, name) + 7)
    np.testing.assert_array_equal(result.dft_vec_re, state.dft_vec_re + 3)
    np.testing.assert_array_equal(result.dft_vec_im, state.dft_vec_im)
    for phase in "he":
        for actual, expected in zip(
            getattr(result, "cpml_psi_" + phase + "_terms"),
            getattr(state, "cpml_psi_" + phase + "_terms"),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axes", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
def test_plain_and_source_entries_restore_state_without_monitors(axes):
    program, state, context = _program_and_state(cpml=True)
    calls = []

    def native(s, ctx, coeffs, *args):
        calls.append(args)
        return s

    plain, full, source = wrap_native_calls(native, native, native, axes)
    groups = (None,) * 9
    outputs = [
        plain(state, context, program.coefficients, 3),
        source(state, context, program.coefficients, groups, 3),
        full(state, context, program.coefficients, groups, None, 3),
    ]
    assert [len(args) for args in calls] == [1, 2, 3]
    for output in outputs:
        for actual, expected in zip(
            jax.tree.leaves(output), jax.tree.leaves(state), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


def test_storage_rejects_reflections_and_unsupported_geometry():
    with pytest.raises(ValueError, match="right-handed"):
        wrap_program_call(None, (2, 1, 0))
    program, state, context = _program_and_state(cpml=True)
    call = wrap_program_call(
        lambda *args: pytest.fail("Native call reached"), (1, 2, 0)
    )
    with pytest.raises(ValueError, match="uniform 3D and diagonal"):
        call(
            state,
            replace(context, is_3d=False),
            program.coefficients,
            (None,) * 9,
            None,
            3,
        )
    cpml = context.boundary.cpml
    bad_term = replace(cpml.h_terms[0], sign=-cpml.h_terms[0].sign)
    context = replace(
        context,
        boundary=replace(
            context.boundary, cpml=replace(cpml, h_terms=(bad_term, *cpml.h_terms[1:]))
        ),
    )
    with pytest.raises(ValueError, match="canonical 3D CPML"):
        call(state, context, program.coefficients, (None,) * 9, None, 3)


@pytest.mark.parametrize("axes", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
@pytest.mark.parametrize("monitor", [False, True])
def test_plan_metadata_without_cpml_or_sources(axes, monitor):
    program, state, _ = _program_and_state(cpml=False)
    program = replace(program, config=replace(program.config, cuda_storage_axes=axes))
    groups = (None,) * 9
    monitors = (np.zeros((1, 6), np.int32),) if monitor else None
    shapes, cpml, sources, indices = mstep_plan_inputs(program, groups, monitors)
    components = tuple(2 - axes[2 - c] for c in range(3))
    order = (*components, *(3 + c for c in components))
    fields = (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)
    assert shapes == tuple(tuple(fields[c].shape[a] for a in axes) for c in order)
    assert cpml.h_terms == cpml.e_terms == ()
    assert sources == groups
    if monitor:
        np.testing.assert_array_equal(indices, monitors[0])
    else:
        assert indices is None
    if axes == (0, 1, 2):
        assert cpml is program.boundary.cpml
        assert sources is groups
