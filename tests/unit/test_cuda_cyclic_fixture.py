"""Storage permutations preserve actual bytes and ordered monitor records."""

import numpy as np
import pytest

from scripts.cuda_cyclic_fixture import (
    byte_codes, component_axis, component_map, dft_components, input_mapping,
    leaf_mapping, metadata, monitor_indices, restore_world_leaves, tensor,
)
from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, verify_component_shapes
from scripts.prove_cuda_cyclic_axis_equivariance import AxisMap, cyclic


SHAPES = tuple(tuple(n + d for n, d in zip((4, 5, 6), delta)) for delta in SUPPORT_DELTA)


@pytest.mark.parametrize("step", (0, 1, 2))
def test_every_abi_slot_and_staggered_shape_has_an_exact_inverse(step):
    r = cyclic(step)
    targets = tuple(input_mapping(i, r) for i in range(134))
    assert sorted(targets) == list(range(134))
    assert [input_mapping(j, r.inverse()) for j in targets] == list(range(134))
    out = [None] * 6
    for old, shape in enumerate(SHAPES):
        out[component_map(r)[old]] = r.vector(shape)
    assert verify_component_shapes(out) == r.vector((4, 5, 6))


@pytest.mark.parametrize("step", (0, 1, 2))
def test_spatial_transpose_preserves_all_fp32_bits_including_signed_zero_and_nan(step):
    r = cyclic(step)
    bits = np.arange(120, dtype=np.uint32).reshape(4, 5, 6)
    bits.reshape(-1)[:6] = [0, 0x80000000, 0x7fc00001, 0x7f800000, 0xff800000, 1]
    after = tensor(bits.view(np.float32), r)
    np.testing.assert_array_equal(tensor(after, r.inverse()).view(np.uint32), bits)


@pytest.mark.parametrize("step", (0, 1, 2))
def test_packed_codes_rotate_sample_identities_and_unused_tail_bytes(step):
    r = cyclic(step)
    shape = (3, 5, 7)
    raw = (np.arange(108, dtype=np.uint8) * 17).astype(np.uint8)
    raw[-3:] = [251, 252, 253]
    words = raw.view(np.int32)
    after = byte_codes(words, shape, r)
    np.testing.assert_array_equal(after.view(np.uint8)[:105].reshape(r.vector(shape)), tensor(raw[:105].reshape(shape), r))
    np.testing.assert_array_equal(byte_codes(after, r.vector(shape), r.inverse()), words)


def test_source_group_maps_preserve_waveform_source_id_and_rename_only_component():
    r = cyclic(2)
    assert input_mapping(101, r) == 98
    assert input_mapping(107, r) == 104
    assert input_mapping(102, r) == 99
    assert input_mapping(108, r) == 105


@pytest.mark.parametrize("step", (0, 1, 2))
def test_monitor_addresses_change_but_neighbor_sequence_and_weights_do_not(step):
    r = cyclic(step)
    ids = np.empty((2, 6, 3, 8), np.int32)
    for c in range(6):
        ids[:, c] = np.arange(48).reshape(2, 3, 8) % np.prod(SHAPES[(c + 3) % 6])
    ids[0, 0, 0, 0] = -7
    ids[0, 0, 0, 1] = 1234567
    rotated = monitor_indices(ids, SHAPES, r)
    rotated_shapes = [None] * 6
    for c in range(6):
        rotated_shapes[component_map(r)[c]] = r.vector(SHAPES[c])
    np.testing.assert_array_equal(monitor_indices(rotated, rotated_shapes, r.inverse()), ids)
    weights = np.arange(ids.size, dtype=np.float32).reshape(ids.shape)
    np.testing.assert_array_equal(component_axis(component_axis(weights, r), r.inverse()), weights)
    for c in range(6):
        world = np.arange(np.prod(SHAPES[(c + 3) % 6]), dtype=np.float32).reshape(SHAPES[(c + 3) % 6])
        internal = tensor(world, r)
        old = ids[:, c]
        new = rotated[:, component_map(r)[c]]
        valid = (old >= 0) & (old < world.size)
        np.testing.assert_array_equal(world.reshape(-1)[old[valid]], internal.reshape(-1)[new[valid]])


@pytest.mark.parametrize("step", (0, 1, 2))
def test_dft_component_blocks_retain_frequency_point_and_monitor_offsets(step):
    r = cyclic(step)
    counts = np.asarray([[2, 3, 1, 2, 0], [1, 4, 7, 40, 2]], np.int32)
    value = np.arange(70, dtype=np.float32)
    after = dft_components(value, counts, r)
    np.testing.assert_array_equal(dft_components(after, counts, r.inverse()), value)
    np.testing.assert_array_equal(after[:2], value[:2])
    np.testing.assert_array_equal(after[36+2:40], value[36+2:40])
    np.testing.assert_array_equal(after[64:], value[64:])


def test_dft_aliasing_and_noncyclic_mapping_fail_closed():
    counts = np.asarray([[1, 2, 1, 0, 0], [1, 2, 1, 6, 1]], np.int32)
    with pytest.raises(ValueError, match="Aliased"):
        dft_components(np.zeros(24, np.float32), counts, cyclic(1))
    with pytest.raises(ValueError, match="Proper cyclic"):
        tensor(np.zeros((3, 4, 5)), AxisMap((1, 0, 2)))


def test_independent_cpml_metadata_and_output_slots_follow_derivative_axis():
    rows = np.asarray([[0, 1, 12, 12, 1], [0, 0, 12, 12, -1], [1, 0, 12, 12, 1], [1, 2, 12, 12, -1], [2, 2, 12, 12, 1], [2, 1, 12, 12, -1]], np.int32)
    for step in range(3):
        r = cyclic(step)
        np.testing.assert_array_equal(metadata(rows, r), rows)
        for slot in range(6, 18):
            before = {"kind": "output", "index": slot}
            assert leaf_mapping(leaf_mapping(before, r), r.inverse()) == before


def test_actual_download_inverse_uses_bound_mapping_and_world_shape():
    r = cyclic(2)
    value = np.arange(np.prod(SHAPES[0]), dtype=np.float32).reshape(SHAPES[0])
    fixture = {"cyclic_layout": {"schema": "beamz.cyclic-storage-layout.v1", "xyz_map": r.xyz},
               "state_leaves": [{"name": "hx", "mapping": {"kind": "field", "component": 2},
                                 "world_shape": list(SHAPES[0])}]}
    after = restore_world_leaves({"hx": tensor(value, r)}, fixture, np.empty((0, 5), np.int32))
    np.testing.assert_array_equal(after["hx"], value)
    fixture["state_leaves"][0]["world_shape"][0] += 1
    with pytest.raises(ValueError, match="Restored world shape"):
        restore_world_leaves({"hx": tensor(value, r)}, fixture, np.empty((0, 5), np.int32))


def test_rotation_preserves_order_sensitive_quarter_weight_interpolation():
    r = cyclic(2)
    shape = SHAPES[0]
    world = np.zeros(shape, np.float32)
    positions = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2)]
    for position, value in zip(positions, [4e8, -4e8, 13, 0], strict=True):
        world[position] = value
    ids = np.full((1, 6, 1, 4), -1, np.int32)
    ids[0, 3, 0] = [np.ravel_multi_index(p, shape) for p in positions]
    transformed = monitor_indices(ids, SHAPES, r)
    internal = tensor(world, r)
    gathered = internal.reshape(-1)[transformed[0, component_map(r)[3], 0]]
    result = np.float32(0)
    for value in gathered:
        result = np.float32(result + np.float32(value * np.float32(.25)))
    assert result == np.float32(3.25)
    reordered = gathered[np.argsort(transformed[0, component_map(r)[3], 0])]
    wrong = np.float32(0)
    for value in reordered:
        wrong = np.float32(wrong + np.float32(value * np.float32(.25)))
    assert wrong == 0
