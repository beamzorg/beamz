"""Immutable byte-ID placement, liveness, and exact lookup semantics."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_stationary_storage import (
    ByteMaterialFacts,
    audit_fixture,
    cache_support_certificate,
    decode_bits,
    material_placement,
    pack_ids,
    prove_stationary_cache,
    yee_support_mask,
)
from scripts.cuda_stencil_trace import Expr


def test_stationary_material_lifetime_and_logical_resource_counts():
    report = material_placement()
    assert report == material_placement()
    assert report["private_uint32_per_thread"] == 6
    assert report["initial_vector_id_loads"] == 3072
    assert report["step_vector_id_loads"] == 747
    assert report["uncached_vector_id_loads"] == 37200
    assert report["total_vector_id_loads"] == 15024
    assert report["pure_low_x_stage_extra_bytes"] == 300


@pytest.mark.parametrize("facts", (
    ByteMaterialFacts(layout=2), ByteMaterialFacts(immutable=False),
    ByteMaterialFacts(index_bits=16), ByteMaterialFacts(physical_component_layouts=False),
))
def test_byte_id_backend_admission_rejects_incompatible_encodings(facts):
    with pytest.raises(IllegalSchedule):
        material_placement(facts=facts)


def test_arbitrary_spatial_coefficient_offset_is_not_silently_cached_at_center():
    operator = yee_operator(cpml_axes=(0, 1, 2))

    def shifted(value):
        if value.op == "coefficient" and value.args[0].name == "source_Ex":
            return Expr("coefficient", (replace(value.args[0], offset=(0, 0, 1)),))
        return Expr(value.op, tuple(shifted(x) if isinstance(x, Expr) else x for x in value.args))

    operator = replace(operator, updates=tuple(replace(u, value=shifted(u.value)) for u in operator.updates))
    with pytest.raises(IllegalSchedule, match="point-local"):
        material_placement(operator)


def test_byte_lookup_is_exact_for_every_code_signed_zero_and_nan_payload():
    # Integers represent raw FP32 bits. Lookup never converts/reassociates them.
    table = [0x80000000, 0x7FC12345, 0x3F800000] + list(range(3, 256))
    for code in range(256):
        word = pack_ids((code, (code + 71) % 256, (code + 193) % 256))
        for c, offset in enumerate((0, 71, 193)):
            assert decode_bits(word, c, table) == table[(code + offset) % 256]
            expected = table[(code + offset) % 256] if (code + offset) % 256 < 3 else 0
            assert decode_bits(word, c, table[:3]) == expected
    with pytest.raises(ValueError):
        pack_ids((0, 1, 256))


@pytest.mark.parametrize("cache", ((6, 16, 32), (4, 8, 16), (3, 5, 7)))
def test_circular_retention_including_non_power_of_two_axes(cache):
    assert prove_stationary_cache(cache, depth=64)["result"] == "pass"


@pytest.mark.parametrize("options", (
    {"miss_axis": 0}, {"miss_axis": 1}, {"miss_axis": 2},
    {"refresh_old_coordinate": True},
))
def test_rejects_omitted_face_refresh_and_pre_roll_coordinate(options):
    with pytest.raises(AssertionError):
        prove_stationary_cache(**options)


def test_actual_native_component_layout_audit_and_hash_binding(tmp_path):
    arrays, inputs = {}, []
    shapes = ((3, 3, 2), (3, 2, 3), (2, 3, 3))
    for c, shape in enumerate(shapes):
        inputs.append({"index": 3 + c, "shape": list(shape)})
        table = np.array([0x80000000, 0x7FC12345, 0x3F800000], dtype=np.uint32).view(np.float32)
        codes = np.array([sum(((4 * word + byte + c) % 256) << (8 * byte) for byte in range(4))
                          for word in range((np.prod(shape) + 3) // 4)], dtype=np.uint32).view(np.int32)
        for index, array in ((37 + c, table), (40 + c, codes)):
            key = f"input_{index:03d}"
            arrays[key] = array
            digest = hashlib.sha256(array.dtype.str.encode() + json.dumps(list(array.shape)).encode() + array.tobytes()).hexdigest()
            inputs.append({"index": index, "key": key, "shape": list(array.shape), "sha256": digest})
    data_path = tmp_path / "data.npz"
    np.savez(data_path, **arrays)
    manifest = {"inputs": inputs, "data_file": data_path.name,
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(manifest))
    assert audit_fixture(path)["bitwise_coefficient_checks"] == 54
    arrays["input_040"][0] ^= 1
    np.savez(data_path, **arrays)
    with pytest.raises(IllegalSchedule, match="archive hash"):
        audit_fixture(path)
    manifest["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(IllegalSchedule, match="array hash"):
        audit_fixture(path)


@pytest.mark.parametrize("depth", (4, 16, 64))
def test_native_bulk_admission_covers_every_real_cache_phase(depth):
    cache = (6, 16, 32)
    owned = tuple(c - 1 for c in cache)
    origin = (depth + 1, depth + 3, depth + 7)
    minimum = tuple(o + c for o, c in zip(origin, cache, strict=True))
    certificate = cache_support_certificate(origin, owned, depth, minimum)
    assert certificate["admitted"]
    for n in range(depth + 1):
        for z in range(cache[0]):
            for y in range(cache[1]):
                for x in range(cache[2]):
                    # Independent actual circular head/slot mapping.
                    point = tuple(o - n - 1 + ((slot - ((-n - 1) % c)) % c)
                                  for o, slot, c in zip(origin, (z, y, x), cache, strict=True))
                    assert all(0 <= p < size for p, size in zip(point, minimum, strict=True))
                    assert yee_support_mask(point, minimum) == 63
    # Off-by-one weakened bounds would admit an invalid imported low or v0 high.
    assert not cache_support_certificate((depth,) + origin[1:], owned, depth, minimum)["admitted"]
    weak_high = (origin[0] + owned[0] - 1,) + minimum[1:]
    assert not cache_support_certificate(origin, owned, depth, weak_high)["admitted"]


@pytest.mark.parametrize("minimum", ((2, 3, 5), (3, 2, 4)))
def test_common_support_mask_matches_each_actual_staggered_component(minimum):
    nz, ny, nx = minimum
    shapes = ((nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx),
              (nz + 1, ny + 1, nx), (nz + 1, ny, nx + 1), (nz, ny + 1, nx + 1))
    for z in range(-1, nz + 3):
        for y in range(-1, ny + 3):
            for x in range(-1, nx + 3):
                direct = sum(int(0 <= z < sz and 0 <= y < sy and 0 <= x < sx) << c
                             for c, (sz, sy, sx) in enumerate(shapes))
                assert yee_support_mask((z, y, x), minimum) == direct
