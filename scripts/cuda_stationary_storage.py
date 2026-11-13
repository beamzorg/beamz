"""Place immutable material IDs beside a moving cache's retained physical cells.

This is a compiler placement/proof pass, not a kernel dispatch or a DRAM model.
Native Layout1 byte IDs are exact encodings of a codebook selection, not a
quantized replacement coefficient. Three E IDs share one private uint32 slot.
The ordinary FP32 codebook read and every arithmetic rounding point remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import IllegalSchedule, validate_operator, yee_operator
from scripts.cuda_stencil_trace import coefficient_reads, serialize_updates


@dataclass(frozen=True)
class ByteMaterialFacts:
    """Backend class facts; the actual launch must verify these predicates."""

    layout: int = 1
    immutable: bool = True
    index_bits: int = 8
    physical_component_layouts: bool = True

    def verify(self):
        if self.layout != 1 or self.index_bits != 8:
            raise IllegalSchedule("Private byte-ID placement admits Layout1 only")
        if not self.immutable or not self.physical_component_layouts:
            raise IllegalSchedule("Immutable, individually indexed E supports required")


def yee_support_mask(point, minimum_support):
    """Exact six-component physical support mask under canonical staggering."""
    if any(p < 0 or p > size for p, size in zip(point, minimum_support, strict=True)):
        return 0
    iz, iy, ix = (p < size for p, size in zip(point, minimum_support, strict=True))
    return int(iz and iy) | (int(iz and ix) << 1) | (int(iy and ix) << 2) | (int(ix) << 3) | (int(iy) << 4) | (int(iz) << 5)


def cache_support_certificate(origin, owned, depth, minimum_support):
    """Prove the existing native bulk predicate covers startup/import operands.

    Field shapes may differ by Yee staggering. Their componentwise minimum
    support is sufficient; padding changes addresses, not physical membership.
    This certificate covers physical bounds/native difference branches only,
    not CPML/source/observer exclusion or immutable metadata ownership.
    """
    if (any(len(v) != 3 for v in (origin, owned, minimum_support))
            or any(type(x) is not int for v in (origin, owned, minimum_support) for x in v)
            or type(depth) is not int or depth < 1 or min(owned) < 1 or min(minimum_support) < 1):
        raise IllegalSchedule("Integer 3D supports/origins and positive depth/owned shape required")
    cache_low = tuple(o - depth - 1 for o in origin)
    cache_high = tuple(o + b - 1 for o, b in zip(origin, owned, strict=True))
    admitted = all(low >= 0 and high < size for low, high, size
                   in zip(cache_low, cache_high, minimum_support, strict=True))
    return {"admitted": admitted, "origin_zyx": tuple(origin), "owned_zyx": tuple(owned),
            "depth": depth, "minimum_support_zyx": tuple(minimum_support),
            "all_cache_versions_inclusive_low": cache_low,
            "all_cache_versions_inclusive_high": cache_high,
            "native_predicate": "origin-depth>0 AND origin+owned-1<minimumSupport on all axes",
            "covers": ["startup v0", "all current low imports", "H positive neighbor",
                       "E negative neighbor", "native difference wall branches"],
            "excludes": ["CPML absence", "sources", "observers", "metadata immutability"]}


def material_placement(operator=None, *, facts=None,
                       cache=(6, 16, 32), threads=512, depth=16):
    """Attach checked lowering metadata without changing any arithmetic node."""
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    facts = ByteMaterialFacts() if facts is None else facts
    facts.verify()
    validate_operator(operator)
    if (len(cache) != 3 or any(type(x) is not int or x < 2 for x in cache)
            or type(threads) is not int or threads < 1 or math.prod(cache) % threads
            or type(depth) is not int or depth < 1):
        raise IllegalSchedule("Positive cache, integral fixed slots, and positive depth required")
    required = {"source_E" + c for c in "xyz"}
    reads = [r for update in operator.updates for r in coefficient_reads(update.value)
             if r.name in required]
    if {r.name for r in reads} != required:
        raise IllegalSchedule("All three canonical E material leaves required")
    if any(r.axes != (0, 1, 2) or r.offset != (0, 0, 0) or r.dtype != "float32" for r in reads):
        raise IllegalSchedule("Material IDs require point-local FP32 physical-coordinate reads")
    volume = math.prod(cache)
    owned = math.prod(c - 1 for c in cache)
    low_union = volume - owned
    slots = volume // threads
    report = {
        "operator_sha256": hashlib.sha256(json.dumps(serialize_updates(operator.updates), sort_keys=True).encode()).hexdigest(),
        "coefficients": sorted(required), "layout": 1,
        "storage": "same-thread-fixed-register", "encoding": "Ex | Ey<<8 | Ez<<16",
        "private_uint32_per_thread": slots,
        "cache_zyx": cache, "owned_zyx": tuple(c - 1 for c in cache),
        "threads": threads, "depth": depth,
        "refresh": "startup all cache points; thereafter ONLY recycled low-face union",
        "retained": "unchanged physical coordinate => unchanged immutable ID",
        "consumer": "E update only after actual component support predicate",
        "physical_index": "separate logical E component support/pitches for each byte",
        "invalid_code": "code >= table_size produces exact +0.0f, matching native decoder",
        "initial_vector_id_loads": volume,
        "step_vector_id_loads": low_union,
        "total_vector_id_loads": volume + depth * low_union,
        "uncached_vector_id_loads": depth * owned,
        "codebook_float_loads_per_useful_update": 3,
        "pure_low_x_stage_extra_bytes": (cache[0] - 1) * (cache[1] - 1) * 4,
        "limits": "Counts are logical ID loads, not transactions/DRAM bytes; register allocation unmeasured",
    }
    report["placement_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return report


def pack_ids(codes):
    if len(codes) != 3 or any(type(c) is not int or not 0 <= c < 256 for c in codes):
        raise ValueError("Exactly three unsigned byte IDs required")
    return sum(c << (8 * component) for component, c in enumerate(codes))


def decode_bits(word, component, table_bits):
    """Bit-preserving FP32 lookup, including signed zero and NaN payloads."""
    if component not in (0, 1, 2) or not 0 <= word < 2**32:
        raise ValueError("Unsigned packed word and component0..2 required")
    code = (word >> (8 * component)) & 255
    return table_bits[code] if code < len(table_bits) else 0


def prove_stationary_cache(cache=(6, 16, 32), depth=16, *, miss_axis=None,
                           refresh_old_coordinate=False):
    """Track exact physical-coordinate identity of every reused circular slot."""
    if len(cache) != 3 or any(type(c) is not int or c < 2 for c in cache):
        raise ValueError("Three cache dimensions >=2 required")
    dims = tuple(cache)
    slots = [(z, y, x) for z in range(dims[0]) for y in range(dims[1]) for x in range(dims[2])]
    origin = tuple(3 * c for c in dims)  # Translation does not change the proof.

    def physical(slot, n):
        return tuple(o - n - 1 + ((s + n + 1) % c) for o, s, c in zip(origin, slot, dims, strict=True))

    values = {slot: physical(slot, 0) for slot in slots}
    refreshes = 0
    retained = 0
    for n in range(1, depth + 1):
        head = tuple((-n - 1) % c for c in dims)
        for slot in slots:
            recycled = any(s == h and axis != miss_axis for axis, (s, h) in enumerate(zip(slot, head, strict=True)))
            if recycled:
                values[slot] = physical(slot, n - int(refresh_old_coordinate))
                refreshes += 1
            else:
                retained += 1
            assert values[slot] == physical(slot, n), "Stationary ID refers to evicted physical cell"
        # Every owned E point is covered by the stronger complete cache test.
    assert refreshes == depth * (math.prod(dims) - math.prod(c - 1 for c in dims))
    return {"cache_zyx": dims, "depth": depth, "refreshed_vector_ids": refreshes,
            "retained_vector_ids": retained, "result": "pass"}


def audit_fixture(path):
    """CPU-only exact-bit audit of actual frozen Layout1 native material data."""
    import numpy as np

    path = Path(path)
    manifest = json.loads(path.read_text())
    data_path = path.parent / manifest["data_file"]
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["data_sha256"]:
        raise IllegalSchedule("Fixture archive hash mismatch")
    inputs = {entry["index"]: entry for entry in manifest["inputs"]}
    tables, packed, shapes = [], [], []
    evidence = []
    with np.load(data_path, allow_pickle=False) as data:
        for c in range(3):
            arrays = []
            for index, dtype in ((37 + c, np.dtype("float32")), (40 + c, np.dtype("int32"))):
                entry = inputs[index]
                array = data[entry["key"]]
                if array.dtype != dtype or array.ndim != 1 or list(array.shape) != entry["shape"]:
                    raise IllegalSchedule("Fixture is not canonical Layout1")
                digest_state = hashlib.sha256()
                digest_state.update(array.dtype.str.encode())
                digest_state.update(json.dumps(list(array.shape)).encode())
                digest_state.update(array.tobytes())
                digest = digest_state.hexdigest()
                if digest != entry["sha256"]:
                    raise IllegalSchedule("Fixture material array hash mismatch")
                arrays.append(array.view(np.uint32))
                evidence.append({"input": index, "sha256": digest})
            shape = tuple(inputs[3 + c]["shape"])
            if len(shape) != 3 or arrays[0].size < 1 or arrays[1].size < (math.prod(shape) + 3) // 4:
                raise IllegalSchedule("Fixture material shape/code capacity mismatch")
            tables.append(arrays[0])
            packed.append(arrays[1])
            shapes.append(shape)
    maximum = tuple(max(s[a] for s in shapes) for a in range(3))
    points = {(z, y, x) for z in (-1, 0, maximum[0] - 2, maximum[0] - 1, maximum[0])
              for y in (-1, 0, maximum[1] - 2, maximum[1] - 1, maximum[1])
              for x in (-1, 0, maximum[2] - 2, maximum[2] - 1, maximum[2])}
    rng = np.random.default_rng(815183)
    points.update(map(tuple, rng.integers(0, maximum, size=(4096, 3)).tolist()))
    comparisons = 0
    for point in sorted(points):
        codes, wanted = [], []
        for c in range(3):
            if any(p < 0 or p >= size for p, size in zip(point, shapes[c], strict=True)):
                codes.append(0)
                wanted.append(None)  # Native Valid prevents this consumer.
                continue
            z, y, x = point
            linear = (z * shapes[c][1] + y) * shapes[c][2] + x
            native_word = int(packed[c][linear >> 2])
            code = (native_word >> (8 * (linear & 3))) & 255
            codes.append(code)
            wanted.append(int(tables[c][code]) if code < len(tables[c]) else 0)
        cached = pack_ids(tuple(codes))
        for c in range(3):
            if wanted[c] is not None:
                assert decode_bits(cached, c, tables[c]) == wanted[c]
                comparisons += 1
    return {"fixture": str(path), "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_sha256": manifest["data_sha256"], "array_evidence": evidence,
            "component_shapes_zyx": shapes, "table_sizes": [len(t) for t in tables],
            "sampled_physical_points": len(points), "bitwise_coefficient_checks": comparisons,
            "result": "pass", "scope": "Actual immutable Layout1 data; no GPU or timed execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Optional frozen native JSON+NPZ Layout1 material audit")
    args = parser.parse_args()
    cases = [prove_stationary_cache(cache, depth) for cache in ((6, 16, 32), (4, 8, 16), (3, 5, 7))
             for depth in (4, 16, 64)]
    negatives = []
    for name, options in (("missing low Z refresh", {"miss_axis": 0}),
                          ("missing low Y refresh", {"miss_axis": 1}),
                          ("missing low X refresh", {"miss_axis": 2}),
                          ("stale pre-roll coordinate", {"refresh_old_coordinate": True})):
        try:
            prove_stationary_cache(**options)
        except AssertionError:
            negatives.append(name)
        else:
            raise AssertionError(f"Negative control escaped: {name}")
    result = {"placement": material_placement(), "cases": cases,
              "negative_controls_rejected": negatives}
    if args.fixture:
        result["fixture_audit"] = audit_fixture(args.fixture)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
