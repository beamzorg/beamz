"""Exact storage-coordinate transforms for immutable native CUDA fixtures.

This module never constructs physical coefficients or interpolation weights.
It permutes captured bytes and addresses, preserving source ID order and each
monitor's point/frequency/neighbor reduction order.  The native moving runtime
and ordinary in-place tail do not consume the unchanged legacy D2 plan/scratch.
It is not an admission for the legacy Mstep execution method.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_fixture_store import ArrayReader, ArrayWriter, digest_array, drop_pages
from scripts.cuda_native_geometry_ir import verify_component_shapes
from scripts.export_fdtdx_comparison import file_digest
from scripts.prove_cuda_cyclic_axis_equivariance import AxisMap, cyclic


def axes(rotation):
    if not isinstance(rotation, AxisMap) or rotation.odd:
        raise ValueError("Proper cyclic layout required")
    return tuple(rotation.zyx.index(a) for a in range(3))


def tensor(value, rotation):
    """Return a view; the writer performs the only contiguous materialization."""
    if value.ndim != 3:
        raise ValueError("Spatial tensor must have three axes")
    return value.transpose(axes(rotation))


def component_map(rotation):
    return tuple(3 * (c // 3) + rotation.xyz[c % 3] for c in range(6))


def component_axis(value, rotation):
    if value.ndim < 2 or value.shape[1] != 6:
        raise ValueError("Packed monitor component axis must have extent six")
    return value[:, np.argsort(component_map(rotation)), ...]


def byte_codes(value, shape, rotation):
    """Transpose uint8 identities; preserve unused tail bytes as well."""
    if value.dtype != np.int32 or value.ndim != 1:
        raise ValueError("Only byte-packed E codes are admitted")
    count = int(np.prod(shape))
    if value.size != (count + 3) // 4:
        raise ValueError("Packed code count differs from field support")
    raw = value.view(np.uint8)
    rotated = np.empty_like(raw)
    rotated[:count] = tensor(raw[:count].reshape(shape), rotation).reshape(-1)
    rotated[count:] = raw[count:]
    return rotated.view(np.int32)


def monitor_indices(value, shapes, rotation):
    if value.dtype != np.int32 or value.ndim != 4 or value.shape[1] != 6:
        raise ValueError("Packed monitor indices[M,6,P,K] required")
    out = np.empty_like(value)
    mapping = component_map(rotation)
    # Monitor component order E,H; native field component order H,E.
    for old in range(6):
        before = value[:, old]
        shape = tuple(shapes[(old + 3) % 6])
        valid = (before >= 0) & (before < np.prod(shape))
        after = np.full(before.shape, -1, np.int32)
        z, rest = np.divmod(before[valid], shape[1] * shape[2])
        y, x = np.divmod(rest, shape[2])
        coords = rotation.vector((z, y, x))
        target = rotation.vector(shape)
        after[valid] = (coords[0] * target[1] + coords[1]) * target[2] + coords[2]
        # Invalid nonnegative values are semantically absent. Preserve their bits
        # when possible, but use -1 when the new support might admit the value.
        invalid = ~valid
        after[invalid] = before[invalid]
        out[:, mapping[old]] = after
    return out


def dft_components(value, counts, rotation):
    if value.ndim != 1 or counts.ndim != 2 or counts.shape[1] != 5:
        raise ValueError("Flat DFT values and counts[M,5] required")
    out = np.array(value, copy=True)
    occupied = []
    for nf, points, interval, start, weight in counts:
        nf, points, start = int(nf), int(points), int(start)
        count = 6 * nf * points
        if nf < 0 or points < 0 or start < 0 or start + count > value.size:
            raise ValueError("DFT monitor range exceeds captured values")
        if any(start < b and a < start + count for a, b in occupied):
            raise ValueError("Aliased DFT ranges are not admitted")
        occupied.append((start, start + count))
        # Transpose component blocks, never the frequency or point axes.
        source = value[start : start + count].reshape(6, nf, points)
        dest = out[start : start + count].reshape(6, nf, points)
        dest[np.asarray(component_map(rotation))] = source
    return out


def metadata(value, rotation):
    if value.dtype != np.int32 or value.shape != (6, 5):
        raise ValueError("Canonical CPML metadata[6,5] required")
    canonical = np.asarray([[0, 1, 12, 12, 1], [0, 0, 12, 12, -1],
                            [1, 0, 12, 12, 1], [1, 2, 12, 12, -1],
                            [2, 2, 12, 12, 1], [2, 1, 12, 12, -1]], np.int32)
    if not np.array_equal(value, canonical):
        raise ValueError("Canonical ordered CPML12 term metadata required")
    out = np.empty_like(value)
    for term in range(6):
        row = np.array(value[term], copy=True)
        row[0] = rotation.xyz[int(row[0])]
        row[1] = rotation.zyx[int(row[1])]
        out[rotation.term(term)] = row
    return out


def input_mapping(index, rotation):
    if index < 6:
        return component_map(rotation)[index]
    if 6 <= index < 12:
        return 6 + 3 * ((index - 6) // 3) + rotation.xyz[(index - 6) % 3]
    for base in (6, 37):
        if index == base + 6:
            return index
        if base + 7 <= index < base + 25:
            term, kind = divmod(index - base - 7, 3)
            return base + 7 + 3 * rotation.term(term) + kind
        if base + 25 <= index < base + 31:
            return base + 25 + rotation.term(index - base - 25)
    if 37 <= index < 43:
        return 37 + 3 * ((index - 37) // 3) + rotation.xyz[(index - 37) % 3]
    if 68 <= index < 74:
        return 68 + 3 * ((index - 68) // 3) + rotation.zyx[(index - 68) % 3]
    if 74 <= index < 86:
        return 74 + 6 * ((index - 74) // 6) + component_map(rotation)[(index - 74) % 6]
    if 86 <= index < 113:
        group, kind = divmod(index - 86, 3)
        return 86 + 3 * rotation.source_group(group) + kind
    return index


def transform_input(index, value, rotation, shapes, counts):
    if index < 6 or 74 <= index < 80 or 31 <= index < 37 or 62 <= index < 68:
        return tensor(value, rotation)
    if index in (12, 43):
        return metadata(value, rotation)
    if 13 <= index < 31 or 44 <= index < 62:
        return tensor(value, rotation)
    if 40 <= index < 43:
        return byte_codes(value, shapes[index - 37], rotation)
    if 86 <= index < 113:
        kind = (index - 86) % 3
        if kind == 0:
            return value.transpose((0, *(a + 1 for a in axes(rotation))))
        if kind == 2:
            return value[:, axes(rotation)]
    if index == 113:
        return monitor_indices(value, shapes, rotation)
    if index in (114, 116):
        return component_axis(value, rotation)
    if index in (120, 121):
        return dft_components(value, counts, rotation)
    return value


def transform_leaf(value, mapping, rotation, counts):
    if mapping["kind"] == "field" or (
        mapping["kind"] == "output" and 6 <= mapping["index"] < 18
    ):
        return tensor(value, rotation)
    if mapping == {"kind": "output", "index": 30} or mapping == {"kind": "output", "index": 31}:
        return dft_components(value, counts, rotation)
    return value


def leaf_mapping(mapping, rotation):
    result = dict(mapping)
    if result["kind"] == "field":
        result["component"] = component_map(rotation)[result["component"]]
    if result["kind"] == "output" and 6 <= result["index"] < 18:
        family, term = divmod(result["index"] - 6, 6)
        result["index"] = 6 + 6 * family + rotation.term(term)
    return result


def rotate_packet(inputs, attributes, steps=2):
    """Transform an already captured small packet for actual CUDA validation."""
    if len(inputs) != 134:
        raise ValueError("Complete native CPML packet required")
    rotation = cyclic(steps)
    shapes = tuple(tuple(a.shape) for a in inputs[:6])
    verify_component_shapes(shapes)
    if (attributes["metric_kind"] != 0 or attributes["boundary_code"] != 12 << 8
            or attributes["cuda_flags"] != 128):
        raise ValueError("Initial cyclic layout admits Cartesian, unpadded CPML12 only")
    result = [None] * 134
    for index, value in enumerate(inputs):
        value = transform_input(index, value, rotation, shapes, inputs[117])
        result[input_mapping(index, rotation)] = np.array(value, copy=True, order="C")
    attrs = dict(attributes)
    for name in ("coincident_source_group_mask", "disjoint_source_group_mask"):
        mask = attributes.get(name, 0)
        attrs[name] = sum(1 << rotation.source_group(g) for g in range(9) if mask & (1 << g))
    logical = rotation.vector(tuple(attributes.get("logical_" + a, 0) for a in "zyx"))
    attrs.update({"logical_" + a: n for a, n in zip("zyx", logical, strict=True)})
    return result, attrs


def transform_fixture(path, output, steps=2):
    started = time.monotonic()
    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists() or (output.parent / (output.stem + ".arrays")).exists():
        raise ValueError("A new immutable output fixture is required")
    rotation = cyclic(steps)
    source = json.loads(path.read_text())
    if source["schema"] != "beamz.cuda-native-fixture.v1" or len(source["inputs"]) != 134:
        raise ValueError("Complete native CPML ABI required")
    attributes = source["attributes"]
    if (attributes["metric_kind"] != 0 or attributes["boundary_code"] != 12 << 8
            or attributes["cuda_flags"] != 128):
        raise ValueError("Initial cyclic layout admits Cartesian, unpadded CPML12 only")
    if file_digest(path.parent / source["data_file"]) != source["data_sha256"]:
        raise ValueError("Source data manifest hash differs")
    store = ArrayReader(path.parent / source["data_directory"]) if source.get("data_format") == "npy_directory_v1" else np.load(path.parent / source["data_file"], allow_pickle=False)
    shapes = tuple(tuple(i["shape"]) for i in source["inputs"][:6])
    verify_component_shapes(shapes)
    if any(len(source["inputs"][i]["shape"]) != 1 for i in (40, 41, 42)):
        raise ValueError("Initial cyclic layout admits byte-packed E only")
    f = copy.deepcopy(source)
    writer = ArrayWriter(output.parent / (output.stem + ".arrays"))
    counts = np.array(store[source["inputs"][117]["key"]], copy=True)
    checks = []
    seen = set()
    inverse = rotation.inverse()
    target_shapes = [None] * 6
    for old, shape in enumerate(shapes):
        target_shapes[component_map(rotation)[old]] = rotation.vector(shape)
    verify_component_shapes(target_shapes)
    for item in source["inputs"]:
        index = item["index"]
        value = store[item["key"]]
        if digest_array(value) != item["sha256"]:
            raise ValueError(f"Input digest differs: {item['key']}")
        target = input_mapping(index, rotation)
        key = source["inputs"][target]["key"]
        transformed = transform_input(index, value, rotation, shapes, counts)
        entry = writer.put(key, transformed)
        seen.add(key)
        f["inputs"][target].update(shape=entry["shape"], dtype=entry["dtype"], sha256=entry["sha256"])
        restored = transform_input(target, transformed, inverse, target_shapes, counts)
        if digest_array(restored) != item["sha256"]:
            raise ValueError(f"Input inverse differs: {item['key']}")
        checks.append({"world_input": index, "internal_input": target, "inverse_bitwise_exact": True})
        del transformed, restored
        drop_pages(value)
    for old, leaf in enumerate(source["state_leaves"]):
        mapping = leaf["mapping"]
        f["state_leaves"][old]["mapping"] = leaf_mapping(mapping, rotation)
        f["state_leaves"][old]["world_shape"] = leaf["shape"]
        f["state_leaves"][old]["world_mapping"] = mapping
        keys = [leaf["initial_key"], leaf["reference_key"]]
        continuation = f"continuation_{old:02d}"
        if source.get("continuation_steps"):
            keys.append(continuation)
        for key in keys:
            value = store[key]
            if isinstance(store, ArrayReader) and digest_array(value) != store.metadata[key]["sha256"]:
                raise ValueError(f"World reference digest differs: {key}")
            transformed = transform_leaf(value, mapping, rotation, counts)
            restored = transform_leaf(transformed, leaf_mapping(mapping, rotation), inverse, counts)
            if digest_array(value) != digest_array(restored):
                raise ValueError(f"World state inverse differs: {key}")
            entry = writer.put(key, transformed)
            if key == leaf["reference_key"]:
                f["state_leaves"][old]["shape"] = entry["shape"]
            seen.add(key)
            del transformed, restored
            drop_pages(value)
    # Preserve any auxiliary arrays without interpreting their semantics.
    names = store.metadata if isinstance(store, ArrayReader) else store.files
    for key in names:
        if key not in seen:
            value = store[key]
            writer.put(key, value)
            drop_pages(value)
    for name in ("coincident_source_group_mask", "disjoint_source_group_mask"):
        mask = attributes.get(name, 0)
        f["attributes"][name] = sum(1 << rotation.source_group(g) for g in range(9) if mask & (1 << g))
    logical = rotation.vector(tuple(attributes.get("logical_" + a, 0) for a in "zyx"))
    f["attributes"].update({"logical_" + a: n for a, n in zip("zyx", logical, strict=True)})
    for c in range(6):
        for bank in (0, 18):
            f["outputs"][bank + c]["shape"] = list(target_shapes[c])
    for family, offset in ((0, 31), (1, 62)):
        for term in range(6):
            f["outputs"][6 + 6 * family + term]["shape"] = f["inputs"][offset + term]["shape"]
    manifest = writer.finish()
    f.update(data_format="npy_directory_v1", data_directory=writer.directory.name,
             data_file=str(manifest.relative_to(output.parent)), data_sha256=file_digest(manifest))
    f["cyclic_layout"] = {"schema": "beamz.cyclic-storage-layout.v1", "world_fixture": str(path),
        "world_fixture_sha256": file_digest(path), "xyz_map": rotation.xyz, "transpose_zyx": axes(rotation),
        "same_world_physics_and_ordered_arithmetic": True, "world_reference_inverse_all30": True,
        "monitor_point_neighbor_and_frequency_orders_preserved": True,
        "monitor_component_axis_permuted_with_dft_and_inverse": True,
        "legacy_D2_plan_and_scratch_unchanged_unused": True,
        "only_admitted_execution": ["native_moving", "ordinary_inplace_program"],
        "producer_sha256": file_digest(__file__), "input_inverse_checks": checks,
        "setup_seconds": time.monotonic() - started}
    output.write_text(json.dumps(f, indent=2) + "\n")
    return f["cyclic_layout"]


def restore_world_leaves(state, fixture, counts):
    """Inverse-map actual downloaded logical leaves, retaining world names.

    `state` is the dictionary returned by logical_native_state. A caller may
    compare these arrays directly with the original world's independent oracle.
    """
    layout = fixture.get("cyclic_layout", {})
    if layout.get("schema") != "beamz.cyclic-storage-layout.v1":
        raise ValueError("A bound cyclic storage fixture is required")
    rotation = AxisMap(tuple(layout["xyz_map"]))
    inverse = rotation.inverse()
    result = {}
    for leaf in fixture["state_leaves"]:
        value = transform_leaf(state[leaf["name"]], leaf["mapping"], inverse, counts)
        if list(value.shape) != leaf["world_shape"]:
            raise ValueError(f"Restored world shape differs: {leaf['name']}")
        result[leaf["name"]] = value
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, choices=(0, 1, 2), default=2)
    a = p.parse_args()
    print(json.dumps(transform_fixture(a.fixture, a.output, a.steps), indent=2))


if __name__ == "__main__":
    main()
