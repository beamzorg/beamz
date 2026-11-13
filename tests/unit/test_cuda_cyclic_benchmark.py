"""The separate cyclic timing adapter must prove actual world readbacks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_cuda_cyclic_runtime as worker
from scripts.compare_cuda_cyclic_runtime import check_world_parity
from scripts.cuda_cyclic_fixture import leaf_mapping, transform_leaf
from scripts.cuda_fixture_store import ArrayReader, ArrayWriter
from scripts.export_fdtdx_comparison import file_digest
from scripts.prove_cuda_cyclic_axis_equivariance import cyclic


@pytest.mark.parametrize("corrupt", (False, True))
def test_actual_readbacks_are_inverse_compared_not_just_reference_metadata(tmp_path, monkeypatch, corrupt):
    r = cyclic(2)
    world_writer = ArrayWriter(tmp_path / "world-arrays")
    internal_writer = ArrayWriter(tmp_path / "internal-arrays")
    counts = np.asarray([[2, 3, 1, 0, 0]], np.int32)
    world_writer.put("counts", counts)
    leaves, mapped = [], []
    for i in range(30):
        mapping = ({"kind": "field", "component": i} if i < 6 else
                   {"kind": "output", "index": i} if i < 18 else
                   {"kind": "output", "index": 30 + i - 26} if i in (26, 27) else
                   {"kind": "unchanged"})
        shape = (3, 4, 5) if i < 18 else (36,) if i in (26, 27) else ()
        value = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + np.float32(i))
        key = f"reference_{i:02d}"
        leaf = {"name": f"world{i}", "mapping": mapping, "shape": list(shape), "dtype": "<f4", "reference_key": key}
        leaves.append(leaf)
        world_writer.put(key, value)
        transformed = transform_leaf(value, mapping, r, counts)
        internal_writer.put(key, transformed)
        mapped.append(dict(leaf, shape=list(transformed.shape), mapping=leaf_mapping(mapping, r)))
    world_manifest = world_writer.finish()
    internal_writer.finish()
    world = {"state_leaves": leaves, "data_format": "npy_directory_v1", "data_directory": "world-arrays",
             "data_file": str(world_manifest.relative_to(tmp_path)), "data_sha256": file_digest(world_manifest),
             "inputs": [{"key": "unused"}] * 117 + [{"key": "counts"}]}
    path = tmp_path / "world.json"
    path.write_text(json.dumps(world))
    layout = {"schema": "beamz.cyclic-storage-layout.v1", "xyz_map": r.xyz,
              "world_fixture": str(path), "world_fixture_sha256": file_digest(path)}
    internal = {"state_leaves": mapped, "cyclic_layout": layout}
    store = ArrayReader(internal_writer.directory)

    def fake_native(args, fixture, arrays, inputs, chunks):
        for repeat in range(2):
            for index, leaf in enumerate(fixture["state_leaves"]):
                expected = arrays[leaf["reference_key"]]
                actual = np.array(expected, copy=True)
                if corrupt and index == 0:
                    actual.reshape(-1)[0] += np.float32(1)
                worker.base.compare_arrays(actual, expected)
        return {"status": "ok"}

    monkeypatch.setattr(worker, "ORIGINAL_NATIVE", fake_native)
    module = tmp_path / "candidate.so"
    result = worker.native(SimpleNamespace(method="moving", module=module), internal, store, [], [16])
    assert result["module"] == str(module.resolve())
    assert result["world_parity"]["passed"] is not corrupt
    assert result["world_repeat_parity"]["passed"] is not corrupt
    assert result["status"] == ("failed_world_inverse_parity" if corrupt else "ok")
    assert result["world_inverse_and_comparison_outside_timed_solve"]
    if corrupt:
        with pytest.raises(ValueError, match="Actual first/repeat"):
            check_world_parity(result, world)
    else:
        check_world_parity(result, world)


def test_absent_world_readback_evidence_cannot_inherit_internal_parity():
    with pytest.raises(ValueError, match="Actual first/repeat"):
        check_world_parity({"parity": {"passed": True}}, {"state_leaves": []})
